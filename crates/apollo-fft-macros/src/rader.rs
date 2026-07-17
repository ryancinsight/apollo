use proc_macro::TokenStream as CompilerTokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, LitInt, Result};

use crate::math::{find_primitive_root, mod_inverse_isize, mod_pow, ComplexF64};

struct RaderInput {
    p: LitInt,
}

impl Parse for RaderInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let p = input.parse()?;
        Ok(Self { p })
    }
}

/// Compute `B[k] = (1/M) · Σ_{q=0}^{M-1} b[q] · exp(-2πi·k·q/M)` using the
/// naive O(M²) DFT.  Called at proc-macro expansion time, not at runtime.
fn naive_dft_scaled(b: &[ComplexF64], scale: f64) -> Vec<ComplexF64> {
    let m = b.len();
    let mut out = vec![ComplexF64::zero(); m];
    for (k, out_k) in out.iter_mut().enumerate() {
        let mut acc = ComplexF64::zero();
        for (q, &bq) in b.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * (k * q) as f64 / m as f64;
            let w = ComplexF64::new(angle.cos(), angle.sin());
            acc = acc + bq * w;
        }
        *out_k = acc * ComplexF64::new(scale, 0.0);
    }
    out
}

pub fn generate_rader_fft(input: CompilerTokenStream) -> CompilerTokenStream {
    let input_ast = parse_macro_input!(input as RaderInput);
    let p = input_ast.p.base10_parse::<usize>().unwrap();
    let p_minus_1 = p - 1;

    let g = find_primitive_root(p);
    let g_inv = mod_inverse_isize(g as isize, p as isize) as usize;

    // ── 1. GATHER / SCATTER permutation tables ───────────────────────────────
    let mut gather: Vec<usize> = Vec::with_capacity(p_minus_1);
    let mut scatter: Vec<usize> = Vec::with_capacity(p_minus_1);
    for q in 0..p_minus_1 {
        gather.push(mod_pow(g, q as u64, p as u64));
        scatter.push(mod_pow(g_inv, q as u64, p as u64));
    }

    // ── 2. Rader convolution kernel b and its scaled DFT ────────────────────
    // B_FWD[k] = DFT(b_fwd)[k] / (p-1)   b_fwd[q] = exp(-2πi·g^{-q}/p)
    // B_INV[k] = DFT(b_inv)[k] / (p-1)   b_inv[q] = exp(+2πi·g^{-q}/p)
    // 1/(p-1) folds the IFFT normalization into the constant — zero runtime division.
    let mut b_fwd = vec![ComplexF64::zero(); p_minus_1];
    let mut b_inv = vec![ComplexF64::zero(); p_minus_1];
    for q in 0..p_minus_1 {
        let exp_idx = scatter[q]; // g^{-q} mod p
        let angle_fwd = -2.0 * std::f64::consts::PI * exp_idx as f64 / p as f64;
        let angle_inv = 2.0 * std::f64::consts::PI * exp_idx as f64 / p as f64;
        b_fwd[q] = ComplexF64::new(angle_fwd.cos(), angle_fwd.sin());
        b_inv[q] = ComplexF64::new(angle_inv.cos(), angle_inv.sin());
    }

    let scale = 1.0 / p_minus_1 as f64;
    let b_fwd_dft = naive_dft_scaled(&b_fwd, scale);
    let b_inv_dft = naive_dft_scaled(&b_inv, scale);

    // ── 3. Emit const array tokens ───────────────────────────────────────────
    let gather_consts: Vec<_> = gather
        .iter()
        .map(|&v| {
            let lit = proc_macro2::Literal::usize_unsuffixed(v);
            quote! { #lit }
        })
        .collect();

    let scatter_consts: Vec<_> = scatter
        .iter()
        .map(|&v| {
            let lit = proc_macro2::Literal::usize_unsuffixed(v);
            quote! { #lit }
        })
        .collect();

    let b_fwd_entries: Vec<_> = b_fwd_dft
        .iter()
        .map(|c| {
            let re = proc_macro2::Literal::f64_unsuffixed(c.re);
            let im = proc_macro2::Literal::f64_unsuffixed(c.im);
            quote! { (#re, #im) }
        })
        .collect();

    let b_inv_entries: Vec<_> = b_inv_dft
        .iter()
        .map(|c| {
            let re = proc_macro2::Literal::f64_unsuffixed(c.re);
            let im = proc_macro2::Literal::f64_unsuffixed(c.im);
            quote! { (#re, #im) }
        })
        .collect();

    // ── 4. Identifier construction ───────────────────────────────────────────
    let fn_name = format_ident!("rader_fft_{}", p);
    let gather_name = format_ident!("GATHER_{}", p);
    let scatter_name = format_ident!("SCATTER_{}", p);
    let b_fwd_name = format_ident!("B_FWD_{}", p);
    let b_inv_name = format_ident!("B_INV_{}", p);
    let p_const = proc_macro2::Literal::usize_unsuffixed(p_minus_1);

    // ── 5. Emit ──────────────────────────────────────────────────────────────
    let result = quote! {
        // ── Gather / scatter permutation tables ─────────────────────────────
        const #gather_name: [usize; #p_const] = [#(#gather_consts),*];
        const #scatter_name: [usize; #p_const] = [#(#scatter_consts),*];

        // ── Rader convolution kernel (pre-DFT'd, pre-scaled by 1/(p-1)) ─────
        // B_FWD[k] = DFT(b_fwd)[k] / (p-1)  where  b_fwd[q] = exp(-2πi·g^{-q}/p)
        // B_INV[k] = DFT(b_inv)[k] / (p-1)  where  b_inv[q] = exp(+2πi·g^{-q}/p)
        const #b_fwd_name: [(f64, f64); #p_const] = [#(#b_fwd_entries),*];
        const #b_inv_name: [(f64, f64); #p_const] = [#(#b_inv_entries),*];

        // ── True Rader FFT kernel — zero dispatch overhead ───────────────────
        //
        // Algorithm (O((p-1) log(p-1)) per call):
        //   1. Gather:   rader_in[q] = data[GATHER[q]],  accumulate DC.
        //   2. Sub-FFT:  ShortDft<p-1>::dft::<false>     [forward, no dispatch layers]
        //   3. Convolve: rader_in[k] *= B[k]             [N-1 complex muls]
        //   4. Sub-IFFT: ShortDft<p-1>::dft::<true>      [inverse, no dispatch layers]
        //   5. Scatter:  data[SCATTER[q]] = dc + rader_in[q]
        //
        // Sub-FFTs call ShortDft<p-1>::dft directly — no pow2 check, no
        // short_winograd match, no coprime lookup, no fat-pointer slice overhead.
        // All sub-FFT sizes for static Rader primes live in SHORT_WINOGRAD_SIZES.
        #[inline]
        pub(crate) fn #fn_name<
            F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
                    Complex = eunomia::Complex<F>,
                >
                + crate::application::execution::kernel::components::winograd::ShortWinogradScalar,
            const INVERSE: bool,
        >(data: &mut [F::Complex])
        where
            F: crate::application::execution::kernel::mixed_radix::traits::ShortDft<#p_const>,
        {
            use eunomia::Complex;
            use crate::application::execution::kernel::mixed_radix::traits::ShortDft;

            // Step 1 — Gather + accumulate DC (data[0] = Σ inputs).
            let mut rader_in: [F::Complex; #p_const] =
                [Complex::new(<F as eunomia::NumericElement>::ZERO, <F as eunomia::NumericElement>::ZERO); #p_const];
            let v0 = data[0];
            let mut dc = v0;
            for i in 0..#p_const {
                let x = data[#gather_name[i]];
                rader_in[i] = x;
                dc += x;
            }
            data[0] = dc;

            // Step 2 — Forward sub-FFT: direct ShortDft<p-1> call, zero dispatch overhead.
            <F as ShortDft<#p_const>>::dft::<false>(&mut rader_in);

            // Step 3 — Pointwise multiply by the pre-DFT'd, pre-scaled kernel.
            // Kernel selected at compile time via const INVERSE; N-1 from_precise calls.
            let kernel: &[(f64, f64); #p_const] =
                if INVERSE { &#b_inv_name } else { &#b_fwd_name };
            for k in 0..#p_const {
                let (br, bi) = kernel[k];
                let br = F::from_precise(br);
                let bi = F::from_precise(bi);
                let x = rader_in[k];
                rader_in[k] = Complex::new(
                    br * x.re - bi * x.im,
                    br * x.im + bi * x.re,
                );
            }

            // Step 4 — Inverse sub-FFT (unnormalized; 1/(p-1) folded into kernel).
            <F as ShortDft<#p_const>>::dft::<true>(&mut rader_in);

            // Step 5 — Scatter: data[SCATTER[q]] = dc + rader_in[q].
            for q in 0..#p_const {
                data[#scatter_name[q]] = v0 + rader_in[q];
            }
        }
    };
    result.into()
}

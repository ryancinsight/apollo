//! Winograd prime-power DFT codelets via Rader's algorithm on `(Z/p²Z)*`.
//!
//! For `N = p²` where `p` is prime, uses a modified Rader-like algorithm:
//!
//! 1. Compute `y[r] = Σ_{q=0}^{p-1} x[q·p + r]` for `r ∈ [0, p)`.
//! 2. Compute `Y = DFT_p(y)`.
//! 3. Compute `z[m] = x[m·p]` for `m ∈ [1, p)` and `z[0] = 0`.
//! 4. Compute `Z = DFT_p(z)`.
//! 5. Compute the cyclic convolution `d` of length `φ(p²) = p(p-1)` of:
//!    - `a[s] = x[g^s mod p²]`
//!    - `h[s] = W_{p²}^{g^{-s}}` (forward) or `W_{p²}^{-g^{-s}}` (inverse)
//! 6. Assemble outputs:
//!    - `X[0] = Y[0]`
//!    - `X[k'·p] = Y[k']` for `k' ∈ [1, p)`
//!    - `X[g^{-b}] = x[0] + d[b] + Z[(g mod p)^{-b}]` for `b ∈ [0, φ)`

use quote::{format_ident, quote};
use std::f64::consts::PI;

use crate::math::{mod_pow, ComplexF64};

// ── Primitive root lifting ────────────────────────────────────────────────

/// Lift primitive root `g_mod_p` of `(Z/pZ)*` to a generator of `(Z/p²Z)*`.
/// Lifting theorem: test `g^{p-1} mod p²`; if ≡ 1, use `g + p`.
pub fn primitive_root_mod_p_squared(p: usize, g_mod_p: usize) -> usize {
    let p2 = p * p;
    if mod_pow(g_mod_p, (p - 1) as u64, p2 as u64) == 1 {
        g_mod_p + p
    } else {
        g_mod_p
    }
}

// ── Index tables ──────────────────────────────────────────────────────────

fn build_perm_tables(n: usize, phi: usize, g: usize) -> (Vec<usize>, Vec<usize>) {
    let mut perm_in = vec![0usize; phi]; // perm_in[s] = g^s mod N
    let mut pw = 1usize;
    for p in perm_in.iter_mut() {
        *p = pw;
        pw = (pw * g) % n;
    }
    let mut perm_out = vec![0usize; phi]; // perm_out[k] = g^{-k} mod N
    for (k, out_k) in perm_out.iter_mut().enumerate() {
        *out_k = perm_in[(phi - k) % phi];
    }
    (perm_in, perm_out)
}

// ── Compile-time DFT for H_hat ───────────────────────────────────────────

fn dft_reference(data: &[ComplexF64]) -> Vec<ComplexF64> {
    let n = data.len();
    (0..n)
        .map(|k| {
            data.iter()
                .enumerate()
                .fold(ComplexF64::zero(), |acc, (j, &x)| {
                    let angle = -2.0 * PI * (k * j) as f64 / n as f64;
                    let tw = ComplexF64::new(angle.cos(), angle.sin());
                    ComplexF64::new(
                        acc.re + x.re * tw.re - x.im * tw.im,
                        acc.im + x.re * tw.im + x.im * tw.re,
                    )
                })
        })
        .collect()
}

// ── Code generation ───────────────────────────────────────────────────────

/// Generate a Winograd-Rader prime-power DFT codelet for `N = p²`.
///
/// Requires `ShortDft<p>` and `ShortDft<φ(N)>` to be registered.
pub fn prime_power_winograd_function(
    p: usize,
    g_mod_p: usize,
    inline_attr: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    assert!(p >= 3 && is_prime(p), "p must be prime ≥ 3");
    let n = p * p;
    let phi = p * (p - 1); // φ(p²)
    let g = primitive_root_mod_p_squared(p, g_mod_p);

    let (perm_in, perm_out) = build_perm_tables(n, phi, g);

    // h_fwd[s] = W_N^{g^{-s}} = exp(-2πi·perm_out[s]/N)  [forward DFT convention]
    let h_fwd: Vec<ComplexF64> = (0..phi)
        .map(|s| {
            let angle = -2.0 * PI * perm_out[s] as f64 / n as f64;
            ComplexF64::new(angle.cos(), angle.sin())
        })
        .collect();
    // h_inv[s] = W_N^{-g^{-s}} = exp(+2πi·perm_out[s]/N) [inverse DFT convention]
    let h_inv: Vec<ComplexF64> = (0..phi)
        .map(|s| {
            let angle = 2.0 * PI * perm_out[s] as f64 / n as f64;
            ComplexF64::new(angle.cos(), angle.sin())
        })
        .collect();

    // H_hat = DFT_φ(h) — constant kernel used in frequency-domain multiply
    let hf_hat = dft_reference(&h_fwd);
    let hi_hat = dft_reference(&h_inv);

    let fn_name = format_ident!("dft{}_impl", n);
    let inv_phi = 1.0f64 / phi as f64;

    // ── Token generation ──────────────────────────────────────────────────

    // 1. Initialise y: y[r] = sum_{q=0}^{p-1} data[q*p + r]
    let y_init: Vec<_> = (0..p)
        .map(|r| {
            let terms: Vec<_> = (0..p)
                .map(|q| {
                    let idx = q * p + r;
                    quote! { data[#idx] }
                })
                .collect();
            quote! { #(#terms)+* }
        })
        .collect();

    // 2. Initialise z: z[0] = 0, z[m] = data[m*p]
    let z_init: Vec<_> = (0..p)
        .map(|m| {
            if m == 0 {
                quote! { eunomia::Complex::new(<F as eunomia::NumericElement>::ZERO, <F as eunomia::NumericElement>::ZERO) }
            } else {
                let idx = m * p;
                quote! { data[#idx] }
            }
        })
        .collect();

    // 3. Gather a: a[s] = data[perm_in[s]]
    let gather: Vec<_> = (0..phi)
        .map(|s| {
            let idx = perm_in[s];
            quote! { data[#idx] }
        })
        .collect();

    // H_hat forward/inverse literals
    let hf_lits: Vec<_> = hf_hat
        .iter()
        .map(|c| {
            let re = c.re;
            let im = c.im;
            quote! { eunomia::Complex::new(F::from_precise(#re), F::from_precise(#im)) }
        })
        .collect();
    let hi_lits: Vec<_> = hi_hat
        .iter()
        .map(|c| {
            let re = c.re;
            let im = c.im;
            quote! { eunomia::Complex::new(F::from_precise(#re), F::from_precise(#im)) }
        })
        .collect();

    // 4. Scatter p-multiples: data[k'*p] = y[k'] for k' in 1..p
    let p_multiples_scatter: Vec<_> = (1..p)
        .map(|kp| {
            let idx = kp * p;
            quote! { data[#idx] = y[#kp]; }
        })
        .collect();

    // 5. Scatter units: data[perm_out[b]] = x0 + a[b] + z[k']
    let gp = g % p;
    let mut unit_scatter = Vec::new();
    for (b, &dest_idx) in perm_out.iter().enumerate() {
        let bp = b % (p - 1);
        let kp = mod_pow(gp, (p - 1 - bp) as u64, p as u64);
        unit_scatter.push(quote! {
            data[#dest_idx] = eunomia::Complex::new(
                x0.re + a[#b].re + z[#kp].re,
                x0.im + a[#b].im + z[#kp].im,
            );
        });
    }

    quote! {
        #inline_attr
        #[allow(unused_variables, unused_mut)]
        pub(crate) fn #fn_name<
            F: crate::application::execution::kernel::components::winograd::traits::WinogradScalar
                + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#p>
                + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#phi>,
            const INVERSE: bool,
        >(
            data: &mut [eunomia::Complex<F>; #n],
        ) {
            // Save x[0] for output assembly
            let x0 = data[0];

            // 1. Prepare and compute DFT of y
            let mut y: [eunomia::Complex<F>; #p] = [ #(#y_init),* ];
            <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#p>>::dft::<INVERSE>(&mut y);

            // 2. Prepare and compute DFT of z
            let mut z: [eunomia::Complex<F>; #p] = [ #(#z_init),* ];
            <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#p>>::dft::<INVERSE>(&mut z);

            // 3. Prepare and compute cyclic convolution of length φ
            let mut a: [eunomia::Complex<F>; #phi] = [ #(#gather),* ];
            <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#phi>>::dft::<false>(&mut a);

            let h_hat: [eunomia::Complex<F>; #phi] = if INVERSE {
                [ #(#hi_lits),* ]
            } else {
                [ #(#hf_lits),* ]
            };
            for k in 0..#phi {
                let ak = a[k];
                let hk = h_hat[k];
                a[k] = eunomia::Complex::new(
                    ak.re * hk.re - ak.im * hk.im,
                    ak.re * hk.im + ak.im * hk.re,
                );
            }

            <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#phi>>::dft::<true>(&mut a);
            let inv_phi = F::from_precise(#inv_phi);
            for k in 0..#phi {
                a[k] = eunomia::Complex::new(a[k].re * inv_phi, a[k].im * inv_phi);
            }

            // 4. Assemble outputs
            data[0] = y[0];
            #(#p_multiples_scatter)*
            #(#unit_scatter)*
        }
    }
}

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let mut i = 3;
    while i * i <= n {
        if n.is_multiple_of(i) {
            return false;
        }
        i += 2;
    }
    true
}

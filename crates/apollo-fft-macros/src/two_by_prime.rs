use proc_macro::TokenStream as CompilerTokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{bracketed, parenthesized, parse_macro_input, Ident, LitInt, Result, Token};

struct TwoByPrimeInput {
    pairs: Vec<PrimePair>,
}

struct PrimePair {
    prime: LitInt,
    half: LitInt,
}

impl Parse for PrimePair {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let content;
        parenthesized!(content in input);
        let prime = content.parse()?;
        content.parse::<Token![,]>()?;
        let half = content.parse()?;
        Ok(Self { prime, half })
    }
}

impl Parse for TwoByPrimeInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let pairs_key: Ident = input.parse()?;
        if pairs_key != "pairs" {
            return Err(syn::Error::new(
                pairs_key.span(),
                "expected `pairs: [(prime, half), ..]`",
            ));
        }
        input.parse::<Token![:]>()?;

        let content;
        bracketed!(content in input);
        let pairs = Punctuated::<PrimePair, Token![,]>::parse_terminated(&content)?
            .into_iter()
            .collect::<Vec<_>>();
        if !input.is_empty() {
            input.parse::<Token![,]>()?;
        }

        Ok(Self { pairs })
    }
}

/// Generate fully-inlined 2×P Good-Thomas kernels with compile-time CRT indices.
///
/// Each prime gets a dedicated const-parameterized function with hardcoded
/// gather/scatter index arrays, eliminating all runtime modulo from the hot path.
pub fn generate_two_by_prime_natural_dispatch(input: CompilerTokenStream) -> CompilerTokenStream {
    let input = parse_macro_input!(input as TwoByPrimeInput);

    let route_fns = input.pairs.iter().map(|pair| {
        let p = pair.prime.base10_parse::<usize>().unwrap();
        let h = pair.half.base10_parse::<usize>().unwrap();
        two_by_prime_kernel(p, h)
    });

    let match_arms = input.pairs.iter().map(|pair| {
        let p = pair.prime.base10_parse::<usize>().unwrap();
        let route = route_ident(p);
        let prime = &pair.prime;
        let half = &pair.half;
        quote! {
            #prime => #route::<F, INVERSE>(data, <F as crate::application::execution::kernel::components::winograd::radix::odd_prime_pair::PrimePairTable<#prime, #half>>::cos_table(), <F as crate::application::execution::kernel::components::winograd::radix::odd_prime_pair::PrimePairTable<#prime, #half>>::sin_table()),
        }
    });

    quote! {
        #(#route_fns)*

        #[inline]
        fn fuse_two_prime_natural<
            F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
                Complex = num_complex::Complex<F>,
            >,
            const INVERSE: bool,
        >(
            data: &mut [F::Complex],
            prime: usize,
        ) -> bool {
            match prime {
                #(#match_arms)*
                _ => return false,
            }
            true
        }
    }
    .into()
}

fn route_ident(p: usize) -> Ident {
    format_ident!("__apollo_two_by_prime_fft_{p}")
}

/// Compute the CRT gather index for the odd row in a 2×P Good-Thomas layout.
///
/// Gather even row: `data[2 * i1]` (no modulo needed since `2*i1 < 2*P`).
/// Gather odd row:  `data[(2 * i1 + P) % (2 * P)]`, with potential wrap.
fn gather_odd_indices(p: usize) -> Vec<usize> {
    let n = 2 * p;
    (0..p).map(|i1| (2 * i1 + p) % n).collect()
}

/// Compute CRT scatter positions for a 2×P Good-Thomas layout.
///
/// `scatter_sum[k1]`  = index for `rows[0][k1] + rows[1][k1]`
/// `scatter_diff[k1]` = index for `rows[0][k1] - rows[1][k1]`
fn scatter_positions(p: usize) -> (Vec<usize>, Vec<usize>) {
    let n = 2 * p;
    let mut sum_pos = vec![0usize; p];
    let mut diff_pos = vec![0usize; p];
    for k1 in 0..p {
        let base = (k1 * (p + 1)) % n;
        sum_pos[k1] = base;
        diff_pos[k1] = (base + p) % n;
    }
    (sum_pos, diff_pos)
}

/// Generate a fully-inlined const-parameterized 2×P Good-Thomas kernel.
///
/// Hardcodes all gather and scatter indices as const arrays, eliminating
/// runtime modulo operations on the hot code path.
fn two_by_prime_kernel(p: usize, h: usize) -> proc_macro2::TokenStream {
    let route = route_ident(p);
    let gather_odd: Vec<usize> = gather_odd_indices(p);
    let (scatter_sum, scatter_diff) = scatter_positions(p);

    // Emit const arrays with hardcoded indices
    let gather_odd_tokens: Vec<_> = gather_odd.iter().map(|&idx| quote! { #idx }).collect();
    let scatter_sum_tokens: Vec<_> = scatter_sum.iter().map(|&idx| quote! { #idx }).collect();
    let scatter_diff_tokens: Vec<_> = scatter_diff.iter().map(|&idx| quote! { #idx }).collect();

    quote! {
        #[inline]
        fn #route<
            F: crate::application::execution::kernel::components::winograd::traits::WinogradScalar,
            const INVERSE: bool,
        >(
            data: &mut [num_complex::Complex<F>],
            cos: &[[F; #h]; #h],
            sin: &[[F; #h]; #h],
        ) {
            const GATHER_ODD: [usize; #p] = [#(#gather_odd_tokens),*];
            const SCATTER_SUM: [usize; #p] = [#(#scatter_sum_tokens),*];
            const SCATTER_DIFF: [usize; #p] = [#(#scatter_diff_tokens),*];

            let zero = F::zero();
            let mut rows = [[num_complex::Complex::new(zero, zero); #p]; 2];

            // Gather: even row from even indices, odd row via const CRT map
            for i1 in 0..#p {
                rows[0][i1] = data[2 * i1];
                rows[1][i1] = data[GATHER_ODD[i1]];
            }

            // Transform both rows
            use crate::application::execution::kernel::components::winograd::radix::odd_prime_pair::dft_pair_impl;
            dft_pair_impl::<F, #p, #h, INVERSE>(&mut rows[0], cos, sin);
            dft_pair_impl::<F, #p, #h, INVERSE>(&mut rows[1], cos, sin);

            // Scatter using const CRT maps
            for k1 in 0..#p {
                let a = rows[0][k1];
                let b = rows[1][k1];
                data[SCATTER_SUM[k1]] = num_complex::Complex::new(a.re + b.re, a.im + b.im);
                data[SCATTER_DIFF[k1]] = num_complex::Complex::new(a.re - b.re, a.im - b.im);
            }
        }
    }
}

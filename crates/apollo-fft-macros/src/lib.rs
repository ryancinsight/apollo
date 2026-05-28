//! Code generation macros for Apollo FFT kernels.
//!
//! The macros in this crate generate static dispatch surfaces from compact
//! mathematical specifications. Runtime FFT code keeps the numerical kernels
//! local to `apollo-fft`; this crate owns only token generation.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{bracketed, parse_macro_input, Ident, LitInt, Result, Token};

pub(crate) mod cooley_tukey;
pub(crate) mod good_thomas;
pub(crate) mod math;
pub(crate) mod prime_pair_tables;
pub(crate) mod prime_power_winograd;
pub(crate) mod rader;
pub(crate) mod two_by_prime;
pub(crate) mod winograd_composites;

/// Generate static Rader prime-N FFT codelets with compile-time gather/scatter tables.
#[proc_macro]
pub fn generate_rader_fft(input: TokenStream) -> TokenStream {
    rader::generate_rader_fft(input)
}

/// Generate the Good-Thomas PFA dispatch table for fixed coprime `(n1, n2)` pairs.
#[proc_macro]
pub fn generate_good_thomas_dispatch(input: TokenStream) -> TokenStream {
    good_thomas::generate_good_thomas_dispatch(input)
}

/// Generate the natural-layout Winograd-pair `2*p` dispatch table for promoted primes.
#[proc_macro]
pub fn generate_two_by_prime_natural_dispatch(input: TokenStream) -> TokenStream {
    two_by_prime::generate_two_by_prime_natural_dispatch(input)
}

/// Generate the `PrimePairTable<P, H>` trait impls with compile-time twiddle constants.
#[proc_macro]
pub fn generate_prime_pair_tables(input: TokenStream) -> TokenStream {
    prime_pair_tables::generate_prime_pair_tables(input)
}

struct ThreeByPrimeDispatchInput {
    primes: Vec<LitInt>,
    // Parsed for parse-API symmetry with `generate_good_thomas_dispatch`;
    // the actual 2×prime / 3×prime exclusion filtering happens inside
    // `good_thomas::parse_dispatch_pairs`, which reads its own copy of
    // `direct_pair_primes` from the `GoodThomasDispatchInput`.
    // This field is structurally inert in `ThreeByPrimeDispatchInput`.
    #[allow(dead_code)]
    direct_pair_primes: Option<Vec<LitInt>>,
}

impl Parse for ThreeByPrimeDispatchInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut primes = Vec::new();
        let mut direct_pair_primes = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            match key.to_string().as_str() {
                "primes" => {
                    let content;
                    bracketed!(content in input);
                    primes = Punctuated::<LitInt, Token![,]>::parse_terminated(&content)?
                        .into_iter()
                        .collect();
                }
                "direct_pair_primes" => {
                    let content;
                    bracketed!(content in input);
                    direct_pair_primes = Some(
                        Punctuated::<LitInt, Token![,]>::parse_terminated(&content)?
                            .into_iter()
                            .collect(),
                    );
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `primes` or `direct_pair_primes`",
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        if primes.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "`primes` is required",
            ));
        }

        Ok(Self {
            primes,
            direct_pair_primes,
        })
    }
}

/// Generate compact Good-Thomas `3*p` route kernels.
///
/// The generated code keeps one authoritative prime list for support
/// detection, monomorphized kernel selection, CRT gather, short-codelet
/// execution, and CRT scatter.
/// Generate all Winograd composite codelets from one canonical specification.
///
/// Accepts `gt_pairs` (coprime factors → Good-Thomas PFA, no twiddles),
/// `ct_pairs` (non-coprime factors → Cooley-Tukey DIT, twiddle constants),
/// and `pp_pairs` (prime-power p² → Winograd-Rader, twiddle-free Rader convolution)
/// and emits every `dftN_impl` function.
#[proc_macro]
pub fn generate_winograd_composites(input: TokenStream) -> TokenStream {
    winograd_composites::generate_winograd_composites(input)
}

/// Generate compact Good-Thomas `3*p` route kernels for the supplied prime list.
#[proc_macro]
pub fn generate_three_by_prime_dispatch(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ThreeByPrimeDispatchInput);
    let primes = input
        .primes
        .iter()
        .map(|prime| prime.base10_parse::<usize>())
        .collect::<Result<Vec<_>>>();
    let primes = match primes {
        Ok(primes) => primes,
        Err(error) => return error.to_compile_error().into(),
    };
    let route_fns = primes.iter().map(|prime| route_kernel(*prime));
    let match_arms = primes.iter().map(|prime| {
        let route = route_ident(*prime);
        quote! {
            #prime => #route::<F, INVERSE>(data),
        }
    });

    quote! {
        #(#route_fns)*

        #[inline]
        pub(super) fn supports(n1: usize, n2: usize) -> bool {
            n2 == 3 && matches!(n1, #(#primes)|*)
        }

        pub(super) fn try_fft<
            F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
                Complex = num_complex::Complex<F>,
            >,
            const INVERSE: bool,
        >(
            data: &mut [F::Complex],
            n1: usize,
            n2: usize,
        ) -> bool {
            if !supports(n1, n2) {
                return false;
            }

            match n1 {
                #(#match_arms)*
                _ => return false,
            }
            true
        }
    }
    .into()
}

fn route_kernel(p: usize) -> proc_macro2::TokenStream {
    let route = route_ident(p);
    let positions = linear_output_positions(p);
    let row_tokens: Vec<_> = positions.iter().map(|&(row, _)| quote! { #row }).collect();
    let col_tokens: Vec<_> = positions.iter().map(|&(_, col)| quote! { #col }).collect();

    quote! {
        #[inline(always)]
        fn #route<
            F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
                Complex = num_complex::Complex<F>,
            >,
            const INVERSE: bool,
        >(
            data: &mut [F::Complex],
        ) {
            debug_assert_eq!(data.len(), 3 * #p);
            const SCATTER_ROW: [usize; 3 * #p] = [#(#row_tokens),*];
            const SCATTER_COL: [usize; 3 * #p] = [#(#col_tokens),*];

            let zero = F::complex(0.0, 0.0);
            let mut rows = [[zero; #p]; 3];

            for col in 0..#p {
                let input0 = 3 * col;
                let mut input1 = #p + 3 * col;
                if input1 >= 3 * #p {
                    input1 -= 3 * #p;
                }
                let mut input2 = 2 * #p + 3 * col;
                if input2 >= 3 * #p {
                    input2 -= 3 * #p;
                }
                let mut column = [data[input0], data[input1], data[input2]];
                <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<3>>::dft::<INVERSE>(&mut column);
                rows[0][col] = column[0];
                rows[1][col] = column[1];
                rows[2][col] = column[2];
            }

            for row in 0..3usize {
                <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#p>>::dft::<INVERSE>(&mut rows[row]);
            }

            for dst in 0..(3 * #p) {
                data[dst] = rows[SCATTER_ROW[dst]][SCATTER_COL[dst]];
            }
        }
    }
}

fn route_ident(p: usize) -> Ident {
    format_ident!("__apollo_three_by_prime_fft_{p}")
}

fn linear_output_positions(p: usize) -> Vec<(usize, usize)> {
    let n = 3 * p;
    let k1_stride = p * inverse_mod(p % 3, 3);
    let k2_stride = 3 * inverse_mod(3, p);
    let mut positions = vec![(0usize, 0usize); n];

    for row in 0..3usize {
        for col in 0..p {
            let dst = (row * k1_stride + col * k2_stride) % n;
            positions[dst] = (row, col);
        }
    }

    positions
}

fn inverse_mod(a: usize, modulus: usize) -> usize {
    (1..modulus)
        .find(|candidate| (a * candidate) % modulus == 1)
        .expect("supported Good-Thomas factors must be coprime")
}

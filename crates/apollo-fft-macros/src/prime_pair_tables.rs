use proc_macro::TokenStream as CompilerTokenStream;
use quote::{format_ident, quote};
use std::f64::consts::TAU;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{parenthesized, parse_macro_input, LitInt, Result, Token};

/// A single `(N, H)` entry in the macro input.
struct PrimePairEntry {
    n: usize,
    h: usize,
}

struct PrimePairEntry_ {
    n: LitInt,
    h: LitInt,
}

impl Parse for PrimePairEntry_ {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let content;
        parenthesized!(content in input);
        let n = content.parse()?;
        content.parse::<Token![,]>()?;
        let h = content.parse()?;
        Ok(Self { n, h })
    }
}

struct PrimePairTablesInput {
    pairs: Vec<PrimePairEntry>,
}

impl Parse for PrimePairTablesInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let entries = Punctuated::<PrimePairEntry_, Token![,]>::parse_terminated(input)?;
        let pairs = entries
            .into_iter()
            .map(|e| {
                Ok(PrimePairEntry {
                    n: e.n.base10_parse()?,
                    h: e.h.base10_parse()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { pairs })
    }
}

/// Generate compile-time static cos/sin tables for a list of `(N, H)` prime pairs.
///
/// Each `(N, H)` pair emits four `static` arrays (two precisions × cos/sin):
/// - `PRIME_PAIR_COS_F64_{N}: [[f64; H]; H]`
/// - `PRIME_PAIR_SIN_F64_{N}: [[f64; H]; H]`
/// - `PRIME_PAIR_COS_F32_{N}: [[f32; H]; H]`
/// - `PRIME_PAIR_SIN_F32_{N}: [[f32; H]; H]`
///
/// And impls of `PrimePairTable<N, H>` for both `f64` and `f32` referencing
/// those statics instead of computing via `OnceLock`.
///
/// Usage:
/// ```rust,ignore
/// apollo_fft_macros::generate_prime_pair_tables![(11, 5), (13, 6), ...]
/// ```
pub fn generate_prime_pair_tables(input: CompilerTokenStream) -> CompilerTokenStream {
    let input = parse_macro_input!(input as PrimePairTablesInput);
    let mut out = proc_macro2::TokenStream::new();

    for entry in &input.pairs {
        let n = entry.n;
        let h = entry.h;

        // Compute all H×H cos and sin table entries at macro-expansion time.
        // cos_table[k][m] = cos(TAU * (k+1) * (m+1) / N)
        // sin_table[k][m] = sin(TAU * (k+1) * (m+1) / N)
        let cos_f64: Vec<Vec<f64>> = (0..h)
            .map(|k| {
                (0..h)
                    .map(|m| (TAU * ((k + 1) * (m + 1)) as f64 / n as f64).cos())
                    .collect()
            })
            .collect();

        let sin_f64: Vec<Vec<f64>> = (0..h)
            .map(|k| {
                (0..h)
                    .map(|m| (TAU * ((k + 1) * (m + 1)) as f64 / n as f64).sin())
                    .collect()
            })
            .collect();

        // Emit the row literals as nested arrays.
        let cos_f64_rows: Vec<proc_macro2::TokenStream> = cos_f64
            .iter()
            .map(|row| {
                let vals = row.iter().map(|&v| quote! { #v });
                quote! { [#(#vals),*] }
            })
            .collect();

        let sin_f64_rows: Vec<proc_macro2::TokenStream> = sin_f64
            .iter()
            .map(|row| {
                let vals = row.iter().map(|&v| quote! { #v });
                quote! { [#(#vals),*] }
            })
            .collect();

        // f32 literals: cast from the f64 values computed above.
        let cos_f32_rows: Vec<proc_macro2::TokenStream> = cos_f64
            .iter()
            .map(|row| {
                let vals = row.iter().map(|&v| {
                    let fv = v as f32;
                    quote! { #fv }
                });
                quote! { [#(#vals),*] }
            })
            .collect();

        let sin_f32_rows: Vec<proc_macro2::TokenStream> = sin_f64
            .iter()
            .map(|row| {
                let vals = row.iter().map(|&v| {
                    let fv = v as f32;
                    quote! { #fv }
                });
                quote! { [#(#vals),*] }
            })
            .collect();

        let cos_f64_name = format_ident!("PRIME_PAIR_COS_F64_{}", n);
        let sin_f64_name = format_ident!("PRIME_PAIR_SIN_F64_{}", n);
        let cos_f32_name = format_ident!("PRIME_PAIR_COS_F32_{}", n);
        let sin_f32_name = format_ident!("PRIME_PAIR_SIN_F32_{}", n);

        out.extend(quote! {
            static #cos_f64_name: [[f64; #h]; #h] = [#(#cos_f64_rows),*];
            static #sin_f64_name: [[f64; #h]; #h] = [#(#sin_f64_rows),*];
            static #cos_f32_name: [[f32; #h]; #h] = [#(#cos_f32_rows),*];
            static #sin_f32_name: [[f32; #h]; #h] = [#(#sin_f32_rows),*];

            impl PrimePairTable<#n, #h> for f64 {
                #[inline]
                fn cos_table() -> &'static [[f64; #h]; #h] {
                    &#cos_f64_name
                }
                #[inline]
                fn sin_table() -> &'static [[f64; #h]; #h] {
                    &#sin_f64_name
                }
            }

            impl PrimePairTable<#n, #h> for f32 {
                #[inline]
                fn cos_table() -> &'static [[f32; #h]; #h] {
                    &#cos_f32_name
                }
                #[inline]
                fn sin_table() -> &'static [[f32; #h]; #h] {
                    &#sin_f32_name
                }
            }
        });
    }

    out.into()
}

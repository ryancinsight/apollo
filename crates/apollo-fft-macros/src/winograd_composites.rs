//! Unified Winograd composite codelet generation.
//!
//! The macro accepts three pair lists and emits every `dftN_impl` function:
//!
//! - `gt_pairs`: coprime (n1, n2) → Good-Thomas PFA, twiddle-free
//! - `ct_pairs`: non-coprime (n1, n2) → Cooley-Tukey DIT, precomputed twiddle constants
//! - `pp_pairs`: prime-power (p, g) → Winograd-Rader on `(Z/p²Z)*`, fully twiddle-free
//!
//! Optional: `inline_attr: always | hint` (default: `always`)

use proc_macro::TokenStream as CompilerTokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{bracketed, parenthesized, parse_macro_input, Ident, LitInt, Result, Token};

use crate::cooley_tukey::cooley_tukey_function;
use crate::good_thomas::good_thomas_function;
use crate::prime_power_winograd::prime_power_winograd_function;

// ── Input parsing ──────────────────────────────────────────────────────────

/// Inlining directive for generated codelets.
///
/// - `Always` → `#[inline]`: forced inlining for small N where
///   stack depth is safe (N ≤ 36).
/// - `Hint`   → `#[inline]`:         optimizer hint only; avoids forced
///   inlining for larger N where debug-mode stack may overflow.
#[derive(Clone, Copy)]
enum InlineAttr {
    Always,
    Hint,
}

impl InlineAttr {
    fn to_tokens(self) -> proc_macro2::TokenStream {
        match self {
            InlineAttr::Always => quote! { #[inline] },
            InlineAttr::Hint => quote! { #[inline] },
        }
    }
}

struct PairInput {
    n1: LitInt,
    n2: LitInt,
}

impl Parse for PairInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let content;
        parenthesized!(content in input);
        let n1 = content.parse()?;
        content.parse::<Token![,]>()?;
        let n2 = content.parse()?;
        Ok(Self { n1, n2 })
    }
}

struct WinogradCompositesInput {
    inline_attr: InlineAttr,
    gt_pairs: Vec<PairInput>,
    ct_pairs: Vec<PairInput>,
    /// `(p, g_mod_p)` — prime `p` with its primitive root `g_mod_p` mod p.
    /// Generates a Winograd-Rader DFT-p² codelet (twiddle-free).
    pp_pairs: Vec<PairInput>,
}

impl Parse for WinogradCompositesInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut inline_attr = InlineAttr::Always;
        let mut gt_pairs = Vec::new();
        let mut ct_pairs = Vec::new();
        let mut pp_pairs = Vec::new();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            match key.to_string().as_str() {
                "inline_attr" => {
                    let value: Ident = input.parse()?;
                    inline_attr = match value.to_string().as_str() {
                        "always" => InlineAttr::Always,
                        "hint" => InlineAttr::Hint,
                        other => {
                            return Err(syn::Error::new(
                                value.span(),
                                format!(
                                    "unknown inline_attr `{other}`; expected `always` or `hint`"
                                ),
                            ));
                        }
                    };
                }
                "gt_pairs" => {
                    let content;
                    bracketed!(content in input);
                    gt_pairs = Punctuated::<PairInput, Token![,]>::parse_terminated(&content)?
                        .into_iter()
                        .collect();
                }
                "ct_pairs" => {
                    let content;
                    bracketed!(content in input);
                    ct_pairs = Punctuated::<PairInput, Token![,]>::parse_terminated(&content)?
                        .into_iter()
                        .collect();
                }
                "pp_pairs" => {
                    let content;
                    bracketed!(content in input);
                    pp_pairs = Punctuated::<PairInput, Token![,]>::parse_terminated(&content)?
                        .into_iter()
                        .collect();
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `inline_attr`, `gt_pairs`, `ct_pairs`, or `pp_pairs`",
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self {
            inline_attr,
            gt_pairs,
            ct_pairs,
            pp_pairs,
        })
    }
}

// ── Entry point ────────────────────────────────────────────────────────────

pub fn generate_winograd_composites(input: CompilerTokenStream) -> CompilerTokenStream {
    let input = parse_macro_input!(input as WinogradCompositesInput);

    let inline_tokens = input.inline_attr.to_tokens();

    let gt_codelets: Vec<_> = input
        .gt_pairs
        .iter()
        .map(|pair| {
            let n1 = pair.n1.base10_parse::<usize>().unwrap();
            let n2 = pair.n2.base10_parse::<usize>().unwrap();
            good_thomas_function(n1, n2, inline_tokens.clone())
        })
        .collect();

    let ct_codelets: Vec<_> = input
        .ct_pairs
        .iter()
        .map(|pair| {
            let n1 = pair.n1.base10_parse::<usize>().unwrap();
            let n2 = pair.n2.base10_parse::<usize>().unwrap();
            cooley_tukey_function(n1, n2, inline_tokens.clone())
        })
        .collect();

    // pp_pairs: (p, g_mod_p) → Winograd-Rader DFT-p² codelet
    let pp_codelets: Vec<_> = input
        .pp_pairs
        .iter()
        .map(|pair| {
            let p = pair.n1.base10_parse::<usize>().unwrap();
            let g = pair.n2.base10_parse::<usize>().unwrap();
            prime_power_winograd_function(p, g, inline_tokens.clone())
        })
        .collect();

    quote! {
        #(#gt_codelets)*
        #(#ct_codelets)*
        #(#pp_codelets)*
    }
    .into()
}

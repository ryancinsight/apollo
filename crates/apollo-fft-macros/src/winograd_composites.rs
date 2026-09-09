//! Unified Winograd composite codelet generation.
//!
//! The macro accepts three pair lists and emits every `dftN_impl` function:
//!
//! - `gt_pairs`: coprime (n1, n2) → Good-Thomas PFA, twiddle-free
//! - `ct_pairs`: non-coprime (n1, n2) → Cooley-Tukey DIT, precomputed twiddle constants
//! - `pp_pairs`: prime-power (p, g) → Winograd-Rader on `(Z/p²Z)*`, fully twiddle-free
//!
//! Optional: `inline_attr: always | hint` (default: `always`)
//! `scheduled_pairs` explicitly opts listed transform pairs into phase scheduling.

use proc_macro::TokenStream as CompilerTokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{bracketed, parenthesized, parse_macro_input, Ident, LitInt, Result, Token};

use crate::cooley_tukey::cooley_tukey_function;
use crate::good_thomas::good_thomas_function;
use crate::phase_emission::PhaseEmission;
use crate::prime_power_winograd::prime_power_winograd_function;

/// Accepted inlining directives; both emit the compiler's `#[inline]` hint.
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
    n1: usize,
    n2: usize,
    span: proc_macro2::Span,
}

impl Parse for PairInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let content;
        parenthesized!(content in input);
        let n1: LitInt = content.parse()?;
        content.parse::<Token![,]>()?;
        let n2: LitInt = content.parse()?;
        Ok(Self {
            n1: n1.base10_parse()?,
            n2: n2.base10_parse()?,
            span: n1.span(),
        })
    }
}

struct WinogradCompositesInput {
    inline_attr: InlineAttr,
    gt_pairs: Vec<PairInput>,
    ct_pairs: Vec<PairInput>,
    /// `(p, g_mod_p)` — prime `p` with its primitive root `g_mod_p` mod p.
    /// Generates a Winograd-Rader DFT-p² codelet (twiddle-free).
    pp_pairs: Vec<PairInput>,
    scheduled_pairs: Vec<PairInput>,
}

impl Parse for WinogradCompositesInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut inline_attr = InlineAttr::Always;
        let mut gt_pairs = Vec::new();
        let mut ct_pairs = Vec::new();
        let mut pp_pairs = Vec::new();
        let mut scheduled_pairs = None;

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
                "scheduled_pairs" => {
                    if scheduled_pairs.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `scheduled_pairs`"));
                    }
                    let content;
                    bracketed!(content in input);
                    scheduled_pairs = Some(
                        Punctuated::<PairInput, Token![,]>::parse_terminated(&content)?
                            .into_iter()
                            .collect(),
                    );
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected `inline_attr`, `gt_pairs`, `ct_pairs`, `pp_pairs`, or `scheduled_pairs`",
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        let parsed = Self {
            inline_attr,
            gt_pairs,
            ct_pairs,
            pp_pairs,
            scheduled_pairs: scheduled_pairs.unwrap_or_default(),
        };
        parsed.check_scheduled_pairs()?;
        Ok(parsed)
    }
}

impl WinogradCompositesInput {
    fn check_scheduled_pairs(&self) -> Result<()> {
        for (index, pair) in self.scheduled_pairs.iter().enumerate() {
            let transform_pairs = match (pair.n1, pair.n2) {
                (2, 25) => &self.gt_pairs,
                (12, 12) => &self.ct_pairs,
                _ => {
                    return Err(syn::Error::new(
                        pair.span,
                        "scheduled pairs must be Good-Thomas (2, 25) or Cooley-Tukey (12, 12)",
                    ));
                }
            };
            if transform_pairs
                .iter()
                .filter(|candidate| candidate.n1 == pair.n1 && candidate.n2 == pair.n2)
                .count()
                != 1
            {
                return Err(syn::Error::new(
                    pair.span,
                    "scheduled pair must occur exactly once in its transform pair list",
                ));
            }
            if self.scheduled_pairs[..index]
                .iter()
                .any(|prior| prior.n1 == pair.n1 && prior.n2 == pair.n2)
            {
                return Err(syn::Error::new(pair.span, "duplicate scheduled pair"));
            }
        }
        Ok(())
    }

    fn phases(&self, pair: &PairInput) -> PhaseEmission {
        if self
            .scheduled_pairs
            .iter()
            .any(|scheduled| scheduled.n1 == pair.n1 && scheduled.n2 == pair.n2)
        {
            PhaseEmission::Scheduled
        } else {
            PhaseEmission::Inline
        }
    }
}

pub fn generate_winograd_composites(input: CompilerTokenStream) -> CompilerTokenStream {
    let input = parse_macro_input!(input as WinogradCompositesInput);
    emit_codelets(&input).into()
}

fn emit_codelets(input: &WinogradCompositesInput) -> proc_macro2::TokenStream {
    let inline_tokens = input.inline_attr.to_tokens();

    let gt_codelets: Vec<_> = input
        .gt_pairs
        .iter()
        .map(|pair| {
            good_thomas_function(pair.n1, pair.n2, inline_tokens.clone(), input.phases(pair))
        })
        .collect();

    let ct_codelets: Vec<_> = input
        .ct_pairs
        .iter()
        .map(|pair| {
            cooley_tukey_function(pair.n1, pair.n2, inline_tokens.clone(), input.phases(pair))
        })
        .collect();

    let pp_codelets: Vec<_> = input
        .pp_pairs
        .iter()
        .map(|pair| prime_power_winograd_function(pair.n1, pair.n2, inline_tokens.clone()))
        .collect();

    quote! {
        #(#gt_codelets)*
        #(#ct_codelets)*
        #(#pp_codelets)*
    }
}

#[cfg(test)]
mod tests;

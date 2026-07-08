use proc_macro::TokenStream as CompilerTokenStream;
use quote::{format_ident, quote};
use std::collections::BTreeSet;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{bracketed, parenthesized, parse_macro_input, Ident, LitInt, Result, Token};

use crate::math::mod_inverse_isize;

struct GoodThomasInput {
    n1: LitInt,
    n2: LitInt,
}

impl Parse for GoodThomasInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let n1 = input.parse()?;
        input.parse::<Token![,]>()?;
        let n2 = input.parse()?;
        Ok(Self { n1, n2 })
    }
}

struct GoodThomasDispatchInput {
    pairs: Vec<GoodThomasInput>,
    short_sizes: Vec<LitInt>,
    max_n: Option<LitInt>,
    direct_pair_primes: Option<Vec<LitInt>>,
}
impl Parse for GoodThomasDispatchInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut pairs = Vec::new();
        let mut short_sizes = Vec::new();
        let mut max_n = None;
        let mut direct_pair_primes = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            match key.to_string().as_str() {
                "pairs" => {
                    let content;
                    bracketed!(content in input);
                    pairs.extend(
                        Punctuated::<ParenthesizedGoodThomasInput, Token![,]>::parse_terminated(
                            &content,
                        )?
                        .into_iter()
                        .map(|pair| pair.0),
                    );
                }
                "short_sizes" => {
                    let content;
                    bracketed!(content in input);
                    short_sizes
                        .extend(Punctuated::<LitInt, Token![,]>::parse_terminated(&content)?);
                }
                "max_n" => {
                    max_n = Some(input.parse()?);
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
                        "expected `pairs`, `short_sizes`, `max_n`, or `direct_pair_primes`",
                    ));
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self {
            pairs,
            short_sizes,
            max_n,
            direct_pair_primes,
        })
    }
}

struct ParenthesizedGoodThomasInput(GoodThomasInput);

impl Parse for ParenthesizedGoodThomasInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let content;
        parenthesized!(content in input);
        Ok(Self(content.parse()?))
    }
}
pub(crate) fn good_thomas_function(
    n1: usize,
    n2: usize,
    inline_attr: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let n = n1 * n2;

    let inv_n2_n1 = mod_inverse_isize(n2 as isize, n1 as isize) as usize;
    let inv_n1_n2 = mod_inverse_isize(n1 as isize, n2 as isize) as usize;

    let fn_name = format_ident!("dft{}_impl", n);
    let stride = (n2 * inv_n2_n1) % n;

    quote! {
        #inline_attr
        #[allow(unused_variables, unused_mut)]
        pub(crate) unsafe fn #fn_name<F: crate::application::execution::kernel::components::winograd::traits::WinogradScalar + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n1> + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n2>, const INVERSE: bool>(
            data: &mut [eunomia::Complex<F>; #n],
        ) {
            // Use MaybeUninit to avoid zero-initialization overhead
            // SAFETY: All scratch positions are written via .write() before any .read()
            let mut scratch = std::mem::MaybeUninit::<[eunomia::Complex<F>; #n]>::uninit();
            let scratch_ptr = scratch.as_mut_ptr() as *mut eunomia::Complex<F>;

            // Gather input using incremental index calculation (no runtime modulo)
            for i1 in 0..#n1 {
                let mut src_idx = i1 * #n2;
                let row_start = i1 * #n2;
                for i2 in 0..#n2 {
                    let dest_idx = row_start + i2;
                    scratch_ptr.add(dest_idx).write(data[src_idx]);
                    src_idx += #n1;
                    if src_idx >= #n {
                        src_idx -= #n;
                    }
                }
            }

            // Transform rows using zero-cost pointer cast
            for i1 in 0..#n1 {
                let row_start = i1 * #n2;
                let row = unsafe { &mut *(scratch_ptr.add(row_start) as *mut [eunomia::Complex<F>; #n2]) };
                unsafe {
                    <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n2>>::dft::<INVERSE>(row);
                }
            }

            // Transform columns & scatter output using incremental index calculation (no runtime modulo)
            let inv_n1_n2 = #inv_n1_n2;
            let inv_n2_n1 = #inv_n2_n1;
            let stride = #stride;
            for i2 in 0..#n2 {
                // SAFETY: All N1 slots of `col` are written by the gather loop before
                // `dft` reads any of them. The loop covers i1=0..N1 unconditionally.
                let mut col = std::mem::MaybeUninit::<[eunomia::Complex<F>; #n1]>::uninit();
                let col_ptr = col.as_mut_ptr() as *mut eunomia::Complex<F>;
                for i1 in 0..#n1 {
                    unsafe { col_ptr.add(i1).write(scratch_ptr.add(i1 * #n2 + i2).read()); }
                }
                let col = unsafe { col.assume_init_mut() };
                unsafe {
                    <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n1>>::dft::<INVERSE>(col);
                }

                let base = (i2 * #n1 * inv_n1_n2) % #n;
                let mut dest_idx = base;
                for i1 in 0..#n1 {
                    data[dest_idx] = col[i1];
                    dest_idx += stride;
                    if dest_idx >= #n {
                        dest_idx -= #n;
                    }
                }
            }
        }
    }
}

pub fn generate_good_thomas_dispatch(input: CompilerTokenStream) -> CompilerTokenStream {
    let input = parse_macro_input!(input as GoodThomasDispatchInput);
    let short_sizes = input.short_sizes.clone();
    let pairs = parse_dispatch_pairs(input);
    let pairs = match pairs {
        Ok(pairs) => pairs,
        Err(error) => return error.to_compile_error().into(),
    };

    let codelets = pairs
        .iter()
        .map(|&(n1, n2)| good_thomas_function(n1, n2, quote! { #[inline] }));

    let supported_pairs = pairs.iter().map(|(n1, n2)| quote! { (#n1, #n2) });
    let match_arms = pairs.iter().map(|&(n1, n2)| {
        let n = n1 * n2;
        let fn_name = format_ident!("dft{}_impl", n);
        quote! {
            (#n1, #n2) => {
                let arr: &mut [F::Complex; #n] = data.try_into().unwrap();
                unsafe { #fn_name::<F, INVERSE>(arr); }
            }
        }
    });

    let trait_bounds = short_sizes.iter().map(|size| {
        quote! {
            + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#size>
        }
    });

    quote! {
        #(#codelets)*

        const SUPPORTED_GOOD_THOMAS_PAIRS: &[(usize, usize)] = &[#(#supported_pairs),*];

        #[inline]
        pub(super) fn supports(n1: usize, n2: usize) -> bool {
            SUPPORTED_GOOD_THOMAS_PAIRS.contains(&(n1, n2))
        }

        pub(super) fn try_fft<
            F: crate::application::execution::kernel::mixed_radix::MixedRadixScalar<
                Complex = eunomia::Complex<F>,
            > #(#trait_bounds)*,
            const INVERSE: bool,
        >(
            data: &mut [F::Complex],
            n1: usize,
            n2: usize,
        ) -> bool {
            match (n1, n2) {
                #(#match_arms)*
                _ => return false,
            }
            true
        }
    }
    .into()
}

fn parse_dispatch_pairs(input: GoodThomasDispatchInput) -> Result<Vec<(usize, usize)>> {
    let mut pairs = BTreeSet::new();
    for pair in input.pairs {
        pairs.insert((pair.n1.base10_parse()?, pair.n2.base10_parse()?));
    }

    let direct_pair_primes: BTreeSet<usize> = input
        .direct_pair_primes
        .unwrap_or_default()
        .into_iter()
        .map(|size| size.base10_parse::<usize>())
        .collect::<Result<_>>()?;

    if !input.short_sizes.is_empty() {
        let max_n = input
            .max_n
            .ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "`short_sizes` requires `max_n`",
                )
            })?
            .base10_parse::<usize>()?;
        let short_sizes = input
            .short_sizes
            .into_iter()
            .map(|size| size.base10_parse::<usize>())
            .collect::<Result<Vec<_>>>()?;
        let short_set = short_sizes.iter().copied().collect::<BTreeSet<_>>();

        for &n1 in &short_sizes {
            for &n2 in &short_sizes {
                let n = n1 * n2;
                if n1 > 1
                    && n2 > 1
                    && n <= max_n
                    && crate::math::gcd(n1, n2) == 1
                    && !short_set.contains(&n)
                    && canonical_coprime_factors(n) == Some((n1, n2))
                {
                    pairs.insert((n1, n2));
                }
            }
        }
    }

    // Exclude 2×prime and 3×prime canonical pairs where `two_by_prime` or
    // `three_by_prime` already handle the dispatch.
    // For each (n1, n2), if min(n1, n2) in [2, 3] and max(n1, n2) is in
    // direct_pair_primes, skip.
    if !direct_pair_primes.is_empty() {
        pairs.retain(|&(a, b)| {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            !((lo == 2 || lo == 3) && direct_pair_primes.contains(&hi))
        });
    }

    Ok(pairs.into_iter().collect())
}

fn canonical_coprime_factors(n: usize) -> Option<(usize, usize)> {
    let mut remaining = n;
    let mut prime_powers = Vec::new();
    let mut p = 2usize;
    while p * p <= remaining {
        if remaining.is_multiple_of(p) {
            let mut power = 1usize;
            while remaining.is_multiple_of(p) {
                power *= p;
                remaining /= p;
            }
            prime_powers.push(power);
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if remaining > 1 {
        prime_powers.push(remaining);
    }
    if prime_powers.len() < 2 {
        return None;
    }
    let n1 = prime_powers.pop()?;
    let n2 = prime_powers.into_iter().product();
    Some((n1, n2))
}

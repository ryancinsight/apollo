use quote::{format_ident, quote};

/// Generate a Cooley-Tukey DIT codelet for `n1 × n2`.
///
/// The generated function is `dft{n1*n2}_impl`, const-generic over `INVERSE`.
/// Column sub-transforms of length `n1` are followed by twiddle-factor
/// multiplication; row sub-transforms of length `n2` write directly back
/// to `data`.
pub(crate) fn cooley_tukey_function(
    n1: usize,
    n2: usize,
    inline_attr: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let n = n1 * n2;
    let fn_name = format_ident!("dft{}_impl", n);

    // 1. Column blocks (DFT + Twiddles, writing to scratch via pointer)
    let mut col_blocks = vec![];
    for j in 0..n2 {
        let mut col_elements = vec![];
        for n1_idx in 0..n1 {
            let idx = j + n1_idx * n2;
            col_elements.push(quote! { data[#idx] });
        }

        let mut twiddle_assignments = vec![];
        for k1 in 0..n1 {
            let exp = k1 * j;
            let scratch_idx = k1 * n2 + j;
            // use_ptr = true: emit `scratch_ptr.add(N).write(val)`
            twiddle_assignments.push(twiddle_expr(n, exp, k1, scratch_idx, true));
        }

        col_blocks.push(quote! {
            {
                let mut col = [ #(#col_elements),* ];
                <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n1>>::dft::<INVERSE>(&mut col);
                #(#twiddle_assignments)*
            }
        });
    }

    // 2. Row blocks (In-place DFT on initialized scratch slice + write to data)
    let mut row_blocks = vec![];
    for k1 in 0..n1 {
        let start = k1 * n2;

        let mut row_stores = vec![];
        for k2 in 0..n2 {
            let src = start + k2;
            let dst = k2 * n1 + k1;
            row_stores.push(quote! {
                data[#dst] = scratch[#src];
            });
        }

        row_blocks.push(quote! {
            {
                let row = unsafe { &mut *(scratch.as_mut_ptr().add(#start) as *mut [eunomia::Complex<F>; #n2]) };
                <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n2>>::dft::<INVERSE>(row);
                #(#row_stores)*
            }
        });
    }

    quote! {
        #inline_attr
        #[allow(unused_variables, unused_mut)]
        pub(crate) unsafe fn #fn_name<
            F: crate::application::execution::kernel::components::winograd::traits::WinogradScalar
                + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n1>
                + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n2>,
            const INVERSE: bool,
        >(
            data: &mut [eunomia::Complex<F>; #n],
        ) {
            // SAFETY: Every element of `scratch` is written by a col_block before
            // any row_block reads it. The nested loop structure guarantees all N
            // positions are covered (col_block for j in 0..n2 writes scratch[k1*n2+j]
            // for all k1 in 0..n1 and all j in 0..n2 = all N indices).
            let mut scratch =
                std::mem::MaybeUninit::<[eunomia::Complex<F>; #n]>::uninit();
            let scratch_ptr = scratch.as_mut_ptr() as *mut eunomia::Complex<F>;
            #(#col_blocks)*
            let scratch = unsafe { scratch.assume_init_mut() };
            #(#row_blocks)*
        }
    }
}

/// Emit the twiddle-multiplication expression for `W_N^{exp}`.
///
/// Special-case `exp = 0` (identity), multiples of N/4 (±i), and
/// multiples of N/8 (±√2/2) to avoid general complex multiplication.
/// Falls through to `apply_twiddle_impl` with the precomputed constant
/// for arbitrary angles.
///
/// When `use_ptr` is true, writes are emitted as
/// `unsafe { scratch_ptr.add(scratch_idx).write(val) }` for use in
/// `MaybeUninit` col_blocks. When false, emits `scratch[scratch_idx] = val`.
fn twiddle_expr(
    n: usize,
    exp: usize,
    k1: usize,
    scratch_idx: usize,
    use_ptr: bool,
) -> proc_macro2::TokenStream {
    // Helper closure: wrap a value expression in the correct assignment form.
    let assign = |val: proc_macro2::TokenStream| -> proc_macro2::TokenStream {
        if use_ptr {
            quote! { unsafe { scratch_ptr.add(#scratch_idx).write(#val); } }
        } else {
            quote! { scratch[#scratch_idx] = #val; }
        }
    };

    if exp == 0 {
        return assign(quote! { col[#k1] });
    }

    let angle = -2.0 * std::f64::consts::PI * (exp as f64) / (n as f64);
    let w_re = angle.cos();
    let w_im = angle.sin();

    // Special-case: ±1
    if (w_re - 1.0).abs() < 1e-6 && w_im.abs() < 1e-6 {
        return assign(quote! { col[#k1] });
    }
    if (w_re + 1.0).abs() < 1e-6 && w_im.abs() < 1e-6 {
        return assign(quote! { eunomia::Complex::new(-col[#k1].re, -col[#k1].im) });
    }

    // Special-case: ±i
    if w_re.abs() < 1e-6 && (w_im - 1.0).abs() < 1e-6 {
        return assign(quote! {
            if INVERSE {
                eunomia::Complex::new(col[#k1].im, -col[#k1].re)
            } else {
                eunomia::Complex::new(-col[#k1].im, col[#k1].re)
            }
        });
    }
    if w_re.abs() < 1e-6 && (w_im + 1.0).abs() < 1e-6 {
        return assign(quote! {
            if INVERSE {
                eunomia::Complex::new(-col[#k1].im, col[#k1].re)
            } else {
                eunomia::Complex::new(col[#k1].im, -col[#k1].re)
            }
        });
    }

    // Special-case: ±√2/2 ± i·√2/2 (45° family)
    let is_sq2o2 = (w_re.abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6
        && (w_im.abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6;

    if is_sq2o2 {
        let s_re_pos = w_re > 0.0;
        let s_im_pos = w_im > 0.0;
        return assign(quote! {
            {
                let sq2o2 = F::sq2o2();
                let a = col[#k1];
                if INVERSE {
                    eunomia::Complex::new(
                        sq2o2 * (
                            if #s_re_pos { a.re } else { -a.re }
                            + if #s_im_pos { a.im } else { -a.im }
                        ),
                        sq2o2 * (
                            if #s_im_pos { -a.re } else { a.re }
                            + if #s_re_pos { a.im } else { -a.im }
                        )
                    )
                } else {
                    eunomia::Complex::new(
                        sq2o2 * (
                            if #s_re_pos { a.re } else { -a.re }
                            - if #s_im_pos { a.im } else { -a.im }
                        ),
                        sq2o2 * (
                            if #s_im_pos { a.re } else { -a.re }
                            + if #s_re_pos { a.im } else { -a.im }
                        )
                    )
                }
            }
        });
    }

    // General case: complex multiplication
    assign(quote! {
        {
            let tw = eunomia::Complex::new(
                F::from_precise(#w_re),
                if INVERSE { F::from_precise(-(#w_im)) } else { F::from_precise(#w_im) }
            );
            crate::application::execution::kernel::components::winograd::traits::apply_twiddle_impl(col[#k1], tw)
        }
    })
}

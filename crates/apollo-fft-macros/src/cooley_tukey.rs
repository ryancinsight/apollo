use quote::{format_ident, quote};

use crate::phase_emission::PhaseEmission;

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
    phases: PhaseEmission,
) -> proc_macro2::TokenStream {
    let n = n1 * n2;
    let fn_name = format_ident!("dft{}_impl", n);

    // Column transforms initialize scratch before row transforms read it.
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
            twiddle_assignments.push(twiddle_expr(n, exp, k1, scratch_idx));
        }

        col_blocks.push(quote! {
            {
                let mut col = [ #(#col_elements),* ];
                <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n1>>::dft::<INVERSE>(&mut col);
                #(#twiddle_assignments)*
            }
        });
    }

    // Each row occupies a disjoint initialized range of N2 elements.
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
                // SAFETY: START = K1 * N2 and K1 < N1 place this row within
                // the initialized N1 * N2 scratch array. Rows do not overlap.
                let row = unsafe { &mut *(scratch.as_mut_ptr().add(#start) as *mut [eunomia::Complex<F>; #n2]) };
                <F as crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n2>>::dft::<INVERSE>(row);
                #(#row_stores)*
            }
        });
    }

    let column_phase = phases.phase(quote! { #(#col_blocks)* });
    let row_phase = phases.phase(quote! { #(#row_blocks)* });
    let schedule_parameter = phases.schedule_parameter();
    let body = quote! {
        // SAFETY: Every element of `scratch` is written by a col_block before
        // any row_block reads it. The nested loop structure guarantees all N
        // positions are covered (col_block for j in 0..n2 writes scratch[k1*n2+j]
        // for all k1 in 0..n1 and all j in 0..n2 = all N indices).
        let mut scratch =
            std::mem::MaybeUninit::<[eunomia::Complex<F>; #n]>::uninit();
        let scratch_ptr = scratch.as_mut_ptr() as *mut eunomia::Complex<F>;
        #column_phase
        // SAFETY: The column phase writes every scratch slot as described
        // above. Generated code invokes the closure synchronously exactly
        // once; the schedule selects only its inlining boundary. All writes
        // therefore complete before this reference forms.
        let scratch = unsafe { scratch.assume_init_mut() };
        #row_phase
    };

    quote! {
        #inline_attr
        #[allow(unused_variables, unused_mut)]
        pub(crate) fn #fn_name<
            F: crate::application::execution::kernel::components::winograd::traits::WinogradScalar
                + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n1>
                + crate::application::execution::kernel::mixed_radix::traits::ShortDft<#n2>,
            const INVERSE: bool,
            #schedule_parameter
        >(
            data: &mut [eunomia::Complex<F>; #n],
        ) {
            #body
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
/// Each expression initializes its column's slot in uninitialized scratch.
fn twiddle_expr(n: usize, exp: usize, k1: usize, scratch_idx: usize) -> proc_macro2::TokenStream {
    let assign = |val: proc_macro2::TokenStream| -> proc_macro2::TokenStream {
        quote! {
            // SAFETY: SCRATCH_IDX = K1 * N2 + J is below N1 * N2.
            // The column loop writes each slot before the row phase reads it.
            unsafe { scratch_ptr.add(#scratch_idx).write(#val); }
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

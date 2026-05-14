import pathlib, os

path = 'd:/apollofft/crates/apollo-fft/src/application/execution/kernel/stockham/precision.rs'
lines = pathlib.Path(path).read_text(encoding='utf-8').splitlines()

base = 'd:/apollofft/crates/apollo-fft/src/application/execution/kernel/stockham/precision'
os.makedirs(base, exist_ok=True)

traits_content = """use num_complex::{Complex32, Complex64};
use super::fusion::{StockhamFused1, StockhamFused2, StockhamFused3, StockhamFused4};

pub(super) mod private {
    pub trait Sealed {}
}

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
pub(super) struct F64Stockham;
#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
pub(super) struct F32Stockham;

#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl private::Sealed for F64Stockham {}
#[cfg(any(
    test,
    not(all(target_arch = "x86_64", target_feature = "avx", target_feature = "fma"))
))]
impl private::Sealed for F32Stockham {}

""" + '\n'.join(lines[34:100]) + '\n' + '\n'.join(lines[156:193])

fusion_content = """use super::traits::private;

pub(super) struct StockhamFused1;
pub(super) struct StockhamFused2;
pub(super) struct StockhamFused3;
pub(super) struct StockhamFused4;

impl private::Sealed for StockhamFused1 {}
impl private::Sealed for StockhamFused2 {}
impl private::Sealed for StockhamFused3 {}
impl private::Sealed for StockhamFused4 {}

""" + '\n'.join(lines[100:156])

f64_content = """use num_complex::Complex64;
use super::traits::*;
use super::super::stage::stage_impl;
use super::super::butterfly::{stage_pair_impl, stage_triple_impl, stage_quad_impl};
use crate::application::execution::kernel::radix_stage::normalize_inplace_c64;
use super::super::avx::*;
use super::super::stage::stockham_f64_stage_is_l1_resident;

""" + '\n'.join(lines[194:265]) + '\n' + '\n'.join(lines[344:551])

f32_content = """use num_complex::Complex32;
use super::traits::*;
use super::super::stage::stage_impl;
use super::super::butterfly::{stage_pair_impl, stage_triple_impl, stage_quad_impl};
use crate::application::execution::kernel::radix_stage::normalize_inplace_c32;
use super::super::avx::*;
use super::super::stage::stockham_f32_stage_is_l1_resident;

""" + '\n'.join(lines[266:343]) + '\n' + '\n'.join(lines[552:764])

pathlib.Path(base + '/traits.rs').write_text(traits_content, encoding='utf-8')
pathlib.Path(base + '/fusion.rs').write_text(fusion_content, encoding='utf-8')
pathlib.Path(base + '/f64_impl.rs').write_text(f64_content, encoding='utf-8')
pathlib.Path(base + '/f32_impl.rs').write_text(f32_content, encoding='utf-8')

mod_content = """pub(crate) mod traits;
pub(crate) mod fusion;
pub(crate) mod f64_impl;
pub(crate) mod f32_impl;

pub(super) use traits::*;
pub(super) use fusion::*;
pub(super) use f64_impl::*;
pub(super) use f32_impl::*;
"""
pathlib.Path(base + '/mod.rs').write_text(mod_content, encoding='utf-8')

triple2_path = 'd:/apollofft/crates/apollo-fft/src/application/execution/kernel/stockham/avx/f64/triple_2.rs'
lines2 = pathlib.Path(triple2_path).read_text(encoding='utf-8').splitlines()
pathlib.Path(triple2_path).write_text('\n'.join(lines2[:301]) + '\n', encoding='utf-8')

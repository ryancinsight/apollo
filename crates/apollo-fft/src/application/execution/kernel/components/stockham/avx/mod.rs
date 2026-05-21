pub(crate) mod backend;
pub(crate) mod f32;
pub(crate) mod f64;
pub(crate) mod generic;

#[cfg(test)]
pub(crate) use f32::base::stage32_groups_one_avx_fma;
pub(crate) use f32::fixed::fixed_len64_32_avx_fma;
#[cfg(test)]
pub(crate) use f32::pair::stage_pair32_quarter_groups_two_avx_fma;
pub(crate) use f64::fixed::{fixed_len32_avx_fma, fixed_len64_avx_fma};

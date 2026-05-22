pub(crate) mod backend;
pub(crate) mod generic;
pub(crate) mod precise;
pub(crate) mod reduced;

pub(crate) use precise::fixed::{fixed_len32_precise_avx_fma, fixed_len64_precise_avx_fma};
#[cfg(test)]
pub(crate) use reduced::base::stage_reduced_groups_one_avx_fma;
pub(crate) use reduced::fixed::fixed_len64_reduced_avx_fma;
#[cfg(test)]
pub(crate) use reduced::pair::stage_pair_quarter_groups_two_reduced_avx_fma;

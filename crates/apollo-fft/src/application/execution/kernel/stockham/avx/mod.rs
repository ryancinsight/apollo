pub(crate) mod backend;
pub(crate) mod f32;
pub(crate) mod f64;
pub(crate) mod generic;

pub(crate) use f32::base::stage32_groups_one_avx_fma;
pub(crate) use f32::fixed::fixed_len64_32_avx_fma;
pub(crate) use f32::pair::{
    stage_pair32_groups_two_avx_fma, stage_pair32_quarter_groups_two_avx_fma,
};
pub(crate) use f32::quad::stockham_quad_groups_eight32;
pub(crate) use f32::triple_1::{
    stage_triple32_avx_fma, stage_triple32_low_live_avx_fma, stage_triple32_radix1_avx_fma,
};
pub(crate) use f32::triple_2::{
    stage_triple32_quarter_groups_one_avx_fma, stage_triple32_quarter_groups_two_avx_fma,
};
pub(crate) use f64::base::stage64_groups_one_avx_fma;
pub(crate) use f64::fixed::fixed_len64_avx_fma;
pub(crate) use f64::pair::stage_pair64_groups_two_avx_fma;
pub(crate) use f64::quad::stockham_quad_groups_eight64_low_live;
pub(crate) use f64::triple_1::{
    stage_triple64_low_live_avx_fma, stage_triple64_quarter_groups_one_avx_fma,
    stage_triple64_radix1_avx_fma,
};
pub(crate) use f64::triple_2::{
    stage_triple64_groups_eight_avx_fma, stage_triple64_throughput_avx_fma,
};

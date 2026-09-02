pub(crate) mod backend;
pub(crate) mod precise;
pub(crate) mod reduced;

#[cfg(test)]
pub(crate) use reduced::pair::stage_pair_quarter_groups_two_reduced_avx_fma;

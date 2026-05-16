use num_complex::Complex;

use super::arity::Radix;
use super::butterfly::stockham_stage;
use super::cache::CompositeCache;
use super::{Fused2, Fused3, Fused4, Fused5, Fused6, stockham_stage_fused};
use crate::application::execution::kernel::mixed_radix::traits::ShortWinogradScalar;
use crate::application::execution::kernel::tuning::RADIX_PARALLEL_CHUNK_THRESHOLD;
use crate::application::execution::policy::{ParallelPolicy, SyncPolicy};

pub(super) fn composite_core_with_radices<F: CompositeCache + ShortWinogradScalar>(
    data: &mut [Complex<F>],
    inverse: bool,
    radices: &[usize],
) {
    let n = data.len();
    if n <= 1 || radices.is_empty() {
        return;
    }
    debug_assert_eq!(radices.iter().product::<usize>(), n);

    let (all_twiddles, stage_offsets) = F::cached_twiddles(inverse, radices);

    F::with_scratch(n, |scratch| {
        let mut src_is_data = true;
        let mut prev_len = 1usize;
        let mut stage_idx = 0;

        while stage_idx < radices.len() {
            let r1 = radices[stage_idx];
            let offset1 = stage_offsets[stage_idx];
            let tw1 = &all_twiddles[offset1..offset1 + prev_len];

            macro_rules! fuse2 {
                ($p:literal, $q:literal) => {
                    if r1 == $p
                        && stage_idx + 1 < radices.len()
                        && radices[stage_idx + 1] == $q
                        && prev_len * ($p * $q) <= 256
                    {
                        let tws: [&[Complex<F>]; 2] = [
                            tw1,
                            &all_twiddles[stage_offsets[stage_idx + 1]..stage_offsets[stage_idx + 1] + prev_len * $p],
                        ];
                        if src_is_data {
                            stockham_stage_fused::<F, SyncPolicy, Fused2<Radix<$p>, Radix<$q>>>(data, scratch, prev_len, &tws, inverse);
                        } else {
                            stockham_stage_fused::<F, SyncPolicy, Fused2<Radix<$p>, Radix<$q>>>(scratch, data, prev_len, &tws, inverse);
                        }
                        src_is_data = !src_is_data;
                        prev_len *= $p * $q;
                        stage_idx += 2;
                        continue;
                    }
                };
            }
            macro_rules! fuse3 {
                ($p:literal, $q:literal, $r:literal) => {
                    if r1 == $p
                        && stage_idx + 1 < radices.len()
                        && radices[stage_idx + 1] == $q
                        && stage_idx + 2 < radices.len()
                        && radices[stage_idx + 2] == $r
                        && prev_len * ($p * $q * $r) <= 256
                    {
                        let tws: [&[Complex<F>]; 3] = [
                            tw1,
                            &all_twiddles[stage_offsets[stage_idx + 1]..stage_offsets[stage_idx + 1] + prev_len * $p],
                            &all_twiddles[stage_offsets[stage_idx + 2]..stage_offsets[stage_idx + 2] + prev_len * $p * $q],
                        ];
                        if src_is_data {
                            stockham_stage_fused::<F, SyncPolicy, Fused3<Radix<$p>, Radix<$q>, Radix<$r>>>(data, scratch, prev_len, &tws, inverse);
                        } else {
                            stockham_stage_fused::<F, SyncPolicy, Fused3<Radix<$p>, Radix<$q>, Radix<$r>>>(scratch, data, prev_len, &tws, inverse);
                        }
                        src_is_data = !src_is_data;
                        prev_len *= $p * $q * $r;
                        stage_idx += 3;
                        continue;
                    }
                };
            }
            macro_rules! fuse4 {
                ($p:literal, $q:literal, $r:literal, $s:literal) => {
                    if r1 == $p
                        && stage_idx + 1 < radices.len()
                        && radices[stage_idx + 1] == $q
                        && stage_idx + 2 < radices.len()
                        && radices[stage_idx + 2] == $r
                        && stage_idx + 3 < radices.len()
                        && radices[stage_idx + 3] == $s
                        && prev_len * ($p * $q * $r * $s) <= 256
                    {
                        let tws: [&[Complex<F>]; 4] = [
                            tw1,
                            &all_twiddles[stage_offsets[stage_idx + 1]..stage_offsets[stage_idx + 1] + prev_len * $p],
                            &all_twiddles[stage_offsets[stage_idx + 2]..stage_offsets[stage_idx + 2] + prev_len * $p * $q],
                            &all_twiddles[stage_offsets[stage_idx + 3]..stage_offsets[stage_idx + 3] + prev_len * $p * $q * $r],
                        ];
                        if src_is_data {
                            stockham_stage_fused::<F, SyncPolicy, Fused4<Radix<$p>, Radix<$q>, Radix<$r>, Radix<$s>>>(data, scratch, prev_len, &tws, inverse);
                        } else {
                            stockham_stage_fused::<F, SyncPolicy, Fused4<Radix<$p>, Radix<$q>, Radix<$r>, Radix<$s>>>(scratch, data, prev_len, &tws, inverse);
                        }
                        src_is_data = !src_is_data;
                        prev_len *= $p * $q * $r * $s;
                        stage_idx += 4;
                        continue;
                    }
                };
            }
            macro_rules! fuse5 {
                ($p:literal, $q:literal, $r:literal, $s:literal, $t:literal) => {
                    if r1 == $p
                        && stage_idx + 1 < radices.len()
                        && radices[stage_idx + 1] == $q
                        && stage_idx + 2 < radices.len()
                        && radices[stage_idx + 2] == $r
                        && stage_idx + 3 < radices.len()
                        && radices[stage_idx + 3] == $s
                        && stage_idx + 4 < radices.len()
                        && radices[stage_idx + 4] == $t
                        && prev_len * ($p * $q * $r * $s * $t) <= 256
                    {
                        let tws: [&[Complex<F>]; 5] = [
                            tw1,
                            &all_twiddles[stage_offsets[stage_idx + 1]..stage_offsets[stage_idx + 1] + prev_len * $p],
                            &all_twiddles[stage_offsets[stage_idx + 2]..stage_offsets[stage_idx + 2] + prev_len * $p * $q],
                            &all_twiddles[stage_offsets[stage_idx + 3]..stage_offsets[stage_idx + 3] + prev_len * $p * $q * $r],
                            &all_twiddles[stage_offsets[stage_idx + 4]..stage_offsets[stage_idx + 4] + prev_len * $p * $q * $r * $s],
                        ];
                        if src_is_data {
                            stockham_stage_fused::<F, SyncPolicy, Fused5<Radix<$p>, Radix<$q>, Radix<$r>, Radix<$s>, Radix<$t>>>(data, scratch, prev_len, &tws, inverse);
                        } else {
                            stockham_stage_fused::<F, SyncPolicy, Fused5<Radix<$p>, Radix<$q>, Radix<$r>, Radix<$s>, Radix<$t>>>(scratch, data, prev_len, &tws, inverse);
                        }
                        src_is_data = !src_is_data;
                        prev_len *= $p * $q * $r * $s * $t;
                        stage_idx += 5;
                        continue;
                    }
                };
            }
            macro_rules! fuse6 {
                ($p:literal, $q:literal, $r:literal, $s:literal, $t:literal, $u:literal) => {
                    if r1 == $p
                        && stage_idx + 1 < radices.len()
                        && radices[stage_idx + 1] == $q
                        && stage_idx + 2 < radices.len()
                        && radices[stage_idx + 2] == $r
                        && stage_idx + 3 < radices.len()
                        && radices[stage_idx + 3] == $s
                        && stage_idx + 4 < radices.len()
                        && radices[stage_idx + 4] == $t
                        && stage_idx + 5 < radices.len()
                        && radices[stage_idx + 5] == $u
                        && prev_len * ($p * $q * $r * $s * $t * $u) <= 256
                    {
                        let tws: [&[Complex<F>]; 6] = [
                            tw1,
                            &all_twiddles[stage_offsets[stage_idx + 1]..stage_offsets[stage_idx + 1] + prev_len * $p],
                            &all_twiddles[stage_offsets[stage_idx + 2]..stage_offsets[stage_idx + 2] + prev_len * $p * $q],
                            &all_twiddles[stage_offsets[stage_idx + 3]..stage_offsets[stage_idx + 3] + prev_len * $p * $q * $r],
                            &all_twiddles[stage_offsets[stage_idx + 4]..stage_offsets[stage_idx + 4] + prev_len * $p * $q * $r * $s],
                            &all_twiddles[stage_offsets[stage_idx + 5]..stage_offsets[stage_idx + 5] + prev_len * $p * $q * $r * $s * $t],
                        ];
                        if src_is_data {
                            stockham_stage_fused::<F, SyncPolicy, Fused6<Radix<$p>, Radix<$q>, Radix<$r>, Radix<$s>, Radix<$t>, Radix<$u>>>(data, scratch, prev_len, &tws, inverse);
                        } else {
                            stockham_stage_fused::<F, SyncPolicy, Fused6<Radix<$p>, Radix<$q>, Radix<$r>, Radix<$s>, Radix<$t>, Radix<$u>>>(scratch, data, prev_len, &tws, inverse);
                        }
                        src_is_data = !src_is_data;
                        prev_len *= $p * $q * $r * $s * $t * $u;
                        stage_idx += 6;
                        continue;
                    }
                };
            }
            fuse3!(5, 3, 3);
            fuse3!(7, 3, 3);
            fuse3!(7, 5, 2);
            fuse3!(5, 5, 3);
            fuse3!(7, 5, 3);
            fuse3!(5, 5, 5);
            fuse3!(7, 3, 2);
            fuse3!(5, 3, 2);
            fuse3!(7, 2, 2);
            fuse3!(5, 2, 2);
            fuse3!(3, 3, 3);
            fuse3!(3, 3, 2);
            fuse3!(3, 2, 2);
            fuse3!(11, 3, 3);
            fuse3!(11, 5, 2);
            fuse3!(11, 3, 2);
            fuse3!(11, 2, 2);
            fuse3!(13, 3, 3);
            fuse3!(13, 5, 2);
            fuse3!(13, 3, 2);
            fuse3!(13, 2, 2);
            fuse3!(17, 3, 2);
            fuse3!(17, 2, 2);
            fuse3!(23, 3, 2);
            fuse3!(23, 2, 2);
            fuse3!(7, 7, 2);
            fuse3!(7, 5, 5);
            fuse3!(7, 7, 3);
            fuse3!(11, 5, 3);
            fuse3!(13, 5, 3);
            fuse3!(11, 7, 2);
            fuse3!(13, 7, 2);
            fuse3!(23, 3, 3);
            fuse3!(5, 5, 2);
            fuse3!(17, 5, 2);
            fuse3!(17, 3, 3);
            fuse3!(11, 7, 3);
            fuse3!(23, 5, 2);
            fuse2!(7, 5);
            fuse2!(7, 3);
            fuse2!(7, 2);
            fuse2!(5, 5);
            fuse2!(5, 3);
            fuse2!(5, 2);
            fuse2!(3, 3);
            fuse2!(3, 2);
            fuse2!(2, 2);
            fuse2!(11, 3);
            fuse2!(11, 5);
            fuse2!(11, 7);
            fuse2!(13, 3);
            fuse2!(13, 5);
            fuse2!(13, 7);
            fuse2!(17, 3);
            fuse2!(17, 5);
            fuse2!(17, 7);
            fuse2!(23, 3);
            fuse2!(23, 5);
            fuse2!(23, 7);
            fuse2!(11, 2);
            fuse2!(13, 2);
            fuse2!(17, 2);
            fuse2!(23, 2);
            fuse4!(5, 3, 3, 2);
            fuse4!(7, 3, 2, 2);
            fuse4!(5, 3, 2, 2);
            fuse4!(7, 2, 2, 2);
            fuse4!(3, 3, 3, 2);
            fuse4!(5, 2, 2, 2);
            fuse4!(3, 3, 2, 2);
            fuse4!(3, 2, 2, 2);
            fuse4!(2, 2, 2, 2);
            fuse4!(7, 3, 3, 2);
            fuse4!(5, 5, 2, 2);
            fuse4!(5, 3, 3, 3);
            fuse4!(11, 3, 2, 2);
            fuse4!(13, 3, 2, 2);
            fuse4!(7, 5, 2, 2);
            fuse4!(5, 5, 3, 2);
            fuse4!(7, 3, 3, 3);
            fuse4!(7, 7, 2, 2);
            fuse4!(11, 3, 3, 2);
            fuse4!(23, 2, 2, 2);
            fuse4!(17, 2, 2, 2);
            fuse4!(11, 2, 2, 2);
            fuse4!(13, 2, 2, 2);
            fuse4!(5, 5, 3, 3);
            fuse4!(7, 5, 3, 2);
            fuse4!(11, 5, 2, 2);
            fuse5!(3, 3, 3, 2, 2);
            fuse5!(5, 3, 2, 2, 2);
            fuse5!(3, 3, 2, 2, 2);
            fuse5!(3, 2, 2, 2, 2);
            fuse5!(2, 2, 2, 2, 2);
            fuse5!(7, 3, 2, 2, 2);
            fuse5!(5, 2, 2, 2, 2);
            fuse5!(7, 2, 2, 2, 2);
            fuse5!(5, 5, 2, 2, 2);
            fuse5!(3, 3, 3, 3, 2);
            fuse5!(5, 3, 3, 2, 2);
            fuse5!(7, 3, 3, 2, 2);
            fuse5!(3, 3, 3, 3, 3);
            fuse5!(13, 2, 2, 2, 2);
            fuse5!(11, 2, 2, 2, 2);
            fuse6!(3, 3, 2, 2, 2, 2);
            fuse6!(3, 2, 2, 2, 2, 2);
            fuse6!(2, 2, 2, 2, 2, 2);
            fuse6!(5, 3, 2, 2, 2, 2);
            fuse6!(7, 2, 2, 2, 2, 2);
            fuse6!(5, 2, 2, 2, 2, 2);
            fuse6!(3, 3, 3, 2, 2, 2);

            // Single-stage fallback
            let stage_len = prev_len * r1;
            let groups = n / stage_len;
            let use_parallel =
                n >= RADIX_PARALLEL_CHUNK_THRESHOLD && stage_len >= 512 && groups >= 4;

            macro_rules! dispatch {
                ($Policy:ty, $src:expr, $dst:expr) => {
                    stockham_stage::<F, $Policy>($src, $dst, r1, prev_len, groups, stage_len, tw1, inverse)
                };
            }

            match (src_is_data, use_parallel) {
                (true,  true)  => dispatch!(ParallelPolicy, data,    scratch),
                (true,  false) => dispatch!(SyncPolicy,    data,    scratch),
                (false, true)  => dispatch!(ParallelPolicy, scratch, data),
                (false, false) => dispatch!(SyncPolicy,    scratch, data),
            }

            src_is_data = !src_is_data;
            prev_len = stage_len;
            stage_idx += 1;
        }

        if !src_is_data {
            data.copy_from_slice(scratch);
        }
    });
}

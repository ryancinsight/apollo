//! Lane execution borrows dead transpose storage across a synchronous join.

use super::dimension_1d::strategy::generic_four_step_applies;
use crate::application::execution::kernel::components::four_step;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::MixedRadixScalar;
use eunomia::Complex;

/// Existing multidimensional crossover, measured in total complex elements.
const PARALLEL_THRESHOLD: usize = 32_768;

/// Bytes of lane data one scheduled task carries when lanes need no workspace.
///
/// One lane per task — the previous shape — made a 64³ pass 3.3x *slower*
/// than running its 4,096 lanes serially on one thread: 734 µs against 225,
/// because moirai's per-task dispatch measures about 180 ns and a length-64
/// lane's codelet is 55 to 130 ns, so the scheduler outweighed the work it
/// scheduled. 64 KiB is 64 such lanes, which amortises the dispatch to well
/// under a percent, stays inside one core's L2, and measured 45 to 55 µs for
/// the same pass — faster than a quarter-megabyte task (57 to 68 µs) and than
/// serial (`dimension_3d::pass_attribution`, 2026-09-09). A lane at or above
/// this size is one task, as before.
const TASK_BYTES: usize = 64 * 1024;

/// Lanes per scheduled task for `lane_len`, never fewer than one.
fn lanes_per_task<T>(lane_len: usize) -> usize {
    let lane_bytes = lane_len.saturating_mul(core::mem::size_of::<T>()).max(1);
    (TASK_BYTES / lane_bytes).max(1)
}

/// Runs contiguous lanes using the same scratch role as a later transpose.
/// Rank-two staging owns the 3D X role and rank-three staging the 2D role,
/// so these borrows remain disjoint even for non-contiguous input views.
pub(super) fn contiguous<F, const FORWARD: bool, const RANK: usize>(
    active: &mut [F::Complex],
    lane_len: usize,
    direct: impl Fn(&mut [F::Complex]) + Send + Sync,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    if workspace(lane_len).is_some_and(|required| required <= active.len()) {
        let total_len = active.len();
        let run = |companion: &mut [F::Complex]| {
            execute::<F, FORWARD>(active, companion, lane_len, direct);
        };
        match RANK {
            2 => F::Complex::with_2d_scratch_impl(total_len, run),
            3 => F::Complex::with_3d_x_scratch_impl(total_len, run),
            _ => unreachable!("invariant: multidimensional FFT rank is two or three"),
        }
    } else {
        execute::<F, FORWARD>(active, &mut [], lane_len, direct);
    }
}

fn workspace(lane_len: usize) -> Option<usize> {
    generic_four_step_applies(lane_len)
        .then(|| four_step::scratch_len(lane_len))
        .flatten()
}

/// The inactive companion is disposable until the caller's next transpose.
/// Each task receives disjoint groups containing enough elements for one
/// complete FFT workspace. It reuses that workspace across its lanes. A final
/// incomplete group runs after the join, borrowing the full companion again.
/// This preserves a full-volume retained-memory bound without worker storage.
pub(super) fn execute<F, const FORWARD: bool>(
    active: &mut [F::Complex],
    companion: &mut [F::Complex],
    lane_len: usize,
    direct: impl Fn(&mut [F::Complex]) + Send + Sync,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
{
    assert!(lane_len > 0 && active.len().is_multiple_of(lane_len));
    crate::ensure_thread_local_scratch_hook_registered();
    let Some(required) = workspace(lane_len).filter(|&required| required <= companion.len()) else {
        // A task is a whole number of lanes, and `active` is a whole number of
        // lanes, so every task boundary is a lane boundary and the shorter
        // final task moirai may hand out still divides exactly.
        let task_len = lane_len * lanes_per_task::<F::Complex>(lane_len);
        moirai::for_each_chunk_mut_with::<moirai::AdaptiveWithThreshold<PARALLEL_THRESHOLD>, _, _>(
            active,
            task_len,
            |task| {
                #[cfg(all(test, not(miri)))]
                crate::application::execution::kernel::worker_quiescence::record_worker();
                let mut lanes = task.chunks_exact_mut(lane_len);
                for lane in &mut lanes {
                    direct(lane);
                }
                debug_assert!(
                    lanes.into_remainder().is_empty(),
                    "invariant: task boundaries fall on lane boundaries"
                );
            },
        );
        return;
    };
    assert!(companion.len() >= active.len());
    // A group is an integral number of lanes; both factors are bounded by
    // the companion length, which is a valid allocated complex slice.
    let group_len = required.div_ceil(lane_len) * lane_len;
    let prefix_len = active.len() / group_len * group_len;
    let (prefix, remainder) = active.split_at_mut(prefix_len);
    moirai::for_each_chunk_pair_mut_enumerated_with::<
        moirai::AdaptiveWithThreshold<PARALLEL_THRESHOLD>,
        _,
        _,
        _,
    >(
        prefix,
        &mut companion[..prefix_len],
        group_len,
        |_, group, scratch| {
            #[cfg(all(test, not(miri)))]
            crate::application::execution::kernel::worker_quiescence::record_worker();
            for lane in group.chunks_exact_mut(lane_len) {
                transform::<F, FORWARD>(lane, scratch);
            }
        },
    );
    for lane in remainder.chunks_exact_mut(lane_len) {
        transform::<F, FORWARD>(lane, &mut companion[..required]);
    }
}

fn transform<F, const FORWARD: bool>(lane: &mut [F::Complex], scratch: &mut [F::Complex])
where
    F: MixedRadixScalar<Complex = Complex<F>>,
{
    if FORWARD {
        four_step::four_step_fft::<F, false, false>(lane, scratch);
    } else {
        four_step::four_step_fft::<F, true, true>(lane, scratch);
    }
}

#[cfg(test)]
mod tests;

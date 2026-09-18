//! Lane execution borrows dead transpose storage across a synchronous join.

use super::dimension_1d::strategy::generic_four_step_applies;
use crate::application::execution::kernel::components::four_step;
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch;
use crate::application::execution::kernel::mixed_radix::{
    dispatch_inplace, forward_inplace, inverse_inplace, MixedRadixScalar,
};
use eunomia::Complex;

/// Existing multidimensional crossover, measured in total complex elements.
const PARALLEL_THRESHOLD: usize = 32_768;

/// The same crossover in bytes of complex f64 lane data, for a pass whose two
/// sides differ in element type and length: a real field beside its half
/// spectrum moves more than its output's element count says.
const PARALLEL_BYTES: usize = PARALLEL_THRESHOLD * core::mem::size_of::<[f64; 2]>();

/// Five logical lane groups expose the packed 32³ inverse's parallel work while
/// keeping the four-task complex 32×32×16 control serial. The element floor
/// remains sufficient: replacing it would also serialize the four-task
/// single-precision 32³ control. See `benches/lane_threshold.rs`.
const PARALLEL_TASKS: usize = 5;

/// Zero-sized direction strategy selecting a lane's kernel direction and
/// whether its inverse normalizes, resolved entirely at compile time so every
/// pass monomorphizes per strategy with no runtime branch (ADR 0067).
pub(crate) trait Direction: Copy + Send + Sync + 'static {
    /// `true` runs the forward (unnormalized) kernel; `false` runs the
    /// inverse kernel.
    const FORWARD: bool;
    /// `true` scales the inverse by `1/N`; meaningless when `FORWARD` is
    /// `true`, where it is always `false`.
    const NORMALIZE: bool;
}

/// The unnormalized forward transform.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Forward;

impl Direction for Forward {
    const FORWARD: bool = true;
    const NORMALIZE: bool = false;
}

/// The inverse transform, normalized by `1/N`: today's `FftwCompatible`
/// inverse.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Inverse;

impl Direction for Inverse {
    const FORWARD: bool = false;
    const NORMALIZE: bool = true;
}

/// The inverse transform, left unnormalized: the unscaled sum, equal to the
/// normalized inverse times the transformed volume's element count.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InverseUnnormalized;

impl Direction for InverseUnnormalized {
    const FORWARD: bool = false;
    const NORMALIZE: bool = false;
}

struct LaneTasks<T>(core::marker::PhantomData<fn() -> T>);

impl<T: 'static> moirai::ExecutionPolicy for LaneTasks<T> {
    fn parallelize(len: usize) -> bool {
        len >= PARALLEL_THRESHOLD
    }

    fn parallelize_chunks(len: usize, chunks: usize) -> bool {
        // Confine the extension to representations whose element floor spans
        // more than four full tasks; narrower types keep their established rule.
        Self::parallelize(len)
            || (core::mem::size_of::<T>()
                > (PARALLEL_TASKS - 1) * moirai::UNIT_TASK_BYTES / PARALLEL_THRESHOLD
                && chunks >= PARALLEL_TASKS)
    }
}

/// Runs contiguous lanes using the same scratch role as a later transpose.
/// Rank-two staging owns the 3D X role and rank-three staging the 2D role,
/// so these borrows remain disjoint even for non-contiguous input views.
pub(super) fn contiguous<F, D: Direction, const RANK: usize>(
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
            execute::<F, D>(active, companion, lane_len, direct);
        };
        match RANK {
            2 => F::Complex::with_2d_scratch_impl(total_len, run),
            3 => F::Complex::with_3d_x_scratch_impl(total_len, run),
            _ => unreachable!("invariant: multidimensional FFT rank is two or three"),
        }
    } else {
        execute::<F, D>(active, &mut [], lane_len, direct);
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
pub(super) fn execute<F, D: Direction>(
    active: &mut [F::Complex],
    companion: &mut [F::Complex],
    lane_len: usize,
    direct: impl Fn(&mut [F::Complex]) + Send + Sync,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
{
    assert!(lane_len > 0 && active.len().is_multiple_of(lane_len));
    crate::application::execution::kernel::scratch_hook::ensure_registered();
    let Some(required) = workspace(lane_len).filter(|&required| required <= companion.len()) else {
        each(active, lane_len, |_, lane| direct(lane));
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
                transform::<F, D>(lane, scratch);
            }
        },
    );
    for lane in remainder.chunks_exact_mut(lane_len) {
        transform::<F, D>(lane, &mut companion[..required]);
    }
}

/// Runs the lanes of `source` into `target`: each lane is copied into place
/// and transformed there while it is cache-resident, so a pass that must also
/// move the volume between buffers pays no separate copy.
///
/// `source` is consumed. Once a task has copied its group, that group of
/// `source` is dead and serves as the group's four-step workspace; the final
/// incomplete group copies first and then borrows the front of `source`,
/// dead by then. The retained-memory bound is `execute`'s, with no companion.
pub(super) fn execute_from<F, D: Direction>(
    source: &mut [F::Complex],
    target: &mut [F::Complex],
    lane_len: usize,
    direct: impl Fn(&mut [F::Complex]) + Send + Sync,
) where
    F: MixedRadixScalar<Complex = Complex<F>>,
{
    assert!(lane_len > 0 && source.len() == target.len() && target.len().is_multiple_of(lane_len));
    crate::application::execution::kernel::scratch_hook::ensure_registered();
    let Some(required) = workspace(lane_len).filter(|&required| required <= source.len()) else {
        paired(target, lane_len, source, lane_len, |outputs, inputs| {
            for (output, input) in outputs
                .chunks_exact_mut(lane_len)
                .zip(inputs.chunks_exact(lane_len))
            {
                output.copy_from_slice(input);
                direct(output);
            }
        });
        return;
    };
    let group_len = required.div_ceil(lane_len) * lane_len;
    let prefix_len = target.len() / group_len * group_len;
    let (prefix, remainder) = target.split_at_mut(prefix_len);
    let (source_prefix, source_remainder) = source.split_at_mut(prefix_len);
    moirai::for_each_chunk_pair_mut_enumerated_with::<
        moirai::AdaptiveWithThreshold<PARALLEL_THRESHOLD>,
        _,
        _,
        _,
    >(prefix, source_prefix, group_len, |_, group, staged| {
        #[cfg(all(test, not(miri)))]
        crate::application::execution::kernel::worker_quiescence::record_worker();
        group.copy_from_slice(staged);
        for lane in group.chunks_exact_mut(lane_len) {
            transform::<F, D>(lane, staged);
        }
    });
    remainder.copy_from_slice(source_remainder);
    for lane in remainder.chunks_exact_mut(lane_len) {
        transform::<F, D>(lane, &mut source[..required]);
    }
}

/// Runs `lane(index, lane)` over every `lane_len`-element lane of `data`,
/// several lanes to a scheduled task.
///
/// A task is a whole number of lanes, and `data` is a whole number of lanes,
/// so every task boundary is a lane boundary and the shorter final task moirai
/// may hand out still divides exactly.
pub(super) fn each<T: Send + 'static>(
    data: &mut [T],
    lane_len: usize,
    lane: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    assert!(lane_len > 0 && data.len().is_multiple_of(lane_len));
    moirai::for_each_unit_task_mut_with::<LaneTasks<T>, _, _, _, _>(
        data,
        lane_len,
        lane_len.saturating_mul(core::mem::size_of::<T>()),
        || (),
        |(), first_lane, lanes| {
            #[cfg(all(test, not(miri)))]
            crate::application::execution::kernel::worker_quiescence::record_worker();
            for (offset, one) in lanes.chunks_exact_mut(lane_len).enumerate() {
                lane(first_lane + offset, one);
            }
        },
    );
}

/// Runs `task(output_chunk, input_chunk)` once per scheduled task over the
/// paired lanes of two volumes holding the same number of lanes at different
/// lengths — a real field beside its half spectrum.
///
/// Parallel over `output`, with a task sized by the bytes both sides of its
/// lanes carry. `task` receives its whole scheduled group of lanes at once —
/// `output_chunk` a whole number of `output_lane`-sized lanes, `input_chunk`
/// the correspondingly-aligned `input_lane`-sized lanes — so a caller that
/// needs per-lane workspace acquires it once (e.g. through a thread-local
/// [`PlanScratch`](crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::PlanScratch)
/// role scoped to the closure) and reuses it across its own
/// `chunks_exact`/`chunks_exact_mut` loop over the group, rather than paying
/// one allocation per task.
pub(super) fn paired<A, B>(
    output: &mut [A],
    output_lane: usize,
    input: &[B],
    input_lane: usize,
    task: impl Fn(&mut [A], &[B]) + Send + Sync,
) where
    A: Send,
    B: Sync,
{
    assert!(output_lane > 0 && input_lane > 0);
    let lanes = output.len() / output_lane;
    assert!(
        output.len() == lanes * output_lane && input.len() == lanes * input_lane,
        "paired lanes: {} and {} elements are not {lanes} lanes of {output_lane} and {input_lane}",
        output.len(),
        input.len()
    );
    // Both sides count: the output's element count alone undercounts a pass
    // that also reads a wider input.
    let pair_bytes =
        output_lane * core::mem::size_of::<A>() + input_lane * core::mem::size_of::<B>();
    units(output, output_lane, pair_bytes, |first_lane, outputs| {
        let input_start = first_lane * input_lane;
        let input_end = input_start + outputs.len() / output_lane * input_lane;
        task(outputs, &input[input_start..input_end]);
    });
}

/// Runs `task(first_unit, units)` once per scheduled task over the
/// `unit_len`-element units of `output`, each unit costing `unit_bytes` of
/// work: its own bytes and whatever it reads elsewhere.
///
/// For a pass whose input is not laid out unit by unit beside its output, so
/// the task finds its input from the index of its first unit.
pub(super) fn units<A: Send>(
    output: &mut [A],
    unit_len: usize,
    unit_bytes: usize,
    task: impl Fn(usize, &mut [A]) + Send + Sync,
) {
    moirai::for_each_unit_task_mut_with::<moirai::WorkBytes<PARALLEL_BYTES>, _, _, _, _>(
        output,
        unit_len,
        unit_bytes,
        || (),
        |(), first_unit, outputs| {
            #[cfg(all(test, not(miri)))]
            crate::application::execution::kernel::worker_quiescence::record_worker();
            task(first_unit, outputs);
        },
    );
}

fn transform<F, D: Direction>(lane: &mut [F::Complex], scratch: &mut [F::Complex])
where
    F: MixedRadixScalar<Complex = Complex<F>>,
{
    // `{ !D::FORWARD }`/`{ D::NORMALIZE }` as inline const-generic arguments
    // would need `generic_const_exprs`, unstable on the pinned toolchain; the
    // three reachable `(FORWARD, NORMALIZE)` combinations are spelled out
    // instead. Every branch is a compile-time fact of the concrete `D`, so
    // this still monomorphizes to one straight-line call per strategy.
    if D::FORWARD {
        four_step::four_step_fft::<F, false, false>(lane, scratch);
    } else if D::NORMALIZE {
        four_step::four_step_fft::<F, true, true>(lane, scratch);
    } else {
        four_step::four_step_fft::<F, true, false>(lane, scratch);
    }
}

/// One direction's transform of a lane of the table's length: the cached
/// power-of-two twiddles where the length has them, the generic mixed radix
/// otherwise.
pub(super) fn lane_over<F, D: Direction>(
    twiddles: Option<&[F::Complex]>,
) -> impl Fn(&mut [F::Complex]) + Send + Sync + '_
where
    F: MixedRadixScalar<Complex = Complex<F>>,
    F::Complex: PlanScratch,
{
    move |lane: &mut [F::Complex]| {
        if D::FORWARD {
            match twiddles {
                Some(twiddles) => dispatch_inplace::<F, false, false>(lane, Some(twiddles)),
                None => forward_inplace::<F>(lane),
            }
        } else if D::NORMALIZE {
            match twiddles {
                Some(twiddles) => dispatch_inplace::<F, true, true>(lane, Some(twiddles)),
                None => inverse_inplace::<F>(lane),
            }
        } else {
            // `InverseUnnormalized`: `dispatch_inplace` routes `None` through
            // the same cached-twiddle lookup `inverse_inplace_unnorm` uses, so
            // this covers both the `Some` and `None` cases correctly.
            dispatch_inplace::<F, true, false>(lane, twiddles)
        }
    }
}

#[cfg(test)]
mod tests;

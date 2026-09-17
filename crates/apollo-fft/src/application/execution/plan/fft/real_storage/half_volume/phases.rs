//! Where the 64³ half-spectrum pair spends its time, sweep by sweep.
//!
//! The consumer census (`backlog.md#apollo-real-3d-pass-count`) put one
//! forward at 178-183 us against 87 us for a single touch of its data, so the
//! pair moves its volume several times. This probe times each sweep the pair
//! makes, on the same buffers the entries use:
//!
//! - `forward z` — the z lanes through the real split into the half volume;
//! - `forward xy` — the x and y chain on it (three moves, two lane passes);
//! - `inverse xy` — the same chain inverted;
//! - `inverse split` and `inverse unpack` — the two z sweeps the inverse made
//!   before `inverse_z_split` fused them, reconstructed here;
//! - `inverse fused` — [`super::inverse_z_split`], the one sweep that replaces
//!   them.
//!
//! Every arm runs once per repeat inside one loop, in alternating order, on a
//! buffer prepared outside the timed region, so host drift reaches every arm
//! alike and each difference stays a reading of the code (the consumer's own
//! first attempt, timed arm by arm in separate loops, reported a negative
//! kernel time). The split and fused inverses must produce the same bits:
//! they run the same arithmetic per lane in the same order.
//!
//! The x and y chain is also timed step by step, replicating
//! `dimension_3d::passes::chain` call for call: the three moves and the two
//! lane passes between them, so the whole-chain arms check the steps' sum.
//!
//! Machine-wide, not pinned: the lanes go to moirai, whose workers span the
//! machine, as `real_volume_arms` explains.

use std::time::{Duration, Instant};

use eunomia::Complex64;

use super::super::split;
use super::{forward_half, inverse_z_split};
use crate::application::execution::kernel::mixed_radix::scalar::plan_scratch::{
    with_3d_x_scratch, with_3d_y_scratch,
};
use crate::application::execution::plan::fft::dimension_3d::FftPlan3D;
use crate::application::execution::plan::fft::lanes;
use crate::application::execution::plan::fft::layout::transpose_matrices;
use crate::{PlanCacheProvider, Shape3D};

/// The x and y chain on a `[N, N, depth]` volume, step by step: move to
/// `(y, z, x)`, x lanes, move to `(z, x, y)`, y lanes, move back.
fn chain_steps<const FORWARD: bool>(
    plan: &FftPlan3D<f64>,
    data: &mut [Complex64],
    depth: usize,
) -> [Duration; 5] {
    let (nx, ny) = (N, N);
    let (x_table, y_table) = if FORWARD {
        (plan.twiddle_x_fwd.as_deref(), plan.twiddle_y_fwd.as_deref())
    } else {
        (plan.twiddle_x_inv.as_deref(), plan.twiddle_y_inv.as_deref())
    };
    let lane_x = lanes::lane_over::<f64, FORWARD>(x_table);
    let lane_y = lanes::lane_over::<f64, FORWARD>(y_table);
    let volume = nx * ny * depth;
    with_3d_x_scratch::<Complex64, _>(volume, |staged_x| {
        with_3d_y_scratch::<Complex64, _>(volume, |staged_y| {
            let mut steps = [Duration::ZERO; 5];
            let start = Instant::now();
            transpose_matrices(data, staged_x, 1, nx, ny * depth);
            steps[0] = start.elapsed();
            let start = Instant::now();
            lanes::execute::<f64, FORWARD>(staged_x, data, nx, &lane_x);
            steps[1] = start.elapsed();
            let start = Instant::now();
            transpose_matrices(staged_x, staged_y, 1, ny, depth * nx);
            steps[2] = start.elapsed();
            let start = Instant::now();
            lanes::execute::<f64, FORWARD>(staged_y, data, ny, &lane_y);
            steps[3] = start.elapsed();
            let start = Instant::now();
            transpose_matrices(staged_y, data, 1, depth, nx * ny);
            steps[4] = start.elapsed();
            steps
        })
    })
}

const N: usize = 64;
const REPEATS: usize = 300;
const WARM: usize = 20;

/// The inverse's z sweeps as they were before the fusion: the split in place
/// over every lane, then the unpack from the spectrum into the output.
fn split_then_unpack(
    plan: &FftPlan3D<f64>,
    bins: &mut [Complex64],
    values: &mut [f64],
) -> (Duration, Duration) {
    let depth = plan.nz_c();
    let half_lane = plan.half_z_lane::<false>();
    let start = Instant::now();
    lanes::each(bins, depth, |_, lane| {
        split::inverse_packed::<f64>(lane, N, plan.split_twiddles().iter().copied(), &half_lane);
    });
    let split_time = start.elapsed();
    let start = Instant::now();
    lanes::paired(values, N, &*bins, depth, |reals_group, lanes_group| {
        for (reals, lane) in reals_group
            .chunks_exact_mut(N)
            .zip(lanes_group.chunks_exact(depth))
        {
            split::unpack(&lane[..N / 2], reals);
        }
    });
    (split_time, start.elapsed())
}

#[derive(Default)]
struct Samples(Vec<Duration>);

impl Samples {
    fn push(&mut self, sample: Duration) {
        self.0.push(sample);
    }

    fn micros(&self) -> (f64, f64) {
        let mut sorted = self.0.clone();
        sorted.sort_unstable();
        let us = |d: Duration| d.as_secs_f64() * 1e6;
        (us(sorted[0]), us(sorted[sorted.len() / 2]))
    }
}

#[test]
#[ignore = "measurement instrument for the 64 cubed half pair's sweeps"]
fn half_pair_sweeps_at_64_cubed() {
    let plan = <f64 as PlanCacheProvider>::get_3d_plan(
        Shape3D::new(N, N, N).expect("invariant: the probe's extents are non-zero"),
    );
    let depth = plan.nz_c();
    let source: Vec<f64> = (0..N * N * N)
        .map(|i| (0.017 * i as f64).sin() + 0.25 * (0.031 * i as f64).cos())
        .collect();
    let mut spectrum = vec![Complex64::default(); N * N * depth];
    forward_half(&plan, &source, &mut spectrum);

    let mut half = vec![Complex64::default(); N * N * depth];
    let mut staged = vec![Complex64::default(); N * N * depth];
    let mut after_xy = vec![Complex64::default(); N * N * depth];
    let mut split_bins = vec![Complex64::default(); N * N * depth];
    let (mut split_out, mut fused_out) = (vec![0.0_f64; N * N * N], vec![0.0_f64; N * N * N]);

    let names = [
        "forward z",
        "forward xy",
        "inverse xy",
        "inverse split",
        "inverse unpack",
        "inverse fused",
        "fwd move 1",
        "fwd x lanes",
        "fwd move 2",
        "fwd y lanes",
        "fwd move 3",
        "inv move 1",
        "inv x lanes",
        "inv move 2",
        "inv y lanes",
        "inv move 3",
        "fwd z serial",
        "inv fused serial",
    ];
    let mut serial_half = vec![Complex64::default(); N * N * depth];
    let mut serial_out = vec![0.0_f64; N * N * N];
    let mut lane_buffer = vec![Complex64::default(); depth];
    let mut stepped = vec![Complex64::default(); N * N * depth];
    let mut stepped_inverse = vec![Complex64::default(); N * N * depth];
    let mut samples: Vec<Samples> = names.iter().map(|_| Samples::default()).collect();

    for repeat in 0..WARM + REPEATS {
        let record = repeat >= WARM;

        // Forward: the z sweep writes the half volume fresh from the field,
        // so the xy chain after it never compounds a previous repeat's growth.
        let start = Instant::now();
        lanes::paired(&mut half, depth, &source, N, |bins_group, reals_group| {
            let half_lane = plan.half_z_lane::<true>();
            for (bins, reals) in bins_group
                .chunks_exact_mut(depth)
                .zip(reals_group.chunks_exact(N))
            {
                split::forward(
                    reals,
                    bins,
                    plan.split_twiddles().iter().copied(),
                    &half_lane,
                );
            }
        });
        let forward_z = start.elapsed();
        let start = Instant::now();
        plan.xy_axes_inplace::<true>(&mut half, depth);
        let forward_xy = start.elapsed();

        // The same chain step by step, on the z sweep's output copied aside,
        // and its inverse on a copy of the forward spectrum; each must land
        // on the bits the whole chain produced.
        lanes::paired(
            &mut stepped,
            depth,
            &source,
            N,
            |bins_group, reals_group| {
                let half_lane = plan.half_z_lane::<true>();
                for (bins, reals) in bins_group
                    .chunks_exact_mut(depth)
                    .zip(reals_group.chunks_exact(N))
                {
                    split::forward(
                        reals,
                        bins,
                        plan.split_twiddles().iter().copied(),
                        &half_lane,
                    );
                }
            },
        );
        let forward_steps = chain_steps::<true>(&plan, &mut stepped, depth);
        assert!(
            stepped == half,
            "the stepped forward chain must match the chain"
        );
        stepped_inverse.copy_from_slice(&spectrum);
        let inverse_steps = chain_steps::<false>(&plan, &mut stepped_inverse, depth);

        // Inverse: the chain on a copy of the forward's spectrum, then both z
        // forms on copies of what it leaves, in an order that alternates.
        staged.copy_from_slice(&spectrum);
        let start = Instant::now();
        plan.xy_axes_inplace::<false>(&mut staged, depth);
        let inverse_xy = start.elapsed();
        assert!(
            stepped_inverse == staged,
            "the stepped inverse chain must match the chain"
        );
        after_xy.copy_from_slice(&staged);
        split_bins.copy_from_slice(&after_xy);

        let (split_time, unpack_time, fused_time);
        if repeat % 2 == 0 {
            (split_time, unpack_time) = split_then_unpack(&plan, &mut split_bins, &mut split_out);
            let start = Instant::now();
            inverse_z_split(&plan, &after_xy, &mut fused_out);
            fused_time = start.elapsed();
        } else {
            let start = Instant::now();
            inverse_z_split(&plan, &after_xy, &mut fused_out);
            fused_time = start.elapsed();
            (split_time, unpack_time) = split_then_unpack(&plan, &mut split_bins, &mut split_out);
        }
        assert_eq!(
            split_out
                .iter()
                .map(|v: &f64| v.to_bits())
                .collect::<Vec<_>>(),
            fused_out
                .iter()
                .map(|v: &f64| v.to_bits())
                .collect::<Vec<_>>(),
            "the fused z inverse must reproduce the two sweeps bit for bit"
        );

        // Serial controls: the same per-lane bodies on the calling thread
        // alone, so the parallel arms read against the work they spread.
        let half_lane = plan.half_z_lane::<true>();
        let start = Instant::now();
        for (bins, reals) in serial_half
            .chunks_exact_mut(depth)
            .zip(source.chunks_exact(N))
        {
            split::forward(
                reals,
                bins,
                plan.split_twiddles().iter().copied(),
                &half_lane,
            );
        }
        let forward_z_serial = start.elapsed();
        let inverse_lane = plan.half_z_lane::<false>();
        let start = Instant::now();
        for (reals, bins) in serial_out
            .chunks_exact_mut(N)
            .zip(after_xy.chunks_exact(depth))
        {
            lane_buffer.copy_from_slice(bins);
            split::inverse_packed::<f64>(
                &mut lane_buffer,
                N,
                plan.split_twiddles().iter().copied(),
                &inverse_lane,
            );
            split::unpack(&lane_buffer[..N / 2], reals);
        }
        let fused_serial = start.elapsed();
        assert!(
            serial_out
                .iter()
                .zip(&fused_out)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the serial fused inverse must match the parallel one bit for bit"
        );

        if record {
            let whole = [
                forward_z,
                forward_xy,
                inverse_xy,
                split_time,
                unpack_time,
                fused_time,
            ];
            for (slot, sample) in samples.iter_mut().zip(
                whole
                    .into_iter()
                    .chain(forward_steps)
                    .chain(inverse_steps)
                    .chain([forward_z_serial, fused_serial]),
            ) {
                slot.push(sample);
            }
        }
    }

    // The round trip lands back on the field, so every arm transformed real
    // data: the bound `tests/real_half_api` derives for the pair.
    let l1: f64 = source.iter().map(|v| v.abs()).sum();
    let bound = 2.0 * 16.0 * ((N * N * N) as f64).log2() * f64::EPSILON / 2.0 * l1;
    let error = fused_out
        .iter()
        .zip(&source)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        error <= bound,
        "round trip departs by {error:e} against {bound:e}"
    );

    println!("HPS 64 cubed half pair, machine-wide, {REPEATS} repeats: min / median us");
    for (name, sample) in names.iter().zip(&samples) {
        let (min, median) = sample.micros();
        println!("HPS {name:<15} {min:>9.1} {median:>9.1}");
    }
}

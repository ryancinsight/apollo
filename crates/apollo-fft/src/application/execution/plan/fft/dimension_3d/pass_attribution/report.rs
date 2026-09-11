use std::sync::atomic::Ordering;

use apollo_bench::{BenchmarkConfig, BenchmarkSuite};

use crate::application::execution::kernel::measurement_cores;
use crate::{FftPlan3D, Shape3D};

use super::{
    arms_for_extent, lane_pass, median_ps, min_ps, volume, EXTENTS, LANES_PER_PROCESSOR,
    TASK_WIDTHS,
};

fn processor_histogram(n: usize) {
    for slot in &LANES_PER_PROCESSOR {
        slot.store(0, Ordering::Relaxed);
    }
    let shape = Shape3D::new(n, n, n).expect("invariant: extents are non-zero");
    let plan = FftPlan3D::<f64>::new(shape);
    let mut data = volume(n);
    lane_pass::<true>(&plan, &mut data, true);

    let total: u32 = LANES_PER_PROCESSOR
        .iter()
        .map(|slot| slot.load(Ordering::Relaxed))
        .sum();
    println!("LANES n={n}: {total} lanes in one forward pass");
    let Some(selection) = measurement_cores::selected() else {
        println!("  host reports no processor class information");
        return;
    };
    for (class_label, class) in [
        ("performance", selection.performance_class()),
        ("efficiency", selection.efficiency_class()),
    ] {
        let Some(class) = class else { continue };
        let members: Vec<u32> = selection.processors_in_class(class).collect();
        let lanes: u32 = members
            .iter()
            .map(|&processor| {
                let slot = usize::try_from(processor).expect("invariant: processor index fits");
                LANES_PER_PROCESSOR[slot].load(Ordering::Relaxed)
            })
            .sum();
        let share = if total == 0 {
            0.0
        } else {
            100.0 * f64::from(lanes) / f64::from(total)
        };
        println!(
            "  {class_label}: {lanes} lanes ({share:.1}%) over {} processors",
            members.len()
        );
    }
    let busy: Vec<String> = LANES_PER_PROCESSOR
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.load(Ordering::Relaxed) > 0)
        .map(|(processor, slot)| format!("{processor}:{}", slot.load(Ordering::Relaxed)))
        .collect();
    println!("  per processor: {}", busy.join(" "));
}

#[test]
#[ignore = "measurement instrument for the 3-D axis-pass attribution"]
fn axis_pass_attribution() {
    if cfg!(debug_assertions) {
        eprintln!(
            "pass_attribution: built without optimization; re-run with  --cargo-profile bench-quick. No timings reported."
        );
        return;
    }

    // A discarded pass warms the freshly linked binary; only the second is
    // reported.
    let mut warmup = BenchmarkSuite::new(BenchmarkConfig::regression());
    for n in EXTENTS {
        arms_for_extent(&mut warmup, n);
    }
    drop(warmup);

    let mut suite = BenchmarkSuite::new(BenchmarkConfig::regression());
    for n in EXTENTS {
        arms_for_extent(&mut suite, n);
    }
    println!("PASS unpinned; every row is a round trip (two of the named operation)");
    let report = suite.report();
    print!("{report}");

    for n in EXTENTS {
        // Per forward: three lane passes and the three-move layout chain the
        // full transform runs; the per-axis pairs print beside it as what the
        // separate passes would pay. Each arm is a round trip, so halve the
        // lane and full rows; the chain is a cycle and counts whole.
        // Printed at the median and at the fastest sample: on a loaded host
        // the median of the full arm carries whatever else ran during it, and
        // the residual it leaves is then contention, not the plan.
        for (statistic, read) in [
            ("median", median_ps as fn(&str, &str) -> f64),
            ("min", min_ps),
        ] {
            let full = read(&report, &format!("unpinned/full/{n}")) / 2.0;
            let lanes = read(&report, &format!("unpinned/lanes-z/{n}")) / 2.0;
            let ty = read(&report, &format!("unpinned/transpose-y/{n}"));
            let tx = read(&report, &format!("unpinned/transpose-x/{n}"));
            let chain = read(&report, &format!("unpinned/transpose-chain/{n}"));
            let rotated = read(&report, &format!("unpinned/rotated-pair/{n}")) / 2.0;
            let pieces = 3.0 * lanes + chain;
            println!(
                "ATTRIBUTION n={n} {statistic}: forward {:.1} us; lanes 3 x {:.1} = {:.1} us ({:.0}%), chain {:.1} us ({:.0}%; the per-axis pairs {:.1} + {:.1} = {:.1}); pieces sum {:.1} us, unaccounted {:.1} us ({:.0}%)",
                full / 1e6,
                lanes / 1e6,
                3.0 * lanes / 1e6,
                100.0 * 3.0 * lanes / full,
                chain / 1e6,
                100.0 * chain / full,
                ty / 1e6,
                tx / 1e6,
                (ty + tx) / 1e6,
                pieces / 1e6,
                (full - pieces) / 1e6,
                100.0 * (full - pieces) / full,
            );
            println!(
                "ROTATED n={n} {statistic}: forward {:.1} us against the C-order {:.1} us ({:+.0}%)",
                rotated / 1e6,
                full / 1e6,
                100.0 * (rotated - full) / full,
            );
            let real = read(&report, &format!("unpinned/real-half-pair/{n}")) / 2.0;
            println!(
                "REAL n={n} {statistic}: half-spectrum transform {:.1} us against the complex {:.1} us ({:+.0}%)",
                real / 1e6,
                full / 1e6,
                100.0 * (real - full) / full,
            );
        }
        let lanes = median_ps(&report, &format!("unpinned/lanes-z/{n}")) / 2.0;
        for (statistic, read) in [
            ("median", median_ps as fn(&str, &str) -> f64),
            ("min", min_ps),
        ] {
            let axis = read(&report, &format!("unpinned/axis1-pass/{n}")) / 2.0;
            let pieces = read(&report, &format!("unpinned/transpose-y/{n}"))
                + read(&report, &format!("unpinned/lanes-z/{n}")) / 2.0;
            println!(
                "INTERACTION n={n} {statistic}: axis-1 pass {:.1} us against its pieces {:.1} us; gap {:.1} us ({:.0}%)",
                axis / 1e6, pieces / 1e6, (axis - pieces) / 1e6, 100.0 * (axis - pieces) / axis
            );
        }
        let ty_serial = median_ps(&report, &format!("unpinned/transpose-y/{n}"));
        let ty_parallel = median_ps(&report, &format!("unpinned/transpose-y-parallel/{n}"));
        println!(
            "TRANSPOSE n={n}: axis-1 pair serial {:.1} us; one matrix per task {:.1} us",
            ty_serial / 1e6,
            ty_parallel / 1e6
        );
        let serial = median_ps(&report, &format!("unpinned/lanes-serial/{n}")) / 2.0;
        let widths: Vec<String> = TASK_WIDTHS
            .iter()
            .map(|width| {
                let pass = median_ps(&report, &format!("unpinned/lanes-task{width}/{n}")) / 2.0;
                format!("{width} lanes/task {:.1} us", pass / 1e6)
            })
            .collect();
        println!(
            "SCHEDULING n={n}: shipped lanes::execute {:.1} us; serial on one thread {:.1} us; {}",
            lanes / 1e6,
            serial / 1e6,
            widths.join("; ")
        );
        processor_histogram(n);
    }
}

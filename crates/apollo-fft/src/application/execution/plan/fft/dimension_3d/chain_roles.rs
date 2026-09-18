//! The C-order chain through one scratch role against the two-role chain it
//! replaced (`APOLLO-MEM-3D-SCRATCH-ROLES`): the same lanes in the same axis
//! order, one binary, arms alternating each round. Unpinned, since the axis
//! passes run on the moirai pool across the machine.

use super::passes::{self, AxisLanes};
use crate::application::execution::plan::fft::lanes::lane_over;
use apollo_bench::{BenchmarkCase, BenchmarkConfig, BenchmarkSuite};
use core::time::Duration;
use eunomia::Complex64;

#[test]
#[ignore = "measurement instrument for the 3-D chain's scratch roles"]
fn chain_one_role_against_two() {
    let config =
        BenchmarkConfig::try_with_budgets(Duration::from_millis(200), Duration::from_millis(1500))
            .expect("invariant: both budgets are non-zero");
    for n in [64_usize, 128] {
        let shape = [n, n, n];
        let src: Vec<Complex64> = (0..n * n * n)
            .map(|i| {
                let x = i as f64;
                Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
            })
            .collect();
        let mut data = src.clone();
        let mut suite = BenchmarkSuite::new(config);
        for round in 0..3 {
            for forward in [true, false] {
                let direction = if forward { "forward" } else { "inverse" };
                suite.run(
                    BenchmarkCase::new("one-role", format!("{direction}-{round}"), n),
                    || {
                        data.copy_from_slice(&src);
                        if forward {
                            passes::all_axes::<f64, true, _, _, _>(
                                &mut data,
                                shape,
                                AxisLanes {
                                    x: lane_over::<f64, true>(None),
                                    y: lane_over::<f64, true>(None),
                                    z: lane_over::<f64, true>(None),
                                },
                            );
                        } else {
                            passes::all_axes::<f64, false, _, _, _>(
                                &mut data,
                                shape,
                                AxisLanes {
                                    x: lane_over::<f64, false>(None),
                                    y: lane_over::<f64, false>(None),
                                    z: lane_over::<f64, false>(None),
                                },
                            );
                        }
                    },
                );
                suite.run(
                    BenchmarkCase::new("two-role", format!("{direction}-{round}"), n),
                    || {
                        data.copy_from_slice(&src);
                        if forward {
                            passes::axis2::<f64, true>(&mut data, n, lane_over::<f64, true>(None));
                            passes::xy_axes::<f64, true>(
                                &mut data,
                                shape,
                                lane_over::<f64, true>(None),
                                lane_over::<f64, true>(None),
                            );
                        } else {
                            passes::xy_axes::<f64, false>(
                                &mut data,
                                shape,
                                lane_over::<f64, false>(None),
                                lane_over::<f64, false>(None),
                            );
                            passes::axis2::<f64, false>(
                                &mut data,
                                n,
                                lane_over::<f64, false>(None),
                            );
                        }
                    },
                );
            }
        }
        print!("{}", suite.report());
    }
}

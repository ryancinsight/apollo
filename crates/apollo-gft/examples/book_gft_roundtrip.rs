//! Graph Fourier transform round-trip on a three-node path graph.
//!
//! The GFT projects a vertex signal onto the Laplacian eigenbasis and the
//! inverse reconstructs it. Forward-then-inverse recovers the signal.

extern crate apollo_gft;
extern crate leto;

use apollo_gft::GftPlan;
use leto::{Array1, Array2};

fn main() {
    let adjacency =
        Array2::from_shape_vec([3, 3], vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            .expect("valid adjacency shape");
    let plan = GftPlan::from_adjacency(adjacency.view()).expect("valid path graph");
    let signal = Array1::from(vec![1.25, -0.5, 2.0]);
    let spectrum = plan.forward(&signal).expect("forward GFT");
    assert_eq!(spectrum.len(), signal.len(), "spectrum covers every vertex");
    let recovered = plan.inverse(&spectrum).expect("inverse GFT");
    for ((recovered, original), index) in recovered.iter().zip(signal.iter()).zip(0..) {
        assert!(
            (recovered - original).abs() < 1e-12,
            "round-trip recovers vertex {index}: {recovered} vs {original}"
        );
    }
    println!("ok");
}

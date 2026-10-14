//! Quantum Fourier transform of the two-qubit |00> basis state.
//!
//! The QFT is unitary, so |00> maps to the uniform superposition: every
//! output amplitude has magnitude 1/sqrt(4). The inverse recovers |00>.

extern crate apollo_qft;
extern crate eunomia;
extern crate leto;

use apollo_qft::{QftPlan, QuantumStateDimension};
use eunomia::Complex64;
use leto::Array1;

fn main() {
    let plan = QftPlan::new(QuantumStateDimension::new(4).expect("valid dimension"));
    let input = Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let output = plan.forward(&input).expect("forward QFT");
    for value in output.iter() {
        assert!(
            (value.norm() - 0.5).abs() < 1e-12,
            "uniform superposition has |amplitude| == 1/2, got {value}"
        );
    }
    let recovered = plan.inverse(&output).expect("inverse QFT");
    for (index, value) in recovered.iter().enumerate() {
        let expected = f64::from(index == 0);
        assert!(
            (value.re - expected).abs() < 1e-12,
            "round-trip recovers |00>: index {index} is {value}"
        );
        assert!(
            value.im.abs() < 1e-12,
            "round-trip has no imaginary residue"
        );
    }
    println!("ok");
}

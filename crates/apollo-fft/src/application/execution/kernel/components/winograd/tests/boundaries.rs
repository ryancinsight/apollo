use crate::application::execution::kernel::components::winograd::*;
use num_complex::Complex64;

// ── Impulse → all-ones ───────────────────────────────────────────────────────

#[test]
fn dft4_impulse_produces_all_ones() {
    let mut buf = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    dft4_array_impl::<f64, false, false>(&mut buf);
    for x in &buf {
        assert!((x - Complex64::new(1.0, 0.0)).norm() < 1e-14, "bin={x:?}");
    }
}

#[test]
fn dft7_impulse_produces_all_ones() {
    let mut buf = [Complex64::new(0.0, 0.0); 7];
    buf[0] = Complex64::new(1.0, 0.0);
    dft7_impl::<f64, false, false>(&mut buf);
    for x in &buf {
        assert!((x - Complex64::new(1.0, 0.0)).norm() < 1e-14, "bin={x:?}");
    }
}

// ── DC signal → energy in bin 0 only ─────────────────────────────────────────

#[test]
fn dft8_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 8];
    dft8_array_impl::<f64, false, false>(&mut buf);
    assert!((buf[0] - Complex64::new(8.0, 0.0)).norm() < 1e-12);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-12, "non-zero bin: {x:?}");
    }
}

#[test]
fn dft16_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 16];
    dft16_impl::<f64, false>(&mut buf);
    assert!((buf[0] - Complex64::new(16.0, 0.0)).norm() < 1e-11);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-11, "non-zero bin: {x:?}");
    }
}

#[test]
fn dft32_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 32];
    dft32_impl::<f64, false>(&mut buf);
    assert!((buf[0] - Complex64::new(32.0, 0.0)).norm() < 1e-11);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-11, "non-zero bin: {x:?}");
    }
}

#[test]
fn dft64_dc_produces_energy_in_bin0() {
    let mut buf = [Complex64::new(1.0, 0.0); 64];
    dft64_impl::<f64, false>(&mut buf);
    assert!((buf[0] - Complex64::new(64.0, 0.0)).norm() < 1e-11);
    for x in &buf[1..] {
        assert!(x.norm() < 1e-11, "non-zero bin: {x:?}");
    }
}

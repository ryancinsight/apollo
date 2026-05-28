//! Extended prime-radix coverage for the 11 and 13 leaves.

use super::{check_forward, check_inverse, check_roundtrip};

#[test]
fn forward_n11() {
    check_forward(11, 1e-13);
}
#[test]
fn forward_n22() {
    check_forward(22, 1e-13);
}
#[test]
fn forward_n33() {
    check_forward(33, 1e-12);
}
#[test]
fn forward_n34() {
    check_forward(34, 1e-12);
}
#[test]
fn forward_n46() {
    check_forward(46, 1e-12);
}
#[test]
fn forward_n121() {
    check_forward(121, 1e-11);
}
#[test]
fn forward_n143() {
    check_forward(143, 1e-11);
}
#[test]
fn forward_n352() {
    check_forward(352, 1e-10);
}
#[test]
fn roundtrip_n22() {
    check_roundtrip(22, 1e-13);
}
#[test]
fn roundtrip_n121() {
    check_roundtrip(121, 1e-11);
}
#[test]
fn roundtrip_n143() {
    check_roundtrip(143, 1e-11);
}
#[test]
fn roundtrip_n352() {
    check_roundtrip(352, 1e-10);
}
#[test]
fn inverse_n34() {
    check_inverse(34, 1e-12);
}
#[test]
fn inverse_n143() {
    check_inverse(143, 1e-11);
}

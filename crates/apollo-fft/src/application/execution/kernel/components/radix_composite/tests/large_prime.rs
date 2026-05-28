//! 17 and 23 prime-radix coverage and composites built on them.

use super::{check_forward, check_inverse, check_roundtrip};

#[test]
fn forward_n17() {
    check_forward(17, 1e-13);
}
#[test]
fn forward_n23() {
    check_forward(23, 1e-13);
}
#[test]
fn forward_n34_via_17() {
    check_forward(34, 1e-12);
}
#[test]
fn forward_n46_via_23() {
    check_forward(46, 1e-12);
}
#[test]
fn forward_n51() {
    check_forward(51, 1e-12);
} // 17×3
#[test]
fn forward_n69() {
    check_forward(69, 1e-12);
} // 23×3
#[test]
fn forward_n242() {
    check_forward(242, 1e-10);
} // 11²×2
#[test]
fn forward_n264() {
    check_forward(264, 1e-10);
} // 11×3×8
#[test]
fn forward_n289() {
    check_forward(289, 1e-10);
} // 17²
#[test]
fn forward_n2200() {
    check_forward(2200, 1e-8);
} // 11×5²×8
#[test]
fn roundtrip_n242() {
    check_roundtrip(242, 1e-11);
}
#[test]
fn roundtrip_n2200() {
    check_roundtrip(2200, 1e-9);
}
#[test]
fn inverse_n242() {
    check_inverse(242, 1e-10);
}

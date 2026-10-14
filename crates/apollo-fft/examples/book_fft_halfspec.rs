//! Half-spectrum descriptor of a 3-D real transform.
//!
//! `HalfSpectrum3D::from_shape` records the full real shape together with
//! the independent complex bin count along z, `nz/2 + 1`.

extern crate apollo_fft;

use apollo_fft::{HalfSpectrum3D, Shape3D};

fn main() {
    let shape = Shape3D::new(8, 8, 8).expect("valid shape");
    let hs = HalfSpectrum3D::from_shape(shape);
    println!(
        "full shape : {} x {} x {}",
        hs.full.nx(),
        hs.full.ny(),
        hs.full.nz()
    );
    println!("nz_c       : {}", hs.nz_c);
    assert_eq!(hs.nz_c, 5);

    let wide = HalfSpectrum3D::from_shape(Shape3D::new(256, 256, 256).expect("valid shape"));
    assert_eq!(wide.nz_c, 129);

    println!("half-spectrum assertions passed");
}

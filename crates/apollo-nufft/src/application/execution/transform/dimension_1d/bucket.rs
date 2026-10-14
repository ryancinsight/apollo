//! Bucket-indexed position sorting for 1D NUFFT spreading.

use crate::domain::metadata::grid::UniformDomain1D;
use crate::infrastructure::kernel::kaiser_bessel::bucket_count;
use eunomia::Complex64;

#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexedPoint1D {
    pub(crate) x: f64,
    pub(crate) value: Complex64,
    bucket: usize,
}

trait BucketIndexed {
    fn bucket(&self) -> usize;
}

impl BucketIndexed for IndexedPoint1D {
    #[inline]
    fn bucket(&self) -> usize {
        self.bucket
    }
}

impl BucketIndexed for (usize, f64, usize) {
    #[inline]
    fn bucket(&self) -> usize {
        self.2
    }
}

#[inline]
fn index_positions_by_bucket<T, F>(
    positions: &[f64],
    domain: UniformDomain1D,
    oversampled_len: usize,
    kernel_width: usize,
    mut project: F,
) -> Vec<T>
where
    T: BucketIndexed,
    F: FnMut(usize, f64, usize) -> T,
{
    let buckets = bucket_count(oversampled_len, kernel_width);
    let bucket_scale = buckets as f64 / domain.length();
    let mut indexed: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(original_index, &x)| {
            let x_mod = x.rem_euclid(domain.length());
            let bucket = ((x_mod * bucket_scale).floor() as usize).min(buckets - 1);
            project(original_index, x_mod, bucket)
        })
        .collect();
    indexed.sort_unstable_by_key(BucketIndexed::bucket);
    indexed
}

pub(crate) fn sort_points_1d(
    positions: &[f64],
    values: &[Complex64],
    domain: UniformDomain1D,
    oversampled_len: usize,
    kernel_width: usize,
) -> Vec<IndexedPoint1D> {
    index_positions_by_bucket(
        positions,
        domain,
        oversampled_len,
        kernel_width,
        |original_index, x_mod, bucket| IndexedPoint1D {
            x: x_mod,
            value: values[original_index],
            bucket,
        },
    )
}

pub(crate) fn sort_positions_1d(
    positions: &[f64],
    domain: UniformDomain1D,
    oversampled_len: usize,
    kernel_width: usize,
) -> Vec<(usize, f64, usize)> {
    index_positions_by_bucket(
        positions,
        domain,
        oversampled_len,
        kernel_width,
        |original_index, x_mod, bucket| (original_index, x_mod, bucket),
    )
}

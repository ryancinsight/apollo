//! Independent direct-DFT oracles for selected-axis STFT verification.

use eunomia::Complex32;

#[derive(Clone, Copy, Debug)]
pub(super) struct ReferenceComplex {
    pub(super) re: f64,
    pub(super) im: f64,
}

pub(super) fn frame_count(signal_len: usize, hop_len: usize) -> usize {
    1 + signal_len.div_ceil(hop_len)
}

pub(super) fn hann(index: usize, frame_len: usize) -> f64 {
    if frame_len <= 1 {
        return 1.0;
    }
    0.5 - 0.5 * (core::f64::consts::TAU * index as f64 / (frame_len - 1) as f64).cos()
}

pub(super) fn forward(signal: &[f32], frame_len: usize, hop_len: usize) -> Vec<ReferenceComplex> {
    let frames = frame_count(signal.len(), hop_len);
    let half = (frame_len / 2) as isize;
    let mut output = Vec::with_capacity(frames * frame_len);
    for frame in 0..frames {
        let center = (frame * hop_len) as isize;
        for frequency in 0..frame_len {
            let mut re = 0.0;
            let mut im = 0.0;
            for local in 0..frame_len {
                let signal_index = center - half + local as isize;
                let sample = usize::try_from(signal_index)
                    .ok()
                    .and_then(|index| signal.get(index))
                    .map_or(0.0, |value| f64::from(*value));
                let phase = -core::f64::consts::TAU * (frequency * local) as f64 / frame_len as f64;
                let windowed = hann(local, frame_len) * sample;
                re = windowed.mul_add(phase.cos(), re);
                im = windowed.mul_add(phase.sin(), im);
            }
            output.push(ReferenceComplex { re, im });
        }
    }
    output
}

pub(super) fn inverse(
    spectrum: &[Complex32],
    signal_len: usize,
    frame_len: usize,
    hop_len: usize,
) -> Vec<f64> {
    let frames = frame_count(signal_len, hop_len);
    assert_eq!(spectrum.len(), frames * frame_len);
    let mut frame_data = vec![0.0; spectrum.len()];
    for (frame, row) in spectrum.chunks_exact(frame_len).enumerate() {
        for local in 0..frame_len {
            let mut value = 0.0;
            for (frequency, bin) in row.iter().enumerate() {
                let phase = core::f64::consts::TAU * (frequency * local) as f64 / frame_len as f64;
                value = f64::from(bin.re).mul_add(
                    phase.cos(),
                    (-f64::from(bin.im)).mul_add(phase.sin(), value),
                );
            }
            frame_data[frame * frame_len + local] =
                value * hann(local, frame_len) / frame_len as f64;
        }
    }

    let half = (frame_len / 2) as isize;
    (0..signal_len)
        .map(|sample| {
            let mut overlap = 0.0;
            let mut weight = 0.0;
            for frame in 0..frames {
                let start = (frame * hop_len) as isize - half;
                let local = sample as isize - start;
                let Ok(local) = usize::try_from(local) else {
                    continue;
                };
                if local >= frame_len {
                    continue;
                }
                let window = hann(local, frame_len);
                overlap += frame_data[frame * frame_len + local];
                weight = window.mul_add(window, weight);
            }
            if weight > 0.0 {
                overlap / weight
            } else {
                0.0
            }
        })
        .collect()
}

pub(super) fn operation_bound(frame_len: usize, scale: f64) -> f64 {
    let radix_extent = if frame_len.is_power_of_two() {
        frame_len
    } else {
        (2 * frame_len - 1).next_power_of_two()
    };
    let transform_count = if frame_len.is_power_of_two() { 1 } else { 3 };
    let levels = radix_extent.ilog2() + 1;
    // A complex butterfly contributes at most 32 rounded real operations per
    // level. Bluestein traverses three radix transforms. The first-order
    // gamma(k) bound is therefore k*epsilon while k*epsilon remains far below
    // one for these test sizes. `scale` is the relevant input one-norm.
    let rounded_operations = 32 * transform_count * levels;
    f64::from(f32::EPSILON) * f64::from(rounded_operations) * scale.max(1.0)
}

pub(super) fn spectrum(frame_count: usize, frame_len: usize, phase_seed: usize) -> Vec<Complex32> {
    (0..frame_count * frame_len)
        .map(|index| {
            let frame = index / frame_len;
            let frequency = index % frame_len;
            let phase = core::f32::consts::TAU * (frequency * (frame + phase_seed + 1)) as f32
                / frame_len as f32;
            let amplitude = 0.25 + 0.125 * (frame + 1) as f32;
            Complex32::new(amplitude * phase.cos(), amplitude * phase.sin())
        })
        .collect()
}

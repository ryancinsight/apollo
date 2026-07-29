//! Thread-local scratch pools and helpers for 1D Short-Time Fourier Transform.

use crate::domain::contracts::error::StftResult;
use eunomia::Complex64;
use mnemosyne::scratch::ScratchPool;

thread_local! {
    pub(crate) static TYPED_SIGNAL64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static TYPED_SPECTRUM64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    pub(crate) static TYPED_FORWARD_OUTPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    pub(crate) static TYPED_INVERSE_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static FORWARD_FRAME_REAL_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static WINDOWED_FRAME_REAL_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static INVERSE_FRAME_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static INVERSE_COMPLEX_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    pub(crate) static INVERSE_OVERLAP_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    pub(crate) static INVERSE_WEIGHT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Below this frame length scalar windowing avoids scratch setup overhead.
pub(crate) const HERMES_WINDOW_FRAME_THRESHOLD: usize = 64;

/// Return `true` when `n > 0`.
#[must_use]
#[inline]
pub fn is_valid_length(n: usize) -> bool {
    n > 0
}

pub(crate) fn window_signal_frame_into(
    start: isize,
    signal: &[f64],
    window: &[f64],
    output: &mut [Complex64],
) {
    debug_assert_eq!(window.len(), output.len());
    if window.len() < HERMES_WINDOW_FRAME_THRESHOLD {
        window_signal_frame_scalar(start, signal, window, output);
        return;
    }
    FORWARD_FRAME_REAL_SCRATCH.with(|cell| {
        cell.with_scratch(window.len(), |frame| {
            WINDOWED_FRAME_REAL_SCRATCH.with(|windowed_cell| {
                windowed_cell.with_scratch(window.len(), |windowed| {
                    for (n, slot) in frame.iter_mut().enumerate() {
                        let signal_index = start + n as isize;
                        *slot = if signal_index >= 0 && (signal_index as usize) < signal.len() {
                            signal[signal_index as usize]
                        } else {
                            0.0
                        };
                    }
                    hermes_simd::elementwise_mul(frame, window, windowed)
                        .expect("STFT frame and window lengths match");
                    for (slot, value) in output.iter_mut().zip(windowed.iter().copied()) {
                        *slot = Complex64::new(value, 0.0);
                    }
                });
            });
        });
    });
}

pub(crate) fn window_signal_frame_scalar(
    start: isize,
    signal: &[f64],
    window: &[f64],
    output: &mut [Complex64],
) {
    for (n, slot) in output.iter_mut().enumerate() {
        let signal_index = start + n as isize;
        *slot = if signal_index >= 0 && (signal_index as usize) < signal.len() {
            Complex64::new(signal[signal_index as usize] * window[n], 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
    }
}

pub(crate) fn window_complex_real_frame_into(
    frame_complex: &[Complex64],
    window: &[f64],
    output: &mut [f64],
) {
    debug_assert_eq!(frame_complex.len(), output.len());
    debug_assert_eq!(window.len(), output.len());
    if output.len() >= HERMES_WINDOW_FRAME_THRESHOLD {
        FORWARD_FRAME_REAL_SCRATCH.with(|cell| {
            cell.with_scratch(output.len(), |frame| {
                for (slot, value) in frame.iter_mut().zip(frame_complex.iter()) {
                    *slot = value.re;
                }
                hermes_simd::elementwise_mul(frame, window, output)
                    .expect("STFT inverse frame and window lengths match");
            });
        });
    } else {
        for (slot, value) in output.iter_mut().zip(frame_complex.iter()) {
            *slot = value.re;
        }
        for (slot, factor) in output.iter_mut().zip(window.iter().copied()) {
            *slot *= factor;
        }
    }
}

pub(crate) fn with_forward_typed_workspaces<R>(
    signal_len: usize,
    spectrum_len: usize,
    f: impl FnOnce(&mut [f64], &mut [Complex64]) -> StftResult<R>,
) -> StftResult<R> {
    TYPED_SIGNAL64_SCRATCH.with(|signal_cell| {
        signal_cell.with_scratch(signal_len, |signal| {
            TYPED_FORWARD_OUTPUT64_SCRATCH.with(|spectrum_cell| {
                spectrum_cell.with_scratch(spectrum_len, |spectrum| f(signal, spectrum))
            })
        })
    })
}

pub(crate) fn with_inverse_typed_workspaces<R>(
    spectrum_len: usize,
    signal_len: usize,
    f: impl FnOnce(&mut [Complex64], &mut [f64]) -> StftResult<R>,
) -> StftResult<R> {
    TYPED_SPECTRUM64_SCRATCH.with(|spectrum_cell| {
        spectrum_cell.with_scratch(spectrum_len, |spectrum| {
            TYPED_INVERSE_OUTPUT64_SCRATCH.with(|signal_cell| {
                signal_cell.with_scratch(signal_len, |signal| f(spectrum, signal))
            })
        })
    })
}

pub(crate) fn with_inverse_wola_workspaces<R>(
    frames: usize,
    frame_len: usize,
    signal_len: usize,
    f: impl FnOnce(&mut [f64], &mut [Complex64], &mut [f64], &mut [f64]) -> StftResult<R>,
) -> StftResult<R> {
    let frame_work_len = frames * frame_len;
    INVERSE_FRAME_SCRATCH.with(|frame_cell| {
        frame_cell.with_scratch(frame_work_len, |flat_frames| {
            INVERSE_COMPLEX_SCRATCH.with(|complex_cell| {
                complex_cell.with_scratch(frame_work_len, |flat_complex| {
                    INVERSE_OVERLAP_SCRATCH.with(|overlap_cell| {
                        overlap_cell.with_scratch(signal_len, |overlap| {
                            INVERSE_WEIGHT_SCRATCH.with(|weight_cell| {
                                weight_cell.with_scratch(signal_len, |weight| {
                                    overlap.fill(0.0);
                                    weight.fill(0.0);
                                    f(flat_frames, flat_complex, overlap, weight)
                                })
                            })
                        })
                    })
                })
            })
        })
    })
}

#[cfg(test)]
pub(crate) fn typed_workspace_capacities() -> (usize, usize, usize, usize) {
    let signal = TYPED_SIGNAL64_SCRATCH.with(|cell| cell.capacity());
    let spectrum = TYPED_SPECTRUM64_SCRATCH.with(|cell| cell.capacity());
    let forward_output = TYPED_FORWARD_OUTPUT64_SCRATCH.with(|cell| cell.capacity());
    let inverse_output = TYPED_INVERSE_OUTPUT64_SCRATCH.with(|cell| cell.capacity());
    (signal, spectrum, forward_output, inverse_output)
}

#[cfg(test)]
pub(crate) fn forward_window_workspace_capacity() -> usize {
    FORWARD_FRAME_REAL_SCRATCH.with(|cell| cell.capacity())
}

#[cfg(test)]
pub(crate) fn inverse_wola_workspace_capacities() -> (usize, usize, usize, usize) {
    let frames = INVERSE_FRAME_SCRATCH.with(|cell| cell.capacity());
    let complex = INVERSE_COMPLEX_SCRATCH.with(|cell| cell.capacity());
    let overlap = INVERSE_OVERLAP_SCRATCH.with(|cell| cell.capacity());
    let weight = INVERSE_WEIGHT_SCRATCH.with(|cell| cell.capacity());
    (frames, complex, overlap, weight)
}

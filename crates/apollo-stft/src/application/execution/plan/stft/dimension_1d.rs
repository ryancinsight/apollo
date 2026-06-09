//! 1D Short-Time Fourier Transform plan.

#![allow(clippy::manual_memcpy)]

use super::storage::{
    validate_profile, StftRealOutputStorage, StftRealStorage, StftSpectrumInput,
    StftSpectrumStorage,
};
use crate::application::execution::kernel::hann::hann_window;
use crate::domain::contracts::error::{StftError, StftResult};
use apollo_fft::{FftPlan1D, PrecisionProfile, Shape1D};
use ndarray::Array1;
use num_complex::Complex64;
use mnemosyne::scratch::ScratchPool;

thread_local! {
    static TYPED_SIGNAL64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static TYPED_SPECTRUM64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    static TYPED_FORWARD_OUTPUT64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    static TYPED_INVERSE_OUTPUT64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static INVERSE_FRAME_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static INVERSE_COMPLEX_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
    static INVERSE_OVERLAP_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static INVERSE_WEIGHT_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
}

/// Return `true` when `n > 0`.
#[must_use]
pub fn is_valid_length(n: usize) -> bool {
    n > 0
}

/// Reusable short-time Fourier transform plan.
///
/// Stores a validated frame length, hop size, Hann analysis window, and FFT plan.
/// Construct with `StftPlan::new`; the FFT plan is allocated once and reused.
pub struct StftPlan {
    frame_len: usize,
    hop_len: usize,
    window: Array1<f64>,
    fft_plan: FftPlan1D<f64>,
}

impl StftPlan {
    /// Create a validated STFT plan with a Hann analysis window.
    ///
    /// # Errors
    /// Returns `Err` if `frame_len == 0`, `hop_len == 0`, or `hop_len > frame_len`.
    pub fn new(frame_len: usize, hop_len: usize) -> StftResult<Self> {
        if frame_len == 0 {
            return Err(StftError::EmptyFrameLength);
        }
        if hop_len == 0 {
            return Err(StftError::EmptyHopSize);
        }
        if hop_len > frame_len {
            return Err(StftError::HopExceedsFrame);
        }
        let window = hann_window(frame_len);
        let fft_plan = FftPlan1D::<f64>::new(
            Shape1D::new(frame_len).expect("STFT frame length must be valid"),
        );
        Ok(Self {
            frame_len,
            hop_len,
            window,
            fft_plan,
        })
    }

    /// Return the frame length.
    #[must_use]
    pub const fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Return the hop length.
    #[must_use]
    pub const fn hop_len(&self) -> usize {
        self.hop_len
    }

    /// Return the analysis window.
    #[must_use]
    pub fn window(&self) -> &Array1<f64> {
        &self.window
    }

    /// Return the number of frequency bins (equal to `frame_len`).
    #[must_use]
    pub const fn spectrum_len(&self) -> usize {
        self.frame_len
    }

    /// Return the number of frames for a signal of length `signal_len`.
    ///
    /// Frames are centered at `m * hop_len` for m in 0..frames.
    /// Includes frames whose non-zero window extent overlaps with \[0, signal_len).
    /// Returns 0 when `signal_len < frame_len`.
    #[must_use]
    pub fn frame_count(&self, signal_len: usize) -> usize {
        if signal_len < self.frame_len {
            0
        } else {
            1 + signal_len.div_ceil(self.hop_len)
        }
    }

    /// Forward STFT of a real-valued signal using the internal Hann window.
    ///
    /// Applies the Hann analysis window to each frame and computes the DFT.
    /// Returns a flat array of shape `[frames * spectrum_len]`.
    ///
    /// # Errors
    /// Returns `Err(StftError::InputTooShort)` when `signal.len() < frame_len`.
    pub fn forward(&self, signal: &Array1<f64>) -> StftResult<Array1<Complex64>> {
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.len());
        let mut output = Array1::<Complex64>::zeros(frames * self.spectrum_len());
        self.forward_into(signal, &mut output)?;
        Ok(output)
    }

    /// Forward STFT with a user-supplied analysis window.
    ///
    /// # Errors
    /// Returns `Err(StftError::WindowLengthMismatch)` when `window.len() != frame_len`.
    /// Returns `Err(StftError::InputTooShort)` when `signal.len() < frame_len`.
    pub fn forward_with_window(
        &self,
        signal: &Array1<f64>,
        window: &[f64],
    ) -> StftResult<Array1<Complex64>> {
        if window.len() != self.frame_len {
            return Err(StftError::WindowLengthMismatch);
        }
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.len());
        let mut output = Array1::<Complex64>::zeros(frames * self.spectrum_len());
        let signal_slice = signal.as_slice().expect("signal buffer must be contiguous");
        let output_slice = output
            .as_slice_mut()
            .expect("output buffer must be contiguous");
        self.forward_with_window_slice_inner(signal_slice, window, output_slice)?;
        Ok(output)
    }

    /// Forward STFT into a pre-allocated output buffer.
    ///
    /// # Errors
    /// Returns `Err(StftError::InputTooShort)` when `signal.len() < frame_len`.
    /// Returns `Err(StftError::LengthMismatch)` when `output.len() != frames * spectrum_len`.
    pub fn forward_into(
        &self,
        signal: &Array1<f64>,
        output: &mut Array1<Complex64>,
    ) -> StftResult<()> {
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let signal_slice = signal.as_slice().expect("signal buffer must be contiguous");
        let output_slice = output
            .as_slice_mut()
            .expect("output buffer must be contiguous");
        self.forward_f64_slice_into(signal_slice, output_slice)
    }

    /// Forward STFT for typed real input and typed complex output storage.
    pub fn forward_typed_into<T: StftRealStorage, O: StftSpectrumStorage>(
        &self,
        signal: &Array1<T>,
        output: &mut Array1<O>,
        profile: PrecisionProfile,
    ) -> StftResult<()> {
        validate_profile(profile, T::PROFILE)?;
        validate_profile(profile, O::PROFILE)?;
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.len());
        if output.len() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        with_forward_typed_workspaces(signal.len(), output.len(), |signal64, output64| {
            for (slot, value) in signal64.iter_mut().zip(signal.iter().copied()) {
                *slot = T::to_f64(value);
            }
            self.forward_f64_slice_into(signal64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = O::from_complex64(value);
            }
            Ok(())
        })
    }

    fn forward_f64_slice_into(&self, signal: &[f64], output: &mut [Complex64]) -> StftResult<()> {
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let window = self.window.as_slice().expect("window must be contiguous");
        self.forward_with_window_slice_inner(signal, window, output)
    }

    fn forward_with_window_slice_inner(
        &self,
        signal: &[f64],
        window: &[f64],
        output: &mut [Complex64],
    ) -> StftResult<()> {
        if window.len() != self.frame_len {
            return Err(StftError::WindowLengthMismatch);
        }
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.len());
        if output.len() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
            output,
            self.spectrum_len(),
            |m, out_chunk| {
                let start = m as isize * self.hop_len as isize - (self.frame_len / 2) as isize;
                for n in 0..self.frame_len {
                    let signal_index = start + n as isize;
                    out_chunk[n] = if signal_index >= 0 && (signal_index as usize) < signal.len() {
                        Complex64::new(signal[signal_index as usize] * window[n], 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                }
                self.fft_plan.forward_complex_slice_inplace(out_chunk);
            },
        );
        Ok(())
    }

    /// Inverse STFT via weighted overlap-add (WOLA).
    ///
    /// Normalization: each sample is divided by the sum of squared window values
    /// across all contributing frames. Returns zeros at positions with zero total weight.
    ///
    /// # Errors
    /// Returns `Err(StftError::LengthMismatch)` when spectrum length is inconsistent.
    pub fn inverse(
        &self,
        spectrum: &Array1<Complex64>,
        signal_len: usize,
    ) -> StftResult<Array1<f64>> {
        let frames = self.frame_count(signal_len);
        if spectrum.len() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        let mut output = Array1::<f64>::zeros(signal_len);
        self.inverse_into(spectrum, signal_len, &mut output)?;
        Ok(output)
    }

    /// Inverse STFT into a pre-allocated output buffer.
    ///
    /// Frame IFFTs are computed in parallel; overlap-add accumulation is sequential
    /// to avoid data races on shared output positions.
    ///
    /// # Errors
    /// Returns `Err(StftError::LengthMismatch)` when lengths are inconsistent.
    /// Returns `Err(StftError::InputTooShort)` when `signal_len < frame_len`.
    pub fn inverse_into(
        &self,
        spectrum: &Array1<Complex64>,
        signal_len: usize,
        output: &mut Array1<f64>,
    ) -> StftResult<()> {
        let spectrum_slice = spectrum
            .as_slice()
            .expect("spectrum buffer must be contiguous");
        let output_slice = output
            .as_slice_mut()
            .expect("output buffer must be contiguous");
        self.inverse_complex64_slice_into(spectrum_slice, signal_len, output_slice)
    }

    fn inverse_complex64_slice_into(
        &self,
        spectrum: &[Complex64],
        signal_len: usize,
        output: &mut [f64],
    ) -> StftResult<()> {
        let frames = self.frame_count(signal_len);
        if spectrum.len() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        if output.len() != signal_len {
            return Err(StftError::LengthMismatch);
        }
        if signal_len < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let window = self.window.as_slice().expect("window must be contiguous");
        with_inverse_wola_workspaces(
            frames,
            self.frame_len,
            signal_len,
            |flat_frames, flat_complex, overlap, weight| {
                moirai::for_each_chunk_pair_mut_enumerated_with::<moirai::Adaptive, _, _, _>(
                    flat_complex,
                    flat_frames,
                    self.frame_len,
                    |m, frame_complex, frame_out| {
                        let offset = m * self.spectrum_len();
                        for k in 0..self.spectrum_len() {
                            frame_complex[k] = spectrum[offset + k];
                        }
                        self.fft_plan.inverse_complex_slice_inplace(frame_complex);
                        for n in 0..self.frame_len {
                            frame_out[n] = frame_complex[n].re * window[n];
                        }
                    },
                );

                // Sequential overlap-add: avoids data races on shared output positions.
                for (m, frame_vals) in flat_frames.chunks(self.frame_len).enumerate() {
                    let start = m as isize * self.hop_len as isize - (self.frame_len / 2) as isize;
                    for n in 0..self.frame_len {
                        let signal_index = start + n as isize;
                        if signal_index >= 0 && (signal_index as usize) < signal_len {
                            let idx = signal_index as usize;
                            overlap[idx] += frame_vals[n];
                            weight[idx] += window[n] * window[n];
                        }
                    }
                }
                for i in 0..signal_len {
                    output[i] = if weight[i] > 0.0 {
                        overlap[i] / weight[i]
                    } else {
                        0.0
                    };
                }
                Ok(())
            },
        )
    }

    /// Inverse STFT for typed complex spectrum and typed real output storage.
    pub fn inverse_typed_into<T: StftSpectrumInput, O: StftRealOutputStorage>(
        &self,
        spectrum: &Array1<T>,
        signal_len: usize,
        output: &mut Array1<O>,
        profile: PrecisionProfile,
    ) -> StftResult<()> {
        validate_profile(profile, T::PROFILE)?;
        validate_profile(profile, O::PROFILE)?;
        let frames = self.frame_count(signal_len);
        if spectrum.len() != frames * self.spectrum_len() || output.len() != signal_len {
            return Err(StftError::LengthMismatch);
        }
        if signal_len < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        with_inverse_typed_workspaces(spectrum.len(), signal_len, |spectrum64, output64| {
            for (slot, value) in spectrum64.iter_mut().zip(spectrum.iter().copied()) {
                *slot = T::to_complex64(value);
            }
            self.inverse_complex64_slice_into(spectrum64, signal_len, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = O::from_f64(value);
            }
            Ok(())
        })
    }
}

fn with_forward_typed_workspaces<R>(
    signal_len: usize,
    spectrum_len: usize,
    f: impl FnOnce(&mut [f64], &mut [Complex64]) -> StftResult<R>,
) -> StftResult<R> {
    TYPED_SIGNAL64_SCRATCH.with(|signal_cell| {
        signal_cell.with_scratch(signal_len, |signal| {
            TYPED_FORWARD_OUTPUT64_SCRATCH.with(|spectrum_cell| {
                spectrum_cell.with_scratch(spectrum_len, |spectrum| {
                    f(signal, spectrum)
                })
            })
        })
    })
}

fn with_inverse_typed_workspaces<R>(
    spectrum_len: usize,
    signal_len: usize,
    f: impl FnOnce(&mut [Complex64], &mut [f64]) -> StftResult<R>,
) -> StftResult<R> {
    TYPED_SPECTRUM64_SCRATCH.with(|spectrum_cell| {
        spectrum_cell.with_scratch(spectrum_len, |spectrum| {
            TYPED_INVERSE_OUTPUT64_SCRATCH.with(|signal_cell| {
                signal_cell.with_scratch(signal_len, |signal| {
                    f(spectrum, signal)
                })
            })
        })
    })
}

fn with_inverse_wola_workspaces<R>(
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
fn typed_workspace_capacities() -> (usize, usize, usize, usize) {
    let signal = TYPED_SIGNAL64_SCRATCH.with(|cell| cell.capacity());
    let spectrum = TYPED_SPECTRUM64_SCRATCH.with(|cell| cell.capacity());
    let forward_output = TYPED_FORWARD_OUTPUT64_SCRATCH.with(|cell| cell.capacity());
    let inverse_output = TYPED_INVERSE_OUTPUT64_SCRATCH.with(|cell| cell.capacity());
    (signal, spectrum, forward_output, inverse_output)
}

#[cfg(test)]
fn inverse_wola_workspace_capacities() -> (usize, usize, usize, usize) {
    let frames = INVERSE_FRAME_SCRATCH.with(|cell| cell.capacity());
    let complex = INVERSE_COMPLEX_SCRATCH.with(|cell| cell.capacity());
    let overlap = INVERSE_OVERLAP_SCRATCH.with(|cell| cell.capacity());
    let weight = INVERSE_WEIGHT_SCRATCH.with(|cell| cell.capacity());
    (frames, complex, overlap, weight)
}

#[cfg(test)]
mod tests;

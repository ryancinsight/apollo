//! 1D Short-Time Fourier Transform plan.

use super::super::storage::{
    validate_profile, StftRealOutputStorage, StftRealStorage, StftSpectrumInput,
    StftSpectrumStorage,
};
use super::helpers::{
    leto_array1_from_slice, leto_view1_cow, window_complex_real_frame_into,
    window_signal_frame_into, with_forward_typed_workspaces, with_inverse_typed_workspaces,
    with_inverse_wola_workspaces,
};
use crate::application::execution::kernel::hann::hann_window;
use crate::domain::contracts::error::{StftError, StftResult};
use apollo_fft::{FftPlan1D, PrecisionProfile, Shape1D};
use leto::Array1;
use eunomia::Complex64;

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
    /// Returns `Err(StftError::InputTooShort)` when `signal.size() < frame_len`.
    pub fn forward(&self, signal: &Array1<f64>) -> StftResult<Array1<Complex64>> {
        if signal.size() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.size());
        let mut output = Array1::<Complex64>::zeros([frames * self.spectrum_len()]);
        self.forward_into(signal, &mut output)?;
        Ok(output)
    }

    /// Forward STFT from a Leto 1D signal view.
    ///
    /// Contiguous Leto views are borrowed directly; strided views are copied once
    /// into logical order before reusing the canonical Leto kernel.
    pub fn forward_leto(
        &self,
        signal: leto::ArrayView1<'_, f64>,
    ) -> StftResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let signal = leto_view1_cow(signal)?;
        let frames = self.frame_count(signal.len());
        let mut output = vec![Complex64::new(0.0, 0.0); frames * self.spectrum_len()];
        self.forward_f64_slice_into(signal.as_ref(), &mut output)?;
        leto_array1_from_slice(&output)
    }

    /// Forward STFT with a user-supplied analysis window.
    ///
    /// # Errors
    /// Returns `Err(StftError::WindowLengthMismatch)` when `window.len() != frame_len`.
    /// Returns `Err(StftError::InputTooShort)` when `signal.size() < frame_len`.
    pub fn forward_with_window(
        &self,
        signal: &Array1<f64>,
        window: &[f64],
    ) -> StftResult<Array1<Complex64>> {
        if window.len() != self.frame_len {
            return Err(StftError::WindowLengthMismatch);
        }
        if signal.size() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.size());
        let mut output = Array1::<Complex64>::zeros([frames * self.spectrum_len()]);
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
    /// Returns `Err(StftError::InputTooShort)` when `signal.size() < frame_len`.
    /// Returns `Err(StftError::LengthMismatch)` when `output.size() != frames * spectrum_len`.
    pub fn forward_into(
        &self,
        signal: &Array1<f64>,
        output: &mut Array1<Complex64>,
    ) -> StftResult<()> {
        if signal.size() < self.frame_len {
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
        if signal.size() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let frames = self.frame_count(signal.size());
        if output.size() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        with_forward_typed_workspaces(signal.size(), output.size(), |signal64, output64| {
            for (slot, value) in signal64.iter_mut().zip(signal.iter().copied()) {
                *slot = T::to_f64(value);
            }
            self.forward_f64_slice_into(signal64, output64)?;
            for (slot, value) in output.as_slice_mut().expect("contiguous output").iter_mut().zip(output64.iter().copied()) {
                *slot = O::from_complex64(value);
            }
            Ok(())
        })
    }

    /// Forward STFT from typed Leto input into typed Leto spectrum storage.
    pub fn forward_leto_typed<T: StftRealStorage, O: StftSpectrumStorage>(
        &self,
        signal: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> StftResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>> {
        validate_profile(profile, T::PROFILE)?;
        validate_profile(profile, O::PROFILE)?;
        let signal = leto_view1_cow(signal)?;
        let signal = Array1::from(signal.into_owned());
        let frames = self.frame_count(signal.size());
        let mut output = Array1::<O>::from_elem(
            [frames * self.spectrum_len()],
            O::from_complex64(Complex64::new(0.0, 0.0)),
        );
        self.forward_typed_into(&signal, &mut output, profile)?;
        leto_array1_from_slice(output.as_slice().expect("STFT output must be contiguous"))
    }

    pub(crate) fn forward_f64_slice_into(
        &self,
        signal: &[f64],
        output: &mut [Complex64],
    ) -> StftResult<()> {
        if signal.len() < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        let window = self.window.as_slice().expect("window must be contiguous");
        self.forward_with_window_slice_inner(signal, window, output)
    }

    pub(crate) fn forward_with_window_slice_inner(
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
                window_signal_frame_into(start, signal, window, out_chunk);
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
        if spectrum.size() != frames * self.spectrum_len() {
            return Err(StftError::LengthMismatch);
        }
        let mut output = Array1::<f64>::zeros([signal_len]);
        self.inverse_into(spectrum, signal_len, &mut output)?;
        Ok(output)
    }

    /// Inverse STFT from a Leto 1D spectrum view.
    ///
    /// Contiguous Leto views are borrowed directly; strided views are copied once
    /// into logical order before reusing the canonical Leto kernel.
    pub fn inverse_leto(
        &self,
        spectrum: leto::ArrayView1<'_, Complex64>,
        signal_len: usize,
    ) -> StftResult<leto::Array<f64, leto::MnemosyneStorage<f64>, 1>> {
        let spectrum = leto_view1_cow(spectrum)?;
        let mut output = vec![0.0; signal_len];
        self.inverse_complex64_slice_into(spectrum.as_ref(), signal_len, &mut output)?;
        leto_array1_from_slice(&output)
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

    pub(crate) fn inverse_complex64_slice_into(
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
                        frame_complex[..self.spectrum_len()]
                            .copy_from_slice(&spectrum[offset..(offset + self.spectrum_len())]);
                        self.fft_plan.inverse_complex_slice_inplace(frame_complex);
                        window_complex_real_frame_into(frame_complex, window, frame_out);
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
        if spectrum.size() != frames * self.spectrum_len() || output.size() != signal_len {
            return Err(StftError::LengthMismatch);
        }
        if signal_len < self.frame_len {
            return Err(StftError::InputTooShort);
        }
        with_inverse_typed_workspaces(spectrum.size(), signal_len, |spectrum64, output64| {
            for (slot, value) in spectrum64.iter_mut().zip(spectrum.iter().copied()) {
                *slot = T::to_complex64(value);
            }
            self.inverse_complex64_slice_into(spectrum64, signal_len, output64)?;
            for (slot, value) in output.as_slice_mut().expect("contiguous output").iter_mut().zip(output64.iter().copied()) {
                *slot = O::from_f64(value);
            }
            Ok(())
        })
    }

    /// Inverse STFT from typed Leto spectrum storage into typed Leto signal storage.
    pub fn inverse_leto_typed<T: StftSpectrumInput, O: StftRealOutputStorage>(
        &self,
        spectrum: leto::ArrayView1<'_, T>,
        signal_len: usize,
        profile: PrecisionProfile,
    ) -> StftResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>> {
        validate_profile(profile, T::PROFILE)?;
        validate_profile(profile, O::PROFILE)?;
        let spectrum = leto_view1_cow(spectrum)?;
        let spectrum = Array1::from(spectrum.into_owned());
        let mut output = Array1::<O>::from_elem([signal_len], O::from_f64(0.0));
        self.inverse_typed_into(&spectrum, signal_len, &mut output, profile)?;
        leto_array1_from_slice(output.as_slice().expect("STFT output must be contiguous"))
    }
}

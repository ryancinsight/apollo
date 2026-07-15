//! Reusable sliding DFT plan and state.
//!
//! `SdftState` owns the current real-valued window and tracked DFT bins. Each
//! update removes the oldest sample, appends the new sample, and updates all
//! tracked bins through the sliding DFT recurrence.

use crate::domain::contracts::error::{SdftError, SdftResult};
use crate::domain::metadata::window::SlidingDftConfig;
use crate::infrastructure::kernel::sliding::{
    direct_bins, direct_bins_into, update_bins, update_twiddles,
};
use apollo_fft::{f16, PrecisionProfile};
use eunomia::{Complex32, Complex64};
use mnemosyne::scratch::ScratchPool;
use std::collections::VecDeque;

thread_local! {
    static TYPED_WINDOW64_SCRATCH: ScratchPool<f64> = const { ScratchPool::new() };
    static TYPED_BINS64_SCRATCH: ScratchPool<Complex64> = const { ScratchPool::new() };
}

/// Reusable SDFT plan.
#[derive(Debug, Clone, PartialEq)]
pub struct SdftPlan {
    config: SlidingDftConfig,
    twiddles: Vec<Complex64>,
}

impl SdftPlan {
    /// Create a validated SDFT plan.
    pub fn new(window_len: usize, bin_count: usize) -> SdftResult<Self> {
        let config = SlidingDftConfig::new(window_len, bin_count)?;
        let twiddles = update_twiddles(window_len, bin_count);
        Ok(Self { config, twiddles })
    }

    /// Return the validated configuration.
    #[must_use]
    pub const fn config(&self) -> SlidingDftConfig {
        self.config
    }

    /// Return the plan window length.
    #[must_use]
    pub const fn window_len(&self) -> usize {
        self.config.window_len()
    }

    /// Return the tracked bin count.
    #[must_use]
    pub const fn bin_count(&self) -> usize {
        self.config.bin_count()
    }

    /// Create zero-initialized streaming state.
    #[must_use]
    pub fn zero_state(&self) -> SdftState {
        let window = vec![0.0; self.window_len()];
        SdftState::from_validated_window(self.clone(), window)
    }

    /// Create streaming state from an initial full window.
    pub fn state_from_window(&self, window: &[f64]) -> SdftResult<SdftState> {
        if window.len() != self.window_len() {
            return Err(SdftError::InitialWindowLengthMismatch);
        }
        Ok(SdftState::from_validated_window(
            self.clone(),
            window.to_vec(),
        ))
    }

    /// Create streaming state from a Leto 1D window view.
    ///
    /// C-contiguous views borrow directly; strided views copy once into
    /// oldest-to-newest logical order before state construction.
    pub fn state_from_window_leto(
        &self,
        window: leto::ArrayView1<'_, f64>,
    ) -> SdftResult<SdftState> {
        let window = apollo_leto_interop::view_cow(&window);
        self.state_from_window(&window)
    }

    /// Compute direct DFT bins for a full window using this plan's bin count.
    pub fn direct_bins(&self, window: &[f64]) -> SdftResult<Vec<Complex64>> {
        if window.len() != self.window_len() {
            return Err(SdftError::InitialWindowLengthMismatch);
        }
        direct_bins(window, self.bin_count())
    }

    /// Compute direct DFT bins from a Leto 1D window view.
    pub fn direct_bins_leto(
        &self,
        window: leto::ArrayView1<'_, f64>,
    ) -> SdftResult<leto::Array<Complex64, leto::MnemosyneStorage<Complex64>, 1>> {
        let window = apollo_leto_interop::view_cow(&window);
        let bins = self.direct_bins(&window)?;
        apollo_leto_interop::try_array1_from_slice(&bins).ok_or(SdftError::OutputBinLengthMismatch)
    }

    /// Compute direct DFT bins for a full window into caller-owned storage.
    pub fn direct_bins_into(&self, window: &[f64], output: &mut [Complex64]) -> SdftResult<()> {
        if window.len() != self.window_len() {
            return Err(SdftError::InitialWindowLengthMismatch);
        }
        if output.len() != self.bin_count() {
            return Err(SdftError::OutputBinLengthMismatch);
        }
        direct_bins_into(window, output)
    }

    /// Compute direct DFT bins for typed real input and typed complex output storage.
    pub fn direct_bins_typed_into<T: SdftRealStorage, O: SdftBinStorage>(
        &self,
        window: &[T],
        output: &mut [O],
        profile: PrecisionProfile,
    ) -> SdftResult<()> {
        validate_profile(profile, T::PROFILE)?;
        validate_profile(profile, O::PROFILE)?;
        if window.len() != self.window_len() {
            return Err(SdftError::InitialWindowLengthMismatch);
        }
        if output.len() != self.bin_count() {
            return Err(SdftError::OutputBinLengthMismatch);
        }
        with_typed_direct_workspaces(window.len(), output.len(), |input64, output64| {
            for (slot, value) in input64.iter_mut().zip(window.iter().copied()) {
                *slot = T::to_f64(value);
            }
            self.direct_bins_into(input64, output64)?;
            for (slot, value) in output.iter_mut().zip(output64.iter().copied()) {
                *slot = O::from_complex64(value);
            }
            Ok(())
        })
    }

    /// Compute direct DFT bins from typed Leto real input storage.
    pub fn direct_bins_leto_typed<T: SdftRealStorage, O: SdftBinStorage>(
        &self,
        window: leto::ArrayView1<'_, T>,
        profile: PrecisionProfile,
    ) -> SdftResult<leto::Array<O, leto::MnemosyneStorage<O>, 1>> {
        let window = apollo_leto_interop::view_cow(&window);
        let mut output = vec![O::from_complex64(Complex64::new(0.0, 0.0)); self.bin_count()];
        self.direct_bins_typed_into(&window, &mut output, profile)?;
        apollo_leto_interop::try_array1_from_slice(&output)
            .ok_or(SdftError::OutputBinLengthMismatch)
    }
}

fn with_typed_direct_workspaces<R>(
    window_len: usize,
    bin_count: usize,
    f: impl FnOnce(&mut [f64], &mut [Complex64]) -> SdftResult<R>,
) -> SdftResult<R> {
    TYPED_WINDOW64_SCRATCH.with(|window_cell| {
        window_cell.with_scratch(window_len, |window| {
            TYPED_BINS64_SCRATCH
                .with(|bins_cell| bins_cell.with_scratch(bin_count, |bins| f(window, bins)))
        })
    })
}

#[cfg(test)]
fn typed_direct_workspace_capacities() -> (usize, usize) {
    let window = TYPED_WINDOW64_SCRATCH.with(|cell| cell.capacity());
    let bins = TYPED_BINS64_SCRATCH.with(|cell| cell.capacity());
    (window, bins)
}

/// Real input storage accepted by typed SDFT direct-bin paths.
pub trait SdftRealStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert storage into owner `f64` arithmetic.
    fn to_f64(self) -> f64;

    /// View slice as `f32` if layout is identical.
    #[inline]
    fn as_f32_slice(slice: &[Self]) -> Option<&[f32]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `f32` if layout is identical.
    #[inline]
    fn as_f32_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        let _ = slice;
        None
    }
}

impl SdftRealStorage for f64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn to_f64(self) -> f64 {
        self
    }
}

impl SdftRealStorage for f32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline]
    fn as_f32_slice(slice: &[Self]) -> Option<&[f32]> {
        Some(slice)
    }

    #[inline]
    fn as_f32_slice_mut(slice: &mut [Self]) -> Option<&mut [f32]> {
        Some(slice)
    }
}

impl SdftRealStorage for f16 {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }
}

/// Complex output storage accepted by typed SDFT direct-bin paths.
pub trait SdftBinStorage: Copy + Send + Sync + 'static {
    /// Required precision profile.
    const PROFILE: PrecisionProfile;

    /// Convert owner arithmetic result back to storage.
    fn from_complex64(value: Complex64) -> Self;

    /// View slice as `Complex32` if layout is identical.
    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        let _ = slice;
        None
    }

    /// View mutable slice as `Complex32` if layout is identical.
    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        let _ = slice;
        None
    }
}

impl SdftBinStorage for Complex64 {
    const PROFILE: PrecisionProfile = PrecisionProfile::HIGH_ACCURACY_F64;

    fn from_complex64(value: Complex64) -> Self {
        value
    }
}

impl SdftBinStorage for Complex32 {
    const PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    fn from_complex64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }

    #[inline]
    fn as_c32_slice(slice: &[Self]) -> Option<&[Complex32]> {
        Some(slice)
    }

    #[inline]
    fn as_c32_slice_mut(slice: &mut [Self]) -> Option<&mut [Complex32]> {
        Some(slice)
    }
}

impl SdftBinStorage for [f16; 2] {
    const PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    fn from_complex64(value: Complex64) -> Self {
        [
            f16::from_f32(value.re as f32),
            f16::from_f32(value.im as f32),
        ]
    }
}

fn validate_profile(actual: PrecisionProfile, expected: PrecisionProfile) -> SdftResult<()> {
    if actual.matches_storage_and_compute(expected) {
        Ok(())
    } else {
        Err(SdftError::PrecisionMismatch)
    }
}

/// Stateful sliding DFT stream.
#[derive(Debug, Clone, PartialEq)]
pub struct SdftState {
    plan: SdftPlan,
    window: VecDeque<f64>,
    bins: Vec<Complex64>,
    updates: usize,
}

impl SdftState {
    fn from_validated_window(plan: SdftPlan, window: Vec<f64>) -> Self {
        let bins = direct_bins(&window, plan.bin_count())
            .expect("invariant: validated plan has consistent window_len and bin_count");
        Self {
            plan,
            window: VecDeque::from(window),
            bins,
            updates: 0,
        }
    }

    /// Push one new sample and return the updated bins.
    pub fn update(&mut self, sample: f64) -> &[Complex64] {
        let outgoing = self
            .window
            .pop_front()
            .expect("validated SDFT window is non-empty");
        self.window.push_back(sample);
        update_bins(&mut self.bins, &self.plan.twiddles, outgoing, sample);
        self.updates += 1;
        &self.bins
    }

    /// Return current tracked bins.
    #[must_use]
    pub fn bins(&self) -> &[Complex64] {
        &self.bins
    }

    /// Return current window in oldest-to-newest order.
    #[must_use]
    pub fn window(&self) -> Vec<f64> {
        self.window.iter().copied().collect()
    }

    /// Return the number of update calls applied to this state.
    #[must_use]
    pub const fn updates(&self) -> usize {
        self.updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn caller_owned_direct_bins_match_allocating_path() {
        let plan = SdftPlan::new(4, 4).expect("plan");
        let window = [1.0, -2.0, 0.5, 3.0];
        let expected = plan.direct_bins(&window).expect("direct");
        let mut output = vec![Complex64::new(0.0, 0.0); plan.bin_count()];
        plan.direct_bins_into(&window, &mut output)
            .expect("caller-owned direct");

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn typed_direct_bins_support_f64_f32_and_mixed_f16_storage() {
        let plan = SdftPlan::new(4, 4).expect("plan");
        let window64 = [1.0_f64, -2.0, 0.5, 3.0];
        let expected = plan.direct_bins(&window64).expect("direct");

        let mut out64 = vec![Complex64::new(0.0, 0.0); plan.bin_count()];
        plan.direct_bins_typed_into(&window64, &mut out64, PrecisionProfile::HIGH_ACCURACY_F64)
            .expect("typed f64 direct");
        for (actual, expected) in out64.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }

        let window32 = window64.map(|value| value as f32);
        let represented32: Vec<f64> = window32.iter().map(|value| f64::from(*value)).collect();
        let expected32 = plan.direct_bins(&represented32).expect("represented f32");
        let mut out32 = vec![Complex32::new(0.0, 0.0); plan.bin_count()];
        plan.direct_bins_typed_into(&window32, &mut out32, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed f32 direct");
        for (actual, expected) in out32.iter().zip(expected32.iter()) {
            assert!((f64::from(actual.re) - expected.re).abs() < 1.0e-5);
            assert!((f64::from(actual.im) - expected.im).abs() < 1.0e-5);
        }

        let window16 = window64.map(|value| f16::from_f32(value as f32));
        let represented16: Vec<f64> = window16
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect();
        let expected16 = plan.direct_bins(&represented16).expect("represented f16");
        let mut out16 = vec![[f16::from_f32(0.0); 2]; plan.bin_count()];
        plan.direct_bins_typed_into(
            &window16,
            &mut out16,
            PrecisionProfile::MIXED_PRECISION_F16_F32,
        )
        .expect("typed f16 direct");
        for (actual, expected) in out16.iter().zip(expected16.iter()) {
            let re_bound = expected.re.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            let im_bound = expected.im.abs() * 2.0_f64.powi(-10) + 2.0_f64.powi(-14);
            assert!((f64::from(actual[0].to_f32()) - expected.re).abs() <= re_bound);
            assert!((f64::from(actual[1].to_f32()) - expected.im).abs() <= im_bound);
        }
    }

    #[test]
    fn typed_direct_bins_reuse_workspace_capacity() {
        let plan = SdftPlan::new(4, 4).expect("plan");
        let window = [1.0_f32, -2.0, 0.5, 3.0];
        let mut first = vec![Complex32::new(0.0, 0.0); plan.bin_count()];
        let mut second = vec![Complex32::new(0.0, 0.0); plan.bin_count()];

        plan.direct_bins_typed_into(&window, &mut first, PrecisionProfile::LOW_PRECISION_F32)
            .expect("first typed direct");
        let after_first = typed_direct_workspace_capacities();
        assert!(after_first.0 >= plan.window_len());
        assert!(after_first.1 >= plan.bin_count());

        plan.direct_bins_typed_into(&window, &mut second, PrecisionProfile::LOW_PRECISION_F32)
            .expect("second typed direct");
        assert_eq!(typed_direct_workspace_capacities(), after_first);
        for (first, second) in first.iter().zip(second.iter()) {
            assert_eq!(first.re.to_bits(), second.re.to_bits());
            assert_eq!(first.im.to_bits(), second.im.to_bits());
        }
    }

    #[test]
    fn typed_direct_bins_reject_profile_storage_mismatch() {
        let plan = SdftPlan::new(4, 2).expect("plan");
        let window = [1.0_f32, -2.0, 0.5, 3.0];
        let mut output = vec![Complex32::new(0.0, 0.0); plan.bin_count()];
        assert!(matches!(
            plan.direct_bins_typed_into(&window, &mut output, PrecisionProfile::HIGH_ACCURACY_F64),
            Err(SdftError::PrecisionMismatch)
        ));
    }

    #[test]
    fn caller_owned_direct_bins_reject_wrong_output_length() {
        let plan = SdftPlan::new(4, 3).expect("plan");
        let window = [1.0, -2.0, 0.5, 3.0];
        let mut output = vec![Complex64::new(0.0, 0.0); 2];
        assert_eq!(
            plan.direct_bins_into(&window, &mut output).unwrap_err(),
            SdftError::OutputBinLengthMismatch
        );
    }

    #[test]
    fn leto_direct_bins_match_slice_reference() {
        let plan = SdftPlan::new(4, 4).expect("plan");
        let window = vec![1.0, -2.0, 0.5, 3.0];
        let input = leto::Array1::from_shape_vec([4], window.clone()).expect("leto input");
        let expected = plan.direct_bins(&window).expect("direct");

        let actual = plan
            .direct_bins_leto(input.view())
            .expect("leto direct bins");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_strided_direct_bins_match_slice_reference() {
        let plan = SdftPlan::new(4, 4).expect("plan");
        let backing = vec![1.0, 99.0, -2.0, 99.0, 0.5, 99.0, 3.0, 99.0];
        let input = leto::Array1::from_shape_vec([8], backing).expect("leto input");
        let strided = input.view().slice(&[(0, 8, 2)]).expect("strided view");
        let expected_window = [1.0, -2.0, 0.5, 3.0];
        let expected = plan.direct_bins(&expected_window).expect("direct");

        let actual = plan.direct_bins_leto(strided).expect("leto direct bins");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn leto_typed_direct_bins_match_typed_slice_reference() {
        let plan = SdftPlan::new(4, 4).expect("plan");
        let window = vec![1.0_f32, -2.0, 0.5, 3.0];
        let input = leto::Array1::from_shape_vec([4], window.clone()).expect("leto input");
        let mut expected = vec![Complex32::new(0.0, 0.0); plan.bin_count()];
        plan.direct_bins_typed_into(&window, &mut expected, PrecisionProfile::LOW_PRECISION_F32)
            .expect("typed slice direct");

        let actual = plan
            .direct_bins_leto_typed::<f32, Complex32>(
                input.view(),
                PrecisionProfile::LOW_PRECISION_F32,
            )
            .expect("typed leto direct");
        let actual = actual.view();
        let actual = actual.as_slice().expect("contiguous leto output");

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }

    #[test]
    fn leto_state_from_window_matches_slice_state() {
        let plan = SdftPlan::new(4, 4).expect("plan");
        let window = vec![1.0, -2.0, 0.5, 3.0];
        let input = leto::Array1::from_shape_vec([4], window.clone()).expect("leto input");
        let expected = plan.state_from_window(&window).expect("slice state");

        let actual = plan
            .state_from_window_leto(input.view())
            .expect("leto state");

        assert_eq!(actual.window(), expected.window());
        for (actual, expected) in actual.bins().iter().zip(expected.bins().iter()) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }
}

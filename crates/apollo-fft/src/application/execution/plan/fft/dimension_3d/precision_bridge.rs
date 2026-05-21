use super::*;
use crate::application::execution::plan::fft::workspace::uninit_copy_vec;

pub(super) trait Plan3dReal32: Copy + UninitWorkspaceElement {
    const NATIVE_PROFILE: PrecisionProfile;

    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    fn from_f32(value: f32) -> Self;
    fn from_f64(value: f64) -> Self;
}

impl Plan3dReal32 for f32 {
    const NATIVE_PROFILE: PrecisionProfile = PrecisionProfile::LOW_PRECISION_F32;

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        value
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl Plan3dReal32 for f16 {
    const NATIVE_PROFILE: PrecisionProfile = PrecisionProfile::MIXED_PRECISION_F16_F32;

    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self.to_f32())
    }

    #[inline]
    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        Self::from_f32(value as f32)
    }
}

impl FftPlan3D {
    pub(super) fn forward_real32<T: Plan3dReal32>(&self, input: &Array3<T>) -> Array3<Complex32> {
        let mut output = self.uninit_complex32_full();
        self.forward_real32_into(input, &mut output);
        output
    }

    pub(super) fn inverse_real32<T: Plan3dReal32>(&self, input: &Array3<Complex32>) -> Array3<T> {
        self.check_full_complex_shape(input.dim(), "inverse input");
        if self.precision == T::NATIVE_PROFILE {
            let mut data = input.clone();
            self.inverse_complex_inplace_f32(&mut data);
            self.project_real32(data)
        } else {
            let mut promoted = Array3::<Complex64>::from_shape_vec(
                (self.nx, self.ny, self.nz),
                uninit_copy_vec(input.len()),
            )
            .expect("uninit Complex64 3D buffer length must match input len");
            ndarray::Zip::from(&mut promoted)
                .and(input)
                .for_each(|out, value| {
                    *out = Complex64::new(f64::from(value.re), f64::from(value.im));
                });
            let inv = self.inverse_complex_to_real(&promoted);
            let mut result = Array3::<T>::from_shape_vec(
                (self.nx, self.ny, self.nz),
                uninit_copy_vec(inv.len()),
            )
            .expect("uninit real32 3D buffer length must match input len");
            ndarray::Zip::from(&mut result)
                .and(&inv)
                .for_each(|out, value| {
                    *out = T::from_f64(*value);
                });
            result
        }
    }

    pub(super) fn uninit_complex32_full(&self) -> Array3<Complex32> {
        Array3::from_shape_vec(
            (self.nx, self.ny, self.nz),
            uninit_copy_vec(self.nx * self.ny * self.nz),
        )
        .expect("uninitialized Complex32 3D buffer length must match plan shape")
    }

    pub(super) fn project_real32<T: Plan3dReal32>(&self, input: Array3<Complex32>) -> Array3<T> {
        let mut values = uninit_copy_vec(input.len());
        for (slot, value) in values.iter_mut().zip(input.iter()) {
            *slot = T::from_f32(value.re);
        }
        Array3::from_shape_vec((self.nx, self.ny, self.nz), values)
            .expect("projected 3D real32 buffer length must match plan shape")
    }

    pub(super) fn forward_real32_into<T: Plan3dReal32>(
        &self,
        input: &Array3<T>,
        output: &mut Array3<Complex32>,
    ) {
        self.check_real_shape(input.dim(), "forward input");
        self.check_full_complex_shape(output.dim(), "forward output");
        if self.precision == T::NATIVE_PROFILE {
            Zip::from(&mut *output).and(input).for_each(|out, &value| {
                *out = Complex32::new(value.to_f32(), 0.0);
            });
            self.forward_complex_inplace_f32(output);
        } else {
            output.assign(
                &self
                    .forward_real_to_complex(&input.mapv(T::to_f64))
                    .mapv(|value| Complex32::new(value.re as f32, value.im as f32)),
            );
        }
    }

    pub(super) fn inverse_real32_into<T: Plan3dReal32>(
        &self,
        input: &Array3<Complex32>,
        output: &mut Array3<T>,
        scratch: &mut Array3<Complex32>,
    ) {
        self.check_full_complex_shape(input.dim(), "inverse input");
        self.check_real_shape(output.dim(), "inverse output");
        self.check_full_complex_shape(scratch.dim(), "inverse scratch");
        if self.precision == T::NATIVE_PROFILE {
            scratch.assign(input);
            self.inverse_complex_inplace_f32(scratch);
            Zip::from(output).and(scratch).for_each(|out, value| {
                *out = T::from_f32(value.re);
            });
        } else {
            output.assign(
                &self
                    .inverse_complex_to_real(
                        &input
                            .mapv(|value| Complex64::new(f64::from(value.re), f64::from(value.im))),
                    )
                    .mapv(T::from_f64),
            );
        }
    }

    pub(super) fn forward_real_to_complex_into_full(
        &self,
        input: &Array3<f64>,
        output: &mut Array3<Complex64>,
    ) {
        self.check_real_shape(input.dim(), "forward input");
        self.check_full_complex_shape(output.dim(), "forward output");
        Zip::from(output.view_mut())
            .and(input.view())
            .for_each(|out, &value| *out = Complex64::new(value, 0.0));
        self.forward_complex_inplace(output);
    }

    pub(super) fn forward_complex_axis_pass(&self, data: &mut Array3<Complex64>) {
        self.axis_pass_forward(data, Axis(2));
        self.axis_pass_forward(data, Axis(1));
        self.axis_pass_forward(data, Axis(0));
    }

    pub(super) fn inverse_complex_axis_pass(&self, data: &mut Array3<Complex64>) {
        self.axis_pass_inverse(data, Axis(0));
        self.axis_pass_inverse(data, Axis(1));
        self.axis_pass_inverse(data, Axis(2));
    }

    pub(super) fn forward_complex_inplace_f32(&self, data: &mut Array3<Complex32>) {
        self.check_full_complex_shape(data.dim(), "forward input");
        self.axis_pass_forward_f32(data, Axis(2));
        self.axis_pass_forward_f32(data, Axis(1));
        self.axis_pass_forward_f32(data, Axis(0));
    }

    pub(super) fn inverse_complex_inplace_f32(&self, data: &mut Array3<Complex32>) {
        self.check_full_complex_shape(data.dim(), "inverse input");
        self.axis_pass_inverse_f32(data, Axis(0));
        self.axis_pass_inverse_f32(data, Axis(1));
        self.axis_pass_inverse_f32(data, Axis(2));
    }

    fn axis_pass_forward(&self, data: &mut Array3<Complex64>, axis: Axis) {
        self.axis_pass_complex(data, axis, true);
    }

    fn axis_pass_inverse(&self, data: &mut Array3<Complex64>, axis: Axis) {
        self.axis_pass_complex(data, axis, false);
    }

    fn axis_pass_forward_f32(&self, data: &mut Array3<Complex32>, axis: Axis) {
        self.axis_pass_complex_f32(data, axis, true);
    }

    fn axis_pass_inverse_f32(&self, data: &mut Array3<Complex32>, axis: Axis) {
        self.axis_pass_complex_f32(data, axis, false);
    }
}

//! NTT residue number theory plan.

#![warn(missing_docs)]
//! WGPU backend boundary for Apollo NTT.
//!
//! The execution scaffold is the shared `apollo-fft` transform transport
//! (ADR 0037); this module owns the NTT kernels, their domain names, and
//! the residue-field surface. The transform is exact modular arithmetic
//! over `u64`/`u32` residues — outside the scaffold's floating-point
//! element families — so the marker implements only the planner contract
//! and the surface lives on [`ModularExecution`].

/// Infrastructure boundary for the NTT kernels.
pub mod infrastructure;

pub struct ResiduePlan {
    len: usize,
    modulus: u64,
    primitive_root: u64,
}

impl ResiduePlan {
    /// Create a residue-plan payload with the canonical modulus and root.
    #[must_use]
    pub const fn new(len: usize) -> Self {
        Self {
            len,
            modulus: DEFAULT_MODULUS,
            primitive_root: DEFAULT_PRIMITIVE_ROOT,
        }
    }

    /// Create a residue-plan payload with an explicit field contract.
    #[must_use]
    pub const fn with_modulus(len: usize, modulus: u64, primitive_root: u64) -> Self {
        Self {
            len,
            modulus,
            primitive_root,
        }
    }

    /// Return the logical transform length carried by this payload.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Return whether the payload carries zero length.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Return the field modulus carried by this payload.
    #[must_use]
    pub const fn modulus(self) -> u64 {
        self.modulus
    }

    /// Return the primitive root carried by this payload.
    #[must_use]
    pub const fn primitive_root(self) -> u64 {
        self.primitive_root
    }

    /// Validate the residue-field contract and derive the transform root
    /// of unity `omega = g^((p-1)/n) mod p`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan rejection naming the violated constraint.
    pub fn validate_field(self) -> WgpuResult<u64> {
        let Self {
            len,
            modulus,
            primitive_root,
        } = self;
        if len == 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: length must be greater than zero"),
            });
        }
        if !len.is_power_of_two() {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: length must be a power of two"),
            });
        }
        if modulus < 2 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: modulus must be at least 2"),
            });
        }
        if modulus > u64::from(u32::MAX) || primitive_root > u64::from(u32::MAX) {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: accelerator storage requires u32 field values"),
            });
        }
        if (modulus - 1) % len as u64 != 0 {
            return Err(WgpuError::InvalidPlan {
                message: format!("invalid plan len={len}, modulus={modulus}, primitive_root={primitive_root}: transform length is not supported by the modulus"),
            });
        }
        Ok(mod_pow_u64(
            primitive_root,
            (modulus - 1) / len as u64,
            modulus,
        ))
    }
}

/// Metadata-preserving WGPU plan descriptor.
pub type NttWgpuPlan = apollo_fft::WgpuTransformPlan<NttGpuKernel>;

/// WGPU backend descriptor.
pub type NttWgpuBackend = apollo_fft::WgpuTransformBackend<NttGpuKernel>;

/// Residue-field surface of the NTT backend.
///
/// Exact finite-field transforms over `u64` residues, with `u32`
/// quantized forms and reusable host state for repeated dispatches of
/// one plan.
pub trait ModularExecution {
    /// Construct reusable host-side state for one validated plan.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan or provider failure.
    fn create_buffers(&self, plan: &NttWgpuPlan) -> WgpuResult<NttGpuBuffers>;

    /// Execute the forward NTT over the configured residue field.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_forward(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>>;

    /// Execute the forward NTT into reusable host state.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_forward_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()>;

    /// Execute the inverse NTT over the configured residue field.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse(&self, plan: &NttWgpuPlan, input: &[u64]) -> WgpuResult<Vec<u64>>;

    /// Execute the inverse NTT into reusable host state.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u64],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()>;

    /// Execute forward from exact `u32` residues into caller-owned
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_forward_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
    ) -> WgpuResult<()>;

    /// Execute inverse from exact `u32` residues into caller-owned
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse_quantized_into(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        output: &mut [u32],
    ) -> WgpuResult<()>;

    /// Execute forward exact residues into reusable host state.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_forward_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()>;

    /// Execute inverse exact residues into reusable host state.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse_quantized_with_buffers(
        &self,
        plan: &NttWgpuPlan,
        input: &[u32],
        buffers: &mut NttGpuBuffers,
    ) -> WgpuResult<()>;

    /// Execute a forward transform from a Leto host view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_forward_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u64>,
    ) -> WgpuResult<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>>;

    /// Execute an inverse transform from a Leto host view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u64>,
    ) -> WgpuResult<leto::Array<u64, leto::MnemosyneStorage<u64>, 1>>;

    /// Execute forward exact residues from a Leto host view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_forward_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>>;

    /// Execute inverse exact residues from a Leto host view.
    ///
    /// # Errors
    ///
    /// Returns an invalid-plan, length-mismatch, or provider failure.
    fn execute_inverse_quantized_leto(
        &self,
        plan: &NttWgpuPlan,
        input: leto::ArrayView1<'_, u32>,
    ) -> WgpuResult<leto::Array<u32, leto::MnemosyneStorage<u32>, 1>>;
}


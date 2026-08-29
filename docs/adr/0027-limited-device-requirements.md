# ADR 0027: Transform device-limit requirements

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking provider-boundary cleanup

## Context

ADR 0026 removed fifteen zero-behavior transform `try_default` factories but
correctly excluded NUFFT and STFT. Their factories originally set a
transform-specific `DeviceLimits::max_storage_buffers_per_shader_stage` lower
bound before delegating acquisition to Hephaestus. Retaining that behavior
inside an Apollo acquisition factory leaves two obsolete public wrappers and
continues to split provider ownership.

The 2026-08-28 STFT dense-FFT cutover removed Apollo's six-binding Bluestein
descriptor. STFT now composes four-binding domain kernels with provider-owned
selected-axis FFT plans, so its former limit theorem no longer exists.

## Decision

Delete `NufftWgpuBackend::try_default` and `StftWgpuBackend::try_default`
without aliases. NUFFT exposes `required_device_limits`, the single
authoritative description of its seven-binding fast-kernel requirement. STFT's
`required_device_limits` returns the Hephaestus default because its dense FFT
requirements are provider-owned. Test and benchmark boundaries acquire
`hephaestus_wgpu::WgpuDevice` directly with the applicable requirement, then
construct the transform backend through `new(device)`.

Hephaestus remains the only owner of adapter discovery, device creation,
feature mapping, native WGPU conversion, dense-FFT limits, and acquisition
errors. Apollo owns only resource lower bounds for its domain kernels.

## Resource-requirement theorem and evidence boundary

Let `b(K)` be the number of storage-buffer declarations visible to shader
stage `K`, and let `L` be the requested
`max_storage_buffers_per_shader_stage`. A dispatch whose descriptor uses `K`
requires `b(K) <= L`; otherwise a conforming provider must reject the device or
pipeline request before dispatch.

The fast NUFFT shader descriptors bind storage buffers `0..=6`, so
`b(K_nufft) = 7` and NUFFT requests `L_nufft = 7`. Hardware-free
value-semantic tests pin this value; Hephaestus performs the provider-side
validation when a device is acquired. STFT's remaining domain kernels bind no
more than four storage buffers, within the Hephaestus default, while prepared
FFT validation owns any denser provider requirement.

This is a resource-precondition proof sketch grounded in the current shader
declarations. It neither proves numerical transform correctness nor guarantees
that a host exposes a device satisfying the lower bound; existing CPU/GPU
differentials and provider-error tests remain the corresponding empirical
evidence.

## Consequences

- Apollo no longer publishes an adapter-acquisition wrapper for either
  limit-bearing backend.
- NUFFT resource metadata has one transform-local home and is reused by all
  direct Hephaestus acquisition callers; STFT uses provider defaults.
- Present-provider failures cannot be converted to a successful backend or an
  absent-adapter skip by the verification helpers.

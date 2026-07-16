# ADR 0027: Transform device-limit requirements

- Status: Accepted
- Date: 2026-07-16
- Change class: pre-1.0 breaking provider-boundary cleanup

## Context

ADR 0026 removed fifteen zero-behavior transform `try_default` factories but
correctly excluded NUFFT and STFT. Their factories also set a
transform-specific `DeviceLimits::max_storage_buffers_per_shader_stage` lower
bound before delegating acquisition to Hephaestus. Retaining that behavior
inside an Apollo acquisition factory leaves two obsolete public wrappers and
continues to split provider ownership.

## Decision

Delete `NufftWgpuBackend::try_default` and `StftWgpuBackend::try_default`
without aliases. Each backend instead exposes `required_device_limits`, the
single authoritative description of its static resource requirement. Test and
benchmark boundaries acquire `hephaestus_wgpu::WgpuDevice` directly with that
requirement, then construct the transform backend through `new(device)`.

Hephaestus remains the only owner of adapter discovery, device creation,
feature mapping, native WGPU conversion, and acquisition errors. Apollo owns
only the kernel resource lower bound.

## Resource-requirement theorem and evidence boundary

Let `b(K)` be the number of storage-buffer declarations visible to shader
stage `K`, and let `L` be the requested
`max_storage_buffers_per_shader_stage`. A dispatch whose descriptor uses `K`
requires `b(K) <= L`; otherwise a conforming provider must reject the device or
pipeline request before dispatch.

The fast NUFFT shader descriptors bind storage buffers `0..=6`, so
`b(K_nufft) = 7`. The STFT Bluestein chirp descriptor binds four working
buffers and two operation buffers, so `b(K_stft) = 6`. The two backend methods
therefore request `L_nufft = 7` and `L_stft = 6`, respectively. Hardware-free
value-semantic tests pin these values; Hephaestus performs the provider-side
validation when a device is acquired.

This is a resource-precondition proof sketch grounded in the current shader
declarations. It neither proves numerical transform correctness nor guarantees
that a host exposes a device satisfying the lower bound; existing CPU/GPU
differentials and provider-error tests remain the corresponding empirical
evidence.

## Consequences

- Apollo no longer publishes an adapter-acquisition wrapper for either
  limit-bearing backend.
- Resource metadata has one transform-local home and is reused by all direct
  Hephaestus acquisition callers.
- Present-provider failures cannot be converted to a successful backend or an
  absent-adapter skip by the verification helpers.

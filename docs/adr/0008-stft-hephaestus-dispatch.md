# ADR 0008: STFT dispatch through Hephaestus

## Status

Accepted on 2026-07-15.

Revised on 2026-08-28 after Hephaestus added selected-axis prepared FFT plans.
The revision removes Apollo's duplicate dense radix-2 and Bluestein ownership.

## Decision

`apollo-stft` retains frame planning, centered Hann packing, split/interleaved
conversion, synthesis-window application, weighted overlap-add, and only the
WGSL kernels for those STFT-domain operations. Hephaestus owns every dense
radix-2 and Bluestein kernel, pipeline, parameter block, bind group, command,
and scratch buffer.

`StftGpuBuffers` prepares two Hephaestus plans over a dense C-order
`[frame_count, frame_len]` frame plane with active axis `[1]`. The forward and
inverse plans therefore transform each row independently and never transform
across frames. Hephaestus inverse normalization uses the product of active
extents, which is exactly `frame_len`; Apollo's synthesis kernel must not apply
a second `1/frame_len` factor.

The same workspace retains five bound Apollo-domain dispatches, fixed GPU
buffers, and host upload/readback capacity. Forward execution records
pack/window, selected-axis FFT, and interleave in one grouped stream. Inverse
execution records deinterleave, selected-axis inverse FFT, synthesis window,
and overlap-add in one grouped stream. Power-of-two and non-power-of-two frame
lengths use this same path. Apollo owns no dense FFT descriptor or shader.

Leto remains the host-array boundary. Hephaestus remains the sole owner of
device acquisition, provider limits, command submission, synchronization, and
transfer. STFT no longer requests the former six-storage-binding Bluestein
limit; provider defaults and prepared-plan validation are authoritative.

## Mathematical contract

For a complete frame DFT, root-of-unity orthogonality makes the normalized
inverse recover the windowed analysis frame exactly. Synthesis applies the
same window, so weighted overlap-add evaluates
`sum_m x[t] w[t-mH]^2 / sum_m w[t-mH]^2`, which equals `x[t]` wherever the
denominator is non-zero. Command stream order establishes the corresponding
device write-before-read dependencies. This is an exact-arithmetic theorem;
CPU differential and reconstruction tests are empirical finite-precision
evidence rather than a machine-checked proof.

Selected-axis direct-DFT tests use deliberately distinct rows for lengths 8
and 6 so an accidental transform across the frame axis cannot pass. Inverse
tests include independent spectra and normalization sentinels. Reusable-buffer
tests execute two different signals and spectra through one workspace and pin
host storage identity. A whole-call host allocation census is not the ownership
oracle because WGPU encoder, submit, and mapping internals allocate opaque host
state; retained-storage identity plus Hephaestus lifecycle counters establish
the source-controlled reuse claim.

// ntt.wgsl — O(N log N) Cooley-Tukey DIT NTT butterfly and inverse scale.
//
// # Mathematical contract
//
// Forward NTT (Pollard 1971):
//   X[k] = sum_{j=0}^{N-1} x[j] * omega^{j k}  mod  m
// where omega is a primitive N-th root of unity in F_m (omega^N ≡ 1 mod m).
//
// The host applies bit-reversal permutation to the input before upload so that
// the in-place Cooley-Tukey DIT butterfly requires no reordering on the GPU.
//
// Butterfly at stage s (0-indexed), thread id (0 <= id < N/2):
//   half   = 1 << s
//   group  = id / half
//   offset = id % half
//   i      = group * (half << 1) + offset        (left  operand index)
//   j      = i + half                             (right operand index)
//   step   = n >> (s + 1)                         (= N / len, twiddle stride)
//   w      = twiddles[offset * step]              (= omega^{offset * N/len})
//   u      = data[i]
//   t      = w * data[j]  mod m
//   data[i] = (u + t)     mod m
//   data[j] = (u - t + m) mod m
//
// After log2(N) butterfly stages the array holds NTT(x).
//
// Inverse NTT appends one scale pass:
//   data[k] = data[k] * N^{-1}  mod m
//
// # Twiddle buffer layout
//
// `twiddles[k] = omega^k mod m`  for  k = 0 .. N/2 - 1.
// For stage s and butterfly offset j: twiddle = twiddles[j * (N >> (s+1))].
// This flat layout covers every stage without a separate per-stage table.
//
// # NttParams encoding
//
// `stage_or_ninv` is dual-purpose:
//   - ntt_butterfly: the butterfly stage index s (0 .. log2(N)-1)
//   - ntt_scale:     N^{-1} mod m
//
// # References
//   Pollard, J. M. (1971). The fast Fourier transform in a finite field.
//   Mathematics of Computation, 25(114), 365-374.
//   Cooley, J. W. & Tukey, J. W. (1965). An algorithm for the machine
//   calculation of complex Fourier series. Mathematics of Computation,
//   19(90), 297-301.

// --------------------------------------------------------------------------
// Uniform parameter block
// --------------------------------------------------------------------------

struct NttParams {
    /// Transform length N.
    n:             u32,
    /// Butterfly entry: stage index s.  Scale entry: N^{-1} mod modulus.
    stage_or_ninv: u32,
    /// Prime modulus m.
    modulus:       u32,
    /// Padding to satisfy 16-byte struct alignment.
    _pad:          u32,
}

// Binding 0: in-place data buffer (read-write).
@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

// Binding 1: precomputed flat twiddle array (read-only).
//   twiddles[k] = omega^k mod modulus,  k = 0 .. N/2 - 1.
@group(0) @binding(1)
var<storage, read> twiddles: array<u32>;

// Binding 2: per-stage uniform parameters (dynamic offset).
@group(0) @binding(2)
var<uniform> params: NttParams;

// --------------------------------------------------------------------------
// Modular arithmetic helpers
// --------------------------------------------------------------------------

/// Modular addition without overflow.
///
/// Precondition: 0 <= a, b < m.
/// a + b fits in u32 because a + b < 2m < 2^31 (m < 2^30 for all supported moduli).
fn mod_add(a: u32, b: u32, m: u32) -> u32 {
    let s = a + b;
    return select(s, s - m, s >= m);
}

/// Modular subtraction without underflow.
///
/// Precondition: 0 <= a, b < m.
/// When a < b: a + m - b = a - b + m, and 0 < a + m - b < m. No overflow because
/// a < m and m < 2^30, so a + m < 2^31.
fn mod_sub(a: u32, b: u32, m: u32) -> u32 {
    // select(false_value, true_value, condition)
    return select(a + m - b, a - b, a >= b);
}

/// Modular multiplication via binary (Russian-peasant) accumulation.
///
/// WGSL exposes only u32 arithmetic; a*b can reach (2^30)^2 = 2^60 which overflows
/// u32.  The binary method keeps every intermediate value strictly below m by
/// replacing multiplication with iterated mod_add, incurring ~30 iterations for
/// the 998244353 modulus.  Each iteration is two branches + two u32 additions,
/// which is fast on GPU compared to a full 64-bit multiply unavailable in WGSL 1.0.
fn mod_mul(a: u32, b: u32, m: u32) -> u32 {
    var acc:  u32 = 0u;
    var base: u32 = a % m;
    var exp:  u32 = b;
    loop {
        if exp == 0u { break; }
        if (exp & 1u) != 0u {
            acc = mod_add(acc, base, m);
        }
        base = mod_add(base, base, m);
        exp  = exp >> 1u;
    }
    return acc;
}

// --------------------------------------------------------------------------
// Entry point 1: Cooley-Tukey DIT butterfly pass
// --------------------------------------------------------------------------

/// One Cooley-Tukey DIT butterfly stage for in-place NTT.
///
/// # Dispatch
/// Dispatch `ceil(N/2 / 64)` workgroups.  Each thread handles one butterfly
/// pair (i, j), so there are exactly N/2 active threads per stage.
/// Threads with `id >= N/2` return immediately.
///
/// # No data races
/// For a fixed stage s, every pair (i, j) is assigned to exactly one thread.
/// i and j are disjoint across all threads because:
///   - i = group * len + offset  with  group in [0, N/len),  offset in [0, half)
///   - j = i + half
///   - Different threads have different (group, offset) pairs, yielding
///     disjoint index pairs.  Therefore simultaneous reads and writes to
///     data[i] and data[j] are race-free within a single butterfly stage.
@compute @workgroup_size(64, 1, 1)
fn ntt_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id     = gid.x;
    let half_n = params.n >> 1u;
    if id >= half_n { return; }

    let stage  = params.stage_or_ninv;       // butterfly stage index s
    let half   = 1u << stage;                // half butterfly width
    let group  = id / half;
    let offset = id % half;
    let i      = group * (half << 1u) + offset;
    let j      = i + half;

    // Twiddle factor: omega^{offset * N/len} = twiddles[offset * step]
    // where step = N / len = N >> (s+1).
    let step = params.n >> (stage + 1u);
    let w    = twiddles[offset * step];

    let u = data[i];
    let t = mod_mul(data[j], w, params.modulus);
    data[i] = mod_add(u, t, params.modulus);
    data[j] = mod_sub(u, t, params.modulus);
}

// --------------------------------------------------------------------------
// Entry point 2: inverse NTT final scale pass
// --------------------------------------------------------------------------

/// Multiply every element by N^{-1} mod m to complete the inverse NTT.
///
/// # Dispatch
/// Dispatch `ceil(N / 64)` workgroups.  Each thread scales one element.
/// `params.stage_or_ninv` carries N^{-1} mod m for this entry point.
@compute @workgroup_size(64, 1, 1)
fn ntt_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    if k >= params.n { return; }
    data[k] = mod_mul(data[k], params.stage_or_ninv, params.modulus);
}

import ast
from pathlib import Path

import numpy as np
import pytest

import pyapollofft as afft


def test_package_ships_typed_public_surface():
    package = Path(afft.__file__).parent
    assert package.joinpath("py.typed").is_file()
    stub = package.joinpath("__init__.pyi")
    assert stub.is_file()
    module = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    declared = {
        node.name
        for node in module.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef))
    }
    declared.update(
        target.id
        for node in module.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(target := node.target, ast.Name)
    )
    assert set(afft.__all__) <= declared
    assert "__version__" in declared


def test_fft1_roundtrip():
    signal = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    spectrum = afft.fft1(signal)
    reconstructed = afft.ifft1(spectrum)
    assert np.allclose(signal, reconstructed, atol=1e-12)


def test_fft3_roundtrip():
    field = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    spectrum = afft.fft3(field)
    reconstructed = afft.ifft3(spectrum)
    assert np.allclose(field, reconstructed, atol=1e-12)


def test_rfft3_roundtrip():
    field = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    spectrum = afft.rfft3(field)
    reconstructed = afft.irfft3(spectrum, field.shape[-1])
    assert np.allclose(field, reconstructed, atol=1e-12)


def test_rfft3_contract_is_full_complex_spectrum():
    field = np.zeros((2, 2, 4), dtype=np.float64)
    field[0, 0, 0] = 1.0
    spectrum = afft.rfft3(field)

    assert spectrum.shape == field.shape
    assert spectrum.dtype == np.complex128
    assert np.allclose(spectrum, np.ones(field.shape, dtype=np.complex128), atol=1e-12)


def test_irfft3_rejects_half_spectrum_shape():
    half_spectrum = np.ones((2, 2, 3), dtype=np.complex128)
    with pytest.raises(ValueError, match="inconsistent"):
        afft.irfft3(half_spectrum, 4)


def test_nufft_1d_fast_tracks_exact():
    positions = np.array([0.01, 0.09, 0.23, 0.51], dtype=np.float64)
    values = np.array([1.0 + 0.0j, 0.5 + 0.2j, -0.2 + 0.3j, 0.1 - 0.4j], dtype=np.complex128)
    exact = afft.nufft_type1_1d(positions, values, 0.05, n_out=16)
    fast = afft.nufft_type1_1d_fast(positions, values, 0.05, n_out=16)
    assert np.allclose(exact, fast, atol=1e-5)


def test_nufft_3d_fast_tracks_exact():
    positions = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.12, 0.11, 0.05],
            [0.21, 0.18, 0.09],
        ],
        dtype=np.float64,
    )
    values = np.array([1.0 + 0.0j, 0.5 + 0.1j, -0.4 + 0.2j], dtype=np.complex128)
    exact = afft.nufft_type1_3d(positions, values, 4, 4, 4, 0.1, 0.1, 0.1)
    fast = afft.nufft_type1_3d_fast(positions, values, 4, 4, 4, 0.1, 0.1, 0.1)
    assert np.allclose(exact, fast, atol=1e-5)


def test_backend_capabilities_are_reported():
    capabilities = afft.backend_capabilities()
    assert capabilities["cpu"]["available"] is True
    assert capabilities["cpu"]["supports_3d"] is True
    assert "wgpu" in capabilities


def test_non_contiguous_input_is_rejected():
    signal = np.linspace(0.0, 1.0, 16, dtype=np.float64)[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        afft.fft1(signal)


def test_dtype_validation_rejects_wrong_real_dtype():
    signal = np.arange(8, dtype=np.float32)
    with pytest.raises(ValueError, match="expects float64/complex128 storage"):
        afft.fft1(signal)


def test_fft1_low_precision_roundtrip_float32():
    signal = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    spectrum = afft.fft1(signal, precision="low_precision")
    reconstructed = afft.ifft1(spectrum, precision="low_precision")
    assert spectrum.dtype == np.complex64
    assert reconstructed.dtype == np.float32
    assert np.allclose(signal, reconstructed, atol=1e-5)


def test_fft3_mixed_precision_roundtrip_float32():
    field = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    spectrum = afft.fft3(field, precision="mixed_precision")
    reconstructed = afft.ifft3(spectrum, precision="mixed_precision")
    assert spectrum.dtype == np.complex64
    assert reconstructed.dtype == np.float32
    assert np.allclose(field, reconstructed, atol=5e-3)


def test_precision_dtype_mismatch_is_rejected():
    signal64 = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    with pytest.raises(ValueError, match="expects float32/complex64 storage"):
        afft.fft1(signal64, precision="low_precision")

    signal32 = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    with pytest.raises(ValueError, match="expects float64/complex128 storage"):
        afft.fft1(signal32, precision="high_accuracy")


def test_plan_precision_dispatches_by_dtype():
    signal32 = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    plan = afft.FftPlan1D(8, precision="mixed_precision")
    spectrum = plan.fft(signal32)
    reconstructed = plan.ifft(spectrum)
    assert spectrum.dtype == np.complex64
    assert reconstructed.dtype == np.float32
    assert np.allclose(signal32, reconstructed, atol=5e-3)


def test_backend_capabilities_report_precision_profiles():
    capabilities = afft.backend_capabilities()
    assert capabilities["cpu"]["supports_mixed_precision"] is True
    assert capabilities["cpu"]["default_precision_profile"] == "high_accuracy"
    assert "low_precision" in capabilities["cpu"]["supported_precision_profiles"]
    assert "mixed_precision" in capabilities["cpu"]["supported_precision_profiles"]


# ── FFT utility functions ──────────────────────────────────────────────────────

def test_fftfreq_matches_numpy():
    n = 8
    expected = np.fft.fftfreq(n)
    got = afft.fftfreq(n)
    assert np.allclose(got, expected, atol=1e-15), f"expected={expected}, got={got}"


def test_fftfreq_with_sample_spacing():
    n, d = 10, 0.5
    expected = np.fft.fftfreq(n, d)
    got = afft.fftfreq(n, d)
    assert np.allclose(got, expected, atol=1e-15)


def test_rfftfreq_matches_numpy():
    n = 8
    expected = np.fft.rfftfreq(n)
    got = afft.rfftfreq(n)
    assert len(got) == n // 2 + 1
    assert np.allclose(got, expected, atol=1e-15)


def test_fftshift_ifftshift_roundtrip():
    x = np.fft.fftfreq(16)
    shifted = afft.fftshift(x)
    recovered = afft.ifftshift(shifted)
    assert np.allclose(recovered, x, atol=1e-15)


def test_fftshift_matches_numpy():
    x = np.fft.fftfreq(16)
    expected = np.fft.fftshift(x)
    got = afft.fftshift(x)
    assert np.allclose(got, expected, atol=1e-15)


# ── Complex-to-complex FFT ────────────────────────────────────────────────────

def test_fft_complex1_matches_numpy():
    rng = np.random.default_rng(42)
    z = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    z = np.asarray(z, dtype=np.complex128)
    expected = np.fft.fft(z)
    got = afft.fft_complex1(z)
    assert np.allclose(got, expected, atol=1e-10), f"max_err={np.max(np.abs(got - expected))}"


def test_ifft_complex1_roundtrip():
    rng = np.random.default_rng(7)
    z = np.asarray(rng.standard_normal(32) + 1j * rng.standard_normal(32), dtype=np.complex128)
    recovered = afft.ifft_complex1(afft.fft_complex1(z))
    assert np.allclose(recovered, z, atol=1e-11)


def test_fft_complex2_matches_numpy():
    rng = np.random.default_rng(13)
    z = np.asarray(
        rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8)), dtype=np.complex128
    )
    expected = np.fft.fft2(z)
    got = afft.fft_complex2(z)
    assert np.allclose(got, expected, atol=1e-10)


def test_fft_complex3_roundtrip():
    rng = np.random.default_rng(99)
    z = np.asarray(
        rng.standard_normal((4, 4, 4)) + 1j * rng.standard_normal((4, 4, 4)),
        dtype=np.complex128,
    )
    recovered = afft.ifft_complex3(afft.fft_complex3(z))
    assert np.allclose(recovered, z, atol=1e-11)


# ── Discrete Hartley Transform ────────────────────────────────────────────────

def test_dht1_involution():
    # DHT is its own inverse up to a factor of n: DHT(DHT(x)) = n * x
    rng = np.random.default_rng(55)
    x = rng.standard_normal(16).astype(np.float64)
    result = afft.dht1(np.ascontiguousarray(afft.dht1(np.ascontiguousarray(x))))
    assert np.allclose(result, len(x) * x, atol=1e-10), f"max_err={np.max(np.abs(result - len(x)*x))}"


def test_dht1_roundtrip():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(32).astype(np.float64)
    recovered = afft.idht1(np.ascontiguousarray(afft.dht1(np.ascontiguousarray(x))))
    assert np.allclose(recovered, x, atol=1e-11)


def test_dht2_roundtrip():
    rng = np.random.default_rng(21)
    x = np.ascontiguousarray(rng.standard_normal((8, 8)).astype(np.float64))
    recovered = afft.idht2(np.ascontiguousarray(afft.dht2(x)))
    assert np.allclose(recovered, x, atol=1e-10)


def test_dht3_roundtrip():
    rng = np.random.default_rng(77)
    x = np.ascontiguousarray(rng.standard_normal((4, 4, 4)).astype(np.float64))
    recovered = afft.idht3(np.ascontiguousarray(afft.dht3(x)))
    assert np.allclose(recovered, x, atol=1e-10)


# ── Fast Walsh-Hadamard Transform ─────────────────────────────────────────────

def test_fwht1_involution():
    # FWHT is its own inverse up to factor of n: FWHT(FWHT(x)) = n * x
    x = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    result = afft.fwht1(np.ascontiguousarray(afft.fwht1(np.ascontiguousarray(x))))
    assert np.allclose(result, len(x) * x, atol=1e-12)


def test_fwht1_roundtrip():
    rng = np.random.default_rng(11)
    x = np.ascontiguousarray(rng.standard_normal(16).astype(np.float64))
    recovered = afft.ifwht1(np.ascontiguousarray(afft.fwht1(x)))
    assert np.allclose(recovered, x, atol=1e-12)


def test_fwht2_roundtrip():
    rng = np.random.default_rng(22)
    x = np.ascontiguousarray(rng.standard_normal((8, 8)).astype(np.float64))
    recovered = afft.ifwht2(np.ascontiguousarray(afft.fwht2(x)))
    assert np.allclose(recovered, x, atol=1e-11)


def test_fwht3_roundtrip():
    rng = np.random.default_rng(33)
    x = np.ascontiguousarray(rng.standard_normal((8, 8, 8)).astype(np.float64))
    recovered = afft.ifwht3(np.ascontiguousarray(afft.fwht3(x)))
    assert np.allclose(recovered, x, atol=1e-11)


# ── DCT / DST ─────────────────────────────────────────────────────────────────

def test_dct2_1d_roundtrip():
    rng = np.random.default_rng(44)
    x = np.ascontiguousarray(rng.standard_normal(32).astype(np.float64))
    x_hat = afft.dct2_1d(x)
    recovered = afft.idct2_1d(np.ascontiguousarray(x_hat))
    assert np.allclose(recovered, x, atol=1e-10), f"max_err={np.max(np.abs(recovered - x))}"


def test_dst2_1d_roundtrip():
    rng = np.random.default_rng(66)
    x = np.ascontiguousarray(rng.standard_normal(32).astype(np.float64))
    x_hat = afft.dst2_1d(x)
    recovered = afft.idst2_1d(np.ascontiguousarray(x_hat))
    assert np.allclose(recovered, x, atol=1e-10), f"max_err={np.max(np.abs(recovered - x))}"

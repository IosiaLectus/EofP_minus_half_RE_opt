"""Tests for purifications, reflected entropy and the HLS family of states."""

import math

import numpy as np
import pytest
import tensornetwork as tn
import torch

from utilities import (
    BuildPurification,
    EntanglementEntropy,
    HLS_reduced_reflected_spectrum,
    HLS_reflected_spectrum,
    HLS_state,
    MinimalPurification,
    ReflectedEntropy,
    ReflectedEntropyNoSVD,
    ReflectedEntropySpectrum,
    reduced_HLS_state,
    split_density_mat,
)

BETAS = [0.3, 1.0, 2.5]
# SVD paths regularise with reg=1e-12 inside a sqrt, i.e. ~1e-6 absolute error.
REFLECTED_ATOL = 1e-4

# logm of rank-deficient (pure) density matrices legitimately warns.
pytestmark = pytest.mark.filterwarnings("ignore:.*logm input matrix.*")


def _pure_rho(psi):
    return torch.einsum("ij,kl->ijkl", psi, psi.conj())


def _nonzero_sorted(p, tol=1e-6):
    p = p.detach().real.sort(descending=True).values
    return p[p > tol]


def _mixed_rho_ab(make_state, dims=(2, 2)):
    """A rank-4 mixed rho_AB obtained by tracing a pure 4-party state."""
    psi = make_state((*dims, 2, 2), seed=21)
    return torch.einsum("ijab,klab->ijkl", psi, psi.conj())


# --------------------------------------------------------------------------
# MinimalPurification / BuildPurification
# --------------------------------------------------------------------------
def test_purification_is_the_matrix_square_root_of_rho(make_state):
    rho = _mixed_rho_ab(make_state)
    left, mid = MinimalPurification(tn.Node(rho.clone()))
    right = tn.Node(left.tensor.clone().conj())
    psi = BuildPurification(left, mid, right).tensor

    assert psi.shape == rho.shape
    sqrt_rho = psi.reshape(4, 4)
    assert torch.allclose(sqrt_rho @ sqrt_rho, rho.reshape(4, 4), atol=REFLECTED_ATOL)


def test_purification_is_normalised(make_state):
    rho = _mixed_rho_ab(make_state)
    left, mid = MinimalPurification(tn.Node(rho))
    psi = BuildPurification(left, mid, tn.Node(left.tensor.conj())).tensor
    assert float(torch.linalg.vector_norm(psi) ** 2) == pytest.approx(1.0, abs=REFLECTED_ATOL)


def test_minimal_purification_regularises_the_spectrum():
    rho = torch.diag(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.cdouble))
    left, mid = MinimalPurification(tn.Node(rho.reshape(2, 2, 2, 2)), reg=1e-8)
    # every entry of mid is sqrt(entry + reg) >= sqrt(reg)
    assert float(mid.tensor.real.min()) == pytest.approx(1e-4)
    assert isinstance(left, tn.Node)


# --------------------------------------------------------------------------
# ReflectedEntropy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("svd", [True, False])
def test_reflected_entropy_of_bell_pair_is_two_bits(bell_state, svd):
    sr, _ = ReflectedEntropy(tn.Node(_pure_rho(bell_state)), [0], svd=svd)
    assert float(sr) == pytest.approx(2.0, abs=REFLECTED_ATOL)


@pytest.mark.parametrize("svd", [True, False])
@pytest.mark.parametrize("q", [1, 2])
def test_reflected_entropy_of_pure_rho_is_twice_the_entanglement_entropy(make_state, svd, q):
    psi = make_state((2, 3), seed=17)
    sr, _ = ReflectedEntropy(tn.Node(_pure_rho(psi)), [0], Renyi_index=q, svd=svd)
    s_a = EntanglementEntropy(tn.Node(psi.clone()), [0], Renyi_index=q)
    assert float(sr) == pytest.approx(2 * float(s_a), abs=REFLECTED_ATOL)


@pytest.mark.parametrize("svd", [True, False])
def test_reflected_entropy_of_product_state_is_zero(svd):
    a = torch.tensor([0.6, 0.8], dtype=torch.cdouble)
    b = torch.tensor([1.0, 0.0], dtype=torch.cdouble)
    rho_a = torch.outer(a, a.conj())
    rho_b = torch.outer(b, b.conj())
    rho = torch.einsum("ik,jl->ijkl", rho_a, rho_b)
    sr, _ = ReflectedEntropy(tn.Node(rho), [0], svd=svd)
    assert float(sr) == pytest.approx(0.0, abs=REFLECTED_ATOL)


@pytest.mark.parametrize("q", [1, 2, 3])
def test_svd_and_no_svd_reflected_entropies_agree(make_state, q):
    rho = _mixed_rho_ab(make_state)
    with_svd, _ = ReflectedEntropy(tn.Node(rho.clone()), [0], Renyi_index=q, svd=True)
    no_svd, _ = ReflectedEntropy(tn.Node(rho.clone()), [0], Renyi_index=q, svd=False)
    assert float(with_svd) == pytest.approx(float(no_svd), abs=REFLECTED_ATOL)


def test_reflected_entropy_is_symmetric_under_swapping_ab(make_state):
    rho = _mixed_rho_ab(make_state)
    sr_a, _ = ReflectedEntropy(tn.Node(rho.clone()), [0])
    sr_b, _ = ReflectedEntropy(tn.Node(rho.clone()), [1])
    assert float(sr_a) == pytest.approx(float(sr_b), abs=REFLECTED_ATOL)


def test_reflected_entropy_returns_value_and_purification(make_state):
    rho = _mixed_rho_ab(make_state)
    for svd in (True, False):
        value, purification = ReflectedEntropy(tn.Node(rho.clone()), [0], svd=svd)
        assert torch.is_tensor(value) and value.dim() == 0
        assert isinstance(purification, tn.Node)
        assert tuple(purification.tensor.shape) == tuple(rho.shape)


def test_reflected_entropy_no_svd_is_called_directly(make_state):
    rho = _mixed_rho_ab(make_state)
    direct, pur = ReflectedEntropyNoSVD(tn.Node(rho.clone()), [0], Renyi_index=2)
    via_dispatch, _ = ReflectedEntropy(tn.Node(rho.clone()), [0], Renyi_index=2, svd=False)
    assert float(direct) == pytest.approx(float(via_dispatch))
    assert tuple(pur.tensor.shape) == tuple(rho.shape)


def test_reflected_entropy_spectrum_is_a_probability_vector(make_state):
    probs, purification = ReflectedEntropySpectrum(tn.Node(_mixed_rho_ab(make_state)), [0])
    assert float(probs.sum()) == pytest.approx(1.0)
    assert bool((probs >= 0).all())
    assert isinstance(purification, tn.Node)


# --------------------------------------------------------------------------
# HLS states (arXiv:2302.10208 family)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("beta", BETAS)
def test_hls_state_is_a_valid_density_matrix(beta):
    t = HLS_state(beta).tensor
    assert tuple(t.shape) == (3, 3, 2, 3, 3, 2)
    rho = t.reshape(18, 18)
    assert torch.allclose(rho, rho.mH, atol=1e-7)
    assert float(rho.trace().real) == pytest.approx(1.0, abs=1e-6)
    assert float(torch.linalg.eigvalsh(rho).min()) > -1e-7


@pytest.mark.parametrize("beta", BETAS)
def test_hls_state_entries(beta):
    t = HLS_state(beta).tensor
    norm = 4 * beta + 2
    assert float(t[0, 0, 0, 0, 0, 0].real) == pytest.approx(beta / norm, rel=1e-5)
    assert float(t[2, 1, 0, 2, 1, 0].real) == pytest.approx(beta / norm, rel=1e-5)
    assert float(t[0, 2, 0, 0, 2, 0].real) == pytest.approx(1 / norm, rel=1e-5)
    assert float(t[1, 2, 1, 1, 2, 1].real) == pytest.approx(1 / norm, rel=1e-5)
    # exactly six non-zero entries
    assert int((t != 0).sum()) == 6


@pytest.mark.parametrize("beta", BETAS)
def test_reduced_hls_state_is_the_trace_over_the_qubit(beta):
    full = HLS_state(beta).tensor
    expected = torch.einsum("abcdec->abde", full)
    reduced = reduced_HLS_state(beta).tensor
    assert tuple(reduced.shape) == (3, 3, 3, 3)
    assert torch.allclose(reduced, expected, atol=1e-7)
    assert float(reduced.reshape(9, 9).trace().real) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize(
    "spectrum_fn", [HLS_reflected_spectrum, HLS_reduced_reflected_spectrum]
)
def test_hls_closed_form_spectra_are_normalised_and_sorted(beta, spectrum_fn):
    spec = spectrum_fn(beta)
    assert spec.shape == (3,)
    assert float(spec.sum()) == pytest.approx(1.0, abs=1e-6)
    assert bool((spec[:-1] >= spec[1:]).all())


@pytest.mark.parametrize("beta", BETAS)
def test_hls_reflected_spectrum_matches_numerical_reflected_spectrum(beta):
    numeric, _ = ReflectedEntropySpectrum(HLS_state(beta), [0])
    assert torch.allclose(
        _nonzero_sorted(numeric), HLS_reflected_spectrum(beta), atol=REFLECTED_ATOL
    )


@pytest.mark.parametrize("beta", BETAS)
def test_hls_reduced_reflected_spectrum_matches_numerical_reflected_spectrum(beta):
    numeric, _ = ReflectedEntropySpectrum(reduced_HLS_state(beta), [0])
    assert torch.allclose(
        _nonzero_sorted(numeric),
        HLS_reduced_reflected_spectrum(beta),
        atol=REFLECTED_ATOL,
    )

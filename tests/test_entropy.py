"""Tests for the entropy / spectrum functions in src/utilities.py."""

import math

import numpy as np
import pytest
import tensornetwork as tn
import torch

import utilities
from utilities import (
    DensityMatrixSpectrum,
    EntanglementEntropy,
    EntanglementEntropyNoSVD,
    EntanglementSpectrum,
    RenyiEntropy,
    ShannonEntropy,
    VNEntropy,
    split_density_mat,
)

QS = [0.5, 1, 2, 2.5, 3]


def _nonzero_sorted(p, tol=1e-9):
    p = p.detach().real.sort(descending=True).values
    return p[p > tol]


def _pure_rho(psi):
    """|psi><psi| as a (d1..dn, d1..dn) tensor, for psi with n indices."""
    n = psi.dim()
    letters = "abcdefgh"[:n]
    primed = "ijklmnop"[:n]
    return torch.einsum(f"{letters},{primed}->{letters}{primed}", psi, psi.conj())


# --------------------------------------------------------------------------
# ShannonEntropy / RenyiEntropy (classical distributions)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 4, 8])
def test_shannon_entropy_of_uniform_distribution_is_log2_n(n):
    probs = torch.full((n,), 1.0 / n, dtype=torch.float64)
    assert float(ShannonEntropy(probs)) == pytest.approx(math.log2(n))


def test_shannon_entropy_of_deterministic_distribution_is_zero():
    assert float(ShannonEntropy(torch.tensor([1.0, 0.0, 0.0]))) == pytest.approx(0.0)


def test_shannon_entropy_ignores_zero_probabilities():
    # 0 * log(0) must be treated as 0, not NaN.
    h = ShannonEntropy(torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64))
    assert torch.isfinite(h)
    assert float(h) == pytest.approx(1.0)


def test_shannon_entropy_reports_nan_input(capsys):
    h = ShannonEntropy(torch.tensor([float("nan"), 0.5]))
    assert torch.isnan(h)
    assert "Problematic probs" in capsys.readouterr().out


def test_renyi_index_one_dispatches_to_shannon():
    p = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    assert float(RenyiEntropy(p, 1)) == pytest.approx(float(ShannonEntropy(p)))


@pytest.mark.parametrize("q", [0, 0.5, 1, 2, 3, 7])
def test_renyi_entropy_of_uniform_distribution_is_independent_of_q(q):
    probs = torch.full((8,), 1 / 8, dtype=torch.float64)
    assert float(RenyiEntropy(probs, q)) == pytest.approx(3.0)


def test_renyi_two_matches_collision_entropy_formula():
    p = np.array([0.5, 0.25, 0.25])
    expected = -math.log2((p**2).sum())
    assert float(RenyiEntropy(torch.tensor(p), 2)) == pytest.approx(expected)


def test_renyi_zero_is_hartley_entropy():
    p = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    assert float(RenyiEntropy(p, 0)) == pytest.approx(math.log2(3))


def test_renyi_entropy_is_non_increasing_in_q():
    p = torch.tensor([0.5, 0.25, 0.15, 0.1], dtype=torch.float64)
    values = [float(RenyiEntropy(p, q)) for q in (0, 0.5, 1, 2, 3, 5)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))


# --------------------------------------------------------------------------
# EntanglementSpectrum
# --------------------------------------------------------------------------
def test_entanglement_spectrum_of_bell_state(bell_state):
    spec = _nonzero_sorted(EntanglementSpectrum(tn.Node(bell_state), [0]))
    assert torch.allclose(spec, torch.tensor([0.5, 0.5], dtype=spec.dtype))


def test_entanglement_spectrum_of_product_state_is_pure():
    psi = torch.einsum(
        "i,j->ij",
        torch.tensor([1, 0], dtype=torch.cdouble),
        torch.tensor([0.6, 0.8], dtype=torch.cdouble),
    )
    spec = _nonzero_sorted(EntanglementSpectrum(tn.Node(psi), [0]))
    assert spec.numel() == 1 and float(spec[0]) == pytest.approx(1.0)


def test_entanglement_spectrum_is_a_normalised_probability_vector(make_state):
    # Deliberately unnormalised input: the function must normalise.
    psi = 3.7 * make_state((2, 2, 3, 3), seed=5)
    probs = EntanglementSpectrum(tn.Node(psi), [0, 1])
    assert not probs.is_complex()
    assert float(probs.sum()) == pytest.approx(1.0)
    assert bool((probs >= 0).all())


def test_entanglement_spectrum_of_complementary_regions_agree(make_state):
    # Schmidt decomposition: the non-zero spectra of A and its complement match.
    psi = make_state((2, 3, 2, 4), seed=11)
    spec_a = _nonzero_sorted(EntanglementSpectrum(tn.Node(psi.clone()), [0, 1]))
    spec_b = _nonzero_sorted(EntanglementSpectrum(tn.Node(psi.clone()), [2, 3]))
    assert torch.allclose(spec_a, spec_b, atol=1e-10)


# --------------------------------------------------------------------------
# EntanglementEntropy / EntanglementEntropyNoSVD
# --------------------------------------------------------------------------
@pytest.mark.parametrize("svd", [True, False])
def test_entanglement_entropy_of_bell_state_is_one_bit(bell_state, svd):
    ee = EntanglementEntropy(tn.Node(bell_state), [0], svd=svd)
    assert float(ee) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("svd", [True, False])
def test_entanglement_entropy_of_product_state_is_zero(svd):
    psi = torch.einsum(
        "i,j->ij",
        torch.tensor([0.6, 0.8], dtype=torch.cdouble),
        torch.tensor([1, 0], dtype=torch.cdouble),
    )
    ee = EntanglementEntropy(tn.Node(psi), [0], svd=svd)
    assert float(ee) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("svd", [True, False])
def test_entanglement_entropy_of_maximally_entangled_qutrits(svd):
    psi = torch.eye(3, dtype=torch.cdouble) / math.sqrt(3)
    ee = EntanglementEntropy(tn.Node(psi), [0], svd=svd)
    assert float(ee) == pytest.approx(math.log2(3), abs=1e-6)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("A", [[0], [1], [0, 1], [0, 2]])
def test_svd_and_no_svd_entropies_agree(make_state, q, A):
    psi = make_state((2, 2, 3, 3), seed=3)
    with_svd = EntanglementEntropy(tn.Node(psi.clone()), A, Renyi_index=q)
    no_svd = EntanglementEntropy(tn.Node(psi.clone()), A, Renyi_index=q, svd=False)
    assert float(with_svd) == pytest.approx(float(no_svd), abs=1e-6)


@pytest.mark.parametrize("q", [1, 2])
def test_entanglement_entropy_of_complement_matches_for_pure_states(make_state, q):
    psi = make_state((2, 3, 2, 3), seed=8)
    s_a = EntanglementEntropy(tn.Node(psi.clone()), [0, 1], Renyi_index=q)
    s_b = EntanglementEntropy(tn.Node(psi.clone()), [2, 3], Renyi_index=q)
    assert float(s_a) == pytest.approx(float(s_b), abs=1e-8)


def test_no_svd_entropy_is_real_valued(make_state):
    psi = make_state((2, 2, 2, 2), seed=2)
    ee = EntanglementEntropyNoSVD(tn.Node(psi), [0, 1], Renyi_index=2)
    assert not ee.is_complex()


def test_entanglement_entropy_no_svd_helper_in_module_compares_both_paths():
    """src/utilities.py ships its own ``test_EntanglementEntropyNoSVD`` helper
    (not collected by pytest because it lives outside tests/).  Calling it here
    both covers it and checks the two code paths agree."""
    torch.manual_seed(0)
    ee_svd, ee_no_svd = utilities.test_EntanglementEntropyNoSVD(q=1)
    assert ee_svd == pytest.approx(ee_no_svd, abs=1e-5)


# --------------------------------------------------------------------------
# split_density_mat
# --------------------------------------------------------------------------
def test_split_density_mat_rejects_odd_number_of_indices(capsys):
    assert split_density_mat(tn.Node(torch.zeros(2, 2, 2))) is None
    assert "odd number of indices" in capsys.readouterr().out


def test_split_density_mat_rejects_unbalanced_dimensions(capsys):
    assert split_density_mat(tn.Node(torch.zeros(2, 3, 2, 2))) is None
    assert "dimensions unbalanced" in capsys.readouterr().out


def test_split_density_mat_left_and_mid_diagonalise_a_complex_state(make_state):
    """rho = U S U^dagger.  This is the part of the SVD the rest of the code
    relies on (MinimalPurification only keeps ``left`` and ``mid``).

    NB: the ``right`` node is NOT checked for complex rho: the tensornetwork
    PyTorch backend does not return a conjugated V^dagger, so ``left @ mid @
    right`` only reproduces rho when rho is real (see the next test).
    """
    rho2 = _pure_rho(make_state((2, 2), seed=9)).reshape(4, 4)
    rho = 0.7 * _pure_rho(make_state((2, 2), seed=4)).reshape(4, 4) + 0.3 * rho2
    left, mid, right = split_density_mat(tn.Node(rho.reshape(2, 2, 2, 2).clone()))

    u = left.tensor.reshape(4, -1)
    s = mid.tensor
    assert torch.allclose(u @ s @ u.mH, rho, atol=1e-10)


def test_split_density_mat_full_reconstruction_for_real_density_matrix():
    rho = torch.diag(torch.tensor([0.5, 0.3, 0.2, 0.0], dtype=torch.cdouble))
    rho = rho.reshape(2, 2, 2, 2)
    left, mid, right = split_density_mat(tn.Node(rho.clone()))
    # split_node_full_svd returns already-connected nodes: just contract.
    rebuilt = (left @ mid) @ right
    assert torch.allclose(rebuilt.tensor, rho, atol=1e-10)


# --------------------------------------------------------------------------
# DensityMatrixSpectrum / VNEntropy
# --------------------------------------------------------------------------
def test_density_matrix_spectrum_matches_eigenvalues(make_state):
    rho = 0.6 * _pure_rho(make_state((2, 2), seed=1)).reshape(4, 4)
    rho = rho + 0.4 * _pure_rho(make_state((2, 2), seed=2)).reshape(4, 4)
    probs = DensityMatrixSpectrum(tn.Node(rho.reshape(2, 2, 2, 2)))
    expected = torch.linalg.eigvalsh(rho).flip(0).real
    assert torch.allclose(_nonzero_sorted(probs, 1e-9), expected[expected > 1e-9], atol=1e-8)


def test_density_matrix_spectrum_is_normalised():
    rho = 5.0 * torch.eye(4, dtype=torch.cdouble).reshape(2, 2, 2, 2)  # trace 20
    probs = DensityMatrixSpectrum(tn.Node(rho))
    assert float(probs.sum()) == pytest.approx(1.0)
    assert torch.allclose(probs, torch.full((4,), 0.25, dtype=probs.dtype))


def test_vn_entropy_of_pure_state_is_zero(make_state):
    rho = _pure_rho(make_state((2, 2), seed=6)).reshape(2, 2, 2, 2)
    assert float(VNEntropy(tn.Node(rho))) == pytest.approx(0.0, abs=1e-6)


def test_vn_entropy_of_maximally_mixed_two_qubits_is_two_bits():
    rho = (torch.eye(4, dtype=torch.cdouble) / 4).reshape(2, 2, 2, 2)
    assert float(VNEntropy(tn.Node(rho))) == pytest.approx(2.0, abs=1e-8)

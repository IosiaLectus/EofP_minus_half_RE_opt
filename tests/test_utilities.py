import math
import torch
import itertools
import tensornetwork as tn
from utilities import TripartiteInformation
from utilities import partial_trace


def _rho(state_vector):
    return torch.outer(state_vector, state_vector.conj())


def _bell_rho():
    phi = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / math.sqrt(2)
    return _rho(phi)


def _pt(rho, traced_parties):
    """Partial trace of a flat (2**n, 2**n) density matrix."""
    n = int(math.log2(rho.shape[0]))
    t = rho.reshape([2] * (2 * n))
    reduced = partial_trace(t, traced_parties)
    d = 2 ** (n - len(traced_parties))
    return reduced.reshape(d, d)


def test_partial_trace_of_bell_state_is_maximally_mixed():
    rho = _bell_rho()
    expected = 0.5 * torch.eye(2, dtype=torch.cfloat)
    for traced_party in ([0], [1]):
        reduced = _pt(rho, traced_party)
        assert reduced.shape == (2, 2)
        assert torch.allclose(reduced, expected, atol=1e-6)


def test_partial_trace_of_product_state_is_pure():
    # |01>: tracing out party 0 leaves |1><1|, tracing out party 1 leaves |0><0|
    rho = _rho(torch.tensor([0, 1, 0, 0], dtype=torch.cfloat))
    zero = torch.tensor([1, 0], dtype=torch.cfloat)
    one = torch.tensor([0, 1], dtype=torch.cfloat)

    assert torch.allclose(_pt(rho, [0]), torch.outer(one, one.conj()), atol=1e-6)
    assert torch.allclose(_pt(rho, [1]), torch.outer(zero, zero.conj()), atol=1e-6)


def test_partial_trace_preserves_total_trace():
    reduced = _pt(_bell_rho(), [0])
    assert torch.isclose(reduced.trace().real, torch.tensor(1.0), atol=1e-6)


def test_partial_trace_of_maximally_mixed_two_qubits():
    rho = torch.eye(4, dtype=torch.cfloat) / 4
    reduced = _pt(rho, [1])
    assert torch.allclose(reduced, 0.5 * torch.eye(2, dtype=torch.cfloat), atol=1e-6)

def test_tripartite_information_symmetric_on_random_four_party_states():
    """For a pure 4-party state, I3 is the same for every choice of three
    parties and every ordering of the arguments. Checked on several
    seeded random states with uneven local dimensions."""

    gen = torch.Generator().manual_seed(3142)
    for dims in [(2, 2, 2, 2), (3, 3, 2, 2), (4, 3, 2, 5)]:
        for _ in range(5):
            psi = torch.randn(dims, dtype=torch.cdouble, generator=gen)
            psi = psi / torch.linalg.vector_norm(psi)
            values = [
                float(TripartiteInformation(tn.Node(psi.clone()), [a], [b], [c]))
                for triple in itertools.combinations(range(4), 3)
                for a, b, c in itertools.permutations(triple)
            ]
            assert max(values) - min(values) < 1e-6, (dims, values)
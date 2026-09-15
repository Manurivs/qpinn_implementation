'''Equivalence of the PyTorch backend against the PennyLane reference.

Covers the statevector-level output, first derivatives with respect to both
trainable parameters and inputs, and the second input derivative that PINN
residuals rely on. Also pins the parameter counts reported in the QCPINN paper.
'''

import itertools

import pytest
import torch

from src.quantum import torch_sim
from src.quantum.ansatz import ANSATZE, build_circuit, n_params_per_layer
from src.quantum.pennylane_ref import make_qnode

N_QUBITS = 5
N_LAYERS = 1
BATCH = 7
ATOL = 1e-10

CASES = list(itertools.product(ANSATZE, ('angle', 'amplitude')))


def _setup(ansatz, encoding, seed=0):
    torch.manual_seed(seed)
    gates, n_params = build_circuit(ansatz, N_QUBITS, N_LAYERS, encoding)
    x = torch.randn(BATCH, N_QUBITS, dtype=torch.float64)
    theta = torch.randn(n_params, dtype=torch.float64)
    qnode = make_qnode(ansatz, N_QUBITS, N_LAYERS, encoding)
    ppl = n_params_per_layer(ansatz, N_QUBITS)
    return gates, x, theta, qnode, ppl


def _run_torch(gates, x, theta, encoding):
    return torch_sim.run(
        gates, x, theta, N_QUBITS, encoding, dtype=torch.complex128
    )


@pytest.mark.parametrize('ansatz,encoding', CASES)
def test_forward_matches_pennylane(ansatz, encoding):
    '''Pauli-Z expectations agree with PennyLane to numerical precision.'''
    gates, x, theta, qnode, ppl = _setup(ansatz, encoding)
    mine = _run_torch(gates, x, theta, encoding)
    ref = qnode(x, theta.reshape(N_LAYERS, ppl))
    assert torch.allclose(mine, ref.to(mine.dtype), atol=ATOL)


@pytest.mark.parametrize('ansatz,encoding', CASES)
def test_param_gradient_matches_pennylane(ansatz, encoding):
    '''d<Z>/dtheta agrees with PennyLane backprop.'''
    gates, x, theta, qnode, ppl = _setup(ansatz, encoding)

    t1 = theta.clone().requires_grad_(True)
    _run_torch(gates, x, t1, encoding).sum().backward()

    t2 = theta.clone().requires_grad_(True)
    qnode(x, t2.reshape(N_LAYERS, ppl)).sum().backward()

    assert torch.allclose(t1.grad, t2.grad, atol=ATOL)


@pytest.mark.parametrize('ansatz,encoding', CASES)
def test_input_gradient_matches_pennylane(ansatz, encoding):
    '''d<Z>/dx agrees with PennyLane backprop.'''
    gates, x, theta, qnode, ppl = _setup(ansatz, encoding)

    x1 = x.clone().requires_grad_(True)
    _run_torch(gates, x1, theta, encoding).sum().backward()

    x2 = x.clone().requires_grad_(True)
    qnode(x2, theta.reshape(N_LAYERS, ppl)).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=ATOL)


@pytest.mark.parametrize('ansatz,encoding', CASES)
def test_second_input_derivative_matches_finite_difference(ansatz, encoding):
    '''d2<Z>/dx2 survives create_graph and matches central differences.

    This is the property PINN residuals depend on and the main reason the
    backend is hand written rather than delegated to a framework.
    '''
    gates, x, theta, _, _ = _setup(ansatz, encoding)
    x = x.requires_grad_(True)

    out = _run_torch(gates, x, theta, encoding).sum()
    g = torch.autograd.grad(out, x, create_graph=True)[0]
    second = torch.autograd.grad(g[:, 0].sum(), x, create_graph=True)[0][:, 0]
    assert torch.isfinite(second).all()

    eps = 1e-5
    with torch.no_grad():
        shift = torch.zeros_like(x)
        shift[:, 0] = eps
        f_p = _run_torch(gates, x + shift, theta, encoding).sum(dim=1)
        f_0 = _run_torch(gates, x, theta, encoding).sum(dim=1)
        f_m = _run_torch(gates, x - shift, theta, encoding).sum(dim=1)
    fd = (f_p - 2 * f_0 + f_m) / eps ** 2
    assert torch.allclose(second, fd, atol=1e-4)


@pytest.mark.parametrize('ansatz', ANSATZE)
def test_parameter_counts_match_paper(ansatz):
    '''Total model parameters reproduce the reference implementation.

    The classical wrapper for Helmholtz is Linear(2, 50) + Linear(50, 5) in the
    preprocessor and Linear(5, 50) + Linear(50, 1) in the postprocessor, giving
    756 classical parameters.

    Only 'cascade' (771) is confirmed by the paper's Helmholtz table in every
    arXiv version, and 'cross_mesh' (796) matches the v6 wave-equation entry.
    The v4 Helmholtz table reports 781/781/836 for alternate/layered/cross_mesh,
    which no integer layer count reproduces from this architecture; those
    entries are treated as unresolved and are not pinned here.
    '''
    expected = {'cascade': 771, 'alternate': 772, 'layered': 776, 'cross_mesh': 796}
    _, n_params = build_circuit(ansatz, N_QUBITS, N_LAYERS)
    assert 756 + n_params == expected[ansatz]


def test_cascade_has_unreachable_parameters():
    '''The reference cascade circuit cannot reach a third of its parameters.

    It allocates ``3 * n_qubits`` parameters per layer but resets its counter
    before the entangling gates, so the CRX angles alias the RX angles and the
    last ``n_qubits`` parameters are never applied.
    '''
    gates, n_params = build_circuit('cascade', N_QUBITS, N_LAYERS)
    used = {g.index for g in gates if g.source == 'theta'}
    assert n_params == 15
    assert used == set(range(10))

    fixed, n_fixed = build_circuit('cascade', N_QUBITS, N_LAYERS, fix_cascade=True)
    assert {g.index for g in fixed if g.source == 'theta'} == set(range(n_fixed))


@pytest.mark.parametrize('ansatz,encoding', CASES)
@pytest.mark.parametrize('fuse', [False, True])
def test_compiled_plan_matches_reference(ansatz, encoding, fuse):
    '''The compiled fast path agrees with PennyLane, fused and unfused.'''
    from src.quantum.compile import compile_circuit

    gates, x, theta, qnode, ppl = _setup(ansatz, encoding)
    plan = compile_circuit(gates, N_QUBITS, fuse=fuse)
    mine = torch_sim.run_plan(
        plan, x, theta, N_QUBITS, encoding, dtype=torch.complex128
    )
    ref = qnode(x, theta.reshape(N_LAYERS, ppl))
    assert torch.allclose(mine, ref.to(mine.dtype), atol=ATOL)


@pytest.mark.parametrize('ansatz,encoding', CASES)
def test_compiled_plan_second_derivative(ansatz, encoding):
    '''The compiled path keeps a finite, correct second input derivative.'''
    from src.quantum.compile import compile_circuit

    gates, x, theta, _, _ = _setup(ansatz, encoding)
    plan = compile_circuit(gates, N_QUBITS)
    x = x.requires_grad_(True)

    def f(xx):
        return torch_sim.run_plan(
            plan, xx, theta, N_QUBITS, encoding, dtype=torch.complex128
        ).sum()

    g = torch.autograd.grad(f(x), x, create_graph=True)[0]
    second = torch.autograd.grad(g[:, 0].sum(), x, create_graph=True)[0][:, 0]
    assert torch.isfinite(second).all()

    eps = 1e-5
    shift = torch.zeros_like(x)
    shift[:, 0] = eps
    with torch.no_grad():
        fd = (f(x + shift) - 2 * f(x) + f(x - shift)) / eps ** 2
    assert torch.allclose(second.sum(), fd, atol=1e-4)


def test_batched_boundary_loss_equals_per_edge_loss():
    '''Sharing one forward pass across the four edges changes no arithmetic.

    The boundary term is four separate per-edge MSEs. Evaluating them from one
    concatenated forward pass is the same computation, and is much cheaper
    because the simulator is dispatch bound rather than FLOP bound.
    '''
    from src.models.qcpinn import QCPINN
    from src.pdes import helmholtz

    torch.manual_seed(0)
    model = QCPINN(2, 1, ansatz='cascade')
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(3)
    xs, us = zip(*(helmholtz.sample_edge(e, 64, generator=gen)
                   for e in helmholtz.EDGES))

    separate = sum(loss_fn(model(x), u) for x, u in zip(xs, us))
    preds = model(torch.cat(xs, 0)).split(64)
    batched = sum(loss_fn(p, u) for p, u in zip(preds, us))

    assert torch.allclose(separate, batched, atol=1e-6)

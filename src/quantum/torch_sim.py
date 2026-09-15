'''Batched statevector simulator in pure PyTorch.

An `n`-qubit pure state is ``2 ** n`` complex amplitudes, so the whole simulator
is dense linear algebra that autograd already handles. The state is carried as
shape ``(batch, 2, 2, ..., 2)`` with qubit `q` on axis ``q + 1``, so a gate
costs ``O(2 ** n)`` per application rather than the ``O(4 ** n)`` of building a
full kron'd unitary.

Wire ordering is big-endian to match PennyLane: qubit 0 is the most significant
bit of the flattened index.

Notes
-----
Expectation values use ``real ** 2 + imag ** 2`` rather than ``abs() ** 2``.
``abs`` on a complex tensor has a gradient singularity at the origin that
produces NaNs in the second derivative; the polynomial form is smooth
everywhere, which matters because PINN residuals differentiate the circuit
output twice with respect to its inputs.
'''

import torch

from src.quantum import gates as G


def zero_state(batch, n_qubits, dtype=torch.complex64, device=None):
    '''Prepare ``|0...0>`` replicated over a batch.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, 2, ..., 2)``.
    '''
    psi = torch.zeros(batch, 2 ** n_qubits, dtype=dtype, device=device)
    psi[:, 0] = 1.0
    return psi.reshape(batch, *([2] * n_qubits))


def amplitude_state(x, n_qubits, dtype=torch.complex64):
    '''Prepare a state from input features, padding with zeros and normalising.

    Parameters
    ----------
    x : torch.Tensor
        Real features, shape ``(batch, n_features)`` with
        ``n_features <= 2 ** n_qubits``.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, 2, ..., 2)``.

    Notes
    -----
    Reproduces ``qml.AmplitudeEmbedding(..., normalize=True, pad_with=0.0)``.
    The reference implementation feeds only ``n_qubits`` features into a
    ``2 ** n_qubits``-dimensional embedding, so all but the first `n_qubits`
    amplitudes are held at zero.
    '''
    dim = 2 ** n_qubits
    pad = dim - x.shape[-1]
    if pad < 0:
        raise ValueError(f'{x.shape[-1]} features exceed dimension {dim}')
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
    return x.to(dtype).reshape(x.shape[0], *([2] * n_qubits))


def apply_1q(psi, u, q):
    '''Apply a single-qubit gate to qubit `q`.

    Parameters
    ----------
    psi : torch.Tensor
        State, shape ``(batch, 2, ..., 2)``.
    u : torch.Tensor
        Gate of shape ``(2, 2)`` when shared across the batch, or
        ``(batch, 2, 2)`` when the angle is input dependent.
    q : int
        Target qubit.

    Returns
    -------
    torch.Tensor
        State with the same shape as `psi`.
    '''
    psi = torch.movedim(psi, q + 1, -1)
    shape = psi.shape
    flat = psi.reshape(shape[0], -1, 2)
    eq = 'ij,bkj->bki' if u.dim() == 2 else 'bij,bkj->bki'
    flat = torch.einsum(eq, u, flat)
    return torch.movedim(flat.reshape(shape), -1, q + 1)


def apply_2q(psi, u, q0, q1):
    '''Apply a two-qubit gate with `q0` as the more significant index.

    Parameters
    ----------
    psi : torch.Tensor
        State, shape ``(batch, 2, ..., 2)``.
    u : torch.Tensor
        Gate of shape ``(4, 4)`` or ``(batch, 4, 4)``, in the basis
        ``|q0 q1>``.
    q0, q1 : int
        Target qubits; for controlled gates ``q0`` is the control.

    Returns
    -------
    torch.Tensor
        State with the same shape as `psi`.
    '''
    psi = torch.movedim(psi, (q0 + 1, q1 + 1), (-2, -1))
    shape = psi.shape
    flat = psi.reshape(shape[0], -1, 4)
    eq = 'ij,bkj->bki' if u.dim() == 2 else 'bij,bkj->bki'
    flat = torch.einsum(eq, u, flat)
    return torch.movedim(flat.reshape(shape), (-2, -1), (q0 + 1, q1 + 1))


def z_signs(n_qubits, device=None, dtype=torch.float32):
    '''Eigenvalue table for single-qubit Pauli-Z observables.

    Returns
    -------
    torch.Tensor
        Shape ``(n_qubits, 2 ** n_qubits)`` with entries in ``{+1, -1}``.
    '''
    idx = torch.arange(2 ** n_qubits, device=device)
    shifts = torch.arange(n_qubits - 1, -1, -1, device=device).unsqueeze(1)
    bits = (idx.unsqueeze(0) >> shifts) & 1
    return 1.0 - 2.0 * bits.to(dtype)


def expval_z(psi, signs):
    '''Expectation values of Pauli-Z on every qubit.

    Parameters
    ----------
    psi : torch.Tensor
        State, shape ``(batch, 2, ..., 2)``.
    signs : torch.Tensor
        Table from :func:`z_signs`.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, n_qubits)``, real valued.
    '''
    flat = psi.reshape(psi.shape[0], -1)
    probs = flat.real ** 2 + flat.imag ** 2
    return probs @ signs.T


def run(gate_list, x, theta, n_qubits, encoding='angle', dtype=torch.complex64):
    '''Execute a circuit description and return Pauli-Z expectations.

    Parameters
    ----------
    gate_list : list of Gate
        Circuit description from :func:`src.quantum.ansatz.build_circuit`.
    x : torch.Tensor
        Input features, shape ``(batch, n_features)``.
    theta : torch.Tensor
        Flat trainable parameter vector.
    n_qubits : int
        Number of qubits.
    encoding : str, optional
        'angle' or 'amplitude'.
    dtype : torch.dtype, optional
        Complex dtype for the state.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, n_qubits)``, real valued.
    '''
    batch = x.shape[0]
    if encoding == 'amplitude':
        psi = amplitude_state(x, n_qubits, dtype)
    else:
        psi = zero_state(batch, n_qubits, dtype, x.device)

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    for g in gate_list:
        if g.kind == 'CNOT':
            psi = apply_2q(psi, G.cnot(dtype, x.device), *g.wires)
            continue
        angle = x[:, g.index] if g.source == 'x' else theta[g.index]
        angle = angle.to(real_dtype)
        if g.kind in G.BUILDERS_1Q:
            psi = apply_1q(psi, G.BUILDERS_1Q[g.kind](angle, dtype), g.wires[0])
        elif g.kind in G.BUILDERS_2Q:
            psi = apply_2q(psi, G.BUILDERS_2Q[g.kind](angle, dtype), *g.wires)
        else:
            raise ValueError(f'unknown gate: {g.kind}')

    signs = z_signs(n_qubits, x.device, real_dtype)
    return expval_z(psi, signs)


def _angle(spec, x, theta, real_dtype):
    '''Resolve one angle spec to (value, is_batched).'''
    _, source, index = spec
    if source == 'x':
        return x[:, index].to(real_dtype), True
    return theta[index].to(real_dtype), False


def _bcast_diag(d, n_qubits, wires, batched):
    '''Reshape a diagonal table so it broadcasts against the state tensor.'''
    wires = list(wires)
    if len(wires) == 2 and wires[0] > wires[1]:
        d = d.transpose(-2, -1)
        wires = wires[::-1]
    shape = [1] * (n_qubits + 1)
    if batched:
        shape[0] = d.shape[0]
    for w in wires:
        shape[w + 1] = 2
    return d.reshape(shape)


def _rz_table(theta, dtype):
    '''Diagonal of RZ(theta) as a (..., 2) complex tensor.'''
    half = theta / 2
    e_m = torch.complex(torch.cos(half), -torch.sin(half)).to(dtype)
    e_p = torch.complex(torch.cos(half), torch.sin(half)).to(dtype)
    return torch.stack([e_m, e_p], dim=-1)


def _fused_matrix(angles, x, theta, dtype, real_dtype):
    '''Product of a run of single-qubit gates, in application order.'''
    mats, batched = [], False
    for spec in angles:
        val, is_b = _angle(spec, x, theta, real_dtype)
        batched = batched or is_b
        mats.append((G.BUILDERS_1Q[spec[0]](val, dtype), is_b))
    out = None
    for m, is_b in mats:
        if not is_b and batched:
            m = m.unsqueeze(0)
        out = m if out is None else m @ out
    return out, batched


def run_plan(plan, x, theta, n_qubits, encoding='angle', dtype=torch.complex64):
    '''Execute a compiled plan and return Pauli-Z expectations.

    Parameters
    ----------
    plan : list of Op
        Output of :func:`src.quantum.compile.compile_circuit`.
    x : torch.Tensor
        Input features, shape ``(batch, n_features)``.
    theta : torch.Tensor
        Flat trainable parameter vector.
    n_qubits : int
        Number of qubits.
    encoding : str, optional
        'angle' or 'amplitude'.
    dtype : torch.dtype, optional
        Complex dtype for the state.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, n_qubits)``, real valued.
    '''
    batch = x.shape[0]
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    if encoding == 'amplitude':
        psi = amplitude_state(x, n_qubits, dtype)
    else:
        psi = zero_state(batch, n_qubits, dtype, x.device)

    for op in plan:
        if op.kind == 'diag1':
            total = None
            is_b = False
            for spec in op.angles:
                val, b = _angle(spec, x, theta, real_dtype)
                is_b = is_b or b
                if total is None:
                    total = val.expand(batch) if b else val
                else:
                    total = total + val
            if is_b and total.dim() == 0:
                total = total.expand(batch)
            d = _rz_table(total, dtype)
            psi = psi * _bcast_diag(d, n_qubits, op.wires, is_b)

        elif op.kind == 'diag2':
            val, is_b = _angle(op.angles[0], x, theta, real_dtype)
            tbl = _rz_table(val, dtype)
            ones = torch.ones_like(tbl)
            d = torch.stack([ones, tbl], dim=-2)
            psi = psi * _bcast_diag(d, n_qubits, op.wires, is_b)

        elif op.kind == 'u1':
            u, is_b = _fused_matrix(op.angles, x, theta, dtype, real_dtype)
            sub = op.sub_batched if is_b else op.sub_shared
            psi = torch.einsum(sub, u, psi)

        elif op.kind == 'u2':
            if op.static is not None:
                u, is_b = op.static.to(dtype=dtype, device=x.device), False
            else:
                val, is_b = _angle(op.angles[0], x, theta, real_dtype)
                u = G.BUILDERS_2Q[op.angles[0][0]](val, dtype)
            u = u.reshape(*u.shape[:-2], 2, 2, 2, 2)
            sub = op.sub_batched if is_b else op.sub_shared
            psi = torch.einsum(sub, u, psi)

        else:
            raise ValueError(f'unknown op kind: {op.kind}')

    signs = z_signs(n_qubits, x.device, real_dtype)
    return expval_z(psi, signs)

'''Symbolic circuit descriptions for the QCPINN discrete-variable ansaetze.

A circuit is described once, as a list of :class:`Gate` records, and executed by
either the fast PyTorch statevector backend or the PennyLane reference backend.
Describing the circuit in one place guarantees the two backends run structurally
identical circuits, and makes it possible to splice noise channels between gates
later without touching either executor.

Notes
-----
Gate conventions follow PennyLane exactly (``RX(t) = exp(-i t X / 2)`` and so
on), because the reference implementation this replicates is written against
PennyLane. Parameter layouts reproduce ``afrah/qcpinn`` gate for gate, including
its two parameter-indexing defects in the ``cascade`` ansatz, which are
reproduced by default and can be switched off with ``fix_cascade``.
'''

from dataclasses import dataclass

ANSATZE = ('cascade', 'alternate', 'layered', 'cross_mesh')
ENCODINGS = ('angle', 'amplitude')


@dataclass(frozen=True)
class Gate:
    '''A single gate in a circuit description.

    Parameters
    ----------
    kind : str
        One of 'RX', 'RY', 'RZ', 'CNOT', 'CRX', 'CRZ'.
    wires : tuple of int
        Target wires. Two-qubit gates are ordered ``(control, target)``.
    source : str or None
        Where the rotation angle comes from: 'theta' for a trainable
        parameter, 'x' for an input feature, None for a fixed gate.
    index : int or None
        Flat index into the trainable parameter vector or the input feature
        vector, depending on `source`.
    '''

    kind: str
    wires: tuple
    source: str = None
    index: int = None


def n_params_per_layer(ansatz, n_qubits):
    '''Trainable parameters allocated per variational layer.

    Parameters
    ----------
    ansatz : str
        One of the entries of :data:`ANSATZE`.
    n_qubits : int
        Number of qubits.

    Returns
    -------
    int
        Parameter count, matching the reference implementation's allocation.

    Notes
    -----
    For 'cascade' this is the *allocated* count. Five of those parameters per
    layer are never referenced by the reference circuit; see :func:`_cascade`.
    '''
    if ansatz == 'cascade':
        return n_qubits * 3
    if ansatz == 'alternate':
        return n_qubits * 4 - 4
    if ansatz == 'layered':
        return n_qubits * 4
    if ansatz == 'cross_mesh':
        return 4 * n_qubits + n_qubits * (n_qubits - 1)
    raise ValueError(f'unknown ansatz: {ansatz}')


def _cascade(n, base, fix_cascade=False):
    '''Cascade (ring) topology: RX and RZ rotations followed by CRX entanglers.

    Notes
    -----
    The reference implementation resets its parameter counter to zero before
    emitting the entangling gates, so the five CRX angles alias the five RX
    angles, and parameters ``2n..3n-1`` are allocated, initialised and handed to
    the optimiser but never applied. Both defects are reproduced when
    `fix_cascade` is False so that replication is faithful; setting it True
    gives the entanglers their own parameters and recovers the unused block.
    '''
    gates = []
    for q in range(n):
        gates.append(Gate('RX', (q,), 'theta', base + q))
    for q in range(n):
        gates.append(Gate('RZ', (q,), 'theta', base + n + q))

    ent_base = base + 2 * n if fix_cascade else base
    gates.append(Gate('CRX', (n - 1, 0), 'theta', ent_base))
    for k, i in enumerate(reversed(range(1, n)), start=1):
        gates.append(Gate('CRX', (i - 1, i), 'theta', ent_base + k))
    return gates


def _layered(n, base):
    '''Layered topology: RZ/RX rotations, a CNOT ring, then RX/RZ rotations.'''
    gates = []
    idx = base
    for q in range(n):
        gates.append(Gate('RZ', (q,), 'theta', idx))
        gates.append(Gate('RX', (q,), 'theta', idx + 1))
        idx += 2
    for q in range(n):
        gates.append(Gate('CNOT', (q, (q + 1) % n)))
    for q in range(n):
        gates.append(Gate('RX', (q,), 'theta', idx))
        gates.append(Gate('RZ', (q,), 'theta', idx + 1))
        idx += 2
    return gates


def _alternate(n, base):
    '''Alternate topology: two sweeps of thinly dressed CNOT blocks.'''
    gates = []
    idx = base

    def tdcnot(ctrl, tgt):
        nonlocal idx
        gates.append(Gate('RY', (ctrl,), 'theta', idx))
        gates.append(Gate('RY', (tgt,), 'theta', idx + 1))
        gates.append(Gate('CNOT', (ctrl, tgt)))
        gates.append(Gate('RZ', (ctrl,), 'theta', idx + 2))
        gates.append(Gate('RZ', (tgt,), 'theta', idx + 3))
        idx += 4

    for i in list(range(n - 1))[::2]:
        tdcnot(i, (i + 1) % n)
    for i in list(range(n))[1::2]:
        tdcnot(i, (i + 1) % n)
    return gates


def _cross_mesh(n, base):
    '''Cross-mesh topology: all-to-all CRZ coupling between rotation blocks.'''
    gates = []
    idx = base
    for q in range(n):
        gates.append(Gate('RX', (q,), 'theta', idx))
        idx += 1
    for q in range(n):
        gates.append(Gate('RZ', (q,), 'theta', idx))
        idx += 1
    for i in reversed(range(n)):
        for j in reversed(range(n)):
            if j != i:
                gates.append(Gate('CRZ', (i, j), 'theta', idx))
                idx += 1
    for q in range(n):
        gates.append(Gate('RX', (q,), 'theta', idx))
        idx += 1
    for q in range(n):
        gates.append(Gate('RZ', (q,), 'theta', idx))
        idx += 1
    return gates


def build_circuit(ansatz, n_qubits, n_layers, encoding='angle', fix_cascade=False):
    '''Build the full gate list for a QCPINN quantum layer.

    Parameters
    ----------
    ansatz : str
        One of :data:`ANSATZE`.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of repetitions of the variational block.
    encoding : str, optional
        'angle' applies RX(x_i) on wire i; 'amplitude' prepares the state
        directly and emits no embedding gates.
    fix_cascade : bool, optional
        If True, repair the reference implementation's cascade parameter
        aliasing. Ignored for other ansaetze.

    Returns
    -------
    gates : list of Gate
        Embedding gates followed by `n_layers` variational blocks.
    n_params : int
        Total trainable parameter count.

    Examples
    --------
    >>> gates, n = build_circuit('cascade', 5, 1)
    >>> n
    15
    '''
    if ansatz not in ANSATZE:
        raise ValueError(f'unknown ansatz: {ansatz}')
    if encoding not in ENCODINGS:
        raise ValueError(f'unknown encoding: {encoding}')

    gates = []
    if encoding == 'angle':
        gates += [Gate('RX', (q,), 'x', q) for q in range(n_qubits)]

    ppl = n_params_per_layer(ansatz, n_qubits)
    builders = {
        'cascade': lambda b: _cascade(n_qubits, b, fix_cascade),
        'alternate': lambda b: _alternate(n_qubits, b),
        'layered': lambda b: _layered(n_qubits, b),
        'cross_mesh': lambda b: _cross_mesh(n_qubits, b),
    }
    for layer in range(n_layers):
        gates += builders[ansatz](layer * ppl)

    return gates, ppl * n_layers

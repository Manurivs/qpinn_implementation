'''Compile a symbolic circuit into a fast execution plan.

The simulator is dispatch bound, not FLOP bound: a five-qubit gate is a few
thousand floating point operations but costs one or more PyTorch kernel
launches. Three transformations cut the number of launches per circuit:

1. Fusion of consecutive single-qubit gates on the same wire into one 2x2
   matrix. Single-qubit gates on different wires commute, so they may be
   reordered freely as long as each wire's run is flushed before a two-qubit
   gate touches it.
2. A diagonal fast path. RZ and CRZ are diagonal, so they apply as a broadcast
   elementwise multiply instead of a contraction.
3. Precomputed einsum subscripts, which remove the ``movedim``/``reshape``
   round trip around every contraction.

The plan is built once per model and reused for every forward pass.
'''

from dataclasses import dataclass, field

import torch

from src.quantum import gates as G

_QLETTERS = 'abcdefghij'
_DIAGONAL = ('RZ',)


@dataclass
class Op:
    '''One executable operation in a compiled plan.

    Attributes
    ----------
    kind : str
        'u1' (dense single qubit), 'diag1' (RZ), 'diag2' (CRZ), or 'u2'
        (dense two qubit).
    wires : tuple of int
        Target wires.
    angles : list of tuple
        ``(gate_kind, source, index)`` entries applied in order. Empty for
        fixed gates such as CNOT.
    sub_shared : str
        einsum subscripts when every angle is a shared parameter.
    sub_batched : str
        einsum subscripts when any angle comes from the input batch.
    static : torch.Tensor or None
        Precomputed matrix for parameter-free gates.
    '''

    kind: str
    wires: tuple
    angles: list = field(default_factory=list)
    sub_shared: str = ''
    sub_batched: str = ''
    static: torch.Tensor = None


def _subs_1q(n, k):
    '''einsum subscripts for a single-qubit contraction on wire `k`.'''
    q = list(_QLETTERS[:n])
    src, dst = list(q), list(q)
    src[k], dst[k] = 'q', 'p'
    return f'pq,z{"".join(src)}->z{"".join(dst)}', \
           f'zpq,z{"".join(src)}->z{"".join(dst)}'


def _subs_2q(n, k0, k1):
    '''einsum subscripts for a two-qubit contraction on wires `k0`, `k1`.'''
    q = list(_QLETTERS[:n])
    src, dst = list(q), list(q)
    src[k0], src[k1] = 'q', 's'
    dst[k0], dst[k1] = 'p', 'r'
    return f'prqs,z{"".join(src)}->z{"".join(dst)}', \
           f'zprqs,z{"".join(src)}->z{"".join(dst)}'


def compile_circuit(gates, n_qubits, fuse=True):
    '''Turn a gate list into an execution plan.

    Parameters
    ----------
    gates : list of Gate
        Circuit description from :func:`src.quantum.ansatz.build_circuit`.
    n_qubits : int
        Number of qubits.
    fuse : bool, optional
        Fuse adjacent single-qubit gates. Disable to compare against the
        unfused path.

    Returns
    -------
    list of Op
        Operations to apply in order.
    '''
    plan = []
    pending = {q: [] for q in range(n_qubits)}

    def flush(q):
        run = pending[q]
        if not run:
            return
        pending[q] = []
        if all(k == 'RZ' for k, _, _ in run):
            sub_s, sub_b = _subs_1q(n_qubits, q)
            plan.append(Op('diag1', (q,), run, sub_s, sub_b))
            return
        sub_s, sub_b = _subs_1q(n_qubits, q)
        plan.append(Op('u1', (q,), run, sub_s, sub_b))

    for g in gates:
        if len(g.wires) == 1:
            pending[g.wires[0]].append((g.kind, g.source, g.index))
            if not fuse:
                flush(g.wires[0])
            continue
        for w in g.wires:
            flush(w)
        sub_s, sub_b = _subs_2q(n_qubits, *g.wires)
        if g.kind == 'CNOT':
            plan.append(Op('u2', g.wires, [], sub_s, sub_b, static=G.cnot()))
        elif g.kind == 'CRZ':
            plan.append(Op('diag2', g.wires, [(g.kind, g.source, g.index)],
                           sub_s, sub_b))
        else:
            plan.append(Op('u2', g.wires, [(g.kind, g.source, g.index)],
                           sub_s, sub_b))

    for q in range(n_qubits):
        flush(q)
    return plan

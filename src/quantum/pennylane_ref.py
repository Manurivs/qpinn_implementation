'''PennyLane reference implementation, transcribed from ``afrah/qcpinn``.

This module exists only to validate the fast PyTorch backend. It is a
deliberately close transcription of ``src/nn/DVQuantumLayer.py`` in the original
repository, written independently of :mod:`src.quantum.ansatz`, so that the
equivalence tests check the circuit *description* as well as the executor. It is
not used during training.
'''

import pennylane as qml
import torch


def _cascade(params, n):
    counter = 0
    for i in range(n):
        qml.RX(params[counter], wires=i)
        counter += 1
    for i in range(n):
        qml.RZ(params[counter], wires=i)
        counter += 1
    counter = 0
    qml.CRX(params[counter], wires=[n - 1, 0])
    counter += 1
    for i in reversed(range(1, n)):
        qml.CRX(params[counter], wires=[i - 1, i])
        counter += 1


def _layered(params, n):
    idx = 0
    for q in range(n):
        qml.RZ(params[idx], wires=q)
        idx += 1
        qml.RX(params[idx], wires=q)
        idx += 1
    for q in range(n):
        qml.CNOT(wires=[q, (q + 1) % n])
    for q in range(n):
        qml.RX(params[idx], wires=q)
        idx += 1
        qml.RZ(params[idx], wires=q)
        idx += 1


def _alternate(params, n):
    idx = 0

    def build_tdcnot(ctrl, tgt):
        nonlocal idx
        qml.RY(params[idx], wires=ctrl)
        idx += 1
        qml.RY(params[idx], wires=tgt)
        idx += 1
        qml.CNOT(wires=[ctrl, tgt])
        qml.RZ(params[idx], wires=ctrl)
        idx += 1
        qml.RZ(params[idx], wires=tgt)
        idx += 1

    for i in range(n - 1)[::2]:
        build_tdcnot(i, (i + 1) % n)
    for i in range(n)[1::2]:
        build_tdcnot(i, (i + 1) % n)


def _cross_mesh(params, n):
    idx = 0
    for i in range(n):
        qml.RX(params[idx], wires=i)
        idx += 1
    for i in range(n):
        qml.RZ(params[idx], wires=i)
        idx += 1
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if j != i:
                qml.CRZ(params[idx], wires=[i, j])
                idx += 1
    for i in range(n):
        qml.RX(params[idx], wires=i)
        idx += 1
    for i in range(n):
        qml.RZ(params[idx], wires=i)
        idx += 1


_BLOCKS = {
    'cascade': _cascade,
    'layered': _layered,
    'alternate': _alternate,
    'cross_mesh': _cross_mesh,
}


def make_qnode(ansatz, n_qubits, n_layers, encoding='angle', diff_method='backprop'):
    '''Build a PennyLane QNode mirroring the reference quantum layer.

    Parameters
    ----------
    ansatz : str
        One of 'cascade', 'alternate', 'layered', 'cross_mesh'.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational blocks.
    encoding : str, optional
        'angle' or 'amplitude'.
    diff_method : str, optional
        Passed through to the QNode.

    Returns
    -------
    callable
        ``f(x, params) -> tensor`` of shape ``(batch, n_qubits)``, where
        `params` has shape ``(n_layers, params_per_layer)``.
    '''
    dev = qml.device('default.qubit', wires=n_qubits, shots=None)

    def circuit(x, params):
        if encoding == 'amplitude':
            qml.templates.AmplitudeEmbedding(
                x, wires=range(n_qubits), normalize=True, pad_with=0.0
            )
        else:
            qml.templates.AngleEmbedding(x, wires=range(n_qubits), rotation='X')
        for layer in range(n_layers):
            _BLOCKS[ansatz](params[layer], n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    qnode = qml.QNode(circuit, dev, interface='torch', diff_method=diff_method)

    def wrapped(x, params):
        out = qnode(x, params)
        return torch.stack(list(out), dim=-1)

    return wrapped

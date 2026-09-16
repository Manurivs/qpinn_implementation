'''QCPINN model: classical preprocessor, quantum layer, classical postprocessor.

Mirrors ``DVPDESolver`` in ``afrah/qcpinn``, with the quantum layer swappable
between the fast PyTorch backend and the PennyLane reference.
'''

import torch
import torch.nn as nn

from src.quantum import torch_sim
from src.quantum.ansatz import build_circuit, n_params_per_layer
from src.quantum.compile import compile_circuit


class QuantumLayer(nn.Module):
    '''Variational quantum layer returning Pauli-Z expectations.

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
    backend : str, optional
        'torch' for the built-in simulator, 'pennylane' for the reference.
    fix_cascade : bool, optional
        Repair the reference cascade parameter aliasing.
    reupload : bool, optional
        Re-apply the angle embedding before every variational block.
    input_scale : bool, optional
        Learn a multiplicative scale on each encoding angle. With `reupload`
        the scale is per layer and per wire, letting each block select its own
        frequency; otherwise it is one scale per wire.
    '''

    def __init__(self, ansatz, n_qubits, n_layers, encoding='angle',
                 backend='torch', fix_cascade=False, reupload=False,
                 input_scale=False):
        super().__init__()
        self.ansatz = ansatz
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.backend = backend
        self.reupload = reupload

        self.gates, n_params = build_circuit(
            ansatz, n_qubits, n_layers, encoding, fix_cascade, reupload
        )

        self.input_scale = None
        if input_scale:
            shape = (n_layers, n_qubits) if reupload else (n_qubits,)
            self.input_scale = nn.Parameter(torch.ones(shape))
        self.plan = compile_circuit(self.gates, n_qubits)
        self.ppl = n_params_per_layer(ansatz, n_qubits)
        self.params = nn.Parameter(torch.empty(n_layers, self.ppl))
        nn.init.xavier_normal_(self.params)

        if backend == 'pennylane':
            from src.quantum.pennylane_ref import make_qnode
            self._qnode = make_qnode(ansatz, n_qubits, n_layers, encoding)

    def encode(self, x):
        '''Expand and scale input features for the embedding gates.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_qubits)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_qubits)``, or ``(batch, n_layers * n_qubits)``
            when re-uploading, matching the feature indices the circuit
            description expects.
        '''
        if not self.reupload:
            return x if self.input_scale is None else x * self.input_scale
        xs = x.unsqueeze(1)
        if self.input_scale is not None:
            xs = xs * self.input_scale.unsqueeze(0)
        else:
            xs = xs.expand(-1, self.n_layers, -1)
        return xs.reshape(x.shape[0], -1)

    def forward(self, x):
        '''Run the circuit on a batch of features.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_qubits)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_qubits)``.
        '''
        if self.backend == 'pennylane':
            return self._qnode(x, self.params).to(x.dtype)
        return torch_sim.run_plan(
            self.plan, self.encode(x), self.params.reshape(-1),
            self.n_qubits, self.encoding
        )


class QCPINN(nn.Module):
    '''Hybrid quantum-classical PINN.

    Parameters
    ----------
    in_dim, out_dim : int
        Input and output dimensions of the PDE solution map.
    hidden_dim : int, optional
        Width of the classical pre/postprocessor.
    n_qubits, n_layers : int, optional
        Quantum layer size.
    ansatz, encoding, backend : str, optional
        Passed to :class:`QuantumLayer`.
    fix_cascade, reupload, input_scale : bool, optional
        Passed to :class:`QuantumLayer`.
    '''

    def __init__(self, in_dim, out_dim, hidden_dim=50, n_qubits=5, n_layers=1,
                 ansatz='cascade', encoding='angle', backend='torch',
                 fix_cascade=False, reupload=False, input_scale=False):
        super().__init__()
        self.preprocessor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_qubits),
        )
        self.quantum_layer = QuantumLayer(
            ansatz, n_qubits, n_layers, encoding, backend, fix_cascade,
            reupload, input_scale
        )
        self.postprocessor = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        '''Xavier-normal weights and zero biases, as in the reference.'''
        for layer in self.preprocessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        '''Map PDE coordinates to solution values.'''
        return self.postprocessor(self.quantum_layer(self.preprocessor(x)))

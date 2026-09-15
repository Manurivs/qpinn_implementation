'''Classical PINN baselines from the QCPINN paper.

`model-1` and `model-2` reproduce ``ClassicalSolver2`` and ``ClassicalSolver``
in the reference implementation, including their asymmetric weight
initialisation: only the first block receives Xavier-normal weights and zero
biases, the rest keep PyTorch defaults.
'''

import torch.nn as nn


class ClassicalPINN(nn.Module):
    '''Fully connected PINN baseline.

    Parameters
    ----------
    in_dim, out_dim : int
        Input and output dimensions.
    hidden_dim : int, optional
        Layer width.
    variant : str, optional
        'model-1' for the deeper five-layer network (7851 parameters at
        ``in_dim=2, out_dim=1``), 'model-2' for the three-layer network
        (2751 parameters), or 'matched' for a width chosen to approximate a
        target parameter count.
    width : int, optional
        Layer width for the 'matched' variant.
    '''

    def __init__(self, in_dim, out_dim, hidden_dim=50, variant='model-2', width=None):
        super().__init__()
        h = width if variant == 'matched' else hidden_dim
        if variant == 'model-1':
            dims = [in_dim, h, h, h, h, out_dim]
        elif variant == 'model-2':
            dims = [in_dim, h, h, out_dim]
        elif variant == 'matched':
            dims = [in_dim, h, h, out_dim]
        else:
            raise ValueError(f'unknown variant: {variant}')

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

        first = self.net[0]
        nn.init.xavier_normal_(first.weight)
        nn.init.zeros_(first.bias)

    def forward(self, x):
        '''Map coordinates to solution values.'''
        return self.net(x)

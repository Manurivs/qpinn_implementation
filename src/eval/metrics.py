'''Error metrics matching the reference implementation.'''

import torch


def relative_l2(pred, exact):
    '''Relative L2 error as a percentage.

    Parameters
    ----------
    pred, exact : torch.Tensor
        Predicted and exact fields of the same shape.

    Returns
    -------
    float
        ``100 * ||pred - exact||_2 / ||exact||_2``.
    '''
    return (torch.norm(pred - exact, 2) / torch.norm(exact, 2) * 100).item()

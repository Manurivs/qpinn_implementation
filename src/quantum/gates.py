'''Batched gate matrices in PennyLane's sign conventions.

All constructors accept an angle tensor of shape ``(...,)`` and return a matrix
of shape ``(..., 2, 2)`` or ``(..., 4, 4)``. Matrices are built with
``torch.stack`` rather than in-place writes so that second-order autograd
through the angle stays clean, which the PINN residual depends on.
'''

import torch

_I2 = [[1.0, 0.0], [0.0, 1.0]]


def _stack2(rows, dtype):
    '''Stack four scalar/batched entries into a (..., 2, 2) matrix.'''
    r0 = torch.stack([rows[0], rows[1]], dim=-1)
    r1 = torch.stack([rows[2], rows[3]], dim=-1)
    return torch.stack([r0, r1], dim=-2).to(dtype)


def rx(theta, dtype=torch.complex64):
    '''RX(theta) = exp(-i theta X / 2).'''
    c = torch.cos(theta / 2).to(dtype)
    s = torch.sin(theta / 2).to(dtype)
    j = torch.tensor(1j, dtype=dtype, device=theta.device)
    return _stack2([c, -j * s, -j * s, c], dtype)


def ry(theta, dtype=torch.complex64):
    '''RY(theta) = exp(-i theta Y / 2).'''
    c = torch.cos(theta / 2).to(dtype)
    s = torch.sin(theta / 2).to(dtype)
    return _stack2([c, -s, s, c], dtype)


def rz(theta, dtype=torch.complex64):
    '''RZ(theta) = exp(-i theta Z / 2).'''
    half = (theta / 2).to(torch.float64 if dtype == torch.complex128 else torch.float32)
    zero = torch.zeros_like(half)
    e_m = torch.complex(torch.cos(half), -torch.sin(half)).to(dtype)
    e_p = torch.complex(torch.cos(half), torch.sin(half)).to(dtype)
    z = torch.complex(zero, zero).to(dtype)
    return _stack2([e_m, z, z, e_p], dtype)


def _controlled(u, dtype):
    '''Embed a (..., 2, 2) gate as the lower block of a (..., 4, 4) matrix.'''
    batch = u.shape[:-2]
    out = torch.zeros(*batch, 4, 4, dtype=dtype, device=u.device)
    eye = torch.tensor(_I2, dtype=dtype, device=u.device).expand(*batch, 2, 2)
    top = torch.cat([eye, torch.zeros_like(eye)], dim=-1)
    bot = torch.cat([torch.zeros_like(u), u], dim=-1)
    return torch.cat([top, bot], dim=-2).to(out.dtype)


def crx(theta, dtype=torch.complex64):
    '''Controlled-RX with wire order (control, target).'''
    return _controlled(rx(theta, dtype), dtype)


def crz(theta, dtype=torch.complex64):
    '''Controlled-RZ with wire order (control, target).'''
    return _controlled(rz(theta, dtype), dtype)


def cnot(dtype=torch.complex64, device=None):
    '''CNOT with wire order (control, target).'''
    m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    return torch.tensor(m, dtype=dtype, device=device)


BUILDERS_1Q = {'RX': rx, 'RY': ry, 'RZ': rz}
BUILDERS_2Q = {'CRX': crx, 'CRZ': crz}

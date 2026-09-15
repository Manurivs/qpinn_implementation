'''2D Helmholtz benchmark.

Solves ``Delta u + lambda * u = f`` on ``[-1, 1]^2`` with Dirichlet boundary
conditions, against the manufactured solution
``u(x, y) = sin(a1 * pi * x) * sin(a2 * pi * y)``.

Constants and sampling reproduce ``src/data/helmholtz_dataset.py`` in the
reference implementation.
'''

import torch

A1 = 1.0
A2 = 4.0
LAMBDA = 1.0

DOMAIN = ((-1.0, -1.0), (1.0, 1.0))
EDGES = (
    ((-1.0, -1.0), (1.0, -1.0)),
    ((1.0, -1.0), (1.0, 1.0)),
    ((1.0, 1.0), (-1.0, 1.0)),
    ((-1.0, 1.0), (-1.0, -1.0)),
)


def u_exact(x, a1=A1, a2=A2):
    '''Manufactured solution evaluated at coordinates `x` of shape (N, 2).'''
    return torch.sin(a1 * torch.pi * x[:, 0:1]) * torch.sin(a2 * torch.pi * x[:, 1:2])


def f_source(x, a1=A1, a2=A2, lam=LAMBDA):
    '''Forcing term consistent with :func:`u_exact`.'''
    u = u_exact(x, a1, a2)
    return -((a1 * torch.pi) ** 2) * u - ((a2 * torch.pi) ** 2) * u + lam * u


def sample_edge(edge, n, device=None, generator=None):
    '''Draw `n` uniform points along a straight boundary segment.

    Parameters
    ----------
    edge : tuple
        ``((x0, y0), (x1, y1))`` endpoints.
    n : int
        Number of points.
    device : torch.device, optional
        Target device.
    generator : torch.Generator, optional
        Seeded generator for reproducibility.

    Returns
    -------
    x : torch.Tensor
        Coordinates, shape ``(n, 2)``.
    y : torch.Tensor
        Exact solution there, shape ``(n, 1)``.
    '''
    lo = torch.tensor(edge[0], device=device)
    hi = torch.tensor(edge[1], device=device)
    r = torch.rand(n, 2, device=device, generator=generator)
    x = lo + (hi - lo) * r
    return x, u_exact(x)


def sample_interior(n, device=None, generator=None):
    '''Draw `n` uniform collocation points and the forcing term there.'''
    lo = torch.tensor(DOMAIN[0], device=device)
    hi = torch.tensor(DOMAIN[1], device=device)
    r = torch.rand(n, 2, device=device, generator=generator)
    x = lo + (hi - lo) * r
    return x, f_source(x)


def residual(model, x1, x2, lam=LAMBDA):
    '''Helmholtz operator applied to the model output.

    Parameters
    ----------
    model : torch.nn.Module
        Maps ``(N, 2)`` coordinates to ``(N, 1)`` solution values.
    x1, x2 : torch.Tensor
        Coordinate columns of shape ``(N, 1)``; both are marked
        ``requires_grad`` in place.
    lam : float, optional
        Wavenumber squared.

    Returns
    -------
    u : torch.Tensor
        Predicted solution, shape ``(N, 1)``.
    r : torch.Tensor
        ``u_xx + u_yy + lam * u``, shape ``(N, 1)``.
    '''
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    u = model(torch.cat((x1, x2), 1))
    u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u), create_graph=True)[0]
    u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u), create_graph=True)[0]
    u_11 = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1), create_graph=True)[0]
    u_22 = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2), create_graph=True)[0]
    return u, u_11 + u_22 + lam * u


def eval_grid(n=100, device=None):
    '''Uniform ``n x n`` evaluation grid over the domain.

    Returns
    -------
    x : torch.Tensor
        Coordinates, shape ``(n * n, 2)``.
    '''
    a = torch.linspace(DOMAIN[0][0], DOMAIN[1][0], n, device=device)
    b = torch.linspace(DOMAIN[0][1], DOMAIN[1][1], n, device=device)
    g0, g1 = torch.meshgrid(a, b, indexing='ij')
    return torch.hstack((g0.flatten().unsqueeze(1), g1.flatten().unsqueeze(1)))

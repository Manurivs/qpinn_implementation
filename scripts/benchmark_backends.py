'''Time a realistic Helmholtz PINN training step on both quantum backends.

The comparison drives the backend decision: the PennyLane reference is the
ground truth for correctness, but the sweep needs a step time that makes 20k
epochs times ten configurations times several seeds affordable on CPU.
'''

import time

import torch

from src.models.qcpinn import QCPINN

BATCH = 64
WARMUP = 3


def helmholtz_step(model, opt):
    '''One optimiser step with a second-order residual and four BC batches.'''
    opt.zero_grad()
    x1 = torch.rand(BATCH, 1) * 2 - 1
    x2 = torch.rand(BATCH, 1) * 2 - 1
    x1.requires_grad_(True)
    x2.requires_grad_(True)
    u = model(torch.cat((x1, x2), 1))
    u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u), create_graph=True)[0]
    u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2), create_graph=True)[0]
    loss = ((u_xx + u_yy + u) ** 2).mean()
    for _ in range(4):
        xb = torch.rand(BATCH, 2) * 2 - 1
        loss = loss + 10.0 * (model(xb) ** 2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    return loss


def bench(backend, ansatz, n_steps):
    '''Return mean seconds per training step.'''
    torch.manual_seed(0)
    model = QCPINN(2, 1, ansatz=ansatz, backend=backend)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(WARMUP):
        helmholtz_step(model, opt)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        helmholtz_step(model, opt)
    return (time.perf_counter() - t0) / n_steps


if __name__ == '__main__':
    n_params = sum(p.numel() for p in QCPINN(2, 1).parameters())
    print(f'cascade QCPINN parameters: {n_params}\n')
    print(f'{"ansatz":<12}{"backend":<12}{"s/step":>10}{"20k epochs":>14}')
    print('-' * 48)
    rows = {}
    for ansatz in ('cascade', 'cross_mesh'):
        for backend, n_steps in (('torch', 30), ('pennylane', 10)):
            dt = bench(backend, ansatz, n_steps)
            rows[(ansatz, backend)] = dt
            print(f'{ansatz:<12}{backend:<12}{dt:>10.4f}{dt * 20000 / 60:>11.1f} min')
    print()
    for ansatz in ('cascade', 'cross_mesh'):
        speedup = rows[(ansatz, 'pennylane')] / rows[(ansatz, 'torch')]
        print(f'{ansatz}: torch backend is {speedup:.1f}x faster')

'''Training loop for the Helmholtz benchmark.

Reproduces ``src/trainer/helmholtz_train.py`` in the reference implementation:
five independent minibatches per step (four boundary edges plus the interior),
an MSE residual loss, a boundary weight applied to the *sum* of the four edge
losses, gradient-norm clipping, and ``ReduceLROnPlateau`` driven by the
training loss.
'''

import time

import torch

from src.pdes import helmholtz


def train(model, epochs=20000, batch_size=64, lr=5e-3, bc_weight=10.0,
          clip_norm=1.0, log_every=1000, generator=None, logger=None):
    '''Fit a model to the Helmholtz benchmark.

    Parameters
    ----------
    model : torch.nn.Module
        Maps ``(N, 2)`` coordinates to ``(N, 1)`` values.
    epochs : int, optional
        Optimiser steps; the reference runs ``epochs + 1`` iterations.
    batch_size : int, optional
        Points drawn per sampler per step.
    lr : float, optional
        Adam learning rate.
    bc_weight : float, optional
        Weight on the summed boundary losses.
    clip_norm : float, optional
        Gradient-norm clipping threshold.
    log_every : int, optional
        Interval for recording progress.
    generator : torch.Generator, optional
        Seeded generator for sampling.
    logger : callable, optional
        Receives ``(epoch, loss_r, loss_bc, lr)`` at each logging interval.

    Returns
    -------
    dict
        ``loss_history``, ``final_loss`` and ``train_seconds``.
    '''
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.9, patience=1000
    )
    loss_fn = torch.nn.MSELoss()

    history = []
    t0 = time.perf_counter()

    for it in range(epochs + 1):
        opt.zero_grad()

        loss_bc = 0.0
        for edge in helmholtz.EDGES:
            xb, ub = helmholtz.sample_edge(edge, batch_size, generator=generator)
            loss_bc = loss_bc + loss_fn(model(xb), ub)

        xr, fr = helmholtz.sample_interior(batch_size, generator=generator)
        _, r_pred = helmholtz.residual(model, xr[:, 0:1], xr[:, 1:2])
        loss_r = loss_fn(r_pred, fr)

        loss = loss_r + bc_weight * loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        opt.step()
        sched.step(loss.item())

        history.append(loss.item())
        if logger is not None and it % log_every == 0:
            logger(it, loss_r.item(), float(loss_bc), opt.param_groups[0]['lr'])

    return {
        'loss_history': history,
        'final_loss': history[-1],
        'train_seconds': time.perf_counter() - t0,
    }


def evaluate(model, n_grid=100):
    '''Relative L2 errors of the solution and forcing fields.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    n_grid : int, optional
        Evaluation grid resolution per axis.

    Returns
    -------
    dict
        ``error_u`` and ``error_f`` as percentages.
    '''
    from src.eval.metrics import relative_l2

    x = helmholtz.eval_grid(n_grid)
    u_star = helmholtz.u_exact(x)
    f_star = helmholtz.f_source(x)
    u_pred, f_pred = helmholtz.residual(model, x[:, 0:1].clone(), x[:, 1:2].clone())
    return {
        'error_u': relative_l2(u_pred.detach(), u_star),
        'error_f': relative_l2(f_pred.detach(), f_star),
    }

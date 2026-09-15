'''Run one Helmholtz experiment and write a JSON result.

Examples
--------
$ PYTHONPATH=. python scripts/run_experiment.py --model qcpinn \
      --ansatz cascade --encoding angle --seed 0 --out results/run.json
'''

import argparse
import json
import os

import torch

from src.models.pinn import ClassicalPINN
from src.models.qcpinn import QCPINN
from src.training.trainer import evaluate, train


def build_model(args):
    '''Instantiate the model named by the parsed arguments.'''
    if args.model == 'qcpinn':
        return QCPINN(
            2, 1, hidden_dim=args.hidden_dim, n_qubits=args.n_qubits,
            n_layers=args.n_layers, ansatz=args.ansatz, encoding=args.encoding,
            backend=args.backend, fix_cascade=args.fix_cascade,
        )
    return ClassicalPINN(2, 1, hidden_dim=args.hidden_dim, variant=args.variant,
                         width=args.width)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=('qcpinn', 'classical'), default='qcpinn')
    p.add_argument('--ansatz', default='cascade')
    p.add_argument('--encoding', default='angle')
    p.add_argument('--variant', default='model-2')
    p.add_argument('--width', type=int, default=None)
    p.add_argument('--backend', default='torch')
    p.add_argument('--fix-cascade', action='store_true')
    p.add_argument('--n-qubits', type=int, default=5)
    p.add_argument('--n-layers', type=int, default=1)
    p.add_argument('--hidden-dim', type=int, default=50)
    p.add_argument('--epochs', type=int, default=20000)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 10_000)

    model = build_model(args)
    n_params = sum(p_.numel() for p_ in model.parameters())

    out = train(model, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, generator=gen)
    out.update(evaluate(model))

    tag = (f'{args.encoding}-{args.ansatz}' if args.model == 'qcpinn'
           else args.variant)
    record = {
        'tag': tag,
        'model': args.model,
        'ansatz': args.ansatz if args.model == 'qcpinn' else None,
        'encoding': args.encoding if args.model == 'qcpinn' else None,
        'variant': args.variant if args.model == 'classical' else None,
        'fix_cascade': args.fix_cascade,
        'seed': args.seed,
        'epochs': args.epochs,
        'lr': args.lr,
        'n_params': n_params,
        'error_u': out['error_u'],
        'error_f': out['error_f'],
        'final_loss': out['final_loss'],
        'train_seconds': out['train_seconds'],
        'loss_history': out['loss_history'][::100],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(record, fh)
    print(f'{tag} seed={args.seed} params={n_params} '
          f'u={record["error_u"]:.2f}% f={record["error_f"]:.2f}% '
          f'loss={record["final_loss"]:.4f} '
          f'({record["train_seconds"] / 60:.1f} min)')


if __name__ == '__main__':
    main()

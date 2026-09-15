'''Drive the Helmholtz replication sweep across configurations and seeds.

Jobs are ordered seed-major, so an interrupted sweep still yields the same
number of seeds for every configuration rather than all seeds for some and none
for others. Completed runs are detected by their output file and skipped, so
the sweep is resumable.

Learning rates differ by model family. The paper specifies 5e-3 for the DV
circuit but never states a rate for the classical baselines; 1e-3 was selected
over {5e-4, 1e-3, 2e-3} by mean relative L2 across three seeds, and is the only
setting at which the baselines escape the trivial solution u = 0.
'''

import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ANSATZE = ('cascade', 'alternate', 'layered', 'cross_mesh')
ENCODINGS = ('angle', 'amplitude')
VARIANTS = ('model-1', 'model-2')

LR_QUANTUM = 5e-3
LR_CLASSICAL = 1e-3


def jobs(out_dir, seeds, epochs):
    '''Yield (name, command) pairs in seed-major order.'''
    for seed in range(seeds):
        for variant in VARIANTS:
            name = f'{variant}_s{seed}'
            yield name, [
                sys.executable, 'scripts/run_experiment.py',
                '--model', 'classical', '--variant', variant,
                '--lr', str(LR_CLASSICAL), '--seed', str(seed),
                '--epochs', str(epochs), '--threads', '1',
                '--out', os.path.join(out_dir, f'{name}.json'),
            ]
        for encoding, ansatz in itertools.product(ENCODINGS, ANSATZE):
            name = f'{encoding}-{ansatz}_s{seed}'
            yield name, [
                sys.executable, 'scripts/run_experiment.py',
                '--model', 'qcpinn', '--ansatz', ansatz, '--encoding', encoding,
                '--lr', str(LR_QUANTUM), '--seed', str(seed),
                '--epochs', str(epochs), '--threads', '1',
                '--out', os.path.join(out_dir, f'{name}.json'),
            ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='results/helmholtz')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--epochs', type=int, default=20000)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    todo = [(n, c) for n, c in jobs(args.out_dir, args.seeds, args.epochs)
            if not os.path.exists(os.path.join(args.out_dir, f'{n}.json'))]
    print(f'{len(todo)} runs queued, {args.workers} workers', flush=True)

    env = dict(os.environ, PYTHONPATH=os.getcwd())
    done = 0

    def run(job):
        name, cmd = job
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return name, r.returncode, (r.stdout or r.stderr).strip().splitlines()[-1:]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, rc, tail in pool.map(run, todo):
            done += 1
            status = 'ok ' if rc == 0 else 'FAIL'
            msg = tail[0] if tail else ''
            print(f'[{done}/{len(todo)}] {status} {msg}', flush=True)


if __name__ == '__main__':
    main()

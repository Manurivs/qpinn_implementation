'''Run the QCPINN improvement ablation.

Each variant is defined by an explicit flag list rather than assembled in the
shell, because the flags are multi-word and word splitting silently regroups
them into wrong commands.

The variants are chosen so each isolates one change. 'depth' and 'reupload'
carry identical parameter counts, so their difference measures re-uploading
rather than capacity.
'''

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

VARIANTS = {
    'depth': ['--n-layers', '2', '--fix-cascade'],
    'reupload': ['--n-layers', '2', '--fix-cascade', '--reupload'],
    'reuploadscale': ['--n-layers', '2', '--fix-cascade', '--reupload',
                      '--input-scale'],
}


def jobs(out_dir, variants, seeds):
    '''Yield (name, command) pairs for every variant and seed not yet run.'''
    for name in variants:
        for seed in range(seeds):
            out = os.path.join(out_dir, f'{name}_s{seed}.json')
            if os.path.exists(out):
                continue
            yield f'{name}_s{seed}', [
                sys.executable, 'scripts/run_experiment.py',
                '--model', 'qcpinn', '--ansatz', 'cascade',
                '--encoding', 'angle', *VARIANTS[name],
                '--lr', '5e-3', '--seed', str(seed),
                '--epochs', '20000', '--threads', '1', '--out', out,
            ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='results/improve')
    p.add_argument('--variants', nargs='+', default=list(VARIANTS))
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    todo = list(jobs(args.out_dir, args.variants, args.seeds))
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
            print(f'[{done}/{len(todo)}] {"ok " if rc == 0 else "FAIL"} '
                  f'{name}: {tail[0] if tail else ""}', flush=True)


if __name__ == '__main__':
    main()

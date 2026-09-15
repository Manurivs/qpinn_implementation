'''Aggregate sweep results into comparison tables.

Reports median alongside mean and standard deviation. Both model families have
a heavy-tailed failure mode on this problem: some seeds collapse to the trivial
solution u = 0, which the boundary conditions admit, and a single such run
dominates a ten-seed mean. The median says what a typical run does; the
collapse count says how often the method fails outright.
'''

import glob
import json
import os
import statistics as st

COLLAPSE_THRESHOLD = 90.0

PAPER_V6 = {
    'angle-cascade': (6.69, 1.51),
    'model-2': (12.12, 7.17),
}


def load(results_dir):
    '''Load every run record from `results_dir`, keyed by tag.

    Returns
    -------
    dict
        Maps tag to a list of records sorted by seed.
    '''
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        r = json.load(open(path))
        out.setdefault(r['tag'], []).append(r)
    return {k: sorted(v, key=lambda r: r['seed']) for k, v in out.items()}


def summarise(records, key='error_u'):
    '''Summary statistics for one configuration.

    Parameters
    ----------
    records : list of dict
        Run records for a single tag.
    key : str, optional
        Metric field to summarise.

    Returns
    -------
    dict
        Count, mean, standard deviation, median, and collapse count.
    '''
    vals = [r[key] for r in records]
    return {
        'n': len(vals),
        'params': records[0]['n_params'],
        'mean': st.mean(vals),
        'std': st.stdev(vals) if len(vals) > 1 else 0.0,
        'median': st.median(vals),
        'collapsed': sum(v >= COLLAPSE_THRESHOLD for v in vals),
    }


def render(results_dir, key='error_u'):
    '''Render a text table of every configuration present.'''
    data = load(results_dir)
    rows = [(tag, summarise(recs, key)) for tag, recs in sorted(data.items())]
    rows.sort(key=lambda r: r[1]['median'])

    head = (f'{"config":<20}{"n":>3}{"params":>8}{"median":>9}{"mean":>9}'
            f'{"std":>8}{"fail":>6}   paper v6')
    lines = [head, '-' * len(head)]
    for tag, s in rows:
        paper = PAPER_V6.get(tag)
        ref = f'{paper[0]:.2f} +/- {paper[1]:.2f}' if paper else ''
        lines.append(
            f'{tag:<20}{s["n"]:>3}{s["params"]:>8}{s["median"]:>9.2f}'
            f'{s["mean"]:>9.2f}{s["std"]:>8.2f}{s["collapsed"]:>4}/{s["n"]}   {ref}'
        )
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys
    print(render(sys.argv[1] if len(sys.argv) > 1 else 'results/helmholtz'))

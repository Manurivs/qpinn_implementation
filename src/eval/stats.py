'''Significance testing for the Helmholtz comparison.

Both model families have a heavy-tailed failure mode on this benchmark, so
means are not a safe summary: a single collapsed run moves a ten-seed mean by
more than the entire spread of the healthy runs. Comparisons therefore use
medians and rank tests, and report the collapse rate separately with an exact
confidence interval rather than folding failures into a mean.
'''

import glob
import json
import statistics as st
from collections import defaultdict

from scipy import stats

COLLAPSE_THRESHOLD = 90.0


def load_lr_study(path='results/lr_study'):
    '''Load the learning-rate study, keyed by (variant, lr).'''
    d = defaultdict(list)
    for f in glob.glob(f'{path}/*.json'):
        r = json.load(open(f))
        d[(r['variant'], r['lr'])].append(r['error_u'])
    return {k: sorted(v) for k, v in d.items()}


def best_lr(study, variant):
    '''Pick the learning rate minimising median error, then collapse count.

    Returns
    -------
    tuple
        ``(lr, errors)`` for the selected rate.
    '''
    cands = [(lr, v) for (var, lr), v in study.items() if var == variant]
    return min(cands, key=lambda kv: (st.median(kv[1]),
                                      sum(x >= COLLAPSE_THRESHOLD for x in kv[1])))


def compare(a, b, name_a, name_b):
    '''Compare two sets of per-seed errors.

    Parameters
    ----------
    a, b : list of float
        Relative L2 errors, one per seed.
    name_a, name_b : str
        Labels for reporting.

    Returns
    -------
    dict
        Collapse counts, medians, and p-values for the rank, collapse-rate and
        variance comparisons.
    '''
    fa = sum(x >= COLLAPSE_THRESHOLD for x in a)
    fb = sum(x >= COLLAPSE_THRESHOLD for x in b)
    ok_a = [x for x in a if x < COLLAPSE_THRESHOLD]
    ok_b = [x for x in b if x < COLLAPSE_THRESHOLD]
    return {
        'pair': f'{name_a} vs {name_b}',
        'median_a': st.median(a), 'median_b': st.median(b),
        'collapsed_a': fa, 'collapsed_b': fb,
        'p_rank': stats.mannwhitneyu(a, b)[1],
        'p_rank_healthy': stats.mannwhitneyu(ok_a, ok_b)[1],
        'p_collapse': stats.fisher_exact([[fa, len(a) - fa],
                                          [fb, len(b) - fb]])[1],
        'p_variance': stats.levene(ok_a, ok_b)[1],
    }


def collapse_ci(n_fail, n):
    '''Exact binomial confidence interval on a collapse rate.'''
    return tuple(stats.binomtest(n_fail, n).proportion_ci())

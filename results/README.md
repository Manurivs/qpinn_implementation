# Sweep results

One JSON per run, named `<tag>_s<seed>.json`, written by
`scripts/run_experiment.py`. Each record carries the configuration, parameter
count, relative L2 errors for `u` and `f`, final loss, wall-clock time, and the
loss history subsampled every 100 epochs.

These are committed deliberately: runs cost roughly 25 minutes of CPU each, so
the records are worth more than the space they take.

## Learning rates

Quantum configurations use lr = 5e-3, as specified in the paper for the DV
circuit. Classical baselines use lr = 1e-3, selected over {5e-4, 1e-3, 2e-3}
across three seeds at full length. The paper does not state a rate for the
baselines, and at the DV rate both collapse to the trivial solution u = 0,
which this problem's boundary conditions admit because the exact solution
vanishes on the entire boundary.

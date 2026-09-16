# Replication notes: QCPINN (arXiv:2503.16678)

Findings from reading the reference implementation at
[`afrah/QCPINN`](https://github.com/afrah/QCPINN) and from building an
independent simulator validated against it.

## Confirmed configuration (Helmholtz)

| Setting | Value |
| --- | --- |
| Qubits / variational layers | 5 / 1 |
| Classical wrapper | `Linear(2,50) → Tanh → Linear(50,5)`, `Linear(5,50) → Tanh → Linear(50,1)` |
| Angle embedding | `AngleEmbedding(..., rotation='X')`, i.e. RX |
| Optimiser | Adam, `ReduceLROnPlateau(factor=0.9, patience=1000)` |
| Epochs / batch | 20 000 / 64 |
| Gradient clipping | max-norm 1.0 (DV), 0.1 (CV) |
| Evaluation grid | 100 × 100 uniform on [-1,1]^2 |
| Metric | `‖pred − exact‖₂ / ‖exact‖₂ × 100` |

Loss is `loss_r + 10.0 * (loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4)`. The
weight multiplies the **sum** of four separate boundary MSEs, so relative to a
mean-based formulation the effective boundary weight is 40, not 10.

## Defect 1 — a third of the cascade parameters cannot affect the output

`DVQuantumLayer.cascade` allocates `3 * n_qubits = 15` parameters per layer.
`add_rotations` consumes indices 0–9 (RX then RZ on each wire), after which
`add_entangling_gates` **resets its counter to zero** and draws the five CRX
angles from indices 0–4 again.

Consequently, per layer:

- indices 0–4 are **aliased** across the RX rotations and the CRX entanglers;
- indices 10–14 are **never applied** — allocated, Xavier-initialised, and
  handed to the optimiser, but absent from the circuit.

The reported headline count of 771 parameters therefore includes 5 that cannot
influence the output, and the ansatz has 10 effective quantum degrees of
freedom rather than 15. Reproduced faithfully by default; `fix_cascade=True`
gives the entanglers their own parameters.

## Defect 2 — amplitude embedding is starved by construction

The preprocessor emits `n_qubits = 5` features, which are passed to
`AmplitudeEmbedding(..., normalize=True, pad_with=0.0)` over 5 wires — a
32-dimensional state. Padding pins 27 of 32 amplitudes to zero, and
normalisation discards the input's scale, so the embedding carries 4 effective
degrees of freedom.

The paper's finding that amplitude embedding underperforms angle embedding is
therefore a property of this padding, not of amplitude encoding. (Both branches
of the `if self.encoding == 'amplitude'` test in `DVPDESolver.__init__` are
also byte-identical, so the conditional is dead code.)

## Unresolved — parameter counts for three of four ansaetze

With the confirmed architecture above, totals are 756 classical parameters plus
the quantum block:

| Ansatz | Per layer | Total (L=1) | Paper v4 | Paper v6 |
| --- | --- | --- | --- | --- |
| cascade | 15 | **771** | 771 ✓ | 771 ✓ |
| alternate | 16 | 772 | 781 ✗ | — |
| layered | 20 | 776 | 781 ✗ | — |
| cross_mesh | 40 | **796** | 836 ✗ | 796 ✓ (wave) |

`cascade` reproduces exactly in both versions and `cross_mesh` matches v6. No
integer layer count reproduces v4's 781/781/836 from this architecture. Treated
as unresolved; replication targets v6.

## PDE specifications (from `src/nn/pde.py`)

- **Helmholtz**: `f = u_xx + u_yy + λu`, λ = 1.0, a₁ = 1, a₂ = 4, Ω = [-1,1]².
- **Wave**: `u_tt − c²u_xx`, c = 2, exact solution
  `sin(πx)cos(cπt) + a·sin(2cπx)cos(4cπt)`.
- **Klein-Gordon**: `u_tt − u_xx + u³`. The nonlinearity is **cubic**
  (α = −1, β = 0, γ = 1, k = 3), not the `m²u` term the paper's Appendix A.4
  states.
- **Convection-diffusion**: `u_t + v_x u_x + v_y u_y − D(u_xx + u_yy)`,
  D = 0.01, v_x = v_y = 1.0 (2D, not 1D).
- **Cavity**: μ = 0.00345 in code, against μ = 0.01 in the paper.

## Backend benchmark

One Helmholtz training step (second-order residual plus four boundary batches,
batch 64), CPU, measured — not estimated:

| Ansatz | torch `complex64` | torch `complex128` | PennyLane `complex128` |
| --- | --- | --- | --- |
| cascade | 40.5 ms | 43.7 ms | 74.9 ms |
| cross_mesh | 56.0 ms | 59.6 ms | 136.0 ms |

At matched precision the in-house backend is **1.7–2.3× faster**, after gate
fusion and a diagonal fast path cut cascade from 20 gate applications to 10 and
cross_mesh from 45 to 30. This is well short of the order of magnitude
originally assumed; PennyLane's `default.qubit` is not the bottleneck it was
taken to be. The in-house backend is retained because the margin is still real,
because the symbolic gate list is what surfaced Defect 1, and because the
planned noise study needs density-matrix evolution spliced between gates.

## Replication result (Helmholtz, 10 seeds)

| Config | Params | Median | Collapsed | Paper v6 |
| --- | --- | --- | --- | --- |
| angle-cascade | 771 | **6.62** | 0/10 | 6.69 ± 1.51 |
| model-2 (lr 2e-3) | 2751 | 7.22 | 2/10 | 12.12 ± 7.17 |
| model-1 (lr 1e-3) | 7851 | 11.60 | 1/10 | — |

The headline replicates: angle-cascade gives 6.72 ± 1.82 against the paper's
6.69 ± 1.51, from an independent implementation and simulator.

### Means are not a safe summary here

Both model families collapse to the trivial solution `u ≡ 0` on some seeds,
which this problem admits because the exact solution vanishes on the entire
boundary. One collapsed run moves a ten-seed mean further than the whole spread
of the healthy runs — model-2 at lr 1e-3 has median 7.96 and mean 31.75.

This is not an artefact of undertuning. A ten-seed study over {1e-3, 2e-3,
5e-3} shows the failure at every rate, minimised at the rates selected above:

| Variant | 1e-3 | 2e-3 | 5e-3 |
| --- | --- | --- | --- |
| model-1 | median 11.60, 1/10 failed | 25.92, 4/10 | 61.89, 5/10 |
| model-2 | median 7.96, 2/10 failed | 7.22, 2/10 | 54.71, 5/10 |

### The paper's margin over model-2 does not survive

v6 claims roughly 45% lower error than model-2 (6.69 against 12.12). With
medians, 6.62 against 7.22 is a statistical tie: Mann-Whitney p = 0.623 over
all runs and p = 0.829 over healthy runs only. A model-2 mean of 12.12 ± 7.17
is what a heavy-tailed sample looks like when one or two runs degrade and the
mean absorbs them — the same shape our own mean shows. The likely explanation
for the reported margin is the choice of mean over median, not a difference in
model quality.

Against model-1 the advantage is real and large: median 6.62 against 11.60,
p < 0.001, at 10.2x fewer parameters.

### Not supported: a robustness advantage

The 0/10 collapse rate for angle-cascade against 2/10 for model-2 is suggestive
but **not significant** (Fisher exact p = 0.474; against model-1, p = 1.000).
With ten seeds, observing zero failures is consistent with a true failure rate
as high as 30.8%. The tighter spread of the healthy runs is likewise not
significant (Levene p = 0.178). Establishing this would need far more seeds
than the benchmark's cost allows, and it is not claimed.

### Summary

At 771 parameters the hybrid model matches a 2751-parameter classical PINN and
beats a 7851-parameter one. The parameter-efficiency claim holds. The accuracy
margin over the stronger baseline does not.

## Improvement 1: untying the cascade parameters does nothing

The reference cascade circuit aliases its five CRX entangler angles onto its
five RX rotation angles and never applies the remaining five parameters of each
layer. Repairing this gives the entanglers their own angles and puts the unused
block to work, at an unchanged total of 771 parameters.

| Variant | Params | Median | Mean ± std | Collapsed |
| --- | --- | --- | --- | --- |
| baseline (aliased) | 771 | 6.62 | 6.72 ± 1.82 | 0/10 |
| untied | 771 | 6.95 | 6.95 ± 1.96 | 0/10 |

Mann-Whitney p = 1.000, variance p = 0.935, ten seeds each. The repair changes
nothing.

The defect is real — five of fifteen parameters per layer provably cannot reach
the circuit — but it costs no accuracy, so the paper's reported 771 does not
overstate what the model can do. At one layer this ansatz is not limited by its
parameter count, which points at the encoding rather than the variational block
as the binding constraint, and makes data re-uploading the more promising
lever. Both defects therefore stand as implementation findings, not as
explanations of the reported numbers.

## Improvement 2: re-uploading is worse than plain depth

Three variants at two layers, screened at five seeds against the ten-seed
baseline (median 6.62 at 771 parameters). `depth` and `reupload` carry the same
786 parameters, so their difference isolates re-uploading from capacity.

| Variant | Params | n | Median | Std | vs baseline |
| --- | --- | --- | --- | --- | --- |
| depth (L=2) | 786 | 5 | **5.42** | 1.05 | p = 0.310 |
| reupload | 786 | 5 | 8.19 | 3.30 | p = 0.513 |
| reupload + input scale | 796 | 3 | 6.57 | 2.80 | p = 0.811 |

The controlled comparison is significant in the wrong direction: re-uploading
is **worse** than plain depth at identical parameter count, median 8.19 against
5.42, Mann-Whitney p = 0.032. It also triples the spread and produces the
ablation's only outlier at 14.2%.

The standard argument for re-uploading is that one angle embedding makes the
circuit a degree-one trigonometric polynomial in its inputs, so re-injecting
inputs widens the frequency spectrum. That argument assumes the circuit is fed
raw coordinates. In QCPINN it is not: the quantum layer consumes the output of
a two-layer tanh preprocessor, which is already free to construct whatever
features the circuit needs. Re-uploading then re-injects a learned feature
rather than a coordinate, adding gates and optimisation difficulty without
adding representational reach. The expressivity argument does not transfer to
hybrid architectures whose classical front end already carries the nonlinearity.

Plain added depth is the only variant that improves the median, and at five
seeds it is not yet significant against the baseline (p = 0.310). Its spread is
the tightest measured so far, 1.05 against the baseline's 1.82. Confirmation at
ten seeds is running.

## Improvement verdict: nothing beats the baseline significantly

Ten seeds per arm, medians and rank tests throughout.

| Variant | Params | n | Median | Std | vs baseline |
| --- | --- | --- | --- | --- | --- |
| baseline (L=1) | 771 | 10 | 6.62 | 1.82 | — |
| untied cascade | 771 | 10 | 6.95 | 1.96 | p = 1.000 |
| **depth (L=2)** | 786 | 10 | **5.53** | 1.50 | **p = 0.345** |
| re-uploading | 786 | 5 | 8.19 | 3.30 | p = 0.513 |
| re-uploading + scale | 796 | 5 | 5.52 | 2.49 | p = 0.768 |

Added depth improves the median by 16.5%, from 6.62 to 5.53, and every depth
seed but one falls below the baseline median. The distributions still overlap
too much to call: Mann-Whitney p = 0.345, and the tighter spread seen at five
seeds does not survive either (Levene p = 0.534). A bootstrap puts the 95%
interval on the median difference at [-3.18, +0.85] percentage points, with 88%
of resamples favouring depth — suggestive, not decisive.

The observed effect size is Cohen d = 0.51, which needs **61 seeds per arm** for
80% power, about 67 core-hours for the pair. That is six times what the whole
improvement phase has cost so far, and it is the honest price of converting
this from a suggestion into a result.

Against the tuned classical baselines the two-layer model gives median 5.53
against model-2's 7.22 (p = 0.273, still not significant) and model-1's 11.60
(p = 0.0002), at 786 parameters against 2751 and 7851.

### Reading

Three of the four mechanisms tried change nothing or hurt. The one that helps
is a capacity increase rather than an architectural insight, and it does not
reach significance. The reasonable conclusion is that QCPINN as published is
already close to what this architecture achieves on Helmholtz, and that the
contribution of this work is the replication and the correction to how the
comparison is reported, not an improved model.

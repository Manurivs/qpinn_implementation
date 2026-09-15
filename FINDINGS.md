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

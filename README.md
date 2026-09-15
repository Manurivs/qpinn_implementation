# QCPINN replication and extension

Replication of *QCPINN: Quantum-Classical Physics-Informed Neural Networks for
Solving PDEs* ([arXiv:2503.16678](https://arxiv.org/abs/2503.16678); MLST,
2025), followed by architectural improvements and a study of the hybrid model
under realistic quantum noise.

See [FINDINGS.md](FINDINGS.md) for configuration details recovered from the
reference implementation and for two defects found in it.

## Layout

```
src/quantum/    circuit descriptions, compiler, simulator, PennyLane reference
src/models/     QCPINN and classical PINN baselines
tests/          backend equivalence suite
scripts/        benchmarks
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install pennylane numpy scipy matplotlib pandas pytest pyyaml
```

## Validation

The PyTorch backend is checked against a PennyLane transcription of the
reference circuits for all four ansaetze and both embeddings, covering forward
values, parameter gradients, input gradients, and the second input derivative
that PINN residuals depend on:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
PYTHONPATH=. .venv/bin/python scripts/benchmark_backends.py
```

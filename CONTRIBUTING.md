# Contributing to GeoCSV-AI

Thank you for your interest in contributing to **GeoCSV-AI**. This repository implements defense-grade, formally verified geometric interpretability harnesses and industrial Computer Systems Validation (CSV) protocols for artificial intelligence models.

To maintain regulatory, mathematical, and cryptographic integrity, all contributions must strictly adhere to the standards outlined below.

---

## 1. Guiding Principles

1. **Mathematical Invariance**: Every parameter transformation and latent steering operator must preserve exact geometric invariants (e.g., isometry on the Stiefel manifold $\mathcal{V}_k(\mathbb{R}^d)$, Lie algebra anti-symmetry $\mathbf{\Omega}^T = -\mathbf{\Omega}$, unimodularity $\det(\mathbf{R}) = 1$).
2. **Computational Tractability**: Algorithms operating on hidden dimensions $d$ must respect $\mathcal{O}(dr^2)$ complexity via low-rank decompositions and Woodbury identities. Naive $\mathcal{O}(d^3)$ full-matrix inversions are rejected.
3. **Traceability and Qualification**: Changes must map to a formal requirement in the Requirements Traceability Matrix (`docs/validation/RTM-001_Requirements_Traceability_Matrix.md`) and pass automated Installation Qualification (IQ) and Operational Qualification (OQ).
4. **No AI Slop / Technical Precision**: PR descriptions and code comments must be technically exact, concise, and ASD-STE 100 compliant. Avoid hyperbolic buzzwords and gratuitous emoji decorators.

---

## 2. Development Environment Setup

### Prerequisites
- Python $\ge$ 3.11
- PyTorch $\ge$ 2.4 (with CUDA support)
- Tinygrad (for AST kernel fusion)
- Elan / Lean 4 toolchain (pinned in `formal/lean-toolchain`)

### Installation from Source
```bash
git clone https://github.com/justinarndtpa/geocsv-ai.git
cd geocsv-ai

# Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"

# Lean 4 formal verification environment
elan toolchain install $(cat formal/lean-toolchain)
cd formal && lake exe cache get && lake build && cd ..
```

---

## 3. Code Standards & Static Analysis

All submitted code must pass automated linting, formatting, and strict type verification before review:

```bash
# 1. Formatting and linting checks
ruff check src/ tests/
ruff format --check src/ tests/

# 2. Static type safety
mypy src/geocsv

# 3. Unit and property test execution
pytest tests/ -v --cov=src/geocsv

# 4. Standalone Installation Qualification (IQ) verification
python iq_stiefel.py
```

### Acceptance Thresholds
- **Frobenius Isometry Drift**: $\|\mathbf{R}^T \mathbf{R} - \mathbf{I}\|_F < 1.0 \times 10^{-6}$ for float32 precision.
- **Lyapunov Spatial Energy Drift**: $\frac{|E_{\text{after}} - E_{\text{before}}|}{E_{\text{before}}} < 1.0 \times 10^{-6}$ for 2D Kronecker operators.
- **Deterministic Hashing**: SHA-256 state dict hashes must be bit-level identical across runs with fixed seed.

---

## 4. Formal Proof Requirements (Lean 4)

If your contribution modifies or introduces a new mathematical operator:
1. Provide the formal algebraic specification in `formal/GeoCSV/`.
2. Update `formal/lakefile.lean` if new package dependencies are introduced.
3. Ensure `lake build` completes with zero unhandled `sorry` axioms in production paths.

---

## 5. Pull Request Submission Protocol

1. Fork the repository and create a feature branch (`feature/sfr-<id>-<description>` or `fix/<issue-id>-<description>`).
2. Populate all sections of the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
3. Confirm that all automated GitHub Actions checks (`ci.yml`, `iq_qualification.yml`, `lean_verify.yml`) pass.
4. Ensure all commits are signed (`git commit -S`) to comply with cryptographic chain-of-custody requirements.

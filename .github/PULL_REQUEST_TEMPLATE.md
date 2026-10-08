## Description of Change

<!-- Provide a concise technical summary of the proposed modification. State clearly what mathematical operator, metrological protocol, or hardware kernel is added or refactored. -->

---

## Traceability & Specification Mapping

- **Requirements Traceability ID (RTM)**: e.g., `SFR-STIEFEL-001`, `SFR-DETERMIN-002`, `URS-001`
- **Specification Document**: e.g., `SPEC-CORE-CAYLEY`, `SPEC-METR-IQ`
- **Issue Reference**: Fixes #

---

## Mathematical & Invariant Verification Checklist

Please verify and check all applicable technical invariants before requesting review:

- [ ] **Exact Isometry Preservation**: Retraction operator satisfies $\|\mathbf{R}^T \mathbf{R} - \mathbf{I}\|_F < 1.0 \times 10^{-6}$ across tested float32 dimensions.
- [ ] **Unimodularity / Energy Conservation**: Determinant $\det(\mathbf{R}) = 1.0 \pm 10^{-6}$ (zero spatial rasterization drift / Lyapunov stability confirmed).
- [ ] **Asymptotic Complexity**: Implementation strictly respects $\mathcal{O}(dr^2)$ operations via Sherman-Morrison-Woodbury inversion (no $\mathcal{O}(d^3)$ full-rank inverses).
- [ ] **Lie Algebra Anti-Symmetry**: Generator $\mathbf{\Omega}$ is skew-symmetric ($\mathbf{\Omega}^T = -\mathbf{\Omega}$) and trace-free ($\mathrm{Tr}(\mathbf{\Omega}) = 0$).

---

## Metrological & Qualification Impact (FDA 21 CFR Part 11 / GAMP 5)

- [ ] **Installation Qualification (IQ)**: Deterministic weight hashing and kernel signatures remain valid (`python iq_stiefel.py` executes with status `QUALIFIED`).
- [ ] **Cryptographic Audit Trail**: Audit events output valid HMAC-SHA-256 signatures with immutable UTC timestamps.
- [ ] **Operational Qualification (OQ)**: Boundary condition tests and perturbation margins are documented.

---

## Formal Proof Status (Lean 4)

- [ ] Lean 4 theorem statements modified or added in `formal/GeoCSV/`.
- [ ] `lake build` executes without unproven `sorry` axioms in modified modules, or newly introduced hypotheses are documented.

---

## Test Suite & Code Hygiene

- [ ] All unit and property tests pass (`pytest tests/ -v`).
- [ ] Ruff linting passes with zero errors (`ruff check src/ tests/`).
- [ ] MyPy static type checks pass with zero errors (`mypy src/geocsv`).
- [ ] ASD-STE 100 docstrings and type annotations added to all public functions and classes.

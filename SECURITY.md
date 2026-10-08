# Security & Cryptographic Integrity Policy

## 1. Scope and Core Model

`GeoCSV-AI` enforces mathematical invariants and cryptographic audit trails for AI systems. In addition to standard software vulnerabilities, our security posture protects against:
- **Numerical Drift & Null-Space Evasion**: Attacks or perturbations designed to exploit floating-point inaccuracies to escape the Stiefel manifold.
- **Audit Ledger Tampering**: Breaches attempting to alter or falsify HMAC-SHA-256 signatures generated during Installation Qualification (IQ) or Operational Qualification (OQ).
- **Weight & Kernel Supply Chain Invalidation**: Unauthorized modification of neural checkpoint tensors or CUDA/Tinygrad AST JIT kernels.

---

## 2. Reporting Vulnerabilities

If you discover a security vulnerability, cryptographic flaw, or severe mathematical violation that permits stealth parameter drift:

- **Do NOT open a public GitHub issue.**
- Submit your encrypted finding via email to: `justinarndtai@gmail.com`
- Please include:
  - Exact nature of the finding (e.g., cryptographic preimage vulnerability, float32 cancellation edge case, null-space leak).
  - Minimal reproducible proof-of-concept script.
  - Expected vs. actual behavior.

We acknowledge receipt of reports within 48 hours and coordinate remediation releases prior to public disclosure.

---

## 3. Cryptographic Verification Standard

All release tags, model audit certificates, and qualification dossiers are cryptographically verified:
- **Audit Trails**: Signed via HMAC-SHA-256 using private validation keys per FDA 21 CFR Part 11 requirements.
- **Weight Checksums**: Verified using canonical SHA-256 digests over sorted tensor state dictionaries.
- **Git Commits**: Release commits are signed with GPG keys published to open keyservers.

# GeoCSV-AI: Formal Mathematical Verification (Lean 4)

This directory contains the formal proofs and verification scaffolding for the Symplectic Stiefel Manifold constraints and the exact Cayley Retraction used in GeoCSV-AI.

## Current Status (Pre-Grant)

The Lean 4 theorems are currently **scaffolded**. Specifically, the primary theorem `cayley_preserves_orthogonality` in `GeoCSV/Cayley.lean` uses the `sorry` keyword.

In formal verification, `sorry` tells the Lean compiler to skip the proof and assume it is true. This is technically a stub. 

**This is intentional.** Completing these rigorous mathematical proofs requires significant time and computation, and is explicitly scoped as **Milestone 4** of the BlueDot Career Transition Grant. 

We include the scaffolding now to prove that the formal verification pipeline compiles, runs in CI/CD, and is syntactically sound.

## How to Build

To verify the syntax and existing lemma scaffolding:

```bash
cd formal
lake build
```
*(Note: Because of the intentional `sorry` in `Cayley.lean`, the build will output a warning. This is expected behavior until Milestone 4 is completed).*

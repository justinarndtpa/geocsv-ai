import Mathlib.LinearAlgebra.Matrix.Orthogonal
import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# GeoCSV-AI: Basic Definitions for Symplectic Manifold Verification
Defines Special Orthogonal group SO(d), Skew-Symmetric Lie algebra so(d),
and Stiefel manifold V_k(R^d).
-/

open Matrix

variable {d : Type*} [DecidableEq d] [Fintype d]

/-- A matrix is skew-symmetric if Ωᵀ = -Ω -/
def IsSkewSymmetric (Ω : Matrix d d ℝ) : Prop :=
  Ωᵀ = -Ω

/-- Special Orthogonal Group SO(d) representation condition -/
def IsSpecialOrthogonal (R : Matrix d d ℝ) : Prop :=
  Rᵀ * R = 1 ∧ det R = 1

/-- Stiefel Manifold V_k(R^d) condition for tall matrices W ∈ R^(d x k) -/
variable {k : Type*} [DecidableEq k] [Fintype k]

def IsOnStiefelManifold (W : Matrix d k ℝ) : Prop :=
  Wᵀ * W = 1

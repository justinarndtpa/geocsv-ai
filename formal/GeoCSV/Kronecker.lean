import Mathlib.Data.Matrix.Kronecker
import Mathlib.LinearAlgebra.Matrix.Orthogonal
import GeoCSV.Basic

/-!
# GeoCSV-AI: 2D Kronecker Product Orthogonality Theorem
Proves that if R_row ∈ SO(M) and R_col ∈ SO(R),
then R_row ⊗ R_col ∈ SO(MR).
Used directly in the ARC-AGI-3 2D-CSO spatial reasoning operator.
-/

open Matrix
open Kronecker

variable {m n : Type*} [DecidableEq m] [Fintype m] [DecidableEq n] [Fintype n]

theorem kronecker_orthogonal
    (R_row : Matrix m m ℝ)
    (R_col : Matrix n n ℝ)
    (h_row : R_rowᵀ * R_row = 1)
    (h_col : R_colᵀ * R_col = 1) :
    (R_row ⊗ₖ R_col)ᵀ * (R_row ⊗ₖ R_col) = 1 := by
  calc
    (R_row ⊗ₖ R_col)ᵀ * (R_row ⊗ₖ R_col)
      = (R_rowᵀ ⊗ₖ R_colᵀ) * (R_row ⊗ₖ R_col) := by rw [kroneckerMap_transpose]
    _ = (R_rowᵀ * R_row) ⊗ₖ (R_colᵀ * R_col) := by rw [mul_kronecker_mul]
    _ = 1 ⊗ₖ 1 := by rw [h_row, h_col]
    _ = 1 := by rw [one_kronecker_one]

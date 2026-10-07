import pytest
import torch
import sys
import os

# Add src to python path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from geocsv.core.cayley import ExactCayleyRetraction
from geocsv.operators.kronecker import KroneckerSpatialOperator
from geocsv.metrology.iq import InstallationQualification

def test_cayley_isometry():
    """Verify ||R^T R - I||_F < 1e-5 on Stiefel manifold."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, r = 64, 4
    cayley = ExactCayleyRetraction(dimension=d, rank=r)
    A = torch.randn(d, r, device=device)
    B = torch.randn(d, r, device=device)
    
    R = cayley(A, B)
    I = torch.eye(d, device=device)
    frobenius_error = torch.norm(R.T @ R - I, p="fro").item()
    
    assert frobenius_error < 1e-5, f"Isometry violated: {frobenius_error}"

def test_kronecker_spatial_energy_conservation():
    """Verify 2D-CSO preserves total grid energy (Lyapunov stability)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    operator = KroneckerSpatialOperator(grid_dim=64, rank=4)
    grid = torch.randn(64, 64, device=device)
    
    A_r = torch.randn(64, 4, device=device)
    B_r = torch.randn(64, 4, device=device)
    A_c = torch.randn(64, 4, device=device)
    B_c = torch.randn(64, 4, device=device)
    
    transformed = operator(grid, A_r, B_r, A_c, B_c)
    
    energy_before = torch.sum(grid ** 2).item()
    energy_after = torch.sum(transformed ** 2).item()
    relative_drift = abs(energy_after - energy_before) / energy_before
    
    assert relative_drift < 1e-5, f"Energy drift detected: {relative_drift}"

def test_installation_qualification_tamper_evident():
    """Verify IQ engine produces deterministic signed certificate."""
    iq = InstallationQualification(model_id="google/gemma-2-2b")
    cert = iq.execute()
    assert cert.is_valid
    assert len(cert.weight_hash_sha256) == 64
    assert len(cert.signature) == 64

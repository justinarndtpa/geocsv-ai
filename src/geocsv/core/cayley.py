import torch
import torch.nn as nn


class ExactCayleyRetraction(nn.Module):
    """
    Implements the exact Cayley retraction on the Stiefel manifold V_k(R^d)
    using the Sherman-Morrison-Woodbury inversion identity in O(d * r^2).
    Omega = A B^T - B A^T in so(d).
    R = (I + (eta/2) Omega)^(-1) (I - (eta/2) Omega) in SO(d).
    """
    def __init__(self, dimension: int, rank: int, eta: float = 0.01):
        super().__init__()
        self.d = dimension
        self.r = rank
        self.eta = eta

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Takes low-rank generator factors A, B in R^(d x r) and returns
        the special orthogonal rotation matrix R in SO(d).
        """
        assert A.shape == (self.d, self.r), f"Expected A shape ({self.d}, {self.r}), got {A.shape}"
        assert B.shape == (self.d, self.r), f"Expected B shape ({self.d}, {self.r}), got {B.shape}"

        U = torch.cat([A, B], dim=1)
        V = torch.cat([B, -A], dim=1)

        I_2r = torch.eye(2 * self.r, device=A.device, dtype=A.dtype)
        C = I_2r + (self.eta / 2.0) * (V.T @ U)
        C_inv = torch.linalg.inv(C)

        I_d = torch.eye(self.d, device=A.device, dtype=A.dtype)
        Omega = A @ B.T - B @ A.T
        Right_term = I_d - (self.eta / 2.0) * Omega

        inv_factor = (self.eta / 2.0) * (U @ (C_inv @ (V.T @ Right_term)))
        R = Right_term - inv_factor
        return R

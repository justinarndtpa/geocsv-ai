import torch
import torch.nn as nn

from geocsv.core.cayley import ExactCayleyRetraction


class KroneckerSpatialOperator(nn.Module):
    """
    2D Kronecker-Cayley Spatial Operator (2D-CSO) for ARC-AGI-3 64x64 grid reasoning.
    Transforms 2D spatial features via R_grid = R_row (x) R_col in SO(4096)
    without constructing the 4096 x 4096 tensor in memory.
    """
    def __init__(self, grid_dim: int = 64, rank: int = 4, eta: float = 0.05):
        super().__init__()
        self.grid_dim = grid_dim
        self.cayley_row = ExactCayleyRetraction(grid_dim, rank, eta=eta)
        self.cayley_col = ExactCayleyRetraction(grid_dim, rank, eta=eta)

    def forward(self, grid: torch.Tensor, A_row: torch.Tensor, B_row: torch.Tensor,
                A_col: torch.Tensor, B_col: torch.Tensor) -> torch.Tensor:
        """
        Applies isotropic grid rotation: Y = R_row @ grid @ R_col^T.
        Complexity: O(d^3 + d * r^2) vs O(N^2) = O(d^4) flat attention.
        """
        R_row = self.cayley_row(A_row, B_row)
        R_col = self.cayley_col(A_col, B_col)
        return R_row @ grid @ R_col.T

# models/transolver.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsAttention(nn.Module):
    """
    Physics-Attention (Transolver)
    Input:  x  (B, N, D)  where N = s*s grid points
    Steps:
      1) slicing:  A = softmax(W_s x / tau) over slices  -> (B, N, S)
      2) tokens:   T = (A^T x) / sum(A)                 -> (B, S, D)
      3) attn:     standard MHA among tokens            -> (B, S, D)
      4) deslice:  y = A T'                             -> (B, N, D)
    """
    def __init__(
        self,
        dim: int,
        num_slices: int = 64,
        num_heads: int = 8,
        tau: float = 0.5,          # smaller -> lower-entropy assignments (sharper)
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_slices = num_slices
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.tau = float(tau)

        # slice assignment logits
        self.slice_proj = nn.Linear(dim, num_slices)

        # token attention projections
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        return: (B, N, D)
        """
        B, N, D = x.shape
        S = self.num_slices

        # (1) slicing weights A: (B, N, S)
        logits = self.slice_proj(x) / max(self.tau, 1e-6)
        A = F.softmax(logits, dim=-1)

        # (2) encode slice tokens T: (B, S, D)
        denom = A.sum(dim=1).clamp_min(1e-6)                 # (B, S)
        T = torch.einsum("bns,bnd->bsd", A, x) / denom.unsqueeze(-1)

        # (3) attention among tokens (MHA on S tokens)
        qkv = self.qkv(T).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                      # (3, B, H, S, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # each (B, H, S, Hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale         # (B, H, S, S)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                        # (B, H, S, Hd)
        out = out.transpose(1, 2).contiguous().reshape(B, S, D)  # (B, S, D)
        out = self.proj(out)
        out = self.proj_drop(out)                             # token_out T': (B, S, D)

        # (4) deslicing back to points: y = A T'  -> (B, N, D)
        y = torch.einsum("bns,bsd->bnd", A, out)
        return y


class TransolverBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_slices: int = 64,
        num_heads: int = 8,
        tau: float = 0.5,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(
            dim=dim,
            num_slices=num_slices,
            num_heads=num_heads,
            tau=tau,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transolver2DGrid(nn.Module):
    """
    For Darcy on regular grid:
      input:  a (B,s,s,1) or (B,1,s,s)
      output: u (B,s,s)
    We feed per-point feature = [x, y, a] into Transolver blocks.
    """
    def __init__(
        self,
        s: int,
        in_channels: int = 1,     # a
        out_channels: int = 1,    # u
        dim: int = 128,
        depth: int = 8,
        num_slices: int = 64,
        num_heads: int = 8,
        tau: float = 0.5,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.s = int(s)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # coords buffer (N,2) in [0,1]
        xs = torch.linspace(0.0, 1.0, self.s)
        ys = torch.linspace(0.0, 1.0, self.s)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N,2)
        self.register_buffer("coords", coords, persistent=False)

        self.embed = nn.Linear(2 + self.in_channels, dim)
        self.blocks = nn.ModuleList([
            TransolverBlock(
                dim=dim,
                num_slices=num_slices,
                num_heads=num_heads,
                tau=tau,
                mlp_ratio=4.0,
                drop=drop,
                attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (B,s,s,1) or (B,1,s,s)
        if a.dim() != 4:
            raise ValueError(f"Expected 4D input, got {tuple(a.shape)}")

        # to NHWC (B,s,s,C)
        if a.shape[1] == 1 and a.shape[-1] == self.s:
            a = a.permute(0, 2, 3, 1)

        B = a.shape[0]
        s = self.s
        N = s * s

        a_flat = a.reshape(B, N, self.in_channels)                  # (B,N,C)
        coords = self.coords.unsqueeze(0).expand(B, -1, -1)         # (B,N,2)

        x = torch.cat([coords, a_flat], dim=-1)                     # (B,N,2+C)
        x = self.embed(x)                                           # (B,N,D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        y = self.head(x).reshape(B, s, s, self.out_channels)        # (B,s,s,out)

        if self.out_channels == 1:
            return y[..., 0]                                        # (B,s,s)
        return y


import math
import torch
import torch.nn as nn

def _xavier_like_normal_(W: torch.Tensor, gen: torch.Generator | None = None):
    d_in, d_out = W.shape
    std = 1.0 / math.sqrt((d_in + d_out) / 2.0)
    with torch.no_grad():
        if gen is None:
            W.normal_(0.0, std)
        else:
            W.copy_(torch.randn(W.shape, generator=gen, device=W.device) * std)

class ModifiedMLP(nn.Module):

    def __init__(self, layers, activation=nn.Tanh):
        super().__init__()
        assert len(layers) >= 2
        self.act = activation()

        in_dim = int(layers[0])
        hidden0 = int(layers[1])

        self.U1 = nn.Parameter(torch.empty(in_dim, hidden0))
        self.b1 = nn.Parameter(torch.zeros(hidden0))
        self.U2 = nn.Parameter(torch.empty(in_dim, hidden0))
        self.b2 = nn.Parameter(torch.zeros(hidden0))

        g1 = torch.Generator(device="cpu"); g1.manual_seed(12345)
        g2 = torch.Generator(device="cpu"); g2.manual_seed(54321)

        _xavier_like_normal_(self.U1, g1)
        _xavier_like_normal_(self.U2, g2)

        linears = []
        for d_in, d_out in zip(layers[:-1], layers[1:]):
            lin = nn.Linear(int(d_in), int(d_out), bias=True)
            _xavier_like_normal_(lin.weight, None)
            nn.init.zeros_(lin.bias)
            linears.append(lin)
        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        U = self.act(x @ self.U1 + self.b1)  # (B, hidden0)
        V = self.act(x @ self.U2 + self.b2)  # (B, hidden0)

        h = x
        for lin in self.linears[:-1]:
            out = self.act(lin(h))
            h = out * U + (1.0 - out) * V
        y = self.linears[-1](h)
        return y


def make_uniform_sensor_idx(s: int, m_target: int):
    if m_target >= s * s:
        return torch.arange(s * s, dtype=torch.long)

    k = int(math.sqrt(m_target))
    k = max(k, 2)
    xs = torch.linspace(0, s - 1, steps=k).round().long()
    ys = torch.linspace(0, s - 1, steps=k).round().long()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    idx = (yy * s + xx).reshape(-1)
    return idx

class DeepONet2DGrid(nn.Module):
    """
    Input : a on grid (B,s,s,1) or (B,1,s,s)
    Output: u on grid (B,s,s)
    """
    def __init__(
        self,
        s: int,
        sensors: int = 256,     # 16x16
        latent_dim: int = 128,
        width: int = 256,
        depth: int = 3,
        activation=nn.Tanh,
    ):
        super().__init__()
        self.s = int(s)

        sensor_idx = make_uniform_sensor_idx(self.s, sensors)
        self.register_buffer("sensor_idx", sensor_idx, persistent=False)

        # coords (P,2) in [0,1]
        xs = torch.linspace(0.0, 1.0, self.s)
        ys = torch.linspace(0.0, 1.0, self.s)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        self.register_buffer("coords", coords, persistent=False)

        m = int(sensor_idx.numel())

        branch_layers = [m] + [width] * max(depth, 1) + [latent_dim]
        trunk_layers  = [2] + [width] * max(depth, 1) + [latent_dim]

        self.branch = ModifiedMLP(branch_layers, activation=activation)
        self.trunk  = ModifiedMLP(trunk_layers,  activation=activation)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, a):
        if a.dim() != 4:
            raise ValueError(f"Expected 4D input, got {tuple(a.shape)}")

        if a.shape[1] == 1 and a.shape[-1] == self.s:
            a = a.permute(0, 2, 3, 1)  # NCHW -> NHWC

        B = a.shape[0]
        s = self.s
        a_flat = a[..., 0].reshape(B, s * s)              # (B,P)
        a_sens = a_flat.index_select(1, self.sensor_idx)  # (B,m)

        b = self.branch(a_sens)       # (B,p)
        t = self.trunk(self.coords)   # (P,p)

        out = torch.einsum("bp,Pp->bP", b, t) + self.bias
        return out.view(B, s, s)

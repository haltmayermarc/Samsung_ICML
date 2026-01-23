from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Models
# =====================================================================

class GNAct(nn.Module):
    """Conv -> GroupNorm -> GELU"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        g = min(groups, out_ch)
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class ResBlock2D(nn.Module):
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.c1 = GNAct(ch, ch, 3, 1, 1, groups)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        g = min(groups, ch)
        self.gn2 = nn.GroupNorm(g, ch)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.c1(x)
        h = self.gn2(self.c2(h))
        return self.act(x + h)


class LODMimeticNet(nn.Module):
    """Mimic LOD logic: fine field -> learned pooling to 9x9 -> coarse mixing -> interior readout (7x7).

    Input:  (B, Cin, 129, 129)
    Output: (B, 49)
    """

    def __init__(self, in_ch: int = 1, base: int = 32, mix_blocks: int = 6, out_dim: int = 49):
        super().__init__()

        # 129 -> 65 -> 33 -> 17 -> 9
        self.stem = nn.Sequential(
            GNAct(in_ch, base, 3, 1, 1),
            GNAct(base, base, 3, 1, 1),
        )
        self.down1 = nn.Sequential(GNAct(base, base * 2, 3, 2, 1), ResBlock2D(base * 2))
        self.down2 = nn.Sequential(GNAct(base * 2, base * 4, 3, 2, 1), ResBlock2D(base * 4))
        self.down3 = nn.Sequential(GNAct(base * 4, base * 6, 3, 2, 1), ResBlock2D(base * 6))
        self.down4 = nn.Sequential(GNAct(base * 6, base * 8, 3, 2, 1), ResBlock2D(base * 8))

        ch = base * 8
        self.mix = nn.Sequential(*[ResBlock2D(ch) for _ in range(mix_blocks)])

        self.head = nn.Conv2d(ch, 1, kernel_size=1)
        self.out_dim = out_dim

    def forward(self, x):
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)  # (B, C, 9, 9)
        h = self.mix(h)
        u9 = self.head(h).squeeze(1)  # (B, 9, 9)
        u7 = u9[:, 1:8, 1:8].contiguous()  # interior
        return u7.view(u7.size(0), -1)

# ==================================================

class SpectralConv2d(nn.Module):
    """Minimal spectral conv used in FNOCoarseNet."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, input, weights):
        # (B, in, x, y), (in, out, x, y) -> (B, out, x, y)
        # input is complex; weights stored as real/imag
        w = torch.view_as_complex(weights)
        return torch.einsum("bixy,ioxy->boxy", input, w)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2])

        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spectral(x) + self.w(x))


class FNOCoarseNet(nn.Module):
    """Downsample to 9x9, then apply a small FNO stack on coarse grid."""

    def __init__(self, in_ch: int = 1, width: int = 64, depth: int = 4, modes: int = 6, out_dim: int = 49):
        super().__init__()
        # Downsample 129->9 via conv strides
        self.down = nn.Sequential(
            GNAct(in_ch, 32, 3, 1, 1),
            GNAct(32, 32, 3, 2, 1),
            GNAct(32, 48, 3, 2, 1),
            GNAct(48, 64, 3, 2, 1),
            GNAct(64, width, 3, 2, 1),
        )
        self.fno = nn.Sequential(*[FNOBlock(width, modes, modes) for _ in range(depth)])
        self.head = nn.Conv2d(width, 1, 1)
        self.out_dim = out_dim

    def forward(self, x):
        h = self.down(x)  # (B,width,9,9)
        h = self.fno(h)
        u9 = self.head(h).squeeze(1)
        u7 = u9[:, 1:8, 1:8].contiguous()
        return u7.view(u7.size(0), -1)


# =====================================================================
# Utilities: preprocessing + (de)normalization as modules/buffers
# =====================================================================


def _infer_hw(a: torch.Tensor) -> Tuple[int, int]:
    """Accept (H,W), (1,H,W), (B,H,W), (B,1,H,W) and return (H,W)."""
    if a.dim() == 2:
        return int(a.shape[0]), int(a.shape[1])
    if a.dim() == 3:
        # (1,H,W) or (B,H,W)
        return int(a.shape[-2]), int(a.shape[-1])
    if a.dim() == 4:
        return int(a.shape[-2]), int(a.shape[-1])
    raise ValueError(f"Unsupported a shape: {tuple(a.shape)}")


def _ensure_b1hw(a: torch.Tensor) -> torch.Tensor:
    """Return a as (B,1,H,W) float tensor."""
    if a.dim() == 2:
        a = a.unsqueeze(0).unsqueeze(0)
    elif a.dim() == 3:
        # (B,H,W) or (1,H,W)
        a = a.unsqueeze(1)
    elif a.dim() == 4:
        # (B,1,H,W) or (B,C,H,W)
        if a.shape[1] != 1:
            raise ValueError(f"Expected channel=1 for raw coeff field, got shape {tuple(a.shape)}")
    else:
        raise ValueError(f"Unsupported a shape: {tuple(a.shape)}")
    return a.float()


def _make_coords(H: int, W: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return xx, yy


class ChannelwiseNorm2D(nn.Module):
    """Channel-wise normalization for (B,C,H,W) tensors.

    mean/std are registered as buffers so they live in state_dict.
    """

    def __init__(self, C: int, eps: float = 1e-6, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        super().__init__()
        self.eps = float(eps)
        if mean is None:
            mean = torch.zeros(C, 1, 1)
        if std is None:
            std = torch.ones(C, 1, 1)
        self.register_buffer('mean', mean.float())
        self.register_buffer('std', std.float())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean


class VectorNorm(nn.Module):
    """Per-component normalization for (B,D) vectors."""

    def __init__(self, D: int, eps: float = 1e-6, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        super().__init__()
        self.eps = float(eps)
        if mean is None:
            mean = torch.zeros(D)
        if std is None:
            std = torch.ones(D)
        self.register_buffer('mean', mean.float())
        self.register_buffer('std', std.float())

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / (self.std + self.eps)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self.std + self.eps) + self.mean


class DarcyInputPreprocess(nn.Module):
    """Make CNN input channels from a raw Darcy coefficient field.

    Produces a tensor (B,Cin,H,W), where:
      - base channel is either a or log(a)
      - optionally append |∇ base| (forward differences)
      - optionally append coords (x,y)
    """

    def __init__(
        self,
        coeff_preproc: str = 'log',
        add_grad: bool = True,
        add_coords: bool = True,
        eps: float = 1e-12,
        default_hw: Tuple[int, int] = (129, 129),
    ):
        super().__init__()
        assert coeff_preproc in ('raw', 'log')
        self.coeff_preproc = coeff_preproc
        self.add_grad = bool(add_grad)
        self.add_coords = bool(add_coords)
        self.eps = float(eps)

        # Cache coords for the default shape for speed; will regenerate if shape differs.
        H0, W0 = int(default_hw[0]), int(default_hw[1])
        xx0, yy0 = _make_coords(H0, W0, device=torch.device('cpu'), dtype=torch.float32)
        self.register_buffer('_xx_default', xx0.unsqueeze(0).unsqueeze(0))  # (1,1,H,W)
        self.register_buffer('_yy_default', yy0.unsqueeze(0).unsqueeze(0))
        self._default_hw = (H0, W0)

    @property
    def Cin(self) -> int:
        return 1 + (1 if self.add_grad else 0) + (2 if self.add_coords else 0)

    def forward(self, a_raw: torch.Tensor) -> torch.Tensor:
        a = _ensure_b1hw(a_raw)  # (B,1,H,W)
        B, _, H, W = a.shape
        dtype = a.dtype
        device = a.device

        # base channel
        if self.coeff_preproc == 'log':
            base = torch.log(a.clamp_min(self.eps))
        else:
            base = a

        chans = [base]

        if self.add_grad:
            # forward differences with zero-pad at boundary (same as train_CNN_v2)
            gx = F.pad(base[:, :, :, 1:] - base[:, :, :, :-1], (0, 1, 0, 0))
            gy = F.pad(base[:, :, 1:, :] - base[:, :, :-1, :], (0, 0, 0, 1))
            g = torch.sqrt(gx * gx + gy * gy + 1e-12)
            chans.append(g)

        if self.add_coords:
            if (H, W) == self._default_hw and self._xx_default.device == device and self._xx_default.dtype == dtype:
                xx = self._xx_default
                yy = self._yy_default
            else:
                xx2, yy2 = _make_coords(H, W, device=device, dtype=dtype)
                xx = xx2.unsqueeze(0).unsqueeze(0)
                yy = yy2.unsqueeze(0).unsqueeze(0)
            xx = xx.expand(B, -1, -1, -1)
            yy = yy.expand(B, -1, -1, -1)
            chans.append(xx)
            chans.append(yy)

        return torch.cat(chans, dim=1)  # (B,Cin,H,W)


# =====================================================================
# Wrapped models: raw a -> (preproc+norm) -> core net -> decoded coeffs
# =====================================================================


class DarcyCoeffModel(nn.Module):
    """A self-contained Darcy coefficient predictor.

    - Input: raw coefficient field a(x)
      shape: (H,W) or (B,H,W) or (B,1,H,W)
    - Output (default): **physical** 49-dim coefficient vector (decoded)

    Training-friendly methods:
      - forward_norm(a): returns normalized coefficients (B,49)
      - encode_u(u_phys), decode_u(u_norm)
    """

    def __init__(
        self,
        core: nn.Module,
        coeff_preproc: str = 'log',
        add_grad: bool = True,
        add_coords: bool = True,
        x_mean: Optional[torch.Tensor] = None,
        x_std: Optional[torch.Tensor] = None,
        u_mean: Optional[torch.Tensor] = None,
        u_std: Optional[torch.Tensor] = None,
        default_hw: Tuple[int, int] = (129, 129),
    ):
        super().__init__()
        self.pre = DarcyInputPreprocess(
            coeff_preproc=coeff_preproc,
            add_grad=add_grad,
            add_coords=add_coords,
            default_hw=default_hw,
        )

        Cin = self.pre.Cin
        # Normalize preprocessed input
        self.x_norm = ChannelwiseNorm2D(C=Cin, mean=x_mean, std=x_std)
        # Normalize output coeffs
        self.u_norm = VectorNorm(D=49, mean=u_mean, std=u_std)
        self.core = core

    @property
    def Cin(self) -> int:
        return self.pre.Cin

    # ---- output helpers ----
    def encode_u(self, u_phys: torch.Tensor) -> torch.Tensor:
        return self.u_norm.encode(u_phys)

    def decode_u(self, u_normed: torch.Tensor) -> torch.Tensor:
        return self.u_norm.decode(u_normed)

    # ---- forward variants ----
    def forward_norm(self, a_raw: torch.Tensor) -> torch.Tensor:
        x = self.pre(a_raw)
        x = self.x_norm.encode(x)
        return self.core(x)

    def forward(self, a_raw: torch.Tensor) -> torch.Tensor:
        # Default forward returns physical coefficients (decoded)
        u_hat_norm = self.forward_norm(a_raw)
        return self.decode_u(u_hat_norm)


def build_darcy_coeff_model(
    model_name: str,
    coeff_preproc: str,
    add_grad: bool,
    add_coords: bool,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    u_mean: torch.Tensor,
    u_std: torch.Tensor,
    default_hw: Tuple[int, int] = (129, 129),
    **core_kwargs,
) -> DarcyCoeffModel:
    """Factory that builds a self-contained model.

    model_name: one of {'UNet','LODMimetic','CoarseResNet','FNOCoarse'}
    core_kwargs: forwarded to the core net (e.g., base=32, mix_blocks=6).

    Note: the core networks expect inputs in (B,Cin,H,W). We build Cin from
    (add_grad, add_coords).
    """

    model_name = str(model_name)
    pre = DarcyInputPreprocess(coeff_preproc=coeff_preproc, add_grad=add_grad, add_coords=add_coords, default_hw=default_hw)
    Cin = pre.Cin

    if model_name == 'LODMimetic':
        core = LODMimeticNet(in_ch=Cin, **core_kwargs)
    elif model_name == 'FNOCoarse':
        core = FNOCoarseNet(in_ch=Cin, **core_kwargs)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return DarcyCoeffModel(
        core=core,
        coeff_preproc=coeff_preproc,
        add_grad=add_grad,
        add_coords=add_coords,
        x_mean=x_mean,
        x_std=x_std,
        u_mean=u_mean,
        u_std=u_std,
        default_hw=default_hw,
    )


def load_darcy_coeff_model_from_checkpoint(
    ckpt_path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[DarcyCoeffModel, dict]:
    """Load a DarcyCoeffModel from a checkpoint.

    - Works with checkpoints saved by train_CNN_v3.py.
    - Also supports *older* checkpoints (train_CNN_v2.py) that stored
      `normalizers` as python objects, by extracting their mean/std.

    Returns:
      model: DarcyCoeffModel (ready for inference)
      args:  training args dict stored in the checkpoint
    """

    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt.get('args', {})

    model_name = args.get('model', 'UNet')
    coeff_preproc = args.get('coeff_preproc', 'log')
    add_grad = bool(args.get('add_grad', True))
    add_coords = bool(args.get('add_coords', True))

    # Determine Cin to create correctly-shaped buffers.
    Cin = 1 + (1 if add_grad else 0) + (2 if add_coords else 0)
    x_mean = torch.zeros(Cin, 1, 1)
    x_std = torch.ones(Cin, 1, 1)
    u_mean = torch.zeros(49)
    u_std = torch.ones(49)

    # Backward-compatible extraction (train_CNN_v2 style)
    if 'normalizers' in ckpt:
        norms = ckpt['normalizers']
        try:
            x_mean = torch.as_tensor(norms['coeffs'].mean).float().view(Cin, 1, 1)
            x_std = torch.as_tensor(norms['coeffs'].std).float().view(Cin, 1, 1)
            u_mean = torch.as_tensor(norms['u'].mean).float().view(49)
            u_std = torch.as_tensor(norms['u'].std).float().view(49)
        except Exception:
            # If anything goes wrong, fall back to zeros/ones and rely on state_dict.
            pass

    model = build_darcy_coeff_model(
        model_name=model_name,
        coeff_preproc=coeff_preproc,
        add_grad=add_grad,
        add_coords=add_coords,
        x_mean=x_mean,
        x_std=x_std,
        u_mean=u_mean,
        u_std=u_std,
    )

    model.load_state_dict(ckpt['model_state_dict'], strict=strict)

    if device is not None:
        model = model.to(device)
    model.eval()

    return model, args

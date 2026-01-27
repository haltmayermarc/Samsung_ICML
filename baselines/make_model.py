from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MatrixNorm(nn.Module):
    """Per-component normalization for (B,D) vectors."""

    def __init__(self, H: int, W: int, eps: float = 1e-6, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        super().__init__()
        self.eps = float(eps)
        if mean is None:
            mean = torch.zeros((H, W))
        if std is None:
            std = torch.ones((H, W))
        self.register_buffer('mean', mean.float())
        self.register_buffer('std', std.float())

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / (self.std + self.eps)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self.std + self.eps) + self.mean

class Identity(nn.Module):
    """Per-component normalization for (B,D) vectors."""

    def __init__(self):
        super().__init__()

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return y


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
        u_normalization: bool = True,
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
        if u_normalization:
            self.u_norm = MatrixNorm(H=default_hw[0], W=default_hw[1], mean=u_mean, std=u_std)
        else:
            self.u_norm = Identity()

        print('u_normalization:', u_normalization)
        self.core = core

    @property
    def Cin(self) -> int:
        return self.pre.Cin

    # ---- output helpers ----
    def apply_constraint(self, u: torch.Tensor) -> torch.Tensor:
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[0, :] = 0.0
        u[-1, :] = 0.0
        return u
    
    def encode_u(self, u_phys: torch.Tensor) -> torch.Tensor:
        return self.apply_constraint(self.u_norm.encode(u_phys))

    def decode_u(self, u_normed: torch.Tensor) -> torch.Tensor:
        return self.apply_constraint(self.u_norm.decode(u_normed))

    # ---- forward variants ----
    def forward_norm(self, a_raw: torch.Tensor) -> torch.Tensor:
        x = self.pre(a_raw)
        x = self.x_norm.encode(x)
        return self.apply_constraint(self.core(x)[..., 0, :, :]) 

    def forward(self, a_raw: torch.Tensor) -> torch.Tensor:
        # Default forward returns physical coefficients (decoded)
        u_hat_norm = self.forward_norm(a_raw)
        return self.decode_u(u_hat_norm)


def build_darcy_model(
    model_name: str,
    coeff_preproc: str,
    u_normalization: bool,
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

    print("Cin:", Cin)
    
    if model_name == "fno":
        from models.fno import FNO
        core = FNO(
            n_modes=(16, 16),
            hidden_channels=64,
            in_channels=Cin,
            out_channels=1
        )
    elif model_name == "cno":
        from models.cno import CNO2d
        core = CNO2d(
            in_dim=Cin,
            out_dim=1,
            size=default_hw[0],
            N_layers=4,
            N_res=2,
            N_res_neck=4,
            channel_multiplier=16,
            use_bn=True,
        )
    elif model_name == "uno":
        from models.uno import UNO
        core = UNO(
            n_layers=5,
            uno_out_channels=[32, 64, 64, 64, 32],
            uno_n_modes=[[16,16], [12,12], [12,12], [12,12], [16,16]],
            uno_scalings=[[1,1], [0.5,0.5], [1,1], [2,2], [1,1]],
            hidden_channels=64,
            in_channels=1,
            out_channels=1,
            channel_mlp_skip='linear'
        )
    elif model_name == 'mg_tfno':
        from models.mg_tfno import MGTFNO
        num_levels = 2
        tfno_kwargs = dict(
            n_modes=(16, 16),
            hidden_channels=64,
            factorization='tucker',
            implementation='factorized',
            rank=0.05
        )
        core = MGTFNO(
            in_channels=1,
            out_channels=1,
            levels=num_levels,
            kwargs=tfno_kwargs
        )
    elif model_name == 'deeponet':
        from models.deeponet import DeepONet2DGrid
        core = DeepONet2DGrid(
            s=default_hw[0],
            sensors=256,
            latent_dim=128,
            width=256,
            depth=3,
        )
    elif model_name == "transolver":
        from models.transolver import Transolver2DGrid
        core = Transolver2DGrid(
            s=default_hw[0],
            dim=128,
            depth=8,
            num_slices=64,
            num_heads=8,
            tau=0.5,
            drop=0.0,
            attn_drop=0.0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return DarcyCoeffModel(
        core=core,
        coeff_preproc=coeff_preproc,
        u_normalization=u_normalization,
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
    u_mean = torch.zeros(129, 129)
    u_std = torch.ones(129, 129)

    # Backward-compatible extraction (train_CNN_v2 style)
    if 'normalizers' in ckpt:
        norms = ckpt['normalizers']
        try:
            x_mean = torch.as_tensor(norms['coeffs'].mean).float().view(Cin, 1, 1)
            x_std = torch.as_tensor(norms['coeffs'].std).float().view(Cin, 1, 1)
            u_mean = torch.as_tensor(norms['u'].mean).float().view(129, 129)
            u_std = torch.as_tensor(norms['u'].std).float().view(129, 129)
        except Exception:
            # If anything goes wrong, fall back to zeros/ones and rely on state_dict.
            pass

    model = build_darcy_model(
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
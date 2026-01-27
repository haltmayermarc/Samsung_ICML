import os
import time
import datetime
import argparse
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
from make_model import *

parser = argparse.ArgumentParser("Train Darcy")
parser.add_argument("--type", type=str, choices=['quantile', 'lognormal', 'lognormal2', 'fine_checkerboard'])
parser.add_argument("--basis_order", type=str, choices=['1', '2'], default=1)
parser.add_argument("--num_training_data", type=int, default=500)

parser.add_argument("--model", type=str, choices=["fno", "uno", "cno", 'mg_tfno', 'deeponet', 'transolver'], default="fno")
parser.add_argument("--optimizer", type=str, default="AdamW")
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--validate_batch_size", type=int, default=16)
parser.add_argument("--loss", type=str, choices=["mse", "rel_l2", "PI"], default="mse")
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--gpu", type=int, default=0)

parser.add_argument("--coeff_preproc", type=str, choices=['raw', 'log'], default='log')
parser.add_argument("--u_normalization", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--add_grad", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--add_coords", action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()
gparams = args.__dict__


type = gparams['type']
gpu = gparams['gpu']
loss_type = gparams['loss']
basis_order = gparams['basis_order']
num_training_data = gparams['num_training_data']

base = f"data/P{basis_order}_ne0.0625_Darcy_{num_training_data}"

if gparams["type"] is not None:
    npz_path = f"{base}_{type}.npz"
else:
    npz_path = f"{base}.npz"

epochs = int(gparams['epochs'])
train_batch_size = gparams['train_batch_size']
validate_batch_size = gparams['validate_batch_size']

device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")



def compute_stats_with_preprocessor(fixed_npz, chunk_size: int = 64):
    """Compute mean/std for the *preprocessed* input channels, and u mean/std.

    We compute stats using the exact same preprocessing module used in the model
    (DarcyInputPreprocess) to avoid any mismatch.
    """

    a_all = fixed_npz["train_coeffs_a"].reshape(-1, 129, 129).astype(np.float32)
    u_all = fixed_npz["train_u_h_fine"].reshape(-1, 129, 129).astype(np.float32)

    print('add_grad:', gparams['add_grad'])
    print('add_coords:', gparams['add_coords'])
    pre = DarcyInputPreprocess(
        coeff_preproc=gparams['coeff_preproc'],
        add_grad=gparams['add_grad'],
        add_coords=gparams['add_coords'],
        default_hw=(a_all.shape[1], a_all.shape[2]),
    )
    Cin = pre.Cin

    sums = torch.zeros(Cin, dtype=torch.float64)
    sums2 = torch.zeros(Cin, dtype=torch.float64)
    count = 0

    for i in range(0, a_all.shape[0], chunk_size):
        a = torch.from_numpy(a_all[i:i + chunk_size])  # (B,H,W)
        x = pre(a)  # (B,C,H,W) on CPU
        # accumulate per-channel sums over pixels and batch
        sums += x.double().sum(dim=(0, 2, 3))
        sums2 += (x.double() ** 2).sum(dim=(0, 2, 3))
        count += int(x.shape[0] * x.shape[2] * x.shape[3])

    mean = (sums / count).float()  # (C,)
    var = (sums2 / count - (sums / count) ** 2).float()
    std = torch.sqrt(torch.clamp(var, min=1e-12)).float()

    u_mean = torch.from_numpy(u_all.mean(axis=0))
    u_std = torch.from_numpy(u_all.std(axis=0))

    return mean.view(Cin, 1, 1), std.view(Cin, 1, 1), u_mean, u_std


class DarcyDatasetRaw(Dataset):
    """
    Return raw a and raw u.
    Normalization is handled inside the model (for input) and in the loss (for u).
    """

    def __init__(self, npz_data, split: str):
        assert split in ["train", "validate"]
        self.coeffs = npz_data[f"{split}_coeffs_a"].reshape(-1, 129, 129).astype(np.float32)
        # self.u = npz_data[f"{split}_u"].astype(np.float32)
        self.matrices = npz_data[f"{split}_matrices"].astype(np.float32)
        self.loads = npz_data[f"{split}_load_vectors"].astype(np.float32)
        # if split == "validate":
        #     self.Q = npz_data[f"{split}_Q"].astype(np.float32)
        # else:
        #     self.Q = np.zeros((500,100,100))
        self.u_fine = npz_data[f"{split}_u_h_fine"].reshape(-1, 129, 129).astype(np.float32)

    def __len__(self):
        return self.coeffs.shape[0]

    def __getitem__(self, idx):
        coeffs = torch.from_numpy(self.coeffs[idx]).float()  # (H,W)
        # u = torch.from_numpy(self.u[idx]).float()            # (49,)
        matrix = torch.from_numpy(self.matrices[idx]).float()
        load = torch.from_numpy(self.loads[idx]).float()
        u_fine = torch.from_numpy(self.u_fine[idx]).float()
        # Q_h = torch.from_numpy(self.Q[idx]).float()
        return {"coeffs": coeffs, "matrix": matrix, "load": load, "u_fine": u_fine}


fixed = np.load(npz_path, allow_pickle=True)

INTERIOR_IDX = torch.tensor(
    [j * 17 + i for j in range(1, 16) for i in range(1, 16)],
    dtype=torch.long
).to(device)

x_mean, x_std, u_mean, u_std = compute_stats_with_preprocessor(fixed)

train_dataset = DarcyDatasetRaw(fixed, "train")
val_dataset = DarcyDatasetRaw(fixed, "validate")
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=validate_batch_size, shuffle=False)


model = build_darcy_model(
    model_name=gparams["model"],
    coeff_preproc=gparams['coeff_preproc'],
    u_normalization=gparams['u_normalization'],
    add_grad=gparams['add_grad'],
    add_coords=gparams['add_coords'],
    x_mean=x_mean,
    x_std=x_std,
    u_mean=u_mean,
    u_std=u_std,
    default_hw=(fixed["train_coeffs_a"].shape[1], fixed["train_coeffs_a"].shape[2]),
)
model = model.to(device)



def init_optim_lbfgs(model):
    params = {
        'history_size': 10,
        'max_iter': 20,
        'tolerance_grad': 1e-15,
        'tolerance_change': 1e-15,
        'max_eval': 10,
    }
    return torch.optim.LBFGS(model.parameters(), **params)

def init_optim_adam(model, lr=1e-3, weight_decay=0):
    params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    }
    return torch.optim.Adam(model.parameters(), **params)

def init_optim_sgd(model, lr=1e-2, momentum=0.9, weight_decay=0):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

def init_optim_adamw(model, lr=1e-3, weight_decay=1e-2):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def init_optim_adagrad(model, lr=1e-2, weight_decay=0):
    params = {
        'lr': lr,
        'weight_decay': weight_decay,
        'lr_decay': 0,
        'eps': 1e-10,
    }
    return torch.optim.Adagrad(model.parameters(), **params)



if gparams['optimizer'] == "LBFGS":
    optimizer = init_optim_lbfgs(model)
elif gparams['optimizer'] == "Adam":
    optimizer = init_optim_adam(model)
elif gparams['optimizer'] == "SGD":
    optimizer = init_optim_sgd(model)
elif gparams['optimizer'] == "AdamW":
    optimizer = init_optim_adamw(model)
elif gparams['optimizer'] == "Adagrad":
    optimizer = init_optim_adagrad(model)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def rel_L2_error(u_pred, u_true):
    return torch.norm(u_pred - u_true) / torch.norm(u_true)

def darcy_pde_loss_fvm(u: torch.Tensor, a: torch.Tensor, f_value: float = 1.0) -> torch.Tensor:
    B, s, _ = u.shape
    scale = float((s - 1) ** 2)

    u_c  = u[:, 1:-1, 1:-1]
    u_xm = u[:, 0:-2, 1:-1]
    u_xp = u[:, 2:  , 1:-1]
    u_ym = u[:, 1:-1, 0:-2]
    u_yp = u[:, 1:-1, 2:  ]

    a_c  = a[:, 1:-1, 1:-1]
    a_xm = a[:, 0:-2, 1:-1]
    a_xp = a[:, 2:  , 1:-1]
    a_ym = a[:, 1:-1, 0:-2]
    a_yp = a[:, 1:-1, 2:  ]

    a_nx = 0.5 * (a_c + a_xm)
    a_px = 0.5 * (a_c + a_xp)
    a_ny = 0.5 * (a_c + a_ym)
    a_py = 0.5 * (a_c + a_yp)

    diag = a_nx + a_px + a_ny + a_py
    rhs  = torch.full_like(u_c, float(f_value))

    res = diag*u_c - a_nx*u_xm - a_px*u_xp - a_ny*u_ym - a_py*u_yp - rhs/scale
    return torch.mean(res ** 2)

def compute_loss(model, u_pred_norm, batch, loss_type="mse", eps=1e-12):
    u_true_norm = model.encode_u(batch["u_fine"])

    if loss_type == "mse":
        return torch.mean((u_pred_norm - u_true_norm) ** 2)
    
    if loss_type == "l1":
        return torch.mean(torch.abs(u_pred_norm - u_true_norm))

    elif loss_type == "rel_l2":
        diff = u_pred_norm - u_true_norm
        num = torch.norm(diff, dim=(1, 2))
        denom = torch.norm(u_true_norm, dim=(1, 2)) + eps
        return torch.mean(num / denom)
    
    elif loss_type == "PI":
        data_loss = torch.mean((u_pred_norm - u_true_norm) ** 2)
        u_pred_phys = model.decode_u(u_pred_norm)
        pde_loss = darcy_pde_loss_fvm(u_pred_phys, batch['coeffs'], f_value=1.0)
        return data_loss + pde_loss
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def closure(model, batch, loss_type="mse"):
    u_pred_norm = model.forward_norm(batch["coeffs"])
    loss = compute_loss(model, u_pred_norm, batch, loss_type)
    return loss, u_pred_norm

def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


path = os.path.join(os.getcwd(), 'model', str(type), gparams["model"])
if not os.path.exists(path):
    os.makedirs(path)

log_dir = os.path.join(os.getcwd(), "log", str(type), gparams["model"])
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")

nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Write some basic information into log file (kept)
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write(f"Number of parameters: {nparams}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Trained with loss type {loss_type}!")
        f.write("=" * 60 + "\n")
        f.write(f"Model Architecture:\n{str(model)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Optimizer:\n{str(optimizer)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"gparams:\n{gparams}\n")
        f.write("=" * 60 + "\n")




loss_history = []
test_history = []
train_history = []

start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0

    # =======================
    # Training
    # =======================
    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        if gparams['optimizer'] == "LBFGS":
            # LBFGS requires closure
            def lbfgs_closure():
                optimizer.zero_grad()
                u_pred_norm = model.forward_norm(batch["coeffs"])
                loss = compute_loss(model, u_pred_norm, batch, loss_type)
                loss.backward()
                return loss

            loss = optimizer.step(lbfgs_closure)
            epoch_loss += loss.item()

        else:
            optimizer.zero_grad()
            loss, _ = closure(model, batch, loss_type)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    scheduler.step()

    # =======================
    # Validation + Train Eval
    # =======================
    if epoch % 10 == 0:
        model.eval()
        loss_history.append(epoch_loss)

        # ---------- Validation ----------
        rel_err_total = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)

                u_pred = model(batch["coeffs"])
                u_true = batch["u_fine"]

                diff = torch.norm(u_pred - u_true, dim=(1, 2))
                denom = torch.norm(u_true, dim=(1, 2))

                rel_err_total += torch.sum(diff / denom).item()
                count += diff.shape[0]

        test_relL2 = rel_err_total / count
        test_history.append(test_relL2)

        # ---------- Training error ----------
        rel_err_total = 0.0
        count = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)

                u_pred = model(batch["coeffs"])
                u_true = batch["u_fine"]     

                diff = torch.norm(u_pred - u_true, dim=(1, 2))
                denom = torch.norm(u_true, dim=(1, 2))

                rel_err_total += torch.sum(diff / denom).item()
                count += diff.shape[0]

        train_relL2 = rel_err_total / count
        train_history.append(train_relL2)

        # ---------- Logging ----------
        log_str = (
            f"[Epoch {epoch:04d}] "
            f"Loss={epoch_loss:.6f}   "
            f"Test_relL2={test_relL2:.6f}   "
            f"Train_relL2={train_relL2:.6f}"
        )

        print(log_str)
        with open(log_file, "a") as f:
            f.write(log_str + "\n")


# =======================
# Save model
# =======================
checkpoint = {
    "model_state_dict": model.state_dict(),
    "args": gparams,
}



save_path = os.path.join(path, f"{gparams['model']}_{loss_type}_addgrad{gparams['add_grad']}_addcoords{gparams['add_coords']}_{timestamp}.pt")
torch.save(checkpoint, save_path)

print(f"Model saved to {save_path}")
print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")


a_all = fixed["validate_coeffs_a"].astype(np.float32).reshape(-1, 129, 129)
u_all = fixed["validate_u_h_fine"].astype(np.float32).reshape(-1, 129, 129)

model.eval()

indices = [0, 10, 20, 30]
n = len(indices)

ncols = 4
fig = plt.figure(figsize=(4.2*ncols, 3.2*n))

for r, idx in enumerate(indices):
    a = torch.from_numpy(a_all[idx]).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,s,s)
    u = torch.from_numpy(u_all[idx]).to(device).unsqueeze(0)               # (1,s,s)
    pred = model(a)
    
    a_img = a.squeeze(0).squeeze(0).detach().cpu().numpy()
    u_img = u.squeeze(0).detach().cpu().numpy()
    p_img = pred.squeeze(0).detach().cpu().numpy()
    err = (pred - u).abs().squeeze(0).detach().cpu().numpy()

    vmin = min(u_img.min(), p_img.min())
    vmax = max(u_img.max(), p_img.max())

    ax = fig.add_subplot(n, ncols, r*ncols + 1)
    im = ax.imshow(a_img)
    ax.set_title(f"a (idx={idx})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(n, ncols, r*ncols + 2)
    im = ax.imshow(u_img, vmin=vmin, vmax=vmax)
    ax.set_title("u_true")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(n, ncols, r*ncols + 3)
    im = ax.imshow(p_img, vmin=vmin, vmax=vmax)
    ax.set_title("pred")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(n, ncols, r*ncols + 4)
    im = ax.imshow(err)
    ax.set_title("|pred-u|")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(f"Validation samples", y=1.02)
plt.tight_layout()

fig_file = os.path.join(log_dir, f"test_{timestamp}.png")
fig.savefig(fig_file, dpi=150, bbox_inches="tight")
plt.close(fig)





# # =========================
# # Utils
# # =========================

# def set_seed(seed: int):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def rel_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     # pred,true: (B,s,s)
#     b = pred.shape[0]
#     diff = (pred - true).reshape(b, -1)
#     denom = true.reshape(b, -1)
#     num = torch.norm(diff, p=2, dim=1)
#     den = torch.norm(denom, p=2, dim=1).clamp_min(eps)
#     return (num / den).mean()

# def make_mollifier(s: int, device: torch.device) -> torch.Tensor:
#     # xs = torch.linspace(0.0, 1.0, s, device=device)
#     # ys = torch.linspace(0.0, 1.0, s, device=device)
#     # yy, xx = torch.meshgrid(ys, xs, indexing="ij")
#     # return torch.sin(np.pi * xx) * torch.sin(np.pi * yy)  # (s,s)

#     mask = torch.ones((s, s), device=device)
#     mask[0, :]  = 0
#     mask[-1, :] = 0
#     mask[:, 0]  = 0
#     mask[:, -1] = 0
#     return mask

# def cycle(loader):
#     while True:
#         for batch in loader:
#             yield batch


# # =========================
# # Dataset
# # =========================

# def compute_stats_with_preprocessor(fixed_npz, gparams, chunk_size: int = 64):
#     """Compute mean/std for the *preprocessed* input channels, and u mean/std.

#     We compute stats using the exact same preprocessing module used in the model
#     (DarcyInputPreprocess) to avoid any mismatch.
#     """

#     a_all = fixed_npz["train_coeffs_a"].astype(np.float32) 
#     u_all = fixed_npz["train_u"].astype(np.float32) 

#     pre = DarcyInputPreprocess(
#         coeff_preproc=gparams['coeff_preproc'],
#         add_grad=gparams['add_grad'],
#         add_coords=gparams['add_coords'],
#         default_hw=(a_all.shape[1], a_all.shape[2]),
#     )
#     Cin = pre.Cin

#     sums = torch.zeros(Cin, dtype=torch.float64)
#     sums2 = torch.zeros(Cin, dtype=torch.float64)
#     count = 0

#     for i in range(0, a_all.shape[0], chunk_size):
#         a = torch.from_numpy(a_all[i:i + chunk_size])  # (B,H,W)
#         x = pre(a)  # (B,C,H,W) on CPU
#         # accumulate per-channel sums over pixels and batch
#         sums += x.double().sum(dim=(0, 2, 3))
#         sums2 += (x.double() ** 2).sum(dim=(0, 2, 3))
#         count += int(x.shape[0] * x.shape[2] * x.shape[3])

#     mean = (sums / count).float()  # (C,)
#     var = (sums2 / count - (sums / count) ** 2).float()
#     std = torch.sqrt(torch.clamp(var, min=1e-12)).float()

#     u_mean = torch.from_numpy(u_all.mean(axis=0))
#     u_std = torch.from_numpy(u_all.std(axis=0))

#     return mean.view(Cin, 1, 1), std.view(Cin, 1, 1), u_mean, u_std


# class DarcyDatasetRaw(Dataset):
#     """Return raw a and raw u.

#     Normalization is handled inside the model (for input) and in the loss (for u).
#     """

#     def __init__(self, npz_data, split: str):
#         assert split in ["train", "validate"]
#         self.coeffs = npz_data[f"{split}_coeffs_a"].astype(np.float32)
#         self.u = npz_data[f"{split}_u"].astype(np.float32)
#         self.matrices = npz_data[f"{split}_matrices"].astype(np.float32)
#         self.loads = npz_data[f"{split}_load_vectors"].astype(np.float32)
#         # if split == "validate":
#         #     self.Q = npz_data[f"{split}_Q"].astype(np.float32)
#         # else:
#         #     self.Q = np.zeros((500,100,100))
#         self.u_fine = npz_data[f"{split}_u_h_fine"].astype(np.float32)

#     def __len__(self):
#         return self.u.shape[0]

#     def __getitem__(self, idx):
#         coeffs = torch.from_numpy(self.coeffs[idx]).float()  # (H,W)
#         u = torch.from_numpy(self.u[idx]).float()            # (49,)
#         matrix = torch.from_numpy(self.matrices[idx]).float()
#         load = torch.from_numpy(self.loads[idx]).float()
#         u_fine = torch.from_numpy(self.u_fine[idx]).float()
#         # Q_h = torch.from_numpy(self.Q[idx]).float()
#         return {"coeffs": coeffs, "u": u, "matrix": matrix, "load": load, "u_fine": u_fine}
    




# class DarcyFVMNPZ(Dataset):
#     """
#     npz keys (from your generator):
#       train_coeffs_a: (N,s,s)
#       train_u:        (N,s,s)
#       validate_coeffs_a: (N,s,s)
#       validate_u:        (N,s,s)
#     """
#     def __init__(self, npz_path: str, split: str, s: int):
#         arr = np.load(npz_path, allow_pickle=True)
#         assert split in ["train", "validate"]

#         if split == "train":
#             a = arr["train_coeffs_a"].astype(np.float32).reshape(-1, s, s)
#             u = arr["train_u_h_fine"].astype(np.float32).reshape(-1, s, s)
#         else:
#             a = arr["validate_coeffs_a"].astype(np.float32).reshape(-1, s, s)
#             u = arr["validate_u_h_fine"].astype(np.float32).reshape(-1, s, s)

#         self.a = a
#         self.u = u
#         self.s = s

#     def __len__(self):
#         return self.a.shape[0]

#     def __getitem__(self, idx):
#         a = torch.from_numpy(self.a[idx]).unsqueeze(-1)  # (s,s,1)
#         u = torch.from_numpy(self.u[idx])                # (s,s)
#         return a, u


# def compute_a_stats_from_npz(npz_path: str, mode: str = "logstd"):
#     arr = np.load(npz_path, allow_pickle=True)
#     a = arr["train_coeffs_a"].astype(np.float32)  # (N,s,s)

#     if mode == "logstd":
#         a = np.log(np.maximum(a, 1e-6))

#     mean = float(a.mean())
#     std  = float(a.std() + 1e-6)
#     return mean, std

# class APreprocess:
#     def __init__(self, mode: str, mean: float, std: float, device: torch.device):
#         self.mode = mode
#         self.mean = torch.tensor(mean, device=device)
#         self.std  = torch.tensor(std, device=device)

#     def __call__(self, a_raw: torch.Tensor) -> torch.Tensor:
#         x = a_raw[..., 0] 

#         if self.mode == "logstd":
#             x = torch.log(x.clamp_min(1e-6))
#         elif self.mode == "std":
#             pass
#         else:
#             return a_raw

#         x = (x - self.mean) / self.std
#         return x.unsqueeze(-1)


# # =========================
# # PDE loss (FVM discretization)
# # -div(a grad u) = 1, Dirichlet 0 BC is enforced by mollifier
# # =========================

# def darcy_pde_loss_fvm(u: torch.Tensor, a: torch.Tensor, f_value: float = 1.0) -> torch.Tensor:
#     """
#     u,a: (B,s,s)
#     residual on interior uses arithmetic face average like your solver.
#     """
#     B, s, _ = u.shape
#     scale = float((s - 1) ** 2)
#     # scale = 1.0

#     u_c  = u[:, 1:-1, 1:-1]
#     u_xm = u[:, 0:-2, 1:-1]
#     u_xp = u[:, 2:  , 1:-1]
#     u_ym = u[:, 1:-1, 0:-2]
#     u_yp = u[:, 1:-1, 2:  ]

#     a_c  = a[:, 1:-1, 1:-1]
#     a_xm = a[:, 0:-2, 1:-1]
#     a_xp = a[:, 2:  , 1:-1]
#     a_ym = a[:, 1:-1, 0:-2]
#     a_yp = a[:, 1:-1, 2:  ]

#     a_nx = 0.5 * (a_c + a_xm)
#     a_px = 0.5 * (a_c + a_xp)
#     a_ny = 0.5 * (a_c + a_ym)
#     a_py = 0.5 * (a_c + a_yp)

#     diag = a_nx + a_px + a_ny + a_py
#     rhs  = torch.full_like(u_c, float(f_value))

#     res = diag*u_c - a_nx*u_xm - a_px*u_xp - a_ny*u_ym - a_py*u_yp - rhs/scale
#     return torch.mean(res ** 2)

# # =========================
# # Model import
# # =========================

# def build_model(model_name: str, s: int = 64) -> nn.Module:
#     name = model_name.lower()
#     if name == "fno":
#         from models.fno import FNO
#         return FNO(
#             n_modes=(16, 16),
#             hidden_channels=64,
#             in_channels=1,
#             out_channels=1
#         )
#     elif name == "cno":
#         from models.cno import CNO2d
#         return CNO2d(
#             in_dim=1, out_dim=1,
#             size=s,
#             N_layers=4,
#             N_res=2,
#             N_res_neck=4,
#             channel_multiplier=16,
#             use_bn=True,
#         )
#     elif name == "uno":
#         from models.uno import UNO
#         return UNO(
#             n_layers=5,
#             uno_out_channels=[32, 64, 64, 64, 32],
#             uno_n_modes=[[16,16], [12,12], [12,12], [12,12], [16,16]],
#             uno_scalings=[[1,1], [0.5,0.5], [1,1], [2,2], [1,1]],
#             hidden_channels=64,
#             in_channels=1,
#             out_channels=1,
#             channel_mlp_skip='linear'
#         )
#     elif name == 'mg_tfno':
#         from models.mg_tfno import MGTFNO
#         num_levels = 2
#         tfno_kwargs = dict(
#             n_modes=(16, 16),
#             hidden_channels=64,
#             factorization='tucker',
#             implementation='factorized',
#             rank=0.05
#         )
#         return MGTFNO(
#             in_channels=1,
#             out_channels=1,
#             levels=num_levels,
#             kwargs=tfno_kwargs
#         )
#     elif name == 'deeponet':
#         from models.deeponet import DeepONet2DGrid
#         return DeepONet2DGrid(
#             s=s,
#             sensors=256,
#             latent_dim=128,
#             width=256,
#             depth=3,
#         )

#     elif name == "transolver":
#         from models.transolver import Transolver2DGrid
#         return Transolver2DGrid(
#             s=s,
#             dim=128,
#             depth=8,
#             num_slices=64,
#             num_heads=8,
#             tau=0.5,
#             drop=0.0,
#             attn_drop=0.0,
#         )
#     else:
#         raise ValueError(f"Unknown model: {model_name}")

# def infer_io_format(model: nn.Module, s: int, device: torch.device):
#     model.eval()
#     a_nhwc = torch.zeros(2, s, s, 1, device=device)
#     a_nchw = torch.zeros(2, 1, s, s, device=device)

#     input_kind = None

#     with torch.no_grad():
#         for kind, a in [("nhwc", a_nhwc), ("nchw", a_nchw)]:
#             try:
#                 out = model(a)
#                 input_kind = kind
#                 break
#             except Exception:
#                 continue

#     if input_kind is None:
#         raise RuntimeError(
#             "Could not automatically determine model input format [must be either (B,s,s,1) or (B,1,s,s)]"
#         )

#     def forward(a_batch):
#         # a_batch: (B,s,s,1) from dataset
#         if input_kind == "nhwc":
#             out = model(a_batch)
#         else:
#             out = model(a_batch.permute(0, 3, 1, 2))  # -> (B,1,s,s)

#         # standardize output to (B,s,s)
#         if out.dim() == 4:
#             # (B,s,s,1) or (B,1,s,s)
#             if out.shape[1] == 1 and out.shape[-1] == s:
#                 out = out[:, 0, :, :]           # (B,s,s)
#             else:
#                 out = out.squeeze(-1)           # (B,s,s)
#         return out

#     model.train()
#     return forward

# # =========================
# # Visualization
# # =========================

# @torch.no_grad()
# def save_viz(
#     epoch: int,
#     s: int,
#     model_forward,
#     model,
#     npz_path: str,
#     save_path: str,
#     indices,
#     device,
#     a_pre,
#     mollifier=None,
#     use_mollifier: bool=False,
# ):
#     arr = np.load(npz_path, allow_pickle=True)
#     a_all = arr["validate_coeffs_a"].astype(np.float32).reshape(-1, s, s)
#     u_all = arr["validate_u_h_fine"].astype(np.float32).reshape(-1, s, s)

#     model.eval()
#     n = len(indices)

#     ncols = 4
#     fig = plt.figure(figsize=(4.2*ncols, 3.2*n))

#     for r, idx in enumerate(indices):
#         a = torch.from_numpy(a_all[idx]).to(device).unsqueeze(0).unsqueeze(-1)  # (1,s,s,1)
#         u = torch.from_numpy(u_all[idx]).to(device).unsqueeze(0)                # (1,s,s)

#         a_in = a_pre(a)
#         pred_raw = model_forward(a_in)  # (1,s,s)
#         if use_mollifier and mollifier is not None:
#             pred = pred_raw * mollifier
#         else:
#             pred = pred_raw

#         err = (pred - u).abs().squeeze(0).detach().cpu().numpy()
#         a_img = a.squeeze(0).squeeze(-1).detach().cpu().numpy()
#         u_img = u.squeeze(0).detach().cpu().numpy()
#         p_img = pred.squeeze(0).detach().cpu().numpy()

#         vmin = min(u_img.min(), p_img.min())
#         vmax = max(u_img.max(), p_img.max())

#         ax = fig.add_subplot(n, ncols, r*ncols + 1)
#         im = ax.imshow(a_img)
#         ax.set_title(f"a (idx={idx})")
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#         ax = fig.add_subplot(n, ncols, r*ncols + 2)
#         im = ax.imshow(u_img, vmin=vmin, vmax=vmax)
#         ax.set_title("u_true")
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#         ax = fig.add_subplot(n, ncols, r*ncols + 3)
#         im = ax.imshow(p_img, vmin=vmin, vmax=vmax)
#         ax.set_title("pred" + (" (raw*mollifier)" if use_mollifier else ""))
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#         ax = fig.add_subplot(n, ncols, r*ncols + 4)
#         im = ax.imshow(err)
#         ax.set_title("|pred-u|")
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     fig.suptitle(f"Validation samples @ epoch {epoch}", y=1.02)
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     fig.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)

#     model.train()


# # =========================
# # Train / Eval
# # =========================

# @torch.no_grad()
# def evaluate(model_forward, model, val_loader, device, a_pre, mollifier, use_mollifier: bool):
#     model.eval()
#     total = 0.0
#     cnt = 0
#     for a, u in val_loader:
#         a_raw = a.to(device)
#         a_in  = a_pre(a_raw)
#         u = u.to(device)

#         pred = model_forward(a_in)  # (B,s,s)
#         if use_mollifier:
#             pred = pred * mollifier

#         b = pred.shape[0]
#         diff = (pred-u).reshape(b,-1)
#         denom = u.reshape(b,-1)
#         err = torch.norm(diff,2,dim=1) / torch.norm(denom,2,dim=1).clamp_min(1e-12)
#         total += err.sum().item()
#         cnt += b

#     model.train()
#     return total / max(cnt, 1)


# def main():
#     parser = argparse.ArgumentParser("Train Darcy")
#     parser.add_argument("--atype", type=str, choices=['quantile', 'lognormal', 'lognormal2', 'fine_checkerboard'], default='quantile')
#     parser.add_argument('--s', type=int, default=129)
#     parser.add_argument("--model", type=str, choices=["fno", "uno", "cno", 'mg_tfno', 'deeponet', 'transolver'], default="fno")
#     parser.add_argument("--loss", type=str, choices=["mse", "rel_l2", "PI"], default="mse")
#     parser.add_argument("--epochs", type=int, default=2000)
#     parser.add_argument("--train_batch_size", type=int, default=16)
#     parser.add_argument("--validate_batch_size", type=int, default=16)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--gpu", type=int, default=0)
#     parser.parse_args()
#     args = parser.parse_args()

#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     set_seed(args.seed)

#     # data
#     data_npz = f'./data/P1_ne0.0625_Darcy_500_{args.atype}.npz'
#     train_set = DarcyFVMNPZ(data_npz, split="train", s=args.s)
#     val_set   = DarcyFVMNPZ(data_npz, split="validate", s=args.s)

#     train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
#     val_loader   = DataLoader(val_set, batch_size=args.validate_batch_size, shuffle=False, num_workers=4)

#     # normalization
#     need_a_norm = True
#     a_norm_mode = "logstd" if need_a_norm else "none"

#     if need_a_norm:
#         a_mean, a_std = compute_a_stats_from_npz(data_npz, mode=a_norm_mode)
#         a_pre = APreprocess(a_norm_mode, a_mean, a_std, device)
#     else:
#         a_pre = APreprocess("none", 0.0, 1.0, device)

#     # model
#     s = train_set.s
#     model = build_model(args.model, s=s).to(device)
#     nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     if args.model == "mg_tfno":
#         def model_forward(a_batch):
#             x = a_batch.permute(0, 3, 1, 2)
#             x_s1 = F.interpolate(x, size=(s-1, s-1), mode="bilinear", align_corners=False)
#             y_s1 = model(x_s1)  # (B, 1, s-1, s-1)
#             y = F.interpolate(y_s1, size=(s, s), mode="bilinear", align_corners=False)
#             return y[:, 0, :, :]  # -> (B, s, s)
#     else:
#         model_forward = infer_io_format(model, s, device)

#     # hard BC
#     # use_mollifier = True if args.loss == "PI" else False
#     use_mollifier = True
#     mollifier = make_mollifier(s, device) if use_mollifier else None

#     model.eval()
#     a, u = next(iter(val_loader))
#     pde_true = darcy_pde_loss_fvm(u, a[..., 0], f_value=1.0).item()
#     print(f"PDE residual check:", pde_true)

#     val0 = evaluate(model_forward, model, val_loader, device, a_pre, mollifier, use_mollifier)
#     print("Val error before training:", val0)

#     # optimizer
#     lr = 1e-3
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     warmup_epochs = 50
#     lambda_pde = 1e-1

#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     # logging dirs
#     log_dir = os.path.join(os.getcwd(), "log", args.model)
#     os.makedirs(log_dir, exist_ok=True)
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = os.path.join(log_dir, f"training_log_s{args.s}_{args.atype}_{args.model}_{args.loss}_{timestamp}.txt")

#     fig_dir = os.path.join(log_dir, "figs", f"s{args.s}_{args.atype}_{args.model}_{args.loss}_{timestamp}")
#     os.makedirs(fig_dir, exist_ok=True)

#     viz_ids = [0, 10, 20, 30]

#     model_dir = os.path.join(os.getcwd(), "model", args.model)
#     os.makedirs(model_dir, exist_ok=True)
#     model_ckpt = os.path.join(model_dir, f"model_s{args.s}_{args.atype}_{args.model}_{args.loss}_{timestamp}.pt")

#     with open(log_file, "w", encoding="utf-8") as f:
#         f.write(f"Model: {args.model}\n")
#         f.write(f"Loss: {args.loss}\n")
#         f.write(f"Use mollifier (hard BC): {use_mollifier}\n")
#         f.write(f"Params: {nparams}\n")
#         f.write(f"Device: {device}\n")
#         f.write(f"Seed: {args.seed}\n")
#         f.write("=" * 60 + "\n")

#     print("#########################")
#     print("Start training")
#     print("#########################")
#     print(f"Train: {len(train_set)} | Val: {len(val_set)} | Grid s={s}")
#     print(f"Params: {nparams}")
#     print(f"Hard BC (mollifier) -> {use_mollifier}")
#     print(f"Log: {log_file}")

#     eval_every = 10
#     start_time = time.time()

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         epoch_loss = 0.0
#         epoch_data = 0.0
#         epoch_pde  = 0.0
#         nb = 0

#         for a, u in train_loader: 
#             a_raw = a.to(device)  # (B,s,s,1)
#             a_in  = a_pre(a_raw)
#             u = u.to(device)  # (B,s,s)

#             optimizer.zero_grad(set_to_none=True)

#             pred = model_forward(a_in)  # (B,s,s)
#             if use_mollifier:
#                 pred = pred * mollifier

#             # data loss
#             if args.loss in ["mse", "PI"]:
#                 data_loss = F.mse_loss(pred, u)
#             elif args.loss == "rel_l2":
#                 data_loss = rel_l2(pred, u)
#             else:
#                 data_loss = torch.zeros((), device=device)

#             # PDE loss
#             if args.loss in ["PI"] and epoch >= warmup_epochs:
#                 pde_loss = darcy_pde_loss_fvm(pred, a_raw[..., 0], f_value=1.0)
#             else:
#                 pde_loss = torch.zeros((), device=device)

#             if args.loss in ["mse", "rel_l2"]:
#                 loss = data_loss
#             else:  # PI
#                 loss = data_loss + lambda_pde * pde_loss

#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             epoch_data += float(data_loss.item())
#             epoch_pde  += float(pde_loss.item())
#             nb += 1

#         if epoch % eval_every == 0:
#             avg_loss = epoch_loss / max(nb, 1)
#             avg_data = epoch_data / max(nb, 1)
#             avg_pde  = epoch_pde  / max(nb, 1)

#             train_err = evaluate(model_forward, model, train_loader, device, a_pre, mollifier, use_mollifier)
#             val_err = evaluate(model_forward, model, val_loader, device, a_pre, mollifier, use_mollifier)

#             log_str = (
#                 f"[Epoch {epoch:04d}] "
#                 f"Loss={avg_loss:.12f}   "
#                 # f"Data={avg_data:.12f}   "
#                 # f"PDE={avg_pde:.12f}   "
#                 f"Train_RelL2={train_err:.6f}   "
#                 f"Test_RelL2={val_err:.6f}"
#             )
#             print(log_str)
#             with open(log_file, "a", encoding="utf-8") as f:
#                 f.write(log_str + "\n")

#             # if val_err < best_val:
#             #     best_val = val_err
#             #     torch.save({
#             #         "model": model.state_dict(),
#             #         "optimizer": optimizer.state_dict(),
#             #         "epoch": epoch,
#             #         "best_val": best_val,
#             #         "args": vars(args),
#             #     }, best_ckpt)

#             save_path = os.path.join(fig_dir, f"epoch_{epoch:05d}.png")
#             save_viz(epoch, s, model_forward, model, data_npz, save_path, viz_ids, device,
#                      a_pre=a_pre, mollifier=mollifier, use_mollifier=use_mollifier)
        
#         scheduler.step()

#     log_str = f"\nTraining finished in {time.time() - start_time:.2f} seconds."
#     with open(log_file, "a", encoding="utf-8") as f:
#         f.write(log_str + "\n")
#     print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")
#     # print(f"Best Val_RelL2 = {best_val:.6f}")
#     # print(f"Best ckpt: {best_ckpt}")

#     torch.save({
#         "model": model.state_dict(),
#         "args": vars(args),
#     }, model_ckpt)


# if __name__ == "__main__":
#     torch.backends.cudnn.benchmark = True
#     main()

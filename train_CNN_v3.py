import torch
import time
import datetime
import os
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# IMPORTANT: self-contained model with preprocessing+normalization inside
from network_v3 import *


# ARGS (kept compatible with train_CNN_v2)
parser = argparse.ArgumentParser("SEM")
## Data
parser.add_argument("--type", type=str, choices=['quantile', 'coarse_checkerboard', 'fine_checkerboard', 'horizontal', 'vertical'])
parser.add_argument("--basis_order", type=str, choices=['1', '2'], default=1)
parser.add_argument("--num_training_data", type=int, default=5000)

## Train parameters
parser.add_argument("--model", type=str, default='UNet', choices=['LODMimetic', 'FNOCoarse'])
parser.add_argument("--optimizer", type=str, default="AdamW")
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--validate_batch_size", type=int, default=32)
parser.add_argument("--loss", type=str, choices=['mse', 'rel_l2', 'weak_form'], default='mse')
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--gpu", type=int, default=0)

## Input preprocessing
parser.add_argument("--coeff_preproc", type=str, choices=['raw', 'log'], default='log')
parser.add_argument("--add_grad", action='store_true', default=True)
parser.add_argument("--no_add_grad", action='store_false', dest='add_grad')
parser.add_argument("--add_coords", action='store_true', default=True)
parser.add_argument("--no_add_coords", action='store_false', dest='add_coords')

## Optional: keep legacy 2-phase (MSE -> WEAK) behavior
parser.add_argument("--mse_to_weak", action='store_true', default=False)
parser.add_argument("--switch_epoch", type=int, default=150)

args = parser.parse_args()
gparams = args.__dict__


# Training data
type = gparams['type']
gpu = gparams['gpu']
loss_type = gparams['loss']
basis_order = gparams['basis_order']
num_training_data = gparams['num_training_data']

# Base path
base = f"data/P{basis_order}_ne0.125_Darcy_{num_training_data}"

if gparams["type"] is not None:
    npz_path = f"{base}_{type}_CNN.npz"
else:
    npz_path = f"{base}.npz"


epochs = int(gparams['epochs'])
train_batch_size = gparams['train_batch_size']
validate_batch_size = gparams['validate_batch_size']

device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


# KAIMING INITIALIZATION (kept exactly as original)
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)


def compute_stats_with_preprocessor(fixed_npz, chunk_size: int = 64):
    """Compute mean/std for the *preprocessed* input channels, and u mean/std.

    We compute stats using the exact same preprocessing module used in the model
    (DarcyInputPreprocess) to avoid any mismatch.
    """

    a_all = fixed_npz["train_coeffs_a"].astype(np.float32)  # (N,H,W)
    u_all = fixed_npz["train_u"].astype(np.float32)         # (N,49)

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
    """Return raw a and raw u.

    Normalization is handled inside the model (for input) and in the loss (for u).
    """

    def __init__(self, npz_data, split: str):
        assert split in ["train", "validate"]
        self.coeffs = npz_data[f"{split}_coeffs_a"].astype(np.float32)
        self.u = npz_data[f"{split}_u"].astype(np.float32)
        self.matrices = npz_data[f"{split}_matrices"].astype(np.float32)
        self.loads = npz_data[f"{split}_load_vectors"].astype(np.float32)

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        coeffs = torch.from_numpy(self.coeffs[idx]).float()  # (H,W)
        u = torch.from_numpy(self.u[idx]).float()            # (49,)
        matrix = torch.from_numpy(self.matrices[idx]).float()
        load = torch.from_numpy(self.loads[idx]).float()
        return {"coeffs": coeffs, "u": u, "matrix": matrix, "load": load}


fixed = np.load(npz_path, allow_pickle=True)
x_mean, x_std, u_mean, u_std = compute_stats_with_preprocessor(fixed)

train_dataset = DarcyDatasetRaw(fixed, "train")
val_dataset = DarcyDatasetRaw(fixed, "validate")
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=validate_batch_size, shuffle=False)


# Build a *self-contained* model: raw a -> preproc+norm -> core -> (norm) ; forward() -> decoded
model_FEONet = build_darcy_coeff_model(
    model_name=gparams["model"],
    coeff_preproc=gparams['coeff_preproc'],
    add_grad=gparams['add_grad'],
    add_coords=gparams['add_coords'],
    x_mean=x_mean,
    x_std=x_std,
    u_mean=u_mean,
    u_std=u_std,
    default_hw=(fixed["train_coeffs_a"].shape[1], fixed["train_coeffs_a"].shape[2]),
)
model_FEONet = model_FEONet.to(device)
model_FEONet.apply(weights_init)


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
    optimizer = init_optim_lbfgs(model_FEONet)
elif gparams['optimizer'] == "Adam":
    optimizer = init_optim_adam(model_FEONet)
elif gparams['optimizer'] == "SGD":
    optimizer = init_optim_sgd(model_FEONet)
elif gparams['optimizer'] == "AdamW":
    optimizer = init_optim_adamw(model_FEONet)
elif gparams['optimizer'] == "Adagrad":
    optimizer = init_optim_adagrad(model_FEONet)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def rel_L2_error(u_pred, u_true):
    return torch.norm(u_pred - u_true) / torch.norm(u_true)


def compute_loss(model, u_pred_norm, batch, loss_type="mse", eps=1e-8):
    # We keep the *same* training target space as v2: normalized u.
    u_true_norm = model.encode_u(batch["u"])

    if loss_type == "mse":
        return torch.mean((u_pred_norm - u_true_norm) ** 2)

    elif loss_type == "rel_l2":
        diff = u_pred_norm - u_true_norm
        num = torch.norm(diff, dim=1)
        denom = torch.norm(u_true_norm, dim=1) + eps
        return torch.mean(num / denom)
    
    elif loss_type == "weak_form":
        A = batch["matrix"]
        f = batch["load"]
        # decode u before physics
        u_pred_phys = model.decode_u(u_pred_norm)
        r = torch.bmm(A, u_pred_phys.unsqueeze(-1)).squeeze(-1) - f
        num = torch.norm(r, dim=1)
        return torch.mean(num)
    
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

nparams = sum(p.numel() for p in model_FEONet.parameters() if p.requires_grad)

# Write some basic information into log file (kept)
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write(f"Number of parameters: {nparams}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Trained with loss type {loss_type}!")
        f.write("=" * 60 + "\n")
        f.write(f"Model Architecture:\n{str(model_FEONet)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Optimizer:\n{str(optimizer)}\n")
        f.write("=" * 60 + "\n")

print("#########################")
print("Start training CNN")
print("#########################")

loss_history = []
test_history = []
train_history = []

start_time = time.time()

for epoch in range(1, epochs + 1):
    model_FEONet.train()
    epoch_loss = 0.0

    # -----------------------
    # Select optimizer & loss
    # -----------------------

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        if gparams['optimizer'] == "LBFGS":
            # LBFGS requires closure
            def lbfgs_closure():
                optimizer.zero_grad()
                u_pred_norm = model_FEONet.forward_norm(batch["coeffs"])
                loss = compute_loss(model_FEONet, u_pred_norm, batch, loss_type)
                loss.backward()
                return loss

            loss = optimizer.step(lbfgs_closure)
            epoch_loss += loss.item()

        else:
            optimizer.zero_grad()
            loss, _ = closure(model_FEONet, batch, loss_type)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    scheduler.step()
    
    # =======================
    # Validation
    # =======================
    if epoch % 10 == 0:
        model_FEONet.eval()
        rel_err_total = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)

                u_pred = model_FEONet(batch["coeffs"])   # (B,49)  <-- physical
                u_true = batch["u"]                      # (B,49)  <-- physical

                diff = torch.norm(u_pred - u_true, dim=1)
                denom = torch.norm(u_true, dim=1) + 1e-8
                rel_err = torch.mean(diff / denom)

                rel_err_total += rel_err.item()
                count += 1
        
        rel_err = rel_err_total / count
        loss_history.append(epoch_loss)
        test_history.append(rel_err)

        rel_err_total = 0.0
        count = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = move_batch_to_device(batch, device)

                u_pred = model_FEONet(batch["coeffs"])   # (B,49)  <-- physical
                u_true = batch["u"]                      # (B,49)  <-- physical

                diff = torch.norm(u_pred - u_true, dim=1)
                denom = torch.norm(u_true, dim=1) + 1e-8
                rel_err = torch.mean(diff / denom)

                rel_err_total += rel_err.item()
                count += 1

        rel_err_train = rel_err_total / count
        train_history.append(rel_err_train)

        log_str = (
            f"[Epoch {epoch:04d}] "
            f"Loss={epoch_loss:.6e}   "
            f"Test_relL2={rel_err:.6f}   "
            f"Train_relL2={rel_err_train:.6f}"
        )

        print(log_str)

        with open(log_file, "a") as f:
            f.write(log_str + "\n")


checkpoint = {
    "model_state_dict": model_FEONet.state_dict(),
    "args": gparams,
}

save_path = os.path.join(path, f"{gparams['model']}_{loss_type}.pt")
torch.save(checkpoint, save_path)

print(f"Model saved to {save_path}")
print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

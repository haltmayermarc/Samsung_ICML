import torch
import time
import datetime
import re
import os
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from network import *
import torch.nn.utils as utils


# ARGS
parser = argparse.ArgumentParser("SEM")
## Data
parser.add_argument("--type", type=str, choices=['quantile', 'coarse_checkerboard', 'fine_checkerboard', 'horizontal', 'vertical'])
parser.add_argument("--basis_order", type=str, choices=['1', '2'], default=1)
parser.add_argument("--num_training_data", type=int, default=5000)

## Train parameters
#parser.add_argument("--hidden_dims", type=int, default=32)
#parser.add_argument("--out_dims", type=int, default=16)
#parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--model", type=str, default='UNet', choices=['CNN', "UNet"])
parser.add_argument("--optimizer", type=str, default="AdamW")
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--validate_batch_size", type=int, default=32)
parser.add_argument("--loss", type=str, choices=['mse', 'rel_l2', 'weak_form'], default='mse')
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()
gparams = args.__dict__

# Architecture hyperparams
#hidden_dims = gparams["hidden_dims"]
#out_dims = gparams["out_dims"]
#num_layers = gparams["num_layers"]

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

# Load mesh data
#mesh = np.load(npz_path, allow_pickle=True)
#p = mesh["coarse_nodes"]

#Model
models = {
          'UNet': UNetLatentModel
          }

MODEL = models[gparams['model']]

#Train
epochs = int(gparams['epochs'])
train_batch_size = gparams['train_batch_size']
validate_batch_size = gparams['validate_batch_size']


if gparams["model"] == "UNet":
    model_FEONet = MODEL()
    
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
model_FEONet = model_FEONet.to(device)
    

# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

model_FEONet.apply(weights_init)


class Normalizer:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = torch.as_tensor(mean).float()
        self.std = torch.as_tensor(std).float()
        self.eps = eps

    def encode(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / (std + self.eps)

    def decode(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * (std + self.eps) + mean

    

def compute_normalizers(fixed_npz):
    # Coefficients (global scalar stats)
    coeffs_mean = fixed_npz["train_coeffs_a"].mean()
    coeffs_std  = fixed_npz["train_coeffs_a"].std()

    # u (per-component stats)
    u_mean = fixed_npz["train_u"].mean(axis=0)
    u_std  = fixed_npz["train_u"].std(axis=0)

    return {
        "coeffs": Normalizer(coeffs_mean, coeffs_std),
        "u": Normalizer(u_mean, u_std),
    }
    

class DarcyDataset(Dataset):
    def __init__(self, npz_data, split, normalizers):
        """
        split: 'train' or 'validate'
        normalizers: dict returned by compute_normalizers
        """
        assert split in ["train", "validate"]

        self.coeffs = npz_data[f"{split}_coeffs_a"]
        self.u = npz_data[f"{split}_u"]
        self.matrices = npz_data[f"{split}_matrices"]
        self.loads = npz_data[f"{split}_load_vectors"]

        self.norm = normalizers

    def __len__(self):
        return self.u.shape[0]

    def __getitem__(self, idx):
        coeffs = torch.from_numpy(self.coeffs[idx]).float().unsqueeze(0)
        u = torch.from_numpy(self.u[idx]).float()
        matrix = torch.from_numpy(self.matrices[idx]).float()
        load = torch.from_numpy(self.loads[idx]).float()

        coeffs = self.norm["coeffs"].encode(coeffs)
        u = self.norm["u"].encode(u)

        return {
            "coeffs": coeffs,    # normalized
            "u": u,              # normalized
            "matrix": matrix,    # RAW
            "load": load,        # RAW
        }


fixed = np.load(npz_path, allow_pickle=True)

normalizers = compute_normalizers(fixed)

# move all normalizers to GPU
#for k in normalizers:
#    normalizers[k].to(device)


train_dataset = DarcyDataset(fixed, "train", normalizers)
val_dataset   = DarcyDataset(fixed, "validate", normalizers)

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=validate_batch_size, shuffle=False)


def init_optim_lbfgs(model):
    params = {'history_size': 10,
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
        'lr_decay': 0,     # learning rate decay (optional, default 0)
        'eps': 1e-10       # term for numerical stability
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
    
adam_optimizer = init_optim_adam(model_FEONet, lr=1e-3)
lbfgs_optimizer = init_optim_lbfgs(model_FEONet)

switch_epoch = 150

    


def rel_L2_error(u_pred, u_true):
    return torch.norm(u_pred - u_true) / torch.norm(u_true)


def compute_loss(u_pred, batch, loss_type="mse", eps=1e-8):
    u_true = batch["u"]  # (B, 49)

    if loss_type == "mse":
        return torch.mean((u_pred - u_true) ** 2)

    elif loss_type == "rel_l2":
        diff = u_pred - u_true
        num = torch.norm(diff, dim=1)
        denom = torch.norm(u_true, dim=1) + eps
        return torch.mean(num / denom)

    elif loss_type == "weak_form":
        A = batch["matrix"]        # RAW, on GPU
        f = batch["load"]          # RAW, on GPU

        # decode u before physics
        u_pred_phys = normalizers["u"].decode(u_pred)

        r = torch.bmm(A, u_pred_phys.unsqueeze(-1)).squeeze(-1) - f

        num = torch.norm(r, dim=1)
        return torch.mean(num)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def closure(model, batch, loss_type="mse"):
    u_pred = model(batch["coeffs"])
    loss = compute_loss(u_pred, batch, loss_type)
    return loss, u_pred

def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}



path = os.path.join(os.getcwd(), 'model', type, gparams["model"])
if not os.path.exists(path):
    os.makedirs(path)

log_dir = os.path.join(os.getcwd(), "log", type, gparams["model"])
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")

nparams = sum(p.numel() for p in model_FEONet.parameters() if p.requires_grad)

# Write some basic information into log file
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

start_time = time.time()

for epoch in range(1, epochs + 1):
    model_FEONet.train()
    epoch_loss = 0.0

    # -----------------------
    # Select optimizer & loss
    # -----------------------
    if epoch <= switch_epoch:
        optimizer = adam_optimizer
        current_loss = "mse"
    else:
        optimizer = lbfgs_optimizer
        current_loss = "weak_form"

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        if optimizer is lbfgs_optimizer:
            # LBFGS requires closure
            def lbfgs_closure():
                optimizer.zero_grad()
                u_pred = model_FEONet(batch["coeffs"])
                loss = compute_loss(u_pred, batch, current_loss)
                loss.backward()
                return loss

            loss = optimizer.step(lbfgs_closure)
            epoch_loss += loss.item()

        else:
            optimizer.zero_grad()
            loss, u_pred = closure(model_FEONet, batch, current_loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

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

                u_pred = model_FEONet(batch["coeffs"])
                u_true = batch["u"]

                diff = torch.norm(u_pred - u_true, dim=1)
                denom = torch.norm(u_true, dim=1) + 1e-8
                rel_err = torch.mean(diff / denom)

                rel_err_total += rel_err.item()
                count += 1

        rel_err = rel_err_total / count
        loss_history.append(epoch_loss)
        test_history.append(rel_err)

        log_str = (
            f"[Epoch {epoch:04d}] "
            f"Phase={'MSE' if epoch <= switch_epoch else 'WEAK'}   "
            f"Loss={epoch_loss:.6e}   "
            f"Test_relL2={rel_err:.6f}"
        )

        print(log_str)

        with open(log_file, "a") as f:
            f.write(log_str + "\n")
             
checkpoint = {
    "model_state_dict": model_FEONet.state_dict(),
    "normalizers": normalizers,
    "args": gparams,
}

save_path = os.path.join(path, f"{gparams['model']}_{loss_type}.pt")
torch.save(checkpoint, save_path)

print(f"Model saved to {save_path}")


print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

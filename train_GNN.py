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
from dolfin import *
from mshr import *
import torch.nn.utils as utils
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops


# ARGS
parser = argparse.ArgumentParser("SEM")
## Data
parser.add_argument("--type", type=str, choices=['pwc', 'quantile', 'checkerboard', 'horizontal', 'vertical'])
parser.add_argument("--basis_order", type=str, choices=['1', '2'])
parser.add_argument("--num_elems", type=int, default=256)
parser.add_argument("--num_training_data", type=int, default=5000)

## Train parameters
parser.add_argument("--hidden_dims", type=int, default=16)
parser.add_argument("--out_dims", type=int, default=8)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--cpu", type=int, default=0)
parser.add_argument("--model", type=str, default='SimpleDarcyGNN', choices=['SimpleDarcyGNN'])
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--validate_batch_size", type=int, default=32)
parser.add_argument("--loss", type=str, choices=['mse', 'rel_l2', 'weak_form', 'normalized_weak_form'], default='mse')
parser.add_argument("--epochs", type=int, default=50000)
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()
gparams = args.__dict__

hidden_dims = gparams["hidden_dims"]
out_dims = gparams["out_dims"]
num_layers = gparams["num_layers"]
ne_val = gparams['num_elems']
type = gparams['type']
gpu = gparams['gpu']
loss_type = gparams['loss']
basis_order = gparams['basis_order']
num_training_data = gparams['num_training_data']

# Base path
base = f"data/P{basis_order}_ne{ne_val}_Darcy_{num_training_data}"

if gparams["type"] is not None:
    npz_path = f"{base}_{gparams['type']}.npz"
else:
    npz_path = f"{base}.npz"

# Load mesh data
mesh = np.load(npz_path, allow_pickle=True)
num_element, num_pts, p = mesh['ne'], mesh['ng'], mesh['p']

#Model
models = {
          'SimpleDarcyGNN': SimpleDarcyGNN,
          }

MODEL = models[gparams['model']]

#Train
epochs = int(gparams['epochs'])
D_out = num_pts
train_batch_size = gparams['train_batch_size']
validate_batch_size = gparams['train_batch_size']


if gparams["model"] == "SimpleDarcyGNN":
    model_FEONet = MODEL(hidden_dim=hidden_dims, out_dim=out_dims, num_layers=num_layers)
else:
    model_FEONet = MODEL(6, D_out, hidden_dims = [256, 128, 64, 32, 64, 128, 256])
    
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
model_FEONet = model_FEONet.to(device)
    

# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

model_FEONet.apply(weights_init)

class DarcyGraphDataset(InMemoryDataset):
    def __init__(self, root, npz_path, kind="train",
                 transform=None, pre_transform=None, pre_filter=None):

        self.npz_path = npz_path
        self.kind = kind  # "train", "validate"

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.npz_path)]

    @property
    def processed_file_names(self):
        return [f"{self.kind}.pt"]

    def download(self):
        pass

    def process(self):
        arr = np.load(self.npz_path, allow_pickle=True)

        # Shared graph structure
        edge_index = torch.tensor(arr["edges"], dtype=torch.long)
        pos = torch.tensor(arr["p"], dtype=torch.float)

        # Add self-loops (if missing)
        edge_index, _ = add_self_loops(edge_index, num_nodes=pos.shape[0])

        # Extract split
        A_all = arr[f"{self.kind}_matrices"]
        a_all = arr[f"{self.kind}_coeffs_a"]
        u_all = arr[f"{self.kind}_fenics_u"]
        f_all = arr[f"{self.kind}_load_vectors"]

        N = a_all.shape[0]
        data_list = []

        i = edge_index[0]
        j = edge_index[1]

        for k in range(N):
            A_matrix = torch.tensor(A_all[k], dtype=torch.float)
            f = torch.tensor(f_all[k], dtype=torch.float)

            x = torch.tensor(a_all[k], dtype=torch.float).unsqueeze(-1)
            y = torch.tensor(u_all[k], dtype=torch.float).unsqueeze(-1)
            #A_ij = A_matrix[i, j].unsqueeze(-1)

            data = Data(
                x=x,
                edge_index=edge_index,
                #edge_attr=A_ij,
                y=y,
                pos=pos,
                A=A_matrix,      
                f=f       
            )
            data_list.append(data)

        # Save
        self.save(data_list, self.processed_paths[0])

train_dataset = DarcyGraphDataset(
    root="data/",
    npz_path="data/P2_ne200_Darcy_5000_zero.npz",
    kind="train"
)

val_dataset = DarcyGraphDataset(
    root="data/",
    npz_path="data/P2_ne200_Darcy_5000_zero.npz",
    kind="validate"
)

if gparams["optimizer"] == "LBFGS":
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=len(val_dataset), shuffle=False)
else:
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=validate_batch_size, shuffle=False)

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


def rel_L2_error(u_pred, u_true):
    return torch.norm(u_pred - u_true) / torch.norm(u_true)


def compute_loss(u_pred, batch, epoch, loss_type="mse"):
    ptr = batch.ptr
    batch_size = ptr.numel() - 1
    num_nodes = ptr[1] - ptr[0]  # = 441

    u_pred = u_pred.view(batch_size, num_nodes)
    u_true = batch.y.view(batch_size, num_nodes)
    
    if loss_type == "mse":
        return torch.mean((u_pred - u_true) ** 2)
    elif loss_type == "rel_l2":
        return rel_L2_error(u_pred, u_true)
    elif loss_type == "weak_form": 
       r = torch.bmm(batch.A, u_pred.unsqueeze(-1)).squeeze(-1) - batch.f
       return torch.mean(torch.sum(r, dim=1)**2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def closure(model, batch, loss_type="mse"):
    u_pred = model(batch)
    loss = compute_loss(u_pred, batch, loss_type)
    return loss, u_pred

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
print("Start training GNN")
print("#########################")

loss_history = []
test_history = []

start_time = time.time()

for epoch in range(1, epochs + 1):
    model_FEONet.train()
    epoch_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        
        if gparams["optimizer"] == "LBFGS":
            def lbfgs_closure():
                optimizer.zero_grad()
                u_pred = model_FEONet(batch)
                loss = compute_loss(u_pred, batch, epoch, loss_type)
                loss.backward()
                return loss

            loss = optimizer.step(lbfgs_closure)
            epoch_loss += loss.item()

        else:
            optimizer.zero_grad()
            loss, u_pred = closure(model_FEONet, batch, loss_type)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model_FEONet.parameters(), max_norm=5.0)
            optimizer.step()

        epoch_loss += loss.item()

    if epoch % 10 == 0:
        model_FEONet.eval()
        rel_err_total = 0.0
        test_err_total = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                ptr = batch.ptr
                batch_size = ptr.numel() - 1
                num_nodes = ptr[1] - ptr[0]  # = 441
                
                pred = model_FEONet(batch)
                
                rel_err_total += rel_L2_error(pred, batch.y).item()
                count += 1

        rel_err = rel_err_total / count
        loss_history.append(epoch_loss)
        test_history.append(rel_err)

        log_str = (f"[Epoch {epoch:04d}] "
                f"Loss={epoch_loss:.12f}   "
                f"Test_L2={rel_err:.6f}"
                )
            
        print(log_str)
            
        with open(log_file, "a") as f:
            f.write(log_str + "\n")

print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")
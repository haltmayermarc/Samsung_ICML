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

def int_list(arg):
    return [int(x) for x in arg.split(",")]


# ARGS
parser = argparse.ArgumentParser("SEM")
## Data
parser.add_argument("--type", type=str, choices=['quantile', 'coarse_checkerboard', 'fine_checkerboard', 'horizontal', 'vertical'])
parser.add_argument("--basis_order", type=str, choices=['1', '2'], default=1)
parser.add_argument("--num_training_data", type=int, default=5000)

## Train parameters
parser.add_argument("--hidden_dims", type=int, default=32)
parser.add_argument("--out_dims", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--model", type=str, default='SimpleDarcyGNN', choices=['SimpleDarcyGNN', "DarcyGNN", 'GraphtoVec'])
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--validate_batch_size", type=int, default=32)
parser.add_argument("--loss", type=str, choices=['mse', 'rel_l2', 'weak_form'], default='mse')
parser.add_argument("--epochs", type=int, default=50000)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument(
    "--conv_dims",
    type=int_list,
    default=[0],
)
parser.add_argument(
    "--mlp_dims",
    type=int_list,
    default=[0],
)


args = parser.parse_args()
gparams = args.__dict__

hidden_dims = gparams["hidden_dims"]
out_dims = gparams["out_dims"]
num_layers = gparams["num_layers"]
type = gparams['type']
gpu = gparams['gpu']
optimizer = gparams['optimizer']
loss_type = gparams['loss']
basis_order = gparams['basis_order']
num_training_data = gparams['num_training_data']

# Base path
base = f"data/P{basis_order}_ne0.125_Darcy_{num_training_data}"

if gparams["type"] is not None:
    if gparams["model"] != "GraphtoVec":
        npz_path = f"{base}_{type}_FIXED.npz"
    else:
        npz_path = f"{base}_{type}_FINE.npz"
else:
    npz_path = f"{base}.npz"

# Load mesh data
mesh = np.load(npz_path, allow_pickle=True)
p = mesh["coarse_nodes"]

#Model
models = {
          'SimpleDarcyGNN': SimpleDarcyGNN,
          'DarcyGNN': DarcyGNN,
          'GraphtoVec': GraphToVectorGNN
          }

MODEL = models[gparams['model']]

#Train
epochs = int(gparams['epochs'])
train_batch_size = gparams['train_batch_size']
validate_batch_size = gparams['train_batch_size']


if gparams["model"] == "SimpleDarcyGNN":
    model_FEONet = MODEL(hidden_dim=hidden_dims, out_dim=out_dims, num_layers=num_layers)
elif gparams["model"] == "DarcyGNN":
    model_FEONet = MODEL(hidden_dim=hidden_dims, num_layers=num_layers, edge_dim=6)
elif gparams["model"] == "GraphtoVec":
    conv_dims = [3] + gparams['conv_dims']
    mlp_dims = gparams['mlp_dims'] + [49]
    model_FEONet = MODEL(conv_dims=conv_dims, mlp_dims=mlp_dims, dropout=0.1)
    
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
model_FEONet = model_FEONet.to(device)
    

# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

model_FEONet.apply(weights_init)

##################################################################################
### Dataset class definition
##################################################################################

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
        pos = torch.tensor(arr["coarse_nodes"], dtype=torch.float)

        # Extract split
        A_all = arr[f"{self.kind}_matrices"]
        a_all = arr[f"{self.kind}_coeffs_a"]
        u_all = arr[f"{self.kind}_u"]
        f_all = arr[f"{self.kind}_load_vectors"]
        print("A_all.shape:", A_all.shape)
        print("a_all.shape:", a_all.shape)
        print("u_all.shape:", u_all.shape)
        print("f_all.shape:", f_all.shape)

        N = a_all.shape[0]
        print(N)
        data_list = []
        
        i = edge_index[0]
        j = edge_index[1]

        for k in range(N):
            A_matrix = torch.tensor(A_all[k], dtype=torch.float)
            f = torch.tensor(f_all[k], dtype=torch.float)

            x = torch.tensor(a_all[k], dtype=torch.float).unsqueeze(-1)
            y = torch.tensor(u_all[k], dtype=torch.float).unsqueeze(-1)
            
            # Node coordinates
            pos_i = pos[i]        
            pos_j = pos[j]        

            # Node coefficients
            a_i = x[i]            
            a_j = x[j]            
            
            A_ij = A_matrix[i, j].unsqueeze(-1)

            # Edge features
            edge_attr = torch.cat(
                [pos_i, pos_j, a_i, a_j],
                dim=-1
            )
            
            #edge_attr = A_ij

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                pos=pos,
                A=A_matrix,      
                f=f       
            )
            data_list.append(data)

        # Save
        self.save(data_list, self.processed_paths[0])
        
# Dataset class for GraphtoVec
class NewDarcyGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        npz_path,
        kind="train",   # "train" or "validate"
        transform=None,
        pre_transform=None,
    ):
        self.npz_path = npz_path
        self.kind = kind

        super().__init__(root, transform, pre_transform)
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

        # -------------------------------------------------
        # Shared graph structure
        # -------------------------------------------------
        edge_index = torch.tensor(arr["edges"], dtype=torch.long)  
        fine_nodes = torch.tensor(arr["fine_nodes"], dtype=torch.float)

        # -------------------------------------------------
        # Split-specific data
        # -------------------------------------------------
        a_all = arr[f"{self.kind}_coeffs_a"]
        u_all = arr[f"{self.kind}_u"]
        A_all = arr[f"{self.kind}_matrices"]
        f_all = arr[f"{self.kind}_load_vectors"]

        Ns = a_all.shape[0]
        data_list = []

        for k in range(Ns):
            # -------------------------------------------------
            # Node features
            # x_i = [a_i, x_i, y_i]
            # -------------------------------------------------
            a = torch.tensor(a_all[k], dtype=torch.float).unsqueeze(-1)  # (Nf, 1)
            x = torch.cat([a, fine_nodes], dim=1)                        # (Nf, 3)

            # -------------------------------------------------
            # Graph-level quantities (IMPORTANT: add batch dim)
            # -------------------------------------------------
            y = torch.tensor(u_all[k], dtype=torch.float).unsqueeze(0)   # (1, 49)
            A = torch.tensor(A_all[k], dtype=torch.float).unsqueeze(0)   # (1, 49, 49)
            f = torch.tensor(f_all[k], dtype=torch.float).unsqueeze(0)   # (1, 49)

            data = Data(
                x=x,                      # node features
                edge_index=edge_index,    # shared connectivity
                y=y,                      # graph-level target
                A=A,                      # graph-level operator
                f=f,                      # graph-level RHS
                pos=fine_nodes,           
            )

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])



if gparams["model"] != "GraphtoVec":
    train_dataset = DarcyGraphDataset(root="data/",npz_path=npz_path,kind="train")
    val_dataset = DarcyGraphDataset(root="data/",npz_path=npz_path,kind="validate")
else:
    train_dataset = NewDarcyGraphDataset(root="data/",npz_path=npz_path,kind="train")
    val_dataset = NewDarcyGraphDataset(root="data/",npz_path=npz_path,kind="validate")


if optimizer == "LBFGS":
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

"""
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
"""

optimizer = init_optim_adam(model_FEONet, lr=1e-3)
lbfgs_optimizer = init_optim_lbfgs(model_FEONet)

switch_epoch = 1000


def rel_L2_error(u_pred, u_true):
    return torch.norm(u_pred - u_true) / torch.norm(u_true)

def rel_l2_loss(u_pred, u_true, eps=1e-8):
    diff = torch.norm(u_pred - u_true, dim=1)
    denom = torch.norm(u_true, dim=1) + eps
    return torch.mean(diff / denom)

"""
def compute_loss(u_pred, batch, loss_type="rel_l2"):
    ptr = batch.ptr
    batch_size = ptr.numel() - 1
    num_nodes = ptr[1] - ptr[0]

    u_pred = u_pred.view(batch_size, num_nodes)
    u_true = batch.y.view(batch_size, num_nodes)

    if loss_type == "mse":
        return torch.mean((u_pred - u_true) ** 2)

    elif loss_type == "rel_l2":
        diff = u_pred - u_true
        return torch.mean(
            torch.norm(diff, dim=1) /
            (torch.norm(u_true, dim=1) + 1e-8)
        )

    elif loss_type == "weak_form":
        A = batch.A.view(batch_size, num_nodes, num_nodes)   # (B, N, N)
        f = batch.f.view(batch_size, num_nodes)              # (B, N)

        r = torch.bmm(A, u_pred.unsqueeze(-1)).squeeze(-1) - f  # (B, N)

        return torch.mean(
            torch.norm(r, dim=1) /
            (torch.norm(f, dim=1) + 1e-8)
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
"""
    
def compute_loss(u_pred, batch, loss_type="mse", eps=1e-8):
    """
    u_pred : (B, Nc)
    batch.y : (B, Nc)
    batch.A : (B, Nc, Nc)
    batch.f : (B, Nc)
    """

    u_true = batch.y

    if loss_type == "mse":
        # Mean squared error over batch and DOFs
        return torch.mean((u_pred - u_true) ** 2)

    elif loss_type == "rel_l2":
        # Relative L2 error per sample, then averaged
        diff = u_pred - u_true
        num = torch.norm(diff, dim=1)
        denom = torch.norm(u_true, dim=1) + eps
        return torch.mean(num / denom)

    elif loss_type == "weak_form":
        # Residual: A u - f
        r = torch.bmm(batch.A, u_pred.unsqueeze(-1)).squeeze(-1) - batch.f
        return torch.mean(torch.norm(r, dim=1))

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

    if epoch >= switch_epoch:
        optimizer = lbfgs_optimizer

    for batch in train_loader:
        batch = batch.to(device)

        if optimizer is lbfgs_optimizer:
            # LBFGS closure
            def lbfgs_closure():
                optimizer.zero_grad()
                u_pred = model_FEONet(batch)
                loss = compute_loss(u_pred, batch, loss_type)
                loss.backward()
                return loss

            loss = optimizer.step(lbfgs_closure)
            epoch_loss += loss.item()

        else:
            optimizer.zero_grad()
            loss, u_pred = closure(model_FEONet, batch, loss_type)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # =======================
    # Validation
    # =======================
    if epoch % 2 == 0:
        model_FEONet.eval()
        rel_err_total = 0.0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model_FEONet(batch)

                rel_err_total += rel_L2_error(pred, batch.y).item()
                count += 1

        rel_err = rel_err_total / count
        loss_history.append(epoch_loss)
        test_history.append(rel_err)

        log_str = (
            f"[Epoch {epoch:04d}] "
            f"Phase={'MSE' if epoch <= switch_epoch else 'WEAK'}   "
            f"Loss={epoch_loss:.6f}   "
            f"Test_relL2={rel_err:.6f}"
        )

        print(log_str)

        with open(log_file, "a") as f:
            f.write(log_str + "\n")
            
checkpoint = {
    "model_state_dict": model_FEONet.state_dict(),
    "args": gparams,
    "final_phase": "weak_form" if epochs > switch_epoch else "mse",
}

save_path = os.path.join(path, f"{gparams['model']}_MSE_to_WEAK.pt")
torch.save(checkpoint, save_path)

print(f"Model saved to {save_path}")
import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv          
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GlobalAttention

############################################################################
### Simple Darcy GNN
############################################################################

class DarcyMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # sum aggregation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 6, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(m)
    
class SimpleDarcyGNN(nn.Module):
    def __init__(self, hidden_dim= 64, out_dim=16, num_layers=3):
        super().__init__()

        self.embed = nn.Linear(1, hidden_dim)

        self.layers = nn.ModuleList([
            DarcyMessagePassing(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

    def forward(self, data):
        x = self.embed(data.x)

        for layer in self.layers:
            x = x + layer(x, data.edge_index, data.edge_attr)  # residual
            x = F.relu(x)

        return self.out_mlp(x)
    
############################################################################
### DarcyGNN
############################################################################
    
class NewDarcyMessagePassing(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # ensures positivity
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        k_ij = self.edge_mlp(edge_attr)        # (E, 1)
        return k_ij * (x_j - x_i)

class DarcyGNN(nn.Module):
    def __init__(self, hidden_dim, num_layers, edge_dim):
        super().__init__()

        self.embed = nn.Linear(1, hidden_dim)

        self.layers = nn.ModuleList([
            NewDarcyMessagePassing(hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = self.embed(data.x)

        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr) #+x

        return self.out(x)
    
############################################################################
### UNet
############################################################################

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)
        x = self.enc4(x)
        return x
    
class UNetLatentModel(nn.Module):
    def __init__(self, latent_dim=256, out_dim=49):
        super().__init__()

        self.encoder = UNetEncoder()

        self.pool = nn.AdaptiveAvgPool2d(1)  # global pooling

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).flatten(1)  # (B, latent_dim)
        return self.mlp(x)
    
############################################################################
### GraphtoVec
############################################################################

class MLP(nn.Module):
    def __init__(
        self,
        layer_dims,
        activation=nn.Tanh,
        dropout=0.0,
        use_layernorm=False,
    ):
        """
        layer_dims: list[int]
            Example: [128, 256, 256, 128, out_dim]
        """
        super().__init__()

        assert len(layer_dims) >= 2, "Need at least input and output dimension"

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            # no activation after last layer
            if i < len(layer_dims) - 2:
                if use_layernorm:
                    layers.append(nn.LayerNorm(layer_dims[i + 1]))
                layers.append(activation())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class FlexibleGNN(nn.Module):
    def __init__(self, conv_dims, dropout=0.1):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for in_c, out_c in zip(conv_dims[:-1], conv_dims[1:]):
            self.convs.append(GCNConv(in_c, out_c))
            self.norms.append(nn.LayerNorm(out_c))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = F.tanh(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphToVectorGNN(nn.Module):
    def __init__(self, conv_dims, mlp_dims, dropout=0.1):
        super().__init__()

        self.gnn = FlexibleGNN(conv_dims, dropout=dropout)

        self.pre_pool_norm = nn.LayerNorm(conv_dims[-1])

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(conv_dims[-1], 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
        )

        self.mlp = MLP(
            layer_dims=mlp_dims,
            dropout=dropout,
            use_layernorm=True,
        )

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index)
        x = self.pre_pool_norm(x)
        g = self.pool(x, data.batch)
        return self.mlp(g)


class NodeRegressionGNN(nn.Module):
    def __init__(
        self,
        in_dim=3,                 # [a_i, x_i, y_i]
        hidden_dims=[64, 128, 128],
        out_dim=1,
        dropout=0.1,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + hidden_dims

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.convs.append(GCNConv(d_in, d_out))
            self.norms.append(nn.LayerNorm(d_out))

        # Node-wise regression head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Node-wise prediction
        y_pred = self.mlp(x)  # (Nf, 1)

        return y_pred
    
# -----------------------------
# Building Blocks
# -----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetFeatureExtractor(nn.Module):
    """
    Input:  (B, in_ch, H, W)
    Output: (B, latent_ch, H, W)
    Works for arbitrary H, W
    """
    def __init__(self, in_ch=2, base_ch=32, latent_ch=16):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)

        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)

        # Decoder
        self.dec2 = DoubleConv(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = DoubleConv(base_ch * 2 + base_ch, base_ch)

        # Projection
        self.proj = nn.Conv2d(base_ch, latent_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)            # (B, base, H, W)
        e2 = self.enc2(self.pool(e1))# (B, 2base, H/2, W/2)

        # Bottleneck
        b = self.bottleneck(self.pool(e2))

        # Decoder stage 2
        d2 = F.interpolate(
            b, size=e2.shape[-2:], mode="bilinear", align_corners=False
        )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        # Decoder stage 1
        d1 = F.interpolate(
            d2, size=e1.shape[-2:], mode="bilinear", align_corners=False
        )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.proj(d1)

# -----------------------------
# Prediction Head (like Net2D)
# -----------------------------
class UNetHead(nn.Module):
    """
    Input:  (B, d, H, W)
    Output: (B, 1, d_out)
    """
    def __init__(self, resol_in: int, d_in: int, d_out: int, filters: int = 64,
                 kernel_size: int = 7, padding: int = 3, blocks: int = 1):
        super().__init__()
        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(d_in, filters, kernel_size=kernel_size, padding=padding)
        layers = []
        for _ in range(blocks):
            layers += [nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=padding), nn.SiLU(inplace=True)]
        self.mid = nn.Sequential(*layers)
        self.convH = nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=padding)
        self.fc = nn.Linear(filters * (resol_in ** 2), d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d, H, W)
        out = self.act(self.conv1(x))
        if len(self.mid) > 0:
            out = self.mid(out)
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fc(out)                  # (B, d_out)
        return out.view(out.size(0), 1, -1) # (B, 1, d_out)

# -----------------------------
# Full Model (latent only used to feed head)
# -----------------------------
class UNetWithHead(nn.Module):
    def __init__(self, resol_in: int, in_ch: int = 2, base_ch: int = 32, latent_ch: int = 16, d_out: int = 10,
                 head_filters: int = 64, head_blocks: int = 1,
                 head_kernel_size: int = 7, head_padding: int = 3):   
        super().__init__()
        self.feature = UNetFeatureExtractor(in_ch=in_ch, base_ch=base_ch, latent_ch=latent_ch)
        self.head = UNetHead(
            resol_in=resol_in,
            d_in=latent_ch,
            d_out=d_out,
            filters=head_filters,
            blocks=head_blocks,
            kernel_size=head_kernel_size,   
            padding=head_padding            
        )

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.feature(x)   # (B, latent_ch, H, W)
        out = self.head(latent)    # (B, 1, d_out)
        return out
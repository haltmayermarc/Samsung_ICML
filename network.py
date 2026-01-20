import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv          
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool


def conv1d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def conv2d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def conv3d(in_planes, out_planes, stride=1, bias=True, kernel_size=5, padding=2, dialation=1) :
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class NetA(nn.Module) :
    def __init__(self, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0, is_bdrylyaer=False) :
        super(NetA,self).__init__()
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.kern = kernel_size
        self.pad = padding
        self.swish = nn.SiLU()
        self.conv1 = conv1d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv_list = []
        if self.blocks != 0:
            for block in range(self.blocks):
                self.conv_list.append(conv1d(filters, filters, kernel_size=self.kern, padding=self.pad))
                self.conv_list.append(self.swish)
        self.conv_list=nn.Sequential(*self.conv_list)
        self.convH = conv1d(filters, filters, kernel_size=self.kern, padding=self.pad)
        if is_bdrylyaer:
            self.fcH = nn.Linear(filters*(self.d_out-1), self.d_out, bias=True)
        else:
            self.fcH = nn.Linear(filters*self.d_out, self.d_out, bias=True)
    def forward(self, x):
        out = self.swish(self.conv1(x))
        if self.blocks != 0:
            out = self.conv_list(out)
        out = self.convH(out)
        out = out.flatten(start_dim=1)     
        out = self.fcH(out)       
        out = out.view(out.shape[0], 1, self.d_out)        
        return out
    


class Net2D(nn.Module) : # Linear
    def __init__(self, resol_in, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net2D, self).__init__()
        self.resol_in = resol_in
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = nn.SiLU()
        # self.swish = nn.ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv2d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv_list = []
        if self.blocks != 0:
            for block in range(self.blocks):
                self.conv_list.append(conv2d(filters, filters, kernel_size=self.kern, padding=self.pad))
                self.conv_list.append(self.swish)
        self.conv_list=nn.Sequential(*self.conv_list)
        self.convH = conv2d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*resol_in**2, self.d_out, bias=True)

    def forward(self, x):
        out = self.swish(self.conv1(x))
        if self.blocks != 0:
            out = self.conv_list(out)
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out
    


class Net3D(nn.Module) : # Linear
    def __init__(self, resol_in, d_in, filters, d_out, kernel_size=7, padding=3, blocks=0) :
        super(Net3D, self).__init__()
        self.resol_in = resol_in
        self.d_in = d_in
        self.blocks = blocks
        self.filters = filters
        self.d_out = d_out
        self.swish = nn.SiLU()
        # self.swish = nn.ReLU()
        self.kern = kernel_size
        self.pad = padding
        self.conv1 = conv3d(d_in, filters, kernel_size=self.kern, padding=self.pad)
        self.conv_list = []
        if self.blocks != 0:
            for block in range(self.blocks):
                self.conv_list.append(conv3d(filters, filters, kernel_size=self.kern, padding=self.pad))
                self.conv_list.append(self.swish)
        self.conv_list=nn.Sequential(*self.conv_list)
        self.convH = conv3d(filters, filters, kernel_size=self.kern, padding=self.pad)
        self.fcH = nn.Linear(filters*resol_in**3, self.d_out, bias=True)

    def forward(self, x):
        out = self.swish(self.conv1(x))
        if self.blocks != 0:
            out = self.conv_list(out)
        out = self.convH(out)
        out = out.flatten(start_dim=1)
        out = self.fcH(out)
        out = out.view(out.shape[0], 1, self.d_out)
        return out
    
class FCNN(nn.Module):
    def __init__(self, resol_in, output_dim, hidden_dims=[2048, 1024, 512, 1024, 2048, 4096, 4096*2], dropout_prob=0.2):
        super(FCNN, self).__init__()
        
        layers = []
        input_dim = resol_in
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout_prob))

        layers.append(nn.Linear(dims[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1),
        )
    def forward(self, x):
        return self.block(x)

# -----------------------------
# UNet-like Feature Extractor
# -----------------------------
class UNetFeatureExtractor(nn.Module):
    """
    Input:  (B, 2, H, W)
    Output: (B, d, H, W) where d = latent_ch
    Preserves spatial size with strided pooling+upsampling.
    """
    def __init__(self, in_ch=2, base_ch=32, latent_ch=16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        # Project to latent channels d
        self.proj = nn.Conv2d(base_ch, latent_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #B, C, H, W = x.shape
        # Encoder
        e1 = self.enc1(x)              # (B, base_ch, H, W)
        e2 = self.enc2(self.pool1(e1)) # (B, 2*base_ch, H/2, W/2)

        # Bottleneck
        b = self.bottleneck(self.pool2(e2)) # (B, 4*base_ch, H/4, W/4)

        # Decoder with skips
        d2 = self.up2(b)                       # (B, 2*base_ch, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], 1)) # (B, 2*base_ch, H/2, W/2)

        d1 = self.up1(d2)                      # (B, base_ch, H, W)
        d1 = self.dec1(torch.cat([d1, e1], 1)) # (B, base_ch, H, W)

        latent = self.proj(d1)                 # (B, latent_ch, H, W)
        return latent

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
    
class DarcyMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # sum aggregation
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 7, 64),
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
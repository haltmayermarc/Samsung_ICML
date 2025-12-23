import numpy as np
import pandas as pd
import scipy
from scipy import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import time
from datetime import datetime
import os
import random
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dolfin import *
from mshr import *
from scipy.spatial.distance import cdist
from LOD import *

parser = argparse.ArgumentParser("SEM")
parser.add_argument("--type", type=str, choices=['quantile','checkerboard', 'horizontal', 'vertical'])
args = parser.parse_args()
gparams = args.__dict__

TYPE = gparams["type"]

epsi = 0.01
kappa_values = np.array([epsi, 4.0, 8.0, 12.0, 16.0, 20.0])

def f_const(x, y):
    return 1.0

def assign_kappa_quantiles(a_sample, kappa_values):
    K = len(kappa_values)
    edges = np.quantile(a_sample, np.linspace(0, 1, K + 1))
    bins = np.digitize(a_sample, edges[1:-1], right=True)
    bins = np.clip(bins, 0, K - 1)
    
    return kappa_values[bins]

def quantize_field(Z, n, kappa_values=None):
    flat = Z.flatten()
    # Compute quantile boundaries (n+1 edges)
    q_edges = np.quantile(flat, np.linspace(0, 1, n+1))

    if kappa_values is None:
        kappa_values = np.arange(1, n+1)
    kappa_values = np.asarray(kappa_values)

    # Digitize field values based on quantile bins
    bin_indices = np.digitize(flat, q_edges[1:-1], right=False)
    Z_sharp = kappa_values[bin_indices].reshape(Z.shape)

    return Z_sharp, q_edges

def make_LOD_data_quantile(Nx, Ny, refine, grid, g, q_edges, kappa_values):
    coarse_nodes = grid["coarse_nodes"]
    coarse_elems = grid["coarse_elems"]
    fine_nodes   = grid["fine_nodes"]
    fine_elems   = grid["fine_elems"]
    Nx_fine = grid["Nx_fine"]; Ny_fine = grid["Ny_fine"]
    
    g_values = g.vector().get_local()
    q_edges = np.quantile(g_values, np.linspace(0, 1, kappa_values.shape[0]+1))
    
    def quantize_scalar(value, q_edges, kappa_values):
        # Return the kappa level corresponding to value
        idx = np.searchsorted(q_edges[1:], value, side="right")
        return kappa_values[min(idx, len(kappa_values)-1)]
    
    def kappa_sharp(g, q_edges, kappa_values, x, y):
        val = g(Point(x, y))          
        return quantize_scalar(val, q_edges, kappa_values)

    
    A_dc, M_dc, sigma = build_fine_element_matrices_var_kappa_3x3(grid, lambda x, y: kappa_sharp(g, q_edges, kappa_values, x, y))
    Nh = fine_nodes.shape[0]
    A_h, M_h = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)
    
    # RHS
    f_full = assemble_load_quad(fine_nodes, fine_elems, f_const)

    # Dirichlet boundary on fine grid
    bdry = boundary_mask_fine(fine_nodes)
    free_mask = ~bdry
    free_idx = np.where(free_mask)[0]
    A_free = A_h[np.ix_(free_idx, free_idx)]
    f_free = f_full[free_idx]

    # interpolation P and coarse boundary mask
    P = build_P_quad_unique(Nx, Ny, refine)       
    P_h = P.T                                     
    P_free = P[free_idx, :]                      
    N_H = P.shape[1]
    B_H = build_B_H(coarse_nodes, Nx, Ny)

    # build patches per coarse element (no oversampling)
    R_h_list, R_H_list, T_H_list, fine_elems_in_coarse = build_patch_mappings(grid)
    NTH = len(T_H_list)
    
    # compute global correctors Q_h (Algorithm 1)
    Q_h = computeCorrections_algorithm(
        Nh, N_H, NTH,
        A_dc, M_dc, sigma,
        B_H, P_h,
        R_h_list, R_H_list, T_H_list, fine_elems_in_coarse
    )

    # restrict Q_h to free DOFs
    Q_free = Q_h[:, free_idx]   # (N_H x N_free)

    # coarse interior mask (Dirichlet on ∂Ω in coarse space)
    coarse_interior_mask = np.ones(N_H, dtype=bool)
    for j in range(Ny+1):
        for i in range(Nx+1):
            idx = j*(Nx+1) + i
            if i==0 or i==Nx or j==0 or j==Ny:
                coarse_interior_mask[idx] = False
                
    A_lod, f_lod = get_LOD_matrix_rhs(A_free, f_free, P_free, Q_free, coarse_interior_mask)
    
    return A_lod, f_lod

def make_LOD_data(Nx, Ny, refine, grid, g):
    coarse_nodes = grid["coarse_nodes"]
    fine_nodes   = grid["fine_nodes"]
    fine_elems   = grid["fine_elems"]
    
    A_dc, M_dc, sigma = build_fine_element_matrices_var_kappa_3x3(grid, g)
    Nh = fine_nodes.shape[0]
    A_h, M_h = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)
    
    # RHS
    f_full = assemble_load_quad(fine_nodes, fine_elems, f_const)

    # Dirichlet boundary on fine grid
    bdry = boundary_mask_fine(fine_nodes)
    free_mask = ~bdry
    free_idx = np.where(free_mask)[0]
    A_free = A_h[np.ix_(free_idx, free_idx)]
    f_free = f_full[free_idx]

    # interpolation P and coarse boundary mask
    P = build_P_quad_unique(Nx, Ny, refine)       
    P_h = P.T                                     
    P_free = P[free_idx, :]                      
    N_H = P.shape[1]
    B_H = build_B_H(coarse_nodes, Nx, Ny)

    # build patches per coarse element (no oversampling)
    R_h_list, R_H_list, T_H_list, fine_elems_in_coarse = build_patch_mappings(grid)
    NTH = len(T_H_list)
    
    # compute global correctors Q_h (Algorithm 1)
    Q_h = computeCorrections_algorithm(
        Nh, N_H, NTH,
        A_dc, M_dc, sigma,
        B_H, P_h,
        R_h_list, R_H_list, T_H_list, fine_elems_in_coarse
    )

    # restrict Q_h to free DOFs
    Q_free = Q_h[:, free_idx]   # (N_H x N_free)

    # coarse interior mask (Dirichlet on ∂Ω in coarse space)
    coarse_interior_mask = np.ones(N_H, dtype=bool)
    for j in range(Ny+1):
        for i in range(Nx+1):
            idx = j*(Nx+1) + i
            if i==0 or i==Nx or j==0 or j==Ny:
                coarse_interior_mask[idx] = False
                
    A_lod, f_lod = get_LOD_matrix_rhs(A_free, f_free, P_free, Q_free, coarse_interior_mask)
    
    return A_lod, f_lod

def create_dataset(num_input ,num_xy, refine, kappa_values):
    mesh_data = build_uniform_grid(num_xy, num_xy, refine)
    
    coarse_nodes = mesh_data["coarse_nodes"]
    coarse_elems = mesh_data["coarse_elems"]
    fine_nodes   = mesh_data["fine_nodes"]
    fine_elems   = mesh_data["fine_elems"]
    Nx_fine = mesh_data["Nx_fine"]
    Ny_fine = mesh_data["Ny_fine"]
    
    V_dim = coarse_nodes.shape[0]
    
    coarse_row, coarse_col = elems_to_coo(coarse_elems)
    edges = np.vstack([coarse_row, coarse_col])

    ne=coarse_elems.shape[0]
    
    pos = coarse_nodes
    ng=pos.shape[0]

    print("Num of Elements : {}, Num of points : {}".format(ne, ng))
    
    # GRF parameters
    sigma = 1.0     # variance
    ell = 0.3       # correlation length

    # Pairwise distances
    D = cdist(pos, pos)

    # Covariance matrix for GRF
    C = sigma**2 * np.exp(-D**2/(2*ell**2))

    # Generate training and validation data
    train_coeffs_a = []
    #train_values_a = []
    train_matrices = []
    train_load_vectors = []
    train_fenics_u = []
    
    validate_coeffs_a = []
    #validate_values_a = []
    validate_matrices = []
    validate_load_vectors = []
    validate_fenics_u = []
    
    mesh_fenics = UnitSquareMesh(num_xy, num_xy)
    V = FunctionSpace(mesh_fenics, "CG", 1)
    K = np.zeros([num_xy, num_xy])
    
    # TRAINING SET
    np.random.seed(5)
    for _ in tqdm(range(num_input[0])):
        if TYPE == "quantile":
            a_sample = np.random.multivariate_normal(
                mean=np.ones(V_dim),   # mean permeability = 1
                cov=C
            )
            a_sample = np.exp(a_sample)
            a = Function(V)
            a.vector()[:] = a_sample
            train_coeffs_a.append(a_sample)
            q_edges = np.quantile(a_sample, np.linspace(0, 1, kappa_values.shape[0]+1))
            
            A_LOD_matrix, f_LOD_vector = make_LOD_data_quantile(num_xy, num_xy, refine, mesh_data, a, q_edges, kappa_values)
            train_matrices.append(A_LOD_matrix)
            train_load_vectors.append(f_LOD_vector)
            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]), f_LOD_vector, rcond=None)[0]
            train_fenics_u.append(u_lod)
            
        elif TYPE in ["checkerboard", "horizontal", "vertical"]:
            rng = np.random.default_rng()

            # --- checkerboard ---
            if TYPE == "checkerboard":
                for j in range(num_xy):
                    for i in range(num_xy):
                        idx = rng.integers(0, len(kappa_values))
                        K[j, i] = kappa_values[idx]

            # --- horizontal stripes ---
            elif TYPE == "horizontal_stripes":
                n_stripes = 16
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for j in range(num_xy):
                    stripe_id = min(int(j / num_xy * n_stripes), n_stripes - 1)
                    K[j, :] = stripe_vals[stripe_id]

            # --- vertical stripes ---
            elif TYPE == "vertical_stripes":
                n_stripes = 16
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for i in range(num_xy):
                    stripe_id = min(int(i / num_xy * n_stripes), n_stripes - 1)
                    K[:, i] = stripe_vals[stripe_id]

            # --- callable coefficient ---
            def a(x, y):
                x = min(max(x, 0.0), 1.0 - 1e-14)
                y = min(max(y, 0.0), 1.0 - 1e-14)

                i = int(x * num_xy)
                j = int(y * num_xy)

                return K[j, i]

            # --- sample on coarse nodes ---
            a_sample = np.array([a(p[0], p[1]) for p in pos])
            train_coeffs_a.append(a_sample)

            # --- build LOD data ---
            A_LOD_matrix, f_LOD_vector = make_LOD_data(
                num_xy, num_xy, refine, mesh_data, a
            )

            train_matrices.append(A_LOD_matrix)
            train_load_vectors.append(f_LOD_vector)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(
                    A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]),
                    f_LOD_vector,
                    rcond=None
                )[0]
            train_fenics_u.append(u_lod)

            
    # VALIDATION SET
    np.random.seed(10)
    for _ in tqdm(range(num_input[1])):
        if TYPE == "quantile":
            a_sample = np.random.multivariate_normal(
                mean=np.ones(V_dim),   # mean permeability = 1
                cov=C
            )
            a_sample = np.exp(a_sample)
            a = Function(V)
            a.vector()[:] = a_sample
            validate_coeffs_a.append(a_sample)
            q_edges = np.quantile(a_sample, np.linspace(0, 1, kappa_values.shape[0]+1))
            
            A_LOD_matrix, f_LOD_vector = make_LOD_data_quantile(num_xy, num_xy, refine, mesh_data, a, q_edges, kappa_values)
            validate_matrices.append(A_LOD_matrix)
            validate_load_vectors.append(f_LOD_vector)
            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]), f_LOD_vector, rcond=None)[0]
            validate_fenics_u.append(u_lod)
            
        elif TYPE in ["checkerboard", "horizontal_stripes", "vertical_stripes"]:
            rng = np.random.default_rng()

            # --- checkerboard ---
            if TYPE == "checkerboard":
                for j in range(num_xy):
                    for i in range(num_xy):
                        idx = rng.integers(0, len(kappa_values))
                        K[j, i] = kappa_values[idx]

            # --- horizontal stripes ---
            elif TYPE == "horizontal_stripes":
                n_stripes = 16
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for j in range(num_xy):
                    stripe_id = min(int(j / num_xy * n_stripes), n_stripes - 1)
                    K[j, :] = stripe_vals[stripe_id]

            # --- vertical stripes ---
            elif TYPE == "vertical_stripes":
                n_stripes = 16
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for i in range(num_xy):
                    stripe_id = min(int(i / num_xy * n_stripes), n_stripes - 1)
                    K[:, i] = stripe_vals[stripe_id]

            # --- callable coefficient ---
            def a(x, y):
                x = min(max(x, 0.0), 1.0 - 1e-14)
                y = min(max(y, 0.0), 1.0 - 1e-14)

                i = int(x * num_xy)
                j = int(y * num_xy)

                return K[j, i]

            # --- sample on coarse nodes ---
            a_sample = np.array([a(p[0], p[1]) for p in pos])
            validate_coeffs_a.append(a_sample)

            # --- build LOD data ---
            A_LOD_matrix, f_LOD_vector = make_LOD_data(
                num_xy, num_xy, refine, mesh_data, a
            )

            validate_matrices.append(A_LOD_matrix)
            validate_load_vectors.append(f_LOD_vector)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(
                    A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]),
                    f_LOD_vector,
                    rcond=None
                )[0]
            validate_fenics_u.append(u_lod)
    
    return ne, ng, pos, edges, np.array(train_coeffs_a), np.array(train_matrices), np.array(train_load_vectors), np.array(train_fenics_u),  np.array(validate_coeffs_a), np.array(validate_matrices), np.array(validate_load_vectors), np.array(validate_fenics_u)

order='1'
list_num_xy=[16]
num_input=[5000, 1000]
typ='Darcy'
refine = 2
epsi = 0.01
kappa_values = np.array([epsi, 4.0, 8.0, 12.0, 16.0, 20.0])

for idx, num in enumerate(list_num_xy):
    ne, ng, p, edges, train_coeffs_a,  train_matrices, train_load_vectors, train_fenics_u, validate_coeffs_a, validate_matrices, validate_load_vectors, validate_fenics_u = create_dataset(num_input, num, refine, kappa_values)

    # build filename
    base = f"data/P{order}_ne{ne}_{typ}_{num_input[0]}"
    if gparams["type"] is not None:
        mesh_path = f"{base}_{gparams['type']}.npz"
    else:
        mesh_path = f"{base}.npz"

    # save with mesh_path
    np.savez(
        mesh_path,
        ne=ne, ng=ng, p=p, edges=edges,
        train_coeffs_a=train_coeffs_a,
        train_matrices=train_matrices,
        train_load_vectors = train_load_vectors,
        train_fenics_u=train_fenics_u,
        validate_coeffs_a=validate_coeffs_a,
        validate_matrices=validate_matrices,
        validate_load_vectors= validate_load_vectors,
        validate_fenics_u=validate_fenics_u
    )
    print(f"Saved data at {mesh_path} for num_xy = {num}")
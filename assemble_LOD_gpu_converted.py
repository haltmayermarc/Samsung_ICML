import numpy as np
from scipy import io
from tqdm import tqdm
import argparse
#from dolfin import *
#from mshr import *
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma
from scipy.interpolate import LinearNDInterpolator
from LOD_gpu_converted import *

import time


import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy import sparse
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma
from scipy.interpolate import RegularGridInterpolator

import os

from sksparse.cholmod import cholesky
import importlib

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parser = argparse.ArgumentParser("SEM")
parser.add_argument("--type", type=str, choices=['quantile', 'lognormal', 'coarse_checkerboard', 'fine_checkerboard', 'horizontal', 'vertical'])
parser.add_argument("--H", type=int, default=4)
parser.add_argument("--h", type=int, default=7)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--num_training_samples", type=int, default=500)
parser.add_argument("--num_validation_samples", type=int, default=100)
args = parser.parse_args()
gparams = args.__dict__

TYPE = gparams["type"]
H = 2**(-gparams["H"])
h = 2**(-gparams["h"])
k = gparams["k"]
num_train = gparams["num_training_samples"]
num_val = gparams["num_validation_samples"]
print(H)
print(h)
print(k)
print(num_train)
print(num_val)


class SEGaussianSamplerSVD:
    def __init__(self, pos, sigma=1.0, ell=0.3, mean=1.0, tol=1e-8):
        self.pos = np.asarray(pos, dtype=float)
        self.V = self.pos.shape[0]

        # Pairwise distances
        D = cdist(self.pos, self.pos)

        # Covariance matrix for GRF
        C = (sigma**2) * np.exp(-(D**2) / (2 * ell**2))

        U, s, _ = np.linalg.svd(C, full_matrices=False)

        if np.min(s) < -tol:
            raise ValueError(f"Covariance not PSD: min eigen/singular {np.min(s)} < -tol")
        s = np.clip(s, 0.0, None)

        self.A = U * np.sqrt(s)[None, :]
        self.mean = mean

    def sample(self, rng: np.random.Generator):
        z = rng.normal(0.0, 1.0, size=(self.V,))
        return self.mean + self.A @ z
    
def matern_covariance(X, sigma=1.0, nu=1.0, kappa=0.3):
    r = cdist(X, X)
    r[r == 0.0] = 1e-12

    factor = (sigma**2) / (2**(nu - 1) * gamma(nu))
    arg = np.sqrt(2 * nu) * r / kappa

    C = factor * (arg**nu) * kv(nu, arg)
    np.fill_diagonal(C, sigma**2)
    return C

class MaternGaussianField:
    def __init__(self, points, sigma=1.0, nu=1.0, kappa=0.3, mean=0.0, tol=1e-10):
        self.points = np.asarray(points, dtype=float)
        self.N = self.points.shape[0]
        self.mean = mean

        C = matern_covariance(self.points, sigma, nu, kappa)

        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, tol, None)

        self.A = eigvecs @ np.diag(np.sqrt(eigvals))

    def sample(self, rng=np.random.default_rng()):
        z = rng.normal(size=self.N)
        return self.mean + self.A @ z

def make_lognormal_kappa(
    points,                 # (N, 2)
    field,
    rng,
):
    Z = field.sample(rng)
    A = np.exp(Z)   # lognormal field

    # Scattered interpolator
    interpolator = LinearNDInterpolator(points, A)

    def kappa_func(x, y):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        vals = interpolator(np.column_stack([x, y]))

        if vals.size == 1:
            return float(vals[0])   # ← force scalar
        return vals

    return kappa_func, A

# For quantile-type coefficient
def compute_kappa_per_element(coarse_elems, kappa_node):
    """
    Compute elementwise-constant kappa on coarse mesh.

    Returns
    -------
    kappa_elem : (N_elem,) ndarray
    """
    kappa_elem = np.zeros(len(coarse_elems))
    for l, elem in enumerate(coarse_elems):
        kappa_elem[l] = np.mean(kappa_node[list(elem)])
    return kappa_elem

def make_fast_kappa_uniform(h, Nx, kappa_elem):
    def kappa(x, y):
        i = min(int(x // h), Nx - 1)
        j = min(int(y // h), Nx - 1)

        # quad index
        q = j * Nx + i

        # two triangles per quad
        # consistent with quads_to_tris(bl-tr)
        if (x - i*h) + (y - j*h) <= h:
            l = 2*q       # lower-left triangle
        else:
            l = 2*q + 1   # upper-right triangle

        return kappa_elem[l]

    return kappa


def make_LOD_data(grid, adjacency, fine_in_coarse, kappa, B_H, C_h, f_h, P_h):
    fine_nodes   = grid["fine_nodes"]

    A_dc, M_dc, sigma = build_fine_element_matrices(grid, lambda x, y: kappa(x,y))
    Nh = fine_nodes.shape[0]
    A_h, _ = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)
    
    
    # compute global correctors Q_h 
    Q_h = computeCorrections(grid, k, adjacency, fine_in_coarse, A_h, B_H, C_h, kappa, n_jobs=-1)
    
    # Compute Basis
    V_ms = P_h + Q_h  # (N_H x N_h)

    #Compute Change of basis to get LOD stiffness matrix, rhs vector
    A_lod = V_ms @ A_h @ V_ms.T   # (N_H x N_H)
    f = V_ms @ f_h            # (N_H,)

    #restrict to interior
    interior = np.where(np.diag(B_H) > 0.5)[0]  # interior coarse node indices
    A_lod = A_lod[np.ix_(interior, interior)]
    f_lod  = f[interior]
    
    u_h_fine = solve_fine_problem(A_h, f_h, fine_nodes)
    
    return A_lod, f_lod, Q_h, u_h_fine


def create_dataset(num_input, H, h, kappa_values):
    # Create coarse and fine mesh
    Nx = int(1 / H)
    Ny = Nx
    refine = int(H / h)

    mesh_data = build_triangular_mesh(Nx, Ny, refine)
    
    coarse_nodes = mesh_data["coarse_nodes"]
    coarse_elems = mesh_data["coarse_elems"]
    fine_nodes   = mesh_data["fine_nodes"]
    fine_elems   = mesh_data["fine_elems"]
    
    N_H = coarse_nodes.shape[0]
    N_h = fine_nodes.shape[0]
    
    V_dim = coarse_nodes.shape[0]
    
    # Mesh connectivity and adjacency
    adjacency = build_coarse_adjacency_edge(coarse_elems)
    fine_in_coarse = precompute_fine_in_coarse_structured(mesh_data)
    pos = fine_nodes
    print("Meshing finished...")
    
    # Construct SE Kernel sampler for GRF
    if TYPE == "quantile":
        sampler = SEGaussianSamplerSVD(pos, sigma=1.0, ell=0.3, mean=1.0, tol=1e-8)
    elif TYPE == "lognormal":
        field = MaternGaussianField(
                fine_nodes,
                sigma=1.0,
                nu=0.5,
                kappa=0.1,
                mean=0.0,
            )
    
    ###########################################################
    # Generate all data that do not depend 
    # on a given coefficient instance or patch
    ###########################################################
    
    # Interpolation matrix P_h from the 2019 paper
    print("Building P_h...")
    P_h = build_P_triangular(mesh_data)
    # Boundary matrix
    print("Building B_H...")
    B_H = build_B_H(coarse_nodes, Nx, Ny)
    interior = np.where(np.diag(B_H) > 0.5)[0]
    # Fine forcing term rhs vector
    print("Building f_h...")
    f_h = assemble_load_tri(
        fine_nodes, fine_elems, lambda x, y: 1.0
    )
    # From the 2020 SIAM book
    print("Building C_h...")
    C_h = build_IH_quasi_interpolation(mesh_data)
    
    # Generate training and validation data
    train_coeffs_a = []
    train_matrices = []
    train_load_vectors = []
    train_fenics_u = []
    train_Q = []
    train_u_h_fine = []
    
    validate_coeffs_a = []
    validate_matrices = []
    validate_load_vectors = []
    validate_fenics_u = []
    validate_Q = []
    validate_u_h_fine = []
    
    
    # TRAINING SET
    np.random.seed(5)
    for _ in tqdm(range(num_input[0])):
        if TYPE == "quantile":
            rng = np.random.default_rng()
            a_sample = sampler.sample(rng)
            a_sample = np.exp(a_sample)
            q_edges = np.quantile(a_sample, np.linspace(0, 1, kappa_values.shape[0]+1))
            
            kappa_shuffled = np.asarray(kappa_values, dtype=float).copy()
            rng.shuffle(kappa_shuffled)
            
            bins = np.searchsorted(q_edges[1:], a_sample, side="right")
            bins = np.clip(bins, 0, len(kappa_values)-1)
            kappa_node = kappa_shuffled[bins]
            
            kappa_elem = compute_kappa_per_element(fine_elems, kappa_node)
            kappa = make_fast_kappa_uniform(h, int(1 / h), kappa_elem)
            
            train_coeffs_a.append(kappa_node)
    
            A_LOD_matrix, f_LOD_vector, Q_h, u_h_fine = make_LOD_data(
                    mesh_data, adjacency, fine_in_coarse,
                    kappa, 
                    B_H, C_h, f_h, P_h
            )

            train_matrices.append(A_LOD_matrix)
            train_load_vectors.append(f_LOD_vector)
            #train_Q.append(Q_h)
            train_u_h_fine.append(u_h_fine)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]), f_LOD_vector, rcond=None)[0]
            train_fenics_u.append(u_lod)
            
        if TYPE == "lognormal":
            rng = np.random.default_rng()
            
            kappa, kappa_node = make_lognormal_kappa(
                fine_nodes,
                field,
                rng
            )
            
            train_coeffs_a.append(kappa_node)
    
            A_LOD_matrix, f_LOD_vector, Q_h, u_h_fine = make_LOD_data(
                    mesh_data, adjacency, fine_in_coarse,
                    kappa, 
                    B_H, C_h, f_h, P_h
            )

            train_matrices.append(A_LOD_matrix)
            train_load_vectors.append(f_LOD_vector)
            #train_Q.append(Q_h)
            train_u_h_fine.append(u_h_fine)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]), f_LOD_vector, rcond=None)[0]
            train_fenics_u.append(u_lod)
            
        elif TYPE in ["coarse_checkerboard", "fine_checkerboard", "horizontal", "vertical"]:
            rng = np.random.default_rng()
            
            # --- checkerboard ---
            if TYPE == "coarse_checkerboard":
                K = np.zeros([Nx,Ny])
                for j in range(Ny):
                    for i in range(Nx):
                        idx = rng.integers(0, len(kappa_values))
                        K[j, i] = kappa_values[idx]
                        
            elif TYPE == "fine_checkerboard":
                K = np.zeros([int(1/h),int(1/h)])
                for j in range(int(1/h)):
                    for i in range(int(1/h)):
                        idx = rng.integers(0, len(kappa_values))
                        K[j, i] = kappa_values[idx]

            # --- horizontal stripes ---
            elif TYPE == "horizontal":
                K = np.zeros([int(1/h),int(1/h)])
                n_stripes = int(1/h)
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for j in range(Nx):
                    stripe_id = min(int(j / Ny * n_stripes), n_stripes - 1)
                    K[j, :] = stripe_vals[stripe_id]

            # --- vertical stripes ---
            elif TYPE == "vertical":
                K = np.zeros([int(1/h),int(1/h)])
                n_stripes = int(1/h)
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for i in range(Ny):
                    stripe_id = min(int(i / Nx * n_stripes), n_stripes - 1)
                    K[:, i] = stripe_vals[stripe_id]

            # --- callable coefficient ---
            def kappa(x, y):
                x = min(max(x, 0.0), 1.0 - 1e-14)
                y = min(max(y, 0.0), 1.0 - 1e-14)

                i = int(x * int(1/h))
                j = int(y * int(1/h))

                return K[j, i]

            # --- sample on coarse nodes ---
            a_sample = np.array([kappa(p[0], p[1]) for p in fine_nodes])
            train_coeffs_a.append(a_sample)

            A_LOD_matrix, f_LOD_vector, Q_h, u_h_fine = make_LOD_data(
                    mesh_data, adjacency, fine_in_coarse,
                    kappa, 
                    B_H, C_h, f_h, P_h
            )

            train_matrices.append(A_LOD_matrix)
            train_load_vectors.append(f_LOD_vector)
            #train_Q.append(Q_h)
            train_u_h_fine.append(u_h_fine)

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
            rng = np.random.default_rng()
            a_sample = sampler.sample(rng)
            a_sample = np.exp(a_sample)
            q_edges = np.quantile(a_sample, np.linspace(0, 1, kappa_values.shape[0]+1))
            
            kappa_shuffled = np.asarray(kappa_values, dtype=float).copy()
            rng.shuffle(kappa_shuffled)
            
            bins = np.searchsorted(q_edges[1:], a_sample, side="right")
            bins = np.clip(bins, 0, len(kappa_values)-1)
            kappa_node = kappa_shuffled[bins]
            
            kappa_elem = compute_kappa_per_element(fine_elems, kappa_node)
            kappa = make_fast_kappa_uniform(h, int(1 / h), kappa_elem)
            
            validate_coeffs_a.append(kappa_node)

            A_LOD_matrix, f_LOD_vector, Q_h, u_h_fine = make_LOD_data(
                mesh_data, adjacency, fine_in_coarse,
                kappa, B_H, C_h, f_h, P_h
            )

            validate_matrices.append(A_LOD_matrix)
            validate_load_vectors.append(f_LOD_vector)
            validate_Q.append(Q_h)
            validate_u_h_fine.append(u_h_fine)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]), f_LOD_vector, rcond=None)[0]
            validate_fenics_u.append(u_lod)
            
        if TYPE == "lognormal":
            rng = np.random.default_rng()
            
            kappa, kappa_node = make_lognormal_kappa(
                fine_nodes,
                field,
                rng
            )
            
            validate_coeffs_a.append(kappa_node)
    
            A_LOD_matrix, f_LOD_vector, Q_h, u_h_fine = make_LOD_data(
                mesh_data, adjacency, fine_in_coarse,
                kappa, B_H, C_h, f_h, P_h
            )

            validate_matrices.append(A_LOD_matrix)
            validate_load_vectors.append(f_LOD_vector)
            validate_Q.append(Q_h)
            validate_u_h_fine.append(u_h_fine)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]), f_LOD_vector, rcond=None)[0]
            validate_fenics_u.append(u_lod)
            
        elif TYPE in ["coarse_checkerboard", "fine_checkerboard", "horizontal", "vertical"]:
            rng = np.random.default_rng()
            
            # --- checkerboard ---
            if TYPE == "coarse_checkerboard":
                for j in range(Ny):
                    for i in range(Nx):
                        idx = rng.integers(0, len(kappa_values))
                        K[j, i] = kappa_values[idx]
                        
            elif TYPE == "fine_checkerboard":
                for j in range(int(1/h)):
                    for i in range(int(1/h)):
                        idx = rng.integers(0, len(kappa_values))
                        K[j, i] = kappa_values[idx]

            # --- horizontal stripes ---
            elif TYPE == "horizontal":
                n_stripes = int(1/h)
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for j in range(Nx):
                    stripe_id = min(int(j / Ny * n_stripes), n_stripes - 1)
                    K[j, :] = stripe_vals[stripe_id]

            # --- vertical stripes ---
            elif TYPE == "vertical":
                n_stripes = int(1/h)
                stripe_vals = rng.choice(kappa_values, size=n_stripes)

                for i in range(Ny):
                    stripe_id = min(int(i / Nx * n_stripes), n_stripes - 1)
                    K[:, i] = stripe_vals[stripe_id]

            # --- callable coefficient ---
            def kappa(x, y):
                x = min(max(x, 0.0), 1.0 - 1e-14)
                y = min(max(y, 0.0), 1.0 - 1e-14)

                i = int(x * int(1/h))
                j = int(y * int(1/h))

                return K[j, i]

            # --- sample on coarse nodes ---
            a_sample = np.array([kappa(p[0], p[1]) for p in fine_nodes])
            validate_coeffs_a.append(a_sample)

            A_LOD_matrix, f_LOD_vector, Q_h, u_h_fine = make_LOD_data(
                mesh_data, adjacency, fine_in_coarse,
                kappa, B_H, C_h, f_h, P_h
            )

            validate_matrices.append(A_LOD_matrix)
            validate_load_vectors.append(f_LOD_vector)
            validate_Q.append(Q_h)
            validate_u_h_fine.append(u_h_fine)

            try:
                u_lod = np.linalg.solve(A_LOD_matrix, f_LOD_vector)
            except np.linalg.LinAlgError:
                u_lod = np.linalg.lstsq(
                    A_LOD_matrix + 1e-12*np.eye(A_LOD_matrix.shape[0]),
                    f_LOD_vector,
                    rcond=None
                )[0]
            validate_fenics_u.append(u_lod)
    
    #return pos, edges, np.array(train_coeffs_a), np.array(train_matrices), np.array(train_load_vectors), np.array(train_fenics_u),  np.array(validate_coeffs_a), np.array(validate_matrices), np.array(validate_load_vectors), np.array(validate_fenics_u)
    return (
    pos,
    np.array(train_coeffs_a),
    np.array(train_matrices),
    np.array(train_load_vectors),
    np.array(train_fenics_u),
    np.array(train_Q),
    np.array(train_u_h_fine),
    np.array(validate_coeffs_a),
    np.array(validate_matrices),
    np.array(validate_load_vectors),
    np.array(validate_fenics_u),
    np.array(validate_Q),
    np.array(validate_u_h_fine),
    P_h,
    coarse_nodes,
    coarse_elems,
    fine_nodes,
    fine_elems
)


order='1'
list_num_xy=[129]
num_input=[num_train, num_val]
typ='Darcy'

epsi = 0.01
kappa_values = np.array([epsi, 4.0, 8.0, 12.0, 66.0, 100.0])

for idx, num in enumerate(list_num_xy):
    (
        p,
        train_coeffs_a,
        train_matrices,
        train_load_vectors,
        train_fenics_u,
        train_Q,
        train_u_h_fine,
        validate_coeffs_a,
        validate_matrices,
        validate_load_vectors,
        validate_fenics_u,
        validate_Q,
        validate_u_h_fine,
        P_h,
        coarse_nodes,
        coarse_elems,
        fine_nodes,
        fine_elems
    ) = create_dataset(num_input, H, h, kappa_values)


    # build filename
    base = f"data/P{order}_ne{H}_{typ}_{num_input[0]}"
    if gparams["type"] is not None:
        mesh_path = f"{base}_{gparams['type']}_GPU.npz"
    else:
        mesh_path = f"{base}.npz"

    # save with mesh_path
    # save with mesh_path
    np.savez(
        mesh_path,
        p=p,

        coarse_nodes=coarse_nodes,
        coarse_elems=coarse_elems,
        fine_nodes=fine_nodes,
        fine_elems=fine_elems,
        P_h=P_h,
        
        train_coeffs_a=train_coeffs_a,
        train_matrices=train_matrices,
        train_load_vectors=train_load_vectors,
        train_u=train_fenics_u,
        train_Q=train_Q,
        train_u_h_fine=train_u_h_fine,

        validate_coeffs_a=validate_coeffs_a,
        validate_matrices=validate_matrices,
        validate_load_vectors=validate_load_vectors,
        validate_u=validate_fenics_u,
        validate_Q=validate_Q,
        validate_u_h_fine=validate_u_h_fine
    )

    print(f"Saved data at {mesh_path} for num_xy = {num}")
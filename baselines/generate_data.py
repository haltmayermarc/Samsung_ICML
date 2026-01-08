import os
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


# -------------------------
# FVM solver 
# -------------------------

def solve_darcy_2d(coeff, F, boundary_value=None, boundary_flux=None, flux_mask=None):
    s = coeff.shape[0]
    b = F[1:-1, 1:-1].ravel()

    rows, cols, data = [], [], []

    def flatten_idx(i, j):
        return i*(s-2) + j

    scale = (s-1)**2

    if boundary_value is None:
        boundary_value = np.zeros_like(coeff)
    if flux_mask is None:
        flux_mask = np.zeros_like(coeff)

    for i in range(1, s-1):
        for j in range(1, s-1):
            idx = flatten_idx(i-1, j-1)
            a_nx = 0.5 * (coeff[i, j] + coeff[i-1, j])
            a_px = 0.5 * (coeff[i, j] + coeff[i+1, j])
            a_ny = 0.5 * (coeff[i, j] + coeff[i, j-1])
            a_py = 0.5 * (coeff[i, j] + coeff[i, j+1])

            diag = 0.0

            for (di, dj, a) in [(-1, 0, a_nx), (1, 0, a_px), (0, -1, a_ny), (0, 1, a_py)]:
                ii = i+di
                jj = j+dj
                if 0 < ii < s-1 and 0 < jj < s-1:
                    rows.append(idx)
                    cols.append(flatten_idx(ii-1, jj-1))
                    data.append(-a*scale)
                    diag += a*scale
                else:
                    if flux_mask[ii, jj] == 0:
                        b[idx] += a*scale * boundary_value[ii, jj]
                        diag += a*scale
                    else:
                        b[idx] += (s-1) * boundary_flux[ii, jj]

            rows.append(idx)
            cols.append(idx)
            data.append(diag)

    A = coo_matrix((data, (rows, cols)), shape=((s-2)**2, (s-2)**2)).tocsr()

    x, info = cg(A, b, rtol=1e-8, maxiter=50000)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")

    U = boundary_value.copy()
    U[1:-1, 1:-1] = x.reshape(s-2, s-2)
    return U


# -------------------------
# generate coeff
# -------------------------


def coarse_node_positions(num_xy):
    s = num_xy + 1
    xs = np.linspace(0.0, 1.0, s)
    ys = np.linspace(0.0, 1.0, s)
    pos = np.array([(x, y) for y in ys for x in xs], dtype=float)
    return pos

class SEGaussianSamplerSVD:
    
    def __init__(self, pos, sigma=1.0, ell=0.3, mean=1.0, tol=1e-8):
        self.pos = np.asarray(pos, dtype=float)
        self.V = self.pos.shape[0]

        D = cdist(self.pos, self.pos)
        C = (sigma**2) * np.exp(-(D**2) / (2 * ell**2))

        U, s, Vt = np.linalg.svd(C, full_matrices=False)

        if np.min(s) < -tol:
            raise ValueError(f"Covariance not PSD: min eigen/singular {np.min(s)} < -tol")
        s = np.clip(s, 0.0, None)

        self.A = U * np.sqrt(s)[None, :]
        self.mean = mean

    def sample(self, rng: np.random.Generator):
        z = rng.normal(0.0, 1.0, size=(self.V,))
        return self.mean + self.A @ z

def make_coeff_quantile(num_xy, kappa_values, rng, sampler: SEGaussianSamplerSVD, shuffle=True):

    V_dim = (num_xy + 1) ** 2
    assert sampler.V == V_dim

    a = sampler.sample(rng)
    a = np.exp(a)

    q_edges = np.quantile(a, np.linspace(0, 1, len(kappa_values)+1))
    bins = np.digitize(a, q_edges[1:-1], right=True)
    bins = np.clip(bins, 0, len(kappa_values)-1)

    values = np.asarray(kappa_values, dtype=float).copy()
    if shuffle:
        rng.shuffle(values)

    s = num_xy + 1
    coeff = values[bins].reshape(s, s)
    return coeff


def make_coeff_checkerboard(num_xy, kappa_values, rng):

    K = kappa_values[rng.integers(0, len(kappa_values), size=(num_xy, num_xy))]
    s = num_xy + 1
    coeff = np.empty((s, s), dtype=float)

    for j in range(s):
        jj = min(j, num_xy-1)  # y=1에서 last cell로 clamp
        for i in range(s):
            ii = min(i, num_xy-1)
            coeff[j, i] = K[jj, ii]
    return coeff


def make_coeff_horizontal(num_xy, kappa_values, rng, n_stripes=16):

    stripe_vals = rng.choice(kappa_values, size=n_stripes, replace=True)
    K = np.empty((num_xy, num_xy), dtype=float)
    for j in range(num_xy):
        stripe_id = min(int(j / num_xy * n_stripes), n_stripes - 1)
        K[j, :] = stripe_vals[stripe_id]

    s = num_xy + 1
    coeff = np.empty((s, s), dtype=float)
    for j in range(s):
        jj = min(j, num_xy-1)
        for i in range(s):
            ii = min(i, num_xy-1)
            coeff[j, i] = K[jj, ii]
    return coeff


def make_coeff_vertical(num_xy, kappa_values, rng, n_stripes=16):

    stripe_vals = rng.choice(kappa_values, size=n_stripes, replace=True)
    K = np.empty((num_xy, num_xy), dtype=float)
    for i in range(num_xy):
        stripe_id = min(int(i / num_xy * n_stripes), n_stripes - 1)
        K[:, i] = stripe_vals[stripe_id]

    s = num_xy + 1
    coeff = np.empty((s, s), dtype=float)
    for j in range(s):
        jj = min(j, num_xy-1)
        for i in range(s):
            ii = min(i, num_xy-1)
            coeff[j, i] = K[jj, ii]
    return coeff


def make_coeff(coeff_type, num_xy, kappa_values, rng, sampler=None):
    if coeff_type == "quantile":
        return make_coeff_quantile(num_xy, kappa_values, rng, sampler)
    elif coeff_type == "checkerboard":
        return make_coeff_checkerboard(num_xy, kappa_values, rng)
    elif coeff_type == "horizontal":
        return make_coeff_horizontal(num_xy, kappa_values, rng)
    elif coeff_type == "vertical":
        return make_coeff_vertical(num_xy, kappa_values, rng)
    else:
        raise ValueError(f"Unknown coeff_type: {coeff_type}")


# -------------------------
# Dataset Generation
# -------------------------

import time

def generate_dataset(num_train, num_val, num_xy, coeff_type, kappa_values,
                     seed_train=5, seed_val=10, out_path="data/fvm_data.npz"):
    s = num_xy + 1

    F = np.ones((s, s), dtype=float)
    boundary_value = np.zeros((s, s), dtype=float)

    train_ss = np.random.SeedSequence(seed_train)
    val_ss   = np.random.SeedSequence(seed_val)
    train_rngs = [np.random.default_rng(cs) for cs in train_ss.spawn(num_train)]
    val_rngs   = [np.random.default_rng(cs) for cs in val_ss.spawn(num_val)]

    train_coeffs = np.empty((num_train, s, s), dtype=float)
    train_u      = np.empty((num_train, s, s), dtype=float)

    val_coeffs   = np.empty((num_val, s, s), dtype=float)
    val_u        = np.empty((num_val, s, s), dtype=float)

    if coeff_type == "quantile":
        pos = coarse_node_positions(num_xy)
        sampler = SEGaussianSamplerSVD(pos, sigma=1.0, ell=0.3, mean=1.0, tol=1e-8)
    else:
        sampler = None

    for n in tqdm(range(num_train), desc=f"Train ({coeff_type})"):
        coeff = make_coeff(coeff_type, num_xy, kappa_values, train_rngs[n], sampler)
        u = solve_darcy_2d(coeff, F, boundary_value=boundary_value)

        train_coeffs[n] = coeff
        train_u[n] = u

    for n in tqdm(range(num_val), desc=f"Val ({coeff_type})"):
        coeff = make_coeff(coeff_type, num_xy, kappa_values, val_rngs[n], sampler)
        u = solve_darcy_2d(coeff, F, boundary_value=boundary_value)
        val_coeffs[n] = coeff
        val_u[n] = u

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        num_xy=num_xy,
        s=s,
        coeff_type=coeff_type,
        kappa_values=np.asarray(kappa_values),
        train_coeffs_a=train_coeffs,
        train_u=train_u,
        validate_coeffs_a=val_coeffs,
        validate_u=val_u,
    )
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True,
                        choices=["quantile", "checkerboard", "horizontal", "vertical", "hard1"])
    parser.add_argument("--num_xy", type=int, default=127)
    parser.add_argument("--ntrain", type=int, default=5000)
    parser.add_argument("--nval", type=int, default=1000)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    epsi = 0.01
    kappa_values = np.array([epsi, 4.0, 8.0, 12.0, 16.0, 20.0], dtype=float)

    if args.out is None:
        s = args.num_xy + 1
        args.out = f"data/FVM_s{args.num_xy+1}_Darcy_{args.ntrain}_{args.type}.npz"

    generate_dataset(
        num_train=args.ntrain,
        num_val=args.nval,
        num_xy=args.num_xy,
        coeff_type=args.type,
        kappa_values=kappa_values,
        seed_train=5,
        seed_val=10,
        out_path=args.out
    )

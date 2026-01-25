# ============================================================
# LOD_v4.py  (BIG JUMP: CHOLMOD (SuiteSparse) patch solves)
#
# What speeds up computeCorrections:
#   - Replaces SciPy SuperLU factorized(A_l) with CHOLMOD Cholesky
#     when available (scikit-sparse + SuiteSparse).
#
# Math:
#   A_l is SPD (Dirichlet on patch boundary via "interior nodes only"),
#   so we can use Cholesky: A_l = L L^T.
#   Solving A_l x = b via triangular solves is typically much faster
#   and more stable than general LU.
#
# Also includes v3 "fast patch selection":
#   - fine_in_coarse (structured) + node2elems_list precomputed once
#   - avoids scanning all fine elements for each patch
#
# GPU:
#   - Global LOD coarse solve and fine CG can run on GPU (CuPy).
#   - Patch correctors remain CPU (SciPy/CHOLMOD).
# ============================================================

from __future__ import annotations

import os
import numpy as np
from scipy import sparse
import scipy.linalg as la
import scipy.sparse.linalg as spla

from tqdm.auto import tqdm
from joblib import Parallel, delayed
import joblib
from contextlib import contextmanager


# ------------------------------
# Optional CHOLMOD (SuiteSparse)
# ------------------------------
_CHOLMOD_OK = False
try:
    from sksparse.cholmod import cholesky  # type: ignore
    _CHOLMOD_OK = True
except Exception:
    cholesky = None
    _CHOLMOD_OK = False


def cholmod_available() -> bool:
    return bool(_CHOLMOD_OK)


# ------------------------------
# Optional CuPy support
# ------------------------------
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cspla
    _CUPY_OK = True
except Exception:
    cp = None
    csp = None
    cspla = None
    _CUPY_OK = False


def _try_get_mem_info():
    if not _CUPY_OK:
        return None, None
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        return int(free), int(total)
    except Exception:
        return None, None


def cupy_available(min_free_bytes: int = 256 * 1024**2) -> bool:
    if not _CUPY_OK:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return False
        free, _ = _try_get_mem_info()
        if free is not None and free < min_free_bytes:
            return False
        x = cp.empty((1,), dtype=cp.float32)
        del x
        cp.get_default_memory_pool().free_all_blocks()
        return True
    except Exception:
        return False


def cupy_device_info() -> str:
    if not _CUPY_OK:
        return "CuPy not importable"
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        name = props.get("name", b"")
        if isinstance(name, (bytes, bytearray)):
            name = name.decode("utf-8", errors="ignore")
        cc = f"{props.get('major','?')}.{props.get('minor','?')}"
        mem = props.get("totalGlobalMem", 0)
        mem_gb = mem / (1024**3) if mem else 0.0
        free, total = _try_get_mem_info()
        if free is not None and total is not None:
            free_gb = free / (1024**3)
            return f"GPU[{dev.id}] {name} | CC {cc} | VRAM {mem_gb:.2f} GB | free {free_gb:.2f} GB"
        return f"GPU[{dev.id}] {name} | CC {cc} | VRAM {mem_gb:.2f} GB"
    except Exception as e:
        return f"CuPy present but device query failed: {type(e).__name__}: {e}"


def cupy_set_mempool_limit(fraction: float = 0.6, size_bytes: int | None = None) -> bool:
    if not _CUPY_OK:
        return False
    try:
        pool = cp.get_default_memory_pool()
        if size_bytes is not None:
            pool.set_limit(size=int(size_bytes))
        else:
            pool.set_limit(fraction=float(fraction))
        return True
    except Exception:
        return False


def cupy_free_all():
    if not _CUPY_OK:
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ------------------------------
# tqdm <-> joblib integration
# ------------------------------
@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


######################################################
# BUILD GRID / MESH
######################################################

def build_uniform_grid(Nx, Ny, refine):
    dx, dy = 1.0 / Nx, 1.0 / Ny
    coarse_nodes = np.array([(i*dx, j*dy) for j in range(Ny+1) for i in range(Nx+1)], dtype=float)

    def cnode(i, j):
        return j*(Nx+1) + i

    coarse_elems = []
    for j in range(Ny):
        for i in range(Nx):
            bl = cnode(i, j)
            br = cnode(i+1, j)
            tl = cnode(i, j+1)
            tr = cnode(i+1, j+1)
            coarse_elems.append((bl, br, tr, tl))
    coarse_elems = np.array(coarse_elems, dtype=int)

    Nx_fine, Ny_fine = Nx*refine, Ny*refine
    hx, hy = 1.0 / Nx_fine, 1.0 / Ny_fine
    fine_nodes = np.array([(i*hx, j*hy) for j in range(Ny_fine+1) for i in range(Nx_fine+1)], dtype=float)

    def fnode(I, J):
        return J*(Nx_fine+1) + I

    fine_elems = []
    for J in range(Ny_fine):
        for I in range(Nx_fine):
            bl = fnode(I, J)
            br = fnode(I+1, J)
            tl = fnode(I, J+1)
            tr = fnode(I+1, J+1)
            fine_elems.append((bl, br, tr, tl))
    fine_elems = np.array(fine_elems, dtype=int)

    return {
        "Nx": Nx,
        "Ny": Ny,
        "refine": refine,
        "Nx_fine": Nx_fine,
        "Ny_fine": Ny_fine,
        "coarse_nodes": coarse_nodes,
        "coarse_elems": coarse_elems,
        "fine_nodes": fine_nodes,
        "fine_elems": fine_elems,
    }


def quads_to_tris(quad_elems, diagonal="bl-tr"):
    triangles = []
    for bl, br, tr, tl in quad_elems:
        if diagonal == "bl-tr":
            triangles.append((bl, br, tr))
            triangles.append((bl, tr, tl))
        elif diagonal == "br-tl":
            triangles.append((br, tr, tl))
            triangles.append((br, tl, bl))
        else:
            raise ValueError("diagonal must be 'bl-tr' or 'br-tl'")
    return np.array(triangles, dtype=int)


def build_triangular_mesh(Nx, Ny, refine):
    mesh = build_uniform_grid(Nx, Ny, refine)
    mesh["coarse_elems"] = quads_to_tris(mesh["coarse_elems"])
    mesh["fine_elems"]   = quads_to_tris(mesh["fine_elems"])
    return mesh


######################################################
# ADJACENCY / PATCH
######################################################

def build_coarse_adjacency_edge(coarse_elems):
    edge2elems = {}
    for e, tri in enumerate(coarse_elems):
        a, b, c = tri
        edges = [tuple(sorted((a, b))),
                 tuple(sorted((b, c))),
                 tuple(sorted((c, a)))]
        for ed in edges:
            edge2elems.setdefault(ed, []).append(e)

    adjacency = [set() for _ in range(len(coarse_elems))]
    for elems in edge2elems.values():
        if len(elems) > 1:
            for e in elems:
                for ee in elems:
                    if ee != e:
                        adjacency[e].add(ee)
    return adjacency


def coarse_patch_elements(l, k, adjacency):
    patch = {l}
    frontier = {l}
    for _ in range(k):
        new_frontier = set()
        for e in frontier:
            new_frontier.update(adjacency[e])
        new_frontier -= patch
        patch |= new_frontier
        frontier = new_frontier
    return patch


def coarse_global_boundary_nodes(coarse_nodes, tol=1e-12):
    x = coarse_nodes[:, 0]
    y = coarse_nodes[:, 1]
    bdry = (np.abs(x) < tol) | (np.abs(x-1) < tol) | (np.abs(y) < tol) | (np.abs(y-1) < tol)
    return set(np.where(bdry)[0].tolist())


def coarse_nodes_in_patch(patch_elems, coarse_elems):
    nodes = set()
    for e in patch_elems:
        nodes.update(coarse_elems[e])
    return nodes


def coarse_interior_nodes_in_patch(patch_elems, coarse_elems, coarse_nodes):
    nodes_in_patch = coarse_nodes_in_patch(patch_elems, coarse_elems)
    global_boundary = coarse_global_boundary_nodes(coarse_nodes)
    return sorted(nodes_in_patch - global_boundary)


######################################################
# Structured parent coarse triangle mapping
######################################################

def parent_coarse_triangle_index(cx, cy, Nx, Ny):
    H = 1.0 / Nx
    ic = int(np.floor(cx / H))
    jc = int(np.floor(cy / H))
    ic = min(max(ic, 0), Nx-1)
    jc = min(max(jc, 0), Ny-1)
    x0 = ic * H
    y0 = jc * H
    local = 0 if (cy - y0) <= (cx - x0) else 1
    q = jc * Nx + ic
    return 2*q + local


def precompute_fine_in_coarse_structured(mesh, show_tqdm=True):
    Nx, Ny = mesh["Nx"], mesh["Ny"]
    fine_nodes  = mesh["fine_nodes"]
    fine_tris   = mesh["fine_elems"]
    coarse_tris = mesh["coarse_elems"]
    NTH = coarse_tris.shape[0]
    fine_in_coarse = [[] for _ in range(NTH)]

    it = enumerate(fine_tris)
    if show_tqdm:
        it = tqdm(it, total=fine_tris.shape[0], desc="fine_in_coarse (structured)", unit="fine-tri")

    for t, tri in it:
        c = fine_nodes[tri].mean(axis=0)
        L = parent_coarse_triangle_index(c[0], c[1], Nx, Ny)
        fine_in_coarse[L].append(t)
    return fine_in_coarse


######################################################
# node->elements adjacency (precompute ONCE)
######################################################

def build_node2elems_list(fine_elems, n_nodes: int):
    lst = [[] for _ in range(n_nodes)]
    for t, tri in enumerate(fine_elems):
        a, b, c = tri
        lst[a].append(t); lst[b].append(t); lst[c].append(t)
    return [np.asarray(v, dtype=np.int32) for v in lst]


def fine_global_boundary_mask(fine_nodes, tol=1e-12):
    x = fine_nodes[:, 0]; y = fine_nodes[:, 1]
    return (np.abs(x) < tol) | (np.abs(x-1) < tol) | (np.abs(y) < tol) | (np.abs(y-1) < tol)


def fine_interior_nodes_from_patch_fine_elems(patch_fine_elems, fine_elems, node2elems_list, fine_bdry_mask):
    if len(patch_fine_elems) == 0:
        return []

    patch_fine_elems = np.asarray(patch_fine_elems, dtype=np.int32)
    nT = fine_elems.shape[0]
    in_patch = np.zeros(nT, dtype=np.bool_)
    in_patch[patch_fine_elems] = True

    patch_nodes = np.unique(fine_elems[patch_fine_elems].ravel())
    interior = []
    for node in patch_nodes:
        if fine_bdry_mask[node]:
            continue
        adj = node2elems_list[node]
        if adj.size == 0:
            continue
        if in_patch[adj].all():
            interior.append(int(node))
    return interior


######################################################
# ASSEMBLY of stiffness/mass
######################################################

def p1_local_matrices_var_kappa_tri(xy, kappa_func):
    x = xy[:, 0]
    y = xy[:, 1]

    grad_hat = np.array([[-1.0,  1.0,  0.0],
                         [-1.0,  0.0,  1.0]])

    J = np.array([[x[1] - x[0], x[2] - x[0]],
                  [y[1] - y[0], y[2] - y[0]]])
    detJ = np.linalg.det(J)
    invJT = np.linalg.inv(J).T
    grad = invJT @ grad_hat

    quad_pts = [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)]
    quad_w = [1/6, 1/6, 1/6]

    A = np.zeros((3, 3))
    M = np.zeros((3, 3))

    for (xi, eta), w in zip(quad_pts, quad_w):
        N = np.array([1.0 - xi - eta, xi, eta])
        x_gp = N @ x
        y_gp = N @ y
        kappa_gp = kappa_func(x_gp, y_gp)

        for i in range(3):
            gi = grad[:, i]
            for j in range(3):
                A[i, j] += kappa_gp * (gi @ grad[:, j]) * detJ * w
                M[i, j] += (N[i] * N[j]) * detJ * w
    return A, M


def compute_single_element_matrices(t, elem, fine_nodes, kappa_func):
    xy = fine_nodes[elem]
    A_t, M_t = p1_local_matrices_var_kappa_tri(xy, kappa_func)
    return t, A_t, M_t


def build_fine_element_matrices(grid, kappa_func, n_jobs=-1, show_tqdm=True, backend="loky"):
    fine_nodes = grid["fine_nodes"]
    fine_elems = grid["fine_elems"]
    num_elems  = fine_elems.shape[0]

    if show_tqdm:
        with tqdm_joblib(tqdm(total=num_elems, desc="Local element matrices", unit="elem")):
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(compute_single_element_matrices)(t, elem, fine_nodes, kappa_func)
                for t, elem in enumerate(fine_elems)
            )
    else:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(compute_single_element_matrices)(t, elem, fine_nodes, kappa_func)
            for t, elem in enumerate(fine_elems)
        )

    A_loc = np.zeros((num_elems, 3, 3), dtype=np.float64)
    M_loc = np.zeros((num_elems, 3, 3), dtype=np.float64)
    for t, A_t, M_t in results:
        A_loc[t] = A_t
        M_loc[t] = M_t

    sigma = fine_elems.copy()
    return A_loc, M_loc, sigma


def assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh):
    A_loc = A_dc
    M_loc = M_dc
    rows = np.repeat(sigma, 3, axis=1).ravel()
    cols = np.tile(sigma, (1, 3)).ravel()
    A_h = sparse.coo_matrix((A_loc.reshape(-1), (rows, cols)), shape=(Nh, Nh)).tocsr()
    M_h = sparse.coo_matrix((M_loc.reshape(-1), (rows, cols)), shape=(Nh, Nh)).tocsr()
    A_h.eliminate_zeros()
    M_h.eliminate_zeros()
    return A_h, M_h


######################################################
# INTERPOLATION P_h (kept)
######################################################

def barycentric_coords(p, tri, tol=1e-12):
    A, B, C = tri
    v0 = B - A
    v1 = C - A
    v2 = p - A

    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1

    denom = d00 * d11 - d01 * d01
    if abs(denom) < tol:
        return None

    inv = 1.0 / denom
    l2 = (d11 * d20 - d01 * d21) * inv
    l3 = (d00 * d21 - d01 * d20) * inv
    l1 = 1.0 - l2 - l3

    if (l1 >= -tol) and (l2 >= -tol) and (l3 >= -tol):
        return np.array([l1, l2, l3])
    return None


def build_P_triangular(grid, show_tqdm=True):
    coarse_nodes = grid["coarse_nodes"]
    coarse_elems = grid["coarse_elems"]
    fine_nodes   = grid["fine_nodes"]

    N_H = coarse_nodes.shape[0]
    N_h = fine_nodes.shape[0]
    P_h = np.zeros((N_H, N_h))

    it = enumerate(fine_nodes)
    if show_tqdm:
        it = tqdm(it, total=N_h, desc="Building P_h", unit="fine-node")

    for i, z in it:
        for elem in coarse_elems:
            tri = coarse_nodes[elem]
            lambdas = barycentric_coords(z, tri)
            if lambdas is not None:
                for k, j in enumerate(elem):
                    P_h[j, i] = lambdas[k]
                break
        else:
            raise RuntimeError("Fine node not found in any coarse element")
    return P_h


######################################################
# BOUNDARY
######################################################

def build_B_H(coarse_nodes, Nx, Ny):
    N_H = coarse_nodes.shape[0]
    B_H = np.zeros((N_H, N_H))
    for j in range(Ny+1):
        for i in range(Nx+1):
            idx = j*(Nx+1) + i
            x, y = coarse_nodes[idx]
            B_H[idx, idx] = 1.0 if (0.0 < x < 1.0 and 0.0 < y < 1.0) else 0.0
    return B_H


def boundary_mask_fine(fine_nodes, tol=1e-12):
    x = fine_nodes[:, 0]; y = fine_nodes[:, 1]
    return (np.abs(x-0.0) < tol) | (np.abs(x-1.0) < tol) | (np.abs(y-0.0) < tol) | (np.abs(y-1.0) < tol)


######################################################
# IH quasi interpolation (same as v3)
######################################################

def triangle_areas(nodes, tris):
    A = nodes[tris[:, 0]]
    B = nodes[tris[:, 1]]
    C = nodes[tris[:, 2]]
    cross = (B[:, 0] - A[:, 0])*(C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1])*(C[:, 0] - A[:, 0])
    return 0.5 * np.abs(cross)


def build_cg2dg(tris, N_cg_nodes):
    nT = tris.shape[0]
    rows = np.arange(3*nT, dtype=int)
    cols = tris.reshape(-1)
    data = np.ones_like(rows, dtype=float)
    return sparse.coo_matrix((data, (rows, cols)), shape=(3*nT, N_cg_nodes)).tocsr()


def barycentric_coords_fast(p, tri, tol=1e-14):
    A, B, C = tri
    v0 = B - A
    v1 = C - A
    v2 = p - A
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00*d11 - d01*d01
    if abs(denom) < tol:
        return None
    inv = 1.0/denom
    l2 = (d11*d20 - d01*d21)*inv
    l3 = (d00*d21 - d01*d20)*inv
    l1 = 1.0 - l2 - l3
    return np.array([l1, l2, l3])


def build_P1dg_structured(mesh, show_tqdm=True):
    coarse_nodes = mesh["coarse_nodes"]
    coarse_tris  = mesh["coarse_elems"]
    fine_nodes   = mesh["fine_nodes"]
    fine_tris    = mesh["fine_elems"]
    Nx, Ny = mesh["Nx"], mesh["Ny"]

    NTH = coarse_tris.shape[0]
    NTh = fine_tris.shape[0]
    N_coarse_dg = 3 * NTH
    N_fine_dg   = 3 * NTh

    rows, cols, data = [], [], []
    it = range(NTh)
    if show_tqdm:
        it = tqdm(it, total=NTh, desc="Building P1dg", unit="fine-tri")

    for t in it:
        tri_f = fine_tris[t]
        pts = fine_nodes[tri_f]
        centroid = pts.mean(axis=0)
        L = parent_coarse_triangle_index(centroid[0], centroid[1], Nx, Ny)
        tri_c = coarse_nodes[coarse_tris[L]]

        for a in range(3):
            lam = barycentric_coords_fast(pts[a], tri_c)
            if lam is None:
                raise RuntimeError("barycentric failed; check mesh mapping")
            r = 3*t + a
            for b in range(3):
                rows.append(r)
                cols.append(3*L + b)
                data.append(lam[b])

    return sparse.coo_matrix((data, (rows, cols)), shape=(N_fine_dg, N_coarse_dg)).tocsr()


def build_IH_quasi_interpolation(mesh, show_tqdm=True):
    coarse_nodes = mesh["coarse_nodes"]
    coarse_tris  = mesh["coarse_elems"]
    fine_nodes   = mesh["fine_nodes"]
    fine_tris    = mesh["fine_elems"]

    N_H  = coarse_nodes.shape[0]
    N_h  = fine_nodes.shape[0]

    cg2dgh  = build_cg2dg(fine_tris,  N_h)
    cg2dgH  = build_cg2dg(coarse_tris, N_H)

    area_h = triangle_areas(fine_nodes, fine_tris)
    area_H = triangle_areas(coarse_nodes, coarse_tris)

    M_ref = np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float) / 12.0
    Minv_ref = np.array([[9,-3,-3],[-3,9,-3],[-3,-3,9]], dtype=float)

    Mhdg = sparse.kron(sparse.diags(area_h, 0, format="csr"),
                       sparse.csr_matrix(M_ref), format="csr")

    B = sparse.kron(sparse.diags(1.0/area_H, 0, format="csr"),
                    sparse.csr_matrix(Minv_ref), format="csr")

    P1dg = build_P1dg_structured(mesh, show_tqdm=show_tqdm)

    tmp = Mhdg @ cg2dgh
    tmp = P1dg.T @ tmp
    PiHdg = B @ tmp

    counts = np.array(cg2dgH.sum(axis=0)).ravel()
    Dinv = sparse.diags(1.0 / counts, 0, format="csr")
    EH = Dinv @ cg2dgH.T

    IH = EH @ PiHdg
    return IH.tocsr()


######################################################
# LOAD VECTOR
######################################################

def assemble_load_tri(fine_nodes, fine_elems, f_func, show_tqdm=True):
    N = fine_nodes.shape[0]
    b = np.zeros(N)

    quad_pts = [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)]
    quad_w = [1/6, 1/6, 1/6]

    it = fine_elems
    if show_tqdm:
        it = tqdm(fine_elems, total=fine_elems.shape[0], desc="Assembling load f_h", unit="fine-tri")

    for elem in it:
        xy = fine_nodes[elem]
        x = xy[:, 0]; y = xy[:, 1]
        J = np.array([[x[1] - x[0], x[2] - x[0]],
                      [y[1] - y[0], y[2] - y[0]]])
        detJ = abs(np.linalg.det(J))

        for (xi, eta), w in zip(quad_pts, quad_w):
            Nvals = np.array([1.0 - xi - eta, xi, eta])
            x_gp = Nvals @ x
            y_gp = Nvals @ y
            f_val = f_func(x_gp, y_gp)
            for i_local, i_global in enumerate(elem):
                b[i_global] += f_val * Nvals[i_local] * detJ * w
    return b


######################################################
# CORRECTIONS Q_h (BIG JUMP: CHOLMOD)
######################################################

def assemble_r_l_elementwise(mesh, l, dofph, fine_in_coarse_l, kappa_func):
    coarse_nodes = mesh["coarse_nodes"]
    coarse_elems = mesh["coarse_elems"]
    fine_nodes   = mesh["fine_nodes"]
    fine_elems   = mesh["fine_elems"]

    triK = coarse_nodes[coarse_elems[l]]
    g2loc = {g: i for i, g in enumerate(dofph)}
    r_l = np.zeros((3, len(dofph)), dtype=float)

    for t in fine_in_coarse_l:
        elem = fine_elems[t]
        xy   = fine_nodes[elem]
        A_t, _ = p1_local_matrices_var_kappa_tri(xy, kappa_func)

        lam_nodes = np.zeros((3, 3))
        for a in range(3):
            lam = barycentric_coords(fine_nodes[elem[a]], triK)
            if lam is None:
                raise RuntimeError("fine node not inside its parent coarse triangle.")
            lam_nodes[a, :] = lam

        for i in range(3):
            v = lam_nodes[:, i]
            loc = -A_t @ v
            for a in range(3):
                g = elem[a]
                j = g2loc.get(g, None)
                if j is not None:
                    r_l[i, j] += loc[a]
    return r_l


def _make_spd(A: sparse.spmatrix) -> sparse.csc_matrix:
    """
    Ensure symmetry for CHOLMOD.
    If A is nearly symmetric, this is cheap-ish.
    """
    A = A.tocsc()
    return 0.5 * (A + A.T)


def _factorize_patch(A_l: sparse.csc_matrix, method: str = "auto"):
    """
    Returns solve_A(rhs) that solves A_l x = rhs for vector or matrix rhs.

    method:
      - "auto": use CHOLMOD if available else SciPy
      - "cholmod": force CHOLMOD
      - "scipy": force SciPy factorized
    """
    method = str(method).lower()
    if method not in ("auto", "cholmod", "scipy"):
        raise ValueError("method must be 'auto'|'cholmod'|'scipy'")

    if method in ("auto", "cholmod") and _CHOLMOD_OK:
        # CHOLMOD expects float64 and symmetric SPD
        if A_l.dtype != np.float64:
            A_l = A_l.astype(np.float64)
        F = cholesky(A_l)  # factorization
        def solve_A(rhs):
            return F.solve_A(rhs)
        return solve_A, "cholmod"

    # SciPy fallback
    solve_A = spla.factorized(A_l)
    return solve_A, "scipy"


def process_single_element_fast(
    l, mesh, k, adjacency, A_h, C_h, fine_in_coarse,
    node2elems_list, fine_bdry_mask, kappa_func, B_H,
    patch_solver: str = "auto", symmetrize_for_cholmod: bool = True
):
    coarse_elems = mesh["coarse_elems"]
    coarse_nodes = mesh["coarse_nodes"]
    fine_elems   = mesh["fine_elems"]

    patch_elems = coarse_patch_elements(l, k, adjacency)

    dofpH = coarse_interior_nodes_in_patch(patch_elems, coarse_elems, coarse_nodes)
    if len(dofpH) == 0:
        return None

    patch_fine_elems = []
    for ee in patch_elems:
        patch_fine_elems.extend(fine_in_coarse[ee])

    dofph = fine_interior_nodes_from_patch_fine_elems(
        patch_fine_elems, fine_elems, node2elems_list, fine_bdry_mask
    )
    if len(dofph) == 0:
        return None

    # Patch stiffness
    A_l = A_h[dofph, :][:, dofph].tocsc()

    # If using CHOLMOD, enforce symmetry (safe)
    if symmetrize_for_cholmod and (patch_solver.lower() in ("auto", "cholmod")) and _CHOLMOD_OK:
        A_l = _make_spd(A_l)

    solve_A, used = _factorize_patch(A_l, method=patch_solver)

    # Constraints
    C_l = C_h[dofpH, :][:, dofph].toarray()

    r_l = assemble_r_l_elementwise(mesh, l, dofph, fine_in_coarse[l], kappa_func)

    # Z = A^{-1} C^T,  S = C Z = C A^{-1} C^T
    Z = solve_A(C_l.T)
    S = C_l @ Z
    S = 0.5 * (S + S.T)
    lu, piv = la.lu_factor(S + 1e-14 * np.eye(S.shape[0]))

    element_updates = []
    for i in range(3):
        q = solve_A(r_l[i])
        mu = la.lu_solve((lu, piv), C_l @ q)
        phi = q - Z @ mu
        global_coarse_idx = coarse_elems[l][i]
        if B_H[global_coarse_idx, global_coarse_idx] != 0.0:
            element_updates.append((global_coarse_idx, dofph, phi))
    return element_updates


def computeCorrections(
    mesh, k, adjacency, fine_in_coarse, A_h, B_H, C_h, kappa_func,
    n_jobs=-1, show_tqdm=True, backend="loky",
    patch_solver: str = "auto", symmetrize_for_cholmod: bool = True
):
    """
    BIG JUMP option:
      patch_solver="auto"  -> CHOLMOD if available else SciPy
      patch_solver="cholmod" -> force CHOLMOD (error if not installed)
      patch_solver="scipy" -> force SciPy
    """
    N_H = mesh["coarse_nodes"].shape[0]
    N_h = mesh["fine_nodes"].shape[0]
    NTH = mesh["coarse_elems"].shape[0]

    A_h = sparse.csr_matrix(A_h)
    fine_bdry_mask = fine_global_boundary_mask(mesh["fine_nodes"])
    node2elems_list = build_node2elems_list(mesh["fine_elems"], n_nodes=N_h)

    if show_tqdm:
        used_txt = "cholmod" if (patch_solver in ("auto", "cholmod") and _CHOLMOD_OK) else "scipy"
        print(f"[computeCorrections] patch_solver={patch_solver} (available cholmod={_CHOLMOD_OK}) -> using {used_txt} if possible")
        with tqdm_joblib(tqdm(total=NTH, desc=f"Corrections (k={k})", unit="coarse-tri")):
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(process_single_element_fast)(
                    l, mesh, k, adjacency, A_h, C_h, fine_in_coarse,
                    node2elems_list, fine_bdry_mask, kappa_func, B_H,
                    patch_solver=patch_solver, symmetrize_for_cholmod=symmetrize_for_cholmod
                )
                for l in range(NTH)
            )
    else:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(process_single_element_fast)(
                l, mesh, k, adjacency, A_h, C_h, fine_in_coarse,
                node2elems_list, fine_bdry_mask, kappa_func, B_H,
                patch_solver=patch_solver, symmetrize_for_cholmod=symmetrize_for_cholmod
            )
            for l in range(NTH)
        )

    Q_h = np.zeros((N_H, N_h), dtype=float)
    it = results
    if show_tqdm:
        it = tqdm(results, total=len(results), desc="Assembling Q_h", unit="coarse-tri")
    for element_res in it:
        if element_res is not None:
            for global_idx, dof_indices, values in element_res:
                Q_h[global_idx, dof_indices] += values
    return Q_h


######################################################
# SOLVERS (GPU if possible)
######################################################

def solveLODSystem(A_h, f_h, P_h, Q_h, B_H, use_gpu=True, gpu_dtype="float32", block_cols=16):
    V_ms = P_h + Q_h
    interior = np.where(np.diag(B_H) > 0.5)[0]

    if use_gpu and cupy_available() and sparse.issparse(A_h):
        dtype = cp.float32 if str(gpu_dtype).lower() in ["float32", "fp32"] else cp.float64
        A_gpu = csp.csr_matrix(A_h).astype(dtype)
        V_gpu = cp.asarray(V_ms, dtype=dtype)
        f_gpu = cp.asarray(f_h, dtype=dtype)

        nH = V_gpu.shape[0]
        A_H = cp.zeros((nH, nH), dtype=dtype)

        for j0 in range(0, nH, int(block_cols)):
            j1 = min(nH, j0 + int(block_cols))
            Vt_blk = V_gpu.T[:, j0:j1]
            tmp = A_gpu @ Vt_blk
            A_H[:, j0:j1] = V_gpu @ tmp

        f_H = V_gpu @ f_gpu

        idx = cp.asarray(interior, dtype=cp.int32)
        A_ii = A_H[cp.ix_(idx, idx)]
        f_i  = f_H[idx]

        u_i = cp.linalg.solve(A_ii, f_i)
        u_H = cp.zeros((A_H.shape[0],), dtype=dtype)
        u_H[idx] = u_i
        u_h = V_gpu.T @ u_H

        cp.cuda.Stream.null.synchronize()
        cupy_free_all()
        return cp.asnumpy(u_h), cp.asnumpy(u_H)

    # CPU
    if sparse.issparse(A_h):
        tmp = A_h @ V_ms.T
        A = V_ms @ tmp
    else:
        A = V_ms @ A_h @ V_ms.T
    f = V_ms @ f_h

    A_ii = A[np.ix_(interior, interior)]
    f_i  = f[interior]
    u_i = np.linalg.solve(A_ii, f_i)

    u_H = np.zeros(A.shape[0])
    u_H[interior] = u_i
    u_h = V_ms.T @ u_H
    return u_h, u_H


def solve_fine_problem(A_h, f_h, fine_nodes, use_gpu=True, tol=1e-10, maxiter=20000, gpu_dtype="float32"):
    N = fine_nodes.shape[0]
    bdry = boundary_mask_fine(fine_nodes)
    interior = np.where(~bdry)[0]

    if use_gpu and cupy_available() and sparse.issparse(A_h):
        dtype = cp.float32 if str(gpu_dtype).lower() in ["float32", "fp32"] else cp.float64
        A_ii = A_h[interior, :][:, interior].tocsr()
        b_i  = f_h[interior]

        A_gpu = csp.csr_matrix(A_ii).astype(dtype)
        b_gpu = cp.asarray(b_i, dtype=dtype)

        u_i, info = cspla.cg(A_gpu, b_gpu, tol=tol, maxiter=maxiter)
        if info != 0:
            print(f"[warn] GPU CG did not fully converge. info={info}")

        u = np.zeros((N,), dtype=np.float64)
        u[interior] = cp.asnumpy(u_i)
        cp.cuda.Stream.null.synchronize()
        cupy_free_all()
        return u

    # CPU sparse solve
    if sparse.issparse(A_h):
        A_ii = A_h[interior, :][:, interior].tocsr()
        b_i  = f_h[interior]
        u_i = spla.spsolve(A_ii, b_i)
        u = np.zeros((N,), dtype=np.float64)
        u[interior] = u_i
        return u

    # Dense legacy
    A = A_h.copy()
    f = f_h.copy()
    for i in range(N):
        if bdry[i]:
            A[i, :] = 0.0
            A[:, i] = 0.0
            A[i, i] = 1.0
            f[i] = 0.0
    return np.linalg.solve(A, f)


def relative_L2_error(u, v, M):
    diff = u - v
    num = diff @ (M @ diff)
    den = v    @ (M @ v)
    return float(np.sqrt(num / den))

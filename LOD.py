from joblib import Parallel, delayed
import numpy as np
from scipy import sparse
import scipy.linalg as la
import scipy.sparse.linalg as spla
from tqdm import tqdm

######################################################
### BUILD GRID
######################################################

def build_uniform_grid(Nx, Ny, refine):
    """
    Build uniform 2D meshes:
      - coarse: Nx x Ny elements, nodes (Nx+1)*(Ny+1)
      - fine  : (Nx*refine) x (Ny*refine) elements, nodes (Nx*refine+1)*(Ny*refine+1)
    Node ordering is row-major: node_id = j*(n+1) + i.
    Element connectivity: (bottom-left, bottom-right, top-right, top-left).
    """
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
        # bottom-left to top-right
        if diagonal == "bl-tr": 
            triangles.append((bl, br, tr))
            triangles.append((bl, tr, tl))
        # bottom-right to top-left
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

def elems_to_coo(elems, symmetric=True, unique=True):
    """
    Build graph connectivity (node adjacency) from element connectivity.

    Parameters
    ----------
    elems : (n_elems, c_d) ndarray
        Element connectivity (e.g. triangles: c_d = 3)
    symmetric : bool
        If True, include both (i,j) and (j,i)
    unique : bool
        If True, remove duplicate edges

    Returns
    -------
    row : ndarray
    col : ndarray
        COO representation of graph edges
    """
    rows = []
    cols = []

    for elem in elems:
        for i in range(len(elem)):
            for j in range(i + 1, len(elem)):
                rows.append(elem[i])
                cols.append(elem[j])
                if symmetric:
                    rows.append(elem[j])
                    cols.append(elem[i])

    row = np.array(rows, dtype=int)
    col = np.array(cols, dtype=int)

    if unique:
        edges = np.stack([row, col], axis=1)
        edges = np.unique(edges, axis=0)
        row, col = edges[:, 0], edges[:, 1]

    return row, col

######################################################
### PATCHING
######################################################

def build_coarse_adjacency(coarse_elems):
    """
    Two elements are neighbors if they share at least one node.
    """
    node_to_elems = {}
    for e, elem in enumerate(coarse_elems):
        for n in elem:
            node_to_elems.setdefault(n, []).append(e)

    adjacency = [set() for _ in range(len(coarse_elems))]
    for elems in node_to_elems.values():
        for e in elems:
            adjacency[e].update(elems)

    # remove self
    for e in range(len(adjacency)):
        adjacency[e].discard(e)

    return adjacency

def build_coarse_adjacency_edge(coarse_elems):
    # coarse_elems: (Nelem, 3)
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
    """
    Returns a set of coarse element indices in U_k(K_l).
    """
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

def point_in_triangle(p, tri, tol=1e-12):
    A, B, C = tri

    v0 = C - A
    v1 = B - A
    v2 = p - A

    dot00 = v0 @ v0
    dot01 = v0 @ v1
    dot02 = v0 @ v2
    dot11 = v1 @ v1
    dot12 = v1 @ v2

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < tol:
        return False  # degenerate triangle

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv

    return (u >= -tol) and (v >= -tol) and (u + v <= 1 + tol)#

def coarse_global_boundary_nodes(coarse_nodes, tol=1e-12):
    boundary = set()
    for i, (x, y) in enumerate(coarse_nodes):
        if (
            abs(x) < tol or abs(x-1) < tol or
            abs(y) < tol or abs(y-1) < tol
        ):
            boundary.add(i)
    return boundary

def coarse_nodes_in_patch(patch_elems, coarse_elems):
    nodes = set()
    for e in patch_elems:
        nodes.update(coarse_elems[e])
    return nodes

def coarse_interior_nodes_in_patch(
    patch_elems, coarse_elems, coarse_nodes
):
    nodes_in_patch = coarse_nodes_in_patch(patch_elems, coarse_elems)
    global_boundary = coarse_global_boundary_nodes(coarse_nodes)

    interior = sorted(nodes_in_patch - global_boundary)
    return interior

def fine_elements_in_patch(patch_elems, coarse_nodes, coarse_elems, fine_nodes, fine_elems):
    patch_tris = [coarse_nodes[coarse_elems[e]] for e in patch_elems]

    fine_elem_ids = set()
    for t, elem in enumerate(fine_elems):
        tri = fine_nodes[elem]
        centroid = tri.mean(axis=0)
        for K in patch_tris:
            if point_in_triangle(centroid, K):
                fine_elem_ids.add(t)
                break

    return fine_elem_ids

def fine_nodes_from_fine_elements(fine_elem_ids, fine_elems):
    nodes = set()
    for t in fine_elem_ids:
        nodes.update(fine_elems[t])
    return nodes

def fine_node_to_elements(fine_elems):
    node2elems = {}
    for t, elem in enumerate(fine_elems):
        for n in elem:
            node2elems.setdefault(n, set()).add(t)
    return node2elems

def fine_global_boundary_nodes(fine_nodes, tol=1e-12):
    boundary = set()
    for i, (x, y) in enumerate(fine_nodes):
        if (
            abs(x) < tol or abs(x-1) < tol or
            abs(y) < tol or abs(y-1) < tol
        ):
            boundary.add(i)
    return boundary

def fine_interior_nodes_in_patch(fine_elem_ids, fine_elems, fine_nodes):
    node2elems = fine_node_to_elements(fine_elems)

    interior = set()
    boundary = set()

    patch_elems = set(fine_elem_ids)

    for node, elems in node2elems.items():
        if elems & patch_elems:
            if elems <= patch_elems:
                interior.add(node)
            else:
                boundary.add(node)
                
    global_bdry = fine_global_boundary_nodes(fine_nodes)
    bad = interior & global_bdry

    interior -= bad
    boundary |= bad

    return interior, boundary


def fine_nodes_in_patch(
    patch_elems, coarse_nodes, coarse_elems, fine_nodes, fine_elems
):
    fine_elem_ids = fine_elements_in_patch(
        patch_elems, coarse_nodes, coarse_elems,
        fine_nodes, fine_elems
    )

    interior, boundary = fine_interior_nodes_in_patch(
        fine_elem_ids, fine_elems, fine_nodes
    )

    return interior, boundary

def precompute_fine_in_coarse(mesh):
    coarse_nodes = mesh["coarse_nodes"]
    coarse_elems = mesh["coarse_elems"]  # (NTH,3)
    fine_nodes   = mesh["fine_nodes"]
    fine_elems   = mesh["fine_elems"]    # (NTh,3)

    NTH = coarse_elems.shape[0]
    fine_in_coarse = [[] for _ in range(NTH)]

    coarse_tris = [coarse_nodes[coarse_elems[L]] for L in range(NTH)]

    for t, elem in enumerate(fine_elems):
        c = fine_nodes[elem].mean(axis=0)  # centroid
        found = False
        for L in range(NTH):
            lam = barycentric_coords(c, coarse_tris[L])
            if lam is not None:
                fine_in_coarse[L].append(t)
                found = True
                break
        if not found:
            raise RuntimeError("fine element centroid not found in any coarse element.")
    return fine_in_coarse

######################################################
### ASSEMBLY OF STIFFNESS AND MASS MATRICES
######################################################

def p1_local_matrices_var_kappa_tri(xy, kappa_func):
    x = xy[:, 0]
    y = xy[:, 1]

    # --- Reference shape functions ---
    # N1 = 1 - xi - eta
    # N2 = xi
    # N3 = eta

    # Gradients in reference coordinates (constant)
    grad_hat = np.array([
        [-1.0,  1.0,  0.0],
        [-1.0,  0.0,  1.0],
    ])  # (2,3)

    # --- Jacobian ---
    J = np.array([
        [x[1] - x[0], x[2] - x[0]],
        [y[1] - y[0], y[2] - y[0]],
    ])
    detJ = np.linalg.det(J)
    invJT = np.linalg.inv(J).T

    grad = invJT @ grad_hat  # (2,3), constant over element

    # --- Quadrature (degree 2 exact) ---
    # 3-point symmetric rule
    quad_pts = [
        (1/6, 1/6),
        (2/3, 1/6),
        (1/6, 2/3),
    ]
    quad_w = [1/6, 1/6, 1/6]

    A = np.zeros((3, 3))
    M = np.zeros((3, 3))

    for (xi, eta), w in zip(quad_pts, quad_w):
        N = np.array([
            1.0 - xi - eta,
            xi,
            eta,
        ])

        # Physical coordinates
        x_gp = N @ x
        y_gp = N @ y

        kappa_gp = kappa_func(x_gp, y_gp)

        for i in range(3):
            for j in range(3):
                A[i, j] += kappa_gp * (grad[:, i] @ grad[:, j]) * detJ * w
                M[i, j] += (N[i] * N[j]) * detJ * w

    return A, M

def compute_single_element_matrices(t, elem, fine_nodes, kappa_func):
    """Worker function to compute local matrices for one element."""
    xy = fine_nodes[elem]
    A_t, M_t = p1_local_matrices_var_kappa_tri(xy, kappa_func)
    return t, A_t, M_t

def build_fine_element_matrices(grid, kappa_func, n_jobs=-1):
    fine_nodes = grid["fine_nodes"]
    fine_elems = grid["fine_elems"]
    num_elems  = fine_elems.shape[0]
    c_d = 3 # Number of vertices in a triangle

    # Run the local computations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_element_matrices)(t, elem, fine_nodes, kappa_func)
        for t, elem in enumerate(fine_elems)
    )

    # Prepare data for sparse matrix construction (COO format)
    # Each element produces c_d*c_d entries
    rows = []
    cols = []
    data_A = []
    data_M = []

    for t, A_t, M_t in results:
        # Calculate the block offset in the DC matrix
        offset = t * c_d
        # Create local indices for the 3x3 block
        for i in range(c_d):
            for j in range(c_d):
                rows.append(offset + i)
                cols.append(offset + j)
                data_A.append(A_t[i, j])
                data_M.append(M_t[i, j])

    # Construct the global Block Diagonal matrices
    size = num_elems * c_d
    A_dc = sparse.csr_matrix((data_A, (rows, cols)), shape=(size, size))
    M_dc = sparse.csr_matrix((data_M, (rows, cols)), shape=(size, size))
    
    sigma = fine_elems.copy()

    return A_dc, M_dc, sigma


def assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh):
    num_elems = sigma.shape[0]
    c_d = sigma.shape[1]

    A_h = np.zeros((Nh, Nh))
    M_h = np.zeros((Nh, Nh))
    for t in range(num_elems):
        elem_dofs = sigma[t]
        i0 = t * c_d
        i1 = (t + 1) * c_d
        A_t = A_dc[i0:i1, i0:i1]
        M_t = M_dc[i0:i1, i0:i1]
        for i_local, i_global in enumerate(elem_dofs):
            for j_local, j_global in enumerate(elem_dofs):
                A_h[i_global, j_global] += A_t[i_local, j_local]
                M_h[i_global, j_global] += M_t[i_local, j_local]
    return A_h, M_h

######################################################
### COMPUTE INTERPOLATION MATRIX P_h
######################################################

def barycentric_coords(p, tri, tol=1e-12):
    """
    Compute barycentric coordinates of point p w.r.t. triangle tri.
    Returns None if p is outside.
    """
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

def build_P_triangular(grid):
    coarse_nodes = grid["coarse_nodes"]
    coarse_elems = grid["coarse_elems"]   # (Nc, 3)
    fine_nodes   = grid["fine_nodes"]

    N_H = coarse_nodes.shape[0]
    N_h = fine_nodes.shape[0]

    P_h = np.zeros((N_H, N_h))

    for i, z in enumerate(fine_nodes):
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
### BOUNDARY HANDLING
######################################################

def build_B_H(coarse_nodes, Nx, Ny):
    """
    Coarse boundary matrix B_H:
      - 0 on Dirichlet boundary nodes (∂Ω)
      - 1 on interior nodes
    """
    N_H = coarse_nodes.shape[0]
    B_H = np.zeros((N_H, N_H))
    for j in range(Ny+1):
        for i in range(Nx+1):
            idx = j*(Nx+1) + i
            x,y = coarse_nodes[idx]
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                B_H[idx, idx] = 1.0
            else:
                B_H[idx, idx] = 0.0
    return B_H

def boundary_mask_fine(fine_nodes, tol=1e-12):
    x = fine_nodes[:,0]; y = fine_nodes[:,1]
    return (np.abs(x-0.0) < tol) | (np.abs(x-1.0) < tol) | (np.abs(y-0.0) < tol) | (np.abs(y-1.0) < tol)


######################################################
### ADDITIONAL CODE FROM MALQUIST AND PETERSEIM
######################################################


def triangle_areas(nodes, tris):
    A = nodes[tris[:, 0]]
    B = nodes[tris[:, 1]]
    C = nodes[tris[:, 2]]
    cross = (B[:,0]-A[:,0])*(C[:,1]-A[:,1]) - (B[:,1]-A[:,1])*(C[:,0]-A[:,0])
    return 0.5 * np.abs(cross)

def build_cg2dg(tris, N_cg_nodes):
    """
    Continuous CG1 dofs (global nodes) -> Discontinuous P1(dg) dofs (elem-local nodes)
    cg2dg : (N_dg, N_cg)
    row = 3*t + a  (dg dof), col = tris[t,a] (global cg node)
    """
    nT = tris.shape[0]
    rows = np.arange(3*nT, dtype=int)
    cols = tris.reshape(-1)
    data = np.ones_like(rows, dtype=float)
    return sparse.coo_matrix((data, (rows, cols)), shape=(3*nT, N_cg_nodes)).tocsr()

def parent_coarse_triangle_index(cx, cy, Nx, Ny):
    """
    uniform coarse square cell -> one of its two triangles (diagonal bl-tr)
    coarse quad index q = jc*Nx + ic
    coarse tri index  = 2*q + local(0 or 1)
      local=0: (bl, br, tr)  (below diagonal)
      local=1: (bl, tr, tl)  (above diagonal)
    """
    H = 1.0 / Nx
    ic = int(np.floor(cx / H))
    jc = int(np.floor(cy / H))
    ic = min(max(ic, 0), Nx-1)
    jc = min(max(jc, 0), Ny-1)

    x0 = ic * H
    y0 = jc * H

    # diagonal: y = y0 + (x - x0)
    local = 0 if (cy - y0) <= (cx - x0) else 1
    q = jc * Nx + ic
    return 2*q + local

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

def build_P1dg_structured(mesh):
    """
    Prolongation from coarse DG(P1) -> fine DG(P1)
    P1dg: (N_fine_dg, N_coarse_dg)
    """
    coarse_nodes = mesh["coarse_nodes"]
    coarse_tris  = mesh["coarse_elems"]   # (NTH,3)
    fine_nodes   = mesh["fine_nodes"]
    fine_tris    = mesh["fine_elems"]     # (NTh,3)
    Nx, Ny = mesh["Nx"], mesh["Ny"]

    NTH = coarse_tris.shape[0]
    NTh = fine_tris.shape[0]
    N_coarse_dg = 3 * NTH
    N_fine_dg   = 3 * NTh

    rows, cols, data = [], [], []

    # for each fine element and each of its local dg-dofs, evaluate coarse barycentric weights
    for t in range(NTh):
        tri_f = fine_tris[t]
        pts = fine_nodes[tri_f]  # (3,2)

        centroid = pts.mean(axis=0)
        L = parent_coarse_triangle_index(centroid[0], centroid[1], Nx, Ny)
        tri_c_nodes = coarse_tris[L]
        tri_c = coarse_nodes[tri_c_nodes]  # (3,2)

        for a in range(3):
            p = pts[a]
            lam = barycentric_coords_fast(p, tri_c)
            if lam is None:
                raise RuntimeError("barycentric failed; check mesh mapping")

            r = 3*t + a  # fine dg dof row
            # coarse dg columns: 3*L + b (b=0,1,2)
            for b in range(3):
                rows.append(r)
                cols.append(3*L + b)
                data.append(lam[b])

    return sparse.coo_matrix((data, (rows, cols)), shape=(N_fine_dg, N_coarse_dg)).tocsr()

def build_IH_quasi_interpolation(mesh):
    """
    Build IH = EH * PiHdg (book-style)
    IH: (N_H, N_h) mapping fine CG nodal values -> coarse CG nodal values
    """
    coarse_nodes = mesh["coarse_nodes"]
    coarse_tris  = mesh["coarse_elems"]
    fine_nodes   = mesh["fine_nodes"]
    fine_tris    = mesh["fine_elems"]

    N_H  = coarse_nodes.shape[0]
    N_h  = fine_nodes.shape[0]
    NTH  = coarse_tris.shape[0]
    NTh  = fine_tris.shape[0]

    # embeddings CG -> DG
    cg2dgh  = build_cg2dg(fine_tris,  N_h)   # (3*NTh, N_h)
    cg2dgH  = build_cg2dg(coarse_tris, N_H)  # (3*NTH, N_H)

    # DG mass matrices (block diagonal)
    area_h = triangle_areas(fine_nodes, fine_tris)
    area_H = triangle_areas(coarse_nodes, coarse_tris)

    M_ref = np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float) / 12.0
    Minv_ref = np.array([[9,-3,-3],[-3,9,-3],[-3,-3,9]], dtype=float)

    Mhdg = sparse.kron(sparse.diags(area_h, 0, format="csr"),
                       sparse.csr_matrix(M_ref), format="csr")

    B = sparse.kron(sparse.diags(1.0/area_H, 0, format="csr"),
                    sparse.csr_matrix(Minv_ref), format="csr")  # inverse coarse DG mass

    # prolongation coarse DG -> fine DG
    P1dg = build_P1dg_structured(mesh)  # (3*NTh, 3*NTH)

    # PiHdg = B * (P1dg^T * Mhdg * cg2dgh)
    tmp = Mhdg @ cg2dgh                 # (3*NTh, N_h)
    tmp = P1dg.T @ tmp                  # (3*NTH, N_h)
    PiHdg = B @ tmp                     # (3*NTH, N_h)

    # EH = nodal averaging: (N_H x 3*NTH)
    counts = np.array(cg2dgH.sum(axis=0)).ravel()  # how many dg dofs per coarse node
    Dinv = sparse.diags(1.0 / counts, 0, format="csr")
    EH = Dinv @ cg2dgH.T                # (N_H, 3*NTH)

    IH = EH @ PiHdg                     # (N_H, N_h)
    return IH.tocsr()

######################################################
### ASSEMBLE FORCING TERM LOAD VECTOR
######################################################

def assemble_load_tri(fine_nodes, fine_elems, f_func):
    """
    Assemble global load vector b for P1 triangular elements:
        b_i = ∫_Ω f(x,y) φ_i(x,y) dx
    """
    N = fine_nodes.shape[0]
    b = np.zeros(N)

    # Reference triangle quadrature (degree 2 exact)
    quad_pts = [
        (1/6, 1/6),
        (2/3, 1/6),
        (1/6, 2/3),
    ]
    quad_w = [1/6, 1/6, 1/6]

    for elem in fine_elems:
        nodes = elem
        xy = fine_nodes[nodes]  # (3,2)
        x = xy[:, 0]
        y = xy[:, 1]

        # Jacobian
        J = np.array([
            [x[1] - x[0], x[2] - x[0]],
            [y[1] - y[0], y[2] - y[0]],
        ])
        detJ = abs(np.linalg.det(J))

        for (xi, eta), w in zip(quad_pts, quad_w):
            # P1 basis on reference triangle
            Nvals = np.array([
                1.0 - xi - eta,
                xi,
                eta,
            ])

            # Physical coordinates
            x_gp = Nvals @ x
            y_gp = Nvals @ y

            f_val = f_func(x_gp, y_gp)

            for i_local, i_global in enumerate(nodes):
                b[i_global] += f_val * Nvals[i_local] * detJ * w

    return b


######################################################
### COMPUTE CORRECTIONS Q_h
######################################################

def assemble_r_l_elementwise(mesh, l, dofph, fine_in_coarse_l, kappa_func):
    coarse_nodes = mesh["coarse_nodes"]
    coarse_elems = mesh["coarse_elems"]
    fine_nodes   = mesh["fine_nodes"]
    fine_elems   = mesh["fine_elems"]

    triK = coarse_nodes[coarse_elems[l]]  # (3,2) coarse triangle vertices in correct order

    # global fine node -> local index in dofph
    g2loc = {g:i for i,g in enumerate(dofph)}

    r_l = np.zeros((3, len(dofph)), dtype=float)  # 3 local coarse vertices

    for t in fine_in_coarse_l:
        elem = fine_elems[t]           # (3,) global fine node ids
        xy   = fine_nodes[elem]        # (3,2)

        # fine element stiffness A_t (3x3)
        A_t, _ = p1_local_matrices_var_kappa_tri(xy, kappa_func)

        # coarse barycentric values of each fine node wrt triK
        lam_nodes = np.zeros((3,3))
        for a in range(3):
            lam = barycentric_coords(fine_nodes[elem[a]], triK)
            if lam is None:
                raise RuntimeError("fine node not inside its parent coarse triangle (unexpected).")
            lam_nodes[a,:] = lam

        # for each local coarse basis i, values at fine element nodes are v = lam_nodes[:,i]
        for i in range(3):
            v = lam_nodes[:, i]          # (3,)
            loc = -A_t @ v               # (3,)  => -a_{K_l}(Phi_i, phi_fine_node_a)

            # add to patch dofs only
            for a in range(3):
                g = elem[a]
                j = g2loc.get(g, None)
                if j is not None:
                    r_l[i, j] += loc[a]

    return r_l


def process_single_element(l, mesh, k, adjacency, A_h, C_h, fine_in_coarse, kappa_func, B_H):
    """Function containing the logic for a single coarse element l."""
    coarse_elems = mesh["coarse_elems"]
    coarse_nodes = mesh["coarse_nodes"]
    fine_nodes   = mesh["fine_nodes"]
    fine_elems   = mesh["fine_elems"]

    patch_elems = coarse_patch_elements(l, k, adjacency)
    dofpH = coarse_interior_nodes_in_patch(patch_elems, coarse_elems, coarse_nodes)
    if len(dofpH) == 0:
        return None

    dofph_set, _ = fine_nodes_in_patch(patch_elems, coarse_nodes, coarse_elems, fine_nodes, fine_elems)
    dofph = sorted(dofph_set)
    if len(dofph) == 0:
        return None

    # Local stiffness and constraints
    A_l = A_h[dofph, :][:, dofph].tocsc()
    solve_A = spla.factorized(A_l)
    C_l = C_h[dofpH, :][:, dofph].toarray()

    r_l = assemble_r_l_elementwise(mesh, l, dofph, fine_in_coarse[l], kappa_func)

    Z = solve_A(C_l.T)
    S = C_l @ Z
    S = 0.5 * (S + S.T)
    lu, piv = la.lu_factor(S + 1e-14 * np.eye(S.shape[0]))

    # Collect updates for this element
    element_updates = []
    for i in range(3):
        q = solve_A(r_l[i])
        mu = la.lu_solve((lu, piv), C_l @ q)
        phi = q - Z @ mu
        
        global_coarse_idx = coarse_elems[l][i]
        if B_H[global_coarse_idx, global_coarse_idx] != 0.0:
            element_updates.append((global_coarse_idx, dofph, phi))
            
    return element_updates

def computeCorrections(mesh, k, adjacency, fine_in_coarse, A_h, B_H, C_h, kappa_func, n_jobs=-1):
    N_H = mesh["coarse_nodes"].shape[0]
    N_h = mesh["fine_nodes"].shape[0]
    NTH = mesh["coarse_elems"].shape[0]
    
    A_h = sparse.csr_matrix(A_h)

    # Execute the loop in parallel
    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_element)(l, mesh, k, adjacency, A_h, C_h, fine_in_coarse, kappa_func, B_H)
        for l in range(NTH)
    )

    # Assembly phase (sequential, but fast)
    Q_h = np.zeros((N_H, N_h), dtype=float)
    for element_res in results:
        if element_res is not None:
            for global_idx, dof_indices, values in element_res:
                Q_h[global_idx, dof_indices] += values

    return Q_h

######################################################
### SOLVE FINE AND LOD SYSTEM
######################################################

def solveLODSystem(A_h, f_h, P_h, Q_h, B_H):
    V_ms = P_h + Q_h  # (N_H x N_h)

    A = V_ms @ A_h @ V_ms.T   # (N_H x N_H)
    f = V_ms @ f_h            # (N_H,)

    interior = np.where(np.diag(B_H) > 0.5)[0]  # interior coarse node indices
    A_ii = A[np.ix_(interior, interior)]
    f_i  = f[interior]

    u_i = np.linalg.solve(A_ii, f_i)

    u_H = np.zeros(A.shape[0])
    u_H[interior] = u_i

    u_h = V_ms.T @ u_H
    return u_h, u_H


def solve_fine_problem(A_h, f_h, fine_nodes):
    N = fine_nodes.shape[0]
    bdry = boundary_mask_fine(fine_nodes)

    A = A_h.copy()
    f = f_h.copy()

    # enforce Dirichlet BC strongly
    for i in range(N):
        if bdry[i]:
            A[i, :] = 0.0
            A[:, i] = 0.0
            A[i, i] = 1.0
            f[i] = 0.0

    u = np.linalg.solve(A, f)
    return u
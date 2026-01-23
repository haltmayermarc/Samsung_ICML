import numpy as np
from tqdm import tqdm
import argparse
from scipy.spatial.distance import cdist
from scipy import sparse
from joblib import Parallel, delayed
from pathlib import Path
from time import time 
from scipy.special import kv, gamma
from scipy.interpolate import LinearNDInterpolator


parser = argparse.ArgumentParser("SEM")
parser.add_argument("--type", type=str, choices=['quantile', 'lognormal', 'coarse_checkerboard', 'fine_checkerboard', 'horizontal', 'vertical'])
parser.add_argument("--h", type=int, default=7)
args = parser.parse_args()
gparams = args.__dict__
d
TYPE = gparams["type"]
h = 2**(-gparams["h"])

class SEGaussianSamplerSVD:
    def __init__(self, pos, sigma=1.0, ell=0.3, mean=1.0, tol=1e-8):
        self.pos = np.asarray(pos, dtype=float)
        self.V = self.pos.shape[0]

        # Pairwise distances
        D = cdist(self.pos, self.pos)

        # Covariance matrix for GRF
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

# -----

def build_fine_uniform_grid(Nx, Ny):
    dx, dy = 1.0 / Nx, 1.0 / Ny
    fine_nodes = np.array([(i*dx, j*dy) for j in range(Ny+1) for i in range(Nx+1)], dtype=float)
    def cnode(i, j): 
        return j*(Nx+1) + i

    fine_elems = []
    for j in range(Ny):
        for i in range(Nx):
            bl = cnode(i, j)
            br = cnode(i+1, j)
            tl = cnode(i, j+1)
            tr = cnode(i+1, j+1)
            fine_elems.append((bl, br, tr, tl))
    fine_elems = np.array(fine_elems, dtype=int)

    return {
        "Nx": Nx,
        "Ny": Ny,
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

def build_fine_triangular_mesh(Nx, Ny):
    mesh = build_fine_uniform_grid(Nx, Ny)
    mesh["fine_elems"] = quads_to_tris(mesh["fine_elems"])
    return mesh

# -----

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

# -----

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
        pts = np.column_stack([x, y])
        return interpolator(pts)

    return kappa_func, A

# -----

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

    # # Run the local computations in parallel
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(compute_single_element_matrices)(t, elem, fine_nodes, kappa_func)
    #     for t, elem in enumerate(fine_elems)
    # )

    results = [
        compute_single_element_matrices(t, elem, fine_nodes, kappa_func)
        for t, elem in enumerate(fine_elems)
    ]

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

# ------

def boundary_mask_fine(fine_nodes, tol=1e-12):
    x = fine_nodes[:,0]; y = fine_nodes[:,1]
    return (np.abs(x-0.0) < tol) | (np.abs(x-1.0) < tol) | (np.abs(y-0.0) < tol) | (np.abs(y-1.0) < tol)

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

# -----

def create_dataset(num_input, h, kappa_values):
    # Create fine mesh
    Nx = int(1 / h)
    Ny = Nx

    mesh_data = build_fine_triangular_mesh(Nx, Ny)

    fine_nodes   = mesh_data["fine_nodes"]
    fine_elems   = mesh_data["fine_elems"]

    Nh = fine_nodes.shape[0]
    
    # GRF parameters
    sigma = 1.0     # variance
    ell = 0.3       # correlation length
    
    # Construct SE Kernel sampler for GRF
    if TYPE == "quantile":
        sampler = SEGaussianSamplerSVD(fine_nodes, sigma=sigma, ell=ell, mean=1.0, tol=1e-8)
    elif TYPE == "lognormal":
        field = MaternGaussianField(
                fine_nodes,
                sigma=1.0,
                nu=0.5,
                kappa=0.1,
                mean=0.0,
            )
    
    
    # Fine forcing term rhs vector
    print("Building f_h...")
    f_h = assemble_load_tri(
        fine_nodes, fine_elems, lambda x, y: 1.0
    )
    
    # Generate training and validation data
    train_coeffs_a = []
    train_fenics_u = []
    
    validate_coeffs_a = []
    validate_fenics_u = []
    
    
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

            A_dc, M_dc, sigma = build_fine_element_matrices(mesh_data, lambda x, y: kappa(x,y))
            A_h, _ = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)

            u = solve_fine_problem(A_h, f_h, fine_nodes)
            train_fenics_u.append(u)

        if TYPE == "lognormal":
            rng = np.random.default_rng()
            
            kappa, kappa_node = make_lognormal_kappa(
                fine_nodes,
                field,
                rng
            )
            
            train_coeffs_a.append(kappa_node)

            A_dc, M_dc, sigma = build_fine_element_matrices(mesh_data, lambda x, y: kappa(x,y))
            A_h, _ = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)

            u = solve_fine_problem(A_h, f_h, fine_nodes)
            train_fenics_u.append(u)
            
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

                i = int(x * Ny)
                j = int(y * Nx)

                return K[j, i]

            # --- sample on fine nodes ---
            a_sample = np.array([kappa(p[0], p[1]) for p in fine_nodes])
            train_coeffs_a.append(a_sample)

            A_dc, M_dc, sigma = build_fine_element_matrices(mesh_data, lambda x, y: kappa(x,y))
            A_h, _ = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)
            
            u = solve_fine_problem(A_h, f_h, fine_nodes)
            train_fenics_u.append(u)

            
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

            A_dc, M_dc, sigma = build_fine_element_matrices(mesh_data, lambda x, y: kappa(x,y))
            A_h, _ = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)

            u = solve_fine_problem(A_h, f_h, fine_nodes)
            validate_fenics_u.append(u)
            
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

                i = int(x * Ny)
                j = int(y * Nx)

                return K[j, i]

            # --- sample on fine nodes ---
            a_sample = np.array([kappa(p[0], p[1]) for p in fine_nodes])
            validate_coeffs_a.append(a_sample)

            A_dc, M_dc, sigma = build_fine_element_matrices(mesh_data, lambda x, y: kappa(x,y))
            A_h, _ = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)

            u = solve_fine_problem(A_h, f_h, fine_nodes)
            validate_fenics_u.append(u)
    
    return (
        np.array(train_coeffs_a),
        np.array(train_fenics_u),
        np.array(validate_coeffs_a),
        np.array(validate_fenics_u),
    )

order='1'
num_input=[5000, 1000]
typ='Darcy'

epsi = 0.01
kappa_values = np.array([epsi, 4.0, 8.0, 12.0, 16.0, 20.0])

(
    train_coeffs_a,
    train_fenics_u,
    validate_coeffs_a,
    validate_fenics_u,
) = create_dataset(num_input, h, kappa_values)


# build filename
base = f"data/P{order}_nolod_ne{int(1/h)}_{typ}_{num_input[0]}"
Path("data").mkdir(parents=True, exist_ok=True)

if gparams["type"] is not None:
    mesh_path = f"{base}_{gparams['type']}.npz"
else:
    mesh_path = f"{base}.npz"

# save with mesh_path
np.savez(
    mesh_path,
    train_coeffs_a=train_coeffs_a,
    train_u=train_fenics_u,
    validate_coeffs_a=validate_coeffs_a,
    validate_u=validate_fenics_u
)

print(f"Saved data at {mesh_path} for num_xy = {int(1/h) - 1}")
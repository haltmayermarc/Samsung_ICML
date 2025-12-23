import numpy as np
from dolfin import Point

def quantize_scalar(value, q_edges, kappa_values):
    idx = np.searchsorted(q_edges[1:], value, side="right")
    return kappa_values[min(idx, len(kappa_values)-1)]

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
    
def elems_to_coo(elems):
    edges = []

    for (bl, br, tr, tl) in elems:
        edges.extend([
            (bl, br),
            (br, tr),
            (tr, tl),
            (tl, bl),
        ])

    edges = np.array(edges, dtype=int)

    # make undirected: add reversed edges
    edges_rev = edges[:, [1, 0]]
    edges = np.vstack([edges, edges_rev])

    # remove duplicates
    edges = np.unique(edges, axis=0)

    row = edges[:, 0]
    col = edges[:, 1]

    return row, col
    
# Matrix assembly

def q1_local_matrices_var_kappa_3x3(xy, kappa_func):
    """
    Q1 local matrices with variable kappa(x,y) using 3x3 Gauss quadrature.
    Exact for kappa up to degree 5 in each direction.
    """
    x = xy[:, 0]
    y = xy[:, 1]

    # 3x3 Gauss-Legendre on [-1,1]
    sqrt3_5 = np.sqrt(3.0/5.0)
    xi_pts = [-sqrt3_5, 0.0, sqrt3_5]
    w1 = 5.0/9.0
    w2 = 8.0/9.0
    weights = [w1, w2, w1]

    # Tensor product: 9 points
    gauss_pts = [(xi, eta) for eta in xi_pts for xi in xi_pts]
    gauss_w   = [wi * wj for wj in weights for wi in weights]

    A = np.zeros((4, 4))
    M = np.zeros((4, 4))

    hx = abs(x[1] - x[0])
    hy = abs(y[3] - y[0])

    J = np.array([[hx / 2.0, 0.0],
                  [0.0, hy / 2.0]])
    detJ = np.linalg.det(J)
    invJT = np.linalg.inv(J).T

    for (xi, eta), w in zip(gauss_pts, gauss_w):
        # === Shape functions N_i(xi, eta) ===
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),  # bl
            0.25 * (1 + xi) * (1 - eta),  # br
            0.25 * (1 + xi) * (1 + eta),  # tr
            0.25 * (1 - xi) * (1 + eta),  # tl
        ])

        # === Gradients in reference space ===
        dN_dxi = np.array([
            -0.25 * (1 - eta),
             0.25 * (1 - eta),
             0.25 * (1 + eta),
            -0.25 * (1 + eta),
        ])
        dN_deta = np.array([
            -0.25 * (1 - xi),
            -0.25 * (1 + xi),
             0.25 * (1 + xi),
             0.25 * (1 - xi),
        ])
        grad_hat = np.vstack([dN_dxi, dN_deta])
        grad = invJT @ grad_hat  # (2,4)
        
        # === Physical (x,y) at Gauss point ===
        x_gp = N @ x  # = N[0]*x[0] + N[1]*x[1] + ...
        y_gp = N @ y
        

        # === Evaluate kappa(x,y) ===
        kappa_gp = kappa_func(x_gp, y_gp)
        
        # === Quadrature contribution ===
        for i in range(4):
            for j in range(4):
                A[i, j] += kappa_gp * (grad[:, i] @ grad[:, j]) * detJ * w
                M[i, j] += (N[i] * N[j]) * detJ * w

    return A, M

def build_fine_element_matrices_var_kappa_3x3(grid, kappa_func):
    """
    Block-diagonal A_dc, M_dc using 3x3 quadrature for variable kappa.
    """
    fine_nodes = grid["fine_nodes"]
    fine_elems = grid["fine_elems"]
    num_elems  = fine_elems.shape[0]
    c_d = 4

    A_dc = np.zeros((num_elems * c_d, num_elems * c_d))
    M_dc = np.zeros((num_elems * c_d, num_elems * c_d))
    sigma = fine_elems.copy()  # (num_elems, 4)

    for t, elem in enumerate(fine_elems):
        xy = fine_nodes[elem]
        A_t, M_t = q1_local_matrices_var_kappa_3x3(xy, kappa_func)
        i0 = t * c_d
        i1 = i0 + c_d
        A_dc[i0:i1, i0:i1] = A_t
        M_dc[i0:i1, i0:i1] = M_t

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

def build_patch_mappings(grid):
    """
    For each coarse element K_ell build:
      - R_h_list[elt]: fine node indices in K_ell
      - R_H_list[elt]: coarse node indices of K_ell
      - T_H_list[elt]: same as coarse nodes of K_ell (element-local coarse DOFs)
      - fine_elems_in_coarse[elt]: fine element indices inside K_ell
    """
    coarse_nodes = grid["coarse_nodes"]
    coarse_elems = grid["coarse_elems"]
    fine_nodes   = grid["fine_nodes"]
    fine_elems   = grid["fine_elems"]

    NTH = coarse_elems.shape[0]
    R_h_list = []
    R_H_list = []
    T_H_list = []
    fine_elems_in_coarse = []

    for ell, c_elem in enumerate(coarse_elems):
        T_H_list.append(list(c_elem))
        coords = coarse_nodes[c_elem]
        x_min, y_min = coords[:,0].min(), coords[:,1].min()
        x_max, y_max = coords[:,0].max(), coords[:,1].max()

        fine_idx = [
            i for i, (x,y) in enumerate(fine_nodes)
            if (x_min - 1e-12) <= x <= (x_max + 1e-12) and
               (y_min - 1e-12) <= y <= (y_max + 1e-12)
        ]
        R_h_list.append(fine_idx)
        R_H_list.append(list(c_elem))

        fe_list = []
        for t, f_elem in enumerate(fine_elems):
            xy = fine_nodes[f_elem]
            cx = xy[:,0].mean(); cy = xy[:,1].mean()
            if (x_min - 1e-12) <= cx < (x_max + 1e-12) and (y_min - 1e-12) <= cy < (y_max + 1e-12):
                fe_list.append(t)
        fine_elems_in_coarse.append(fe_list)

    return R_h_list, R_H_list, T_H_list, fine_elems_in_coarse

def build_P_quad_unique(Nx, Ny, refine):
    """
    P: (N_h x N_H), P[:,j] = coarse basis φ_j sampled at fine nodes
    """
    Nx_f, Ny_f = Nx*refine, Ny*refine
    N_h = (Nx_f+1)*(Ny_f+1)
    N_H = (Nx+1)*(Ny+1)
    P = np.zeros((N_h, N_H))
    def cnode(i,j): return j*(Nx+1) + i
    def fnode(I,J): return J*(Nx_f+1) + I
    for J in range(Ny_f+1):
        j = min(Ny-1, J // refine)
        t = (J - j*refine) / refine
        for I in range(Nx_f+1):
            i = min(Nx-1, I // refine)
            s = (I - i*refine) / refine
            bl = cnode(i,   j  )
            br = cnode(i+1, j  )
            tr = cnode(i+1, j+1)
            tl = cnode(i,   j+1)
            fn = fnode(I, J)
            P[fn, bl] = (1 - s)*(1 - t)
            P[fn, br] = s*(1 - t)
            P[fn, tr] = s*t
            P[fn, tl] = (1 - s)*t
    return P

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

def assemble_load_quad(fine_nodes, fine_elems, f_func):
    N = fine_nodes.shape[0]
    b = np.zeros(N)
    gp = 1.0/np.sqrt(3.0)
    g1 = 0.5*(1 - gp)
    g2 = 0.5*(1 + gp)
    gauss = [(g1, g1, 0.25), (g2, g1, 0.25), (g1, g2, 0.25), (g2, g2, 0.25)]
    def N_bl(s,t): return (1-s)*(1-t)
    def N_br(s,t): return s*(1-t)
    def N_tr(s,t): return s*t
    def N_tl(s,t): return (1-s)*t
    for (bl, br, tr, tl) in fine_elems:
        x_bl,y_bl = fine_nodes[bl]; x_tr,y_tr = fine_nodes[tr]
        width = x_tr - x_bl; height = y_tr - y_bl
        nodes = [bl, br, tr, tl]
        Nf = [N_bl, N_br, N_tr, N_tl]
        for (s,t,w) in gauss:
            x = x_bl + s*width; y = y_bl + t*height
            f_val = f_func(x, y)
            for p in range(4):
                b[nodes[p]] += f_val * Nf[p](s,t) * w * width * height
    return b

def computeCorrections_algorithm(Nh, NH, NTH,
                                  A_dc, M_dc, sigma,
                                  B_H, P_h,
                                  R_h_list, R_H_list, T_H_list, fine_elems_in_coarse):
    """
    NumPy implementation of Algorithm 1 (Engwer et al.):

      1. Assemble global A_h, M_h from A_dc, M_dc, sigma.
      2. C_h = P_h M_h  (global constraint matrix).
      3. Initialize Q_h = 0.
      4. For each coarse element K_ell:
         - Restrict to patch U_ell: A_ell, C_ell.
         - Assemble local load r_ell using A_dc and P_h (Eq. (16)).
         - Solve local saddle problem by Schur complement:
           A_ell w + C_ell^T λ = r_ell,  C_ell w = 0.
         - Accumulate w_ell into Q_h.

    Q_h is returned as shape (N_H, N_h), each row = fine corrector for one coarse basis.
    """
    # 1. Assemble global fine matrices
    A_h, M_h = assemble_global_from_Adc_Mdc(A_dc, M_dc, sigma, Nh)

    # 2. Global constraint matrix C_h = P_h M_h  (N_H x N_h)
    C_h = P_h @ M_h

    # 3. Initialize global corrector Q_h
    Q_h = np.zeros((NH, Nh))

    num_elems = sigma.shape[0]
    c_d = sigma.shape[1]

    # 4. Loop over coarse elements (patches)
    for ell in range(NTH):
        fine_idx_patch   = R_h_list[ell]
        coarse_idx_patch = R_H_list[ell]
        elem_coarse_idx  = T_H_list[ell]
        fine_elems       = fine_elems_in_coarse[ell]

        N_ell_h   = len(fine_idx_patch)
        N_ell_H   = len(coarse_idx_patch)
        c_d_local = len(elem_coarse_idx)
        if N_ell_h == 0:
            continue

        # map fine global index -> local index in patch
        fine_local_pos = {g: i for i, g in enumerate(fine_idx_patch)}

        # 4(a). Local matrices
        A_ell = A_h[np.ix_(fine_idx_patch, fine_idx_patch)]         # (N_ell_h x N_ell_h)
        C_ell = C_h[np.ix_(coarse_idx_patch, fine_idx_patch)]       # (N_ell_H x N_ell_h)

        # 4(b). Assemble local load matrix r_ell (c_d_local x N_ell_h)
        r_ell = np.zeros((c_d_local, N_ell_h))
        for i_loc, coarse_global in enumerate(elem_coarse_idx):
            # skip Dirichlet coarse nodes via B_H
            if B_H[coarse_global, coarse_global] == 0.0:
                continue
            # fine elements inside K_ell
            for t in fine_elems:
                elem_dofs = sigma[t]             # 4 global fine nodes of element t
                i0 = t * c_d
                i1 = (t + 1) * c_d
                A_t = A_dc[i0:i1, i0:i1]        # local 4x4 stiffness
                # coarse basis values at this element's nodes
                phi_vals = P_h[coarse_global, elem_dofs]   # shape (4,)
                # local_contrib[j] = sum_m phi_vals[m] * A_t[m,j]
                local_contrib = phi_vals @ A_t            # shape (4,)
                # accumulate into r_ell at patch indices
                for j_loc, fine_global in enumerate(elem_dofs):
                    if fine_global in fine_local_pos:
                        j_patch = fine_local_pos[fine_global]
                        r_ell[i_loc, j_patch] -= local_contrib[j_loc]

        # 4(c). Solve local saddle-point via Schur complement

        # Invert A_ell (or pseudo-inverse if ill-conditioned)
        try:
            A_ell_inv = np.linalg.inv(A_ell)
        except np.linalg.LinAlgError:
            A_ell_inv = np.linalg.pinv(A_ell)

        # Y_ell = A_ell^{-1} C_ell^T (store as rows)
        Y_ell = np.zeros((N_ell_H, N_ell_h))
        for m in range(N_ell_H):
            rhs = C_ell[m, :].T
            Y_ell[m, :] = A_ell_inv @ rhs

        # Schur complement S_ell = C_ell Y_ell^T
        S_ell = C_ell @ Y_ell.T
        try:
            S_ell_inv = np.linalg.inv(S_ell + 1e-10*np.eye(N_ell_H))
        except np.linalg.LinAlgError:
            S_ell_inv = np.linalg.pinv(S_ell)

        # Solve for each coarse basis on K_ell
        w_ell = np.zeros((c_d_local, N_ell_h))
        for i_loc in range(c_d_local):
            if np.allclose(r_ell[i_loc], 0.0):
                continue
            q = A_ell_inv @ r_ell[i_loc].T           # unconstrained
            lam = S_ell_inv @ (C_ell @ q)            # Lagrange multiplier
            w = q - Y_ell.T @ lam                    # constrained corrector
            w_ell[i_loc, :] = w

        # 4(d). Accumulate local correctors into global Q_h
        for i_loc, coarse_global in enumerate(elem_coarse_idx):
            Q_h[coarse_global, fine_idx_patch] += w_ell[i_loc, :]

    return Q_h

def reconstruct_full_grid(u_free, free_mask, Nx_fine, Ny_fine):
    N_full = (Nx_fine+1)*(Ny_fine+1)
    u_full = np.zeros(N_full)
    free_idx = np.where(free_mask)[0]
    u_full[free_idx] = u_free
    return u_full.reshape((Ny_fine+1, Nx_fine+1))

def get_LOD_matrix_rhs(A_free, f_free, P_free, Q_free, coarse_interior_mask):
    R_full = P_free + Q_free.T
    coarse_idx = np.where(coarse_interior_mask)[0]
    R  = R_full[:, coarse_idx]

    # LOD coarse system
    A_LOD = R.T @ A_free @ R
    f_LOD = R.T @ f_free
    
    return A_LOD, f_LOD

def solve_lod(A_free, f_free, P_free, Q_free, coarse_interior_mask):
    R_full = P_free + Q_free.T
    coarse_idx = np.where(coarse_interior_mask)[0]
    R  = R_full[:, coarse_idx]

    # LOD coarse system
    A_lod = R.T @ A_free @ R
    f_lod = R.T @ f_free
    try:
        u_lod = np.linalg.solve(A_lod, f_lod)
    except np.linalg.LinAlgError:
        u_lod = np.linalg.lstsq(A_lod + 1e-12*np.eye(A_lod.shape[0]), f_lod, rcond=None)[0]
        
    return  A_lod, f_lod, u_lod
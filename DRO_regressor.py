import numpy as np
from scipy.optimize import linprog
import highspy
from typing import Tuple, List, Dict
import scipy.sparse as sp
from scipy.sparse import csr_matrix

def check_LP_solution(B, c, A, x_full, b_rhs, tol=1e-9):
    """
    Check primal + dual feasibility for:
        min c^T x
        s.t. A x = b, x >= 0
    """

    A_csr = csr_matrix(A)
    m, n = A_csr.shape

    B = np.asarray(B, dtype=int)
    sorted_basic = np.sort(B)

    # ---------- Basic index validity ----------
    if sorted_basic[0] < 0 or len(sorted_basic) != m:
        print("Negative basic index or wrong basic size")
        return False

    # ---------- Build A_B ----------
    A_B = A_csr[:, sorted_basic].toarray()
    if A_B.shape != (m, m):
        print("Wrong A_B shape")
        return False

    # ---------- Solve primal basic system ----------
    try:
        x_B = np.linalg.solve(A_B, b_rhs)
    except np.linalg.LinAlgError:
        print("A_B is singular")
        return False

    # Compare with solver output
    if np.max(np.abs(x_B - x_full[sorted_basic])) > tol:
        print("x_B does not match solver output")
        print(np.max(np.abs(x_B - x_full[sorted_basic])))
        return False

    # Full primal feasibility
    if np.min(x_full) < -tol:
        print("x_full has negative entries")
        return False
    if np.linalg.norm(A_csr @ x_full - b_rhs, ord=np.inf) > tol:
        print("A x != b")
        return False

    # ---------- Dual variables ----------
    c_B = c[sorted_basic]

    try:
        y = np.linalg.solve(A_B.T, c_B)
    except np.linalg.LinAlgError:
        print("A_B^T is singular")
        return False

    return True

def construct_linf_A_v_c(X, y, idx_map, epsilon=0.5, l=1.0):
    N, d = X.shape

    # variable: [β_plus (d), β_minus (d), a_new (1), b (N), s_1 (d), s_2 (d), s_3_1 (N), s_3_2 (N)]
    n_vars = (2 * d + 1 + N) + (1 + 2 * N)    # total: 3d + 2N + 2    

    idx_bp = idx_map["bp"]
    idx_bm = idx_map["bm"]
    idx_a = idx_map["a"]
    idx_b = idx_map["b"]
    idx_s1 = idx_map["s1"]
    idx_s2_1 = idx_map["s2_1"]
    idx_s2_2 = idx_map["s2_2"]

    # Objective function: ε * a_old = ε * (a_new + 1) = ε * a_new + ε
    c = np.zeros(n_vars)
    c[idx_a] = epsilon          # coefficient for a_new
    c[idx_b] = 1.0 / N          # coefficient for b_i
    # Constant ε does not affect optimal solution
                    
    # A_eq x = b_eq
    rows = []
    rhs = []

    # (1) |Beta|_1 + 1 <= a -> |Beta|_1 - a + 1 + s_1 = 0 -> sum(B_j_plus + B_j_minus) - a + s_1 = -1
    row = np.zeros(n_vars)
    row[np.vstack((idx_bp, idx_bm))] = 1.0
    row[idx_a] = -1.0
    row[idx_s1] = 1.0
    rows.append(row)
    rhs.append(-1.0)

    # (2) |y_i - x_i^T β| <= b_i
    for i in range(N):
        # y_i - x_i^T β <= b_i => -x_i^T β - b_i + s_3_1[i] = -y_i
        row1 = np.zeros(n_vars)
        row1[idx_bp] = -X[i]
        row1[idx_bm] = X[i]
        row1[idx_b[i]] = -1.0
        row1[idx_s2_1[i]] = 1.0
        rows.append(row1)
        rhs.append(-y[i][0])

    for i in range(N):
        # -(y_i - x_i^T β) <= b_i => x_i^T β - b_i + s_3_2[i] = y_i
        row2 = np.zeros(n_vars)
        row2[idx_bp] = X[i]
        row2[idx_bm] = -X[i]
        row2[idx_b[i]] = -1.0
        row2[idx_s2_2[i]] = 1.0
        rows.append(row2)
        rhs.append(y[i][0])

    A_eq = np.vstack(rows)
    b_eq = np.array(rhs)
    return A_eq, b_eq, c

def DRO_linf_linprog(X, y, epsilon=1.0, l=None, rs=True):

    N, d = X.shape

    # Variable: [β_plus (d), β_minus (d), a_new (1), b (N), s_1 (d), s_2 (d), s_3_1 (N), s_3_2 (N)]
    n_vars = (2 * d + 1 + N) + (1 + 2 * N)  # Tổng: 2d + 3N + 2

    idx_map = {
        "bp": np.arange(0, d),
        "bm": np.arange(d, 2 * d),
        "a": 2 * d,
        "b": np.arange(2 * d + 1, 2 * d + 1 + N),
        "s1": 2 * d + 1 + N,
        "s2_1": np.arange(2 * d + 2 + N, 2 * d + 2 * N + 2),
        "s2_2": np.arange(2 * d + 2 * N + 2, 2 * d + 3 * N + 2)
    }
    idx_bp = idx_map["bp"]
    idx_bm = idx_map["bm"]

    A_eq, b_eq, c = construct_linf_A_v_c(X, y, idx_map, epsilon=epsilon, l=l)

    # bounds: (β_plus, β_minus, a_new, b_i, slack) >= 0
    bounds = [(0, None)] * n_vars

    # solve LP 
    if rs:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex', options={'maxiter': 100000})
    else:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='simplex', options={'maxiter': 100000})

    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")

    beta_hat = res.x[idx_bp] - res.x[idx_bm]
    B = res["basis"]
    B.sort()
    return beta_hat, B, A_eq, b_eq, res.x, c

def DRO_linf_highspy(X: np.ndarray, y: np.ndarray, epsilon: float = 0.5) \
-> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Solve the DRO problem with Linf Wasserstein norm using HiGHS solver.
    """
    N, d = X.shape
    inf = highspy.kHighsInf

    # === variable index ===
    idx_bp   = np.arange(0, d)                                          # β_plus
    idx_bm   = np.arange(d, 2 * d)                                      # β_minus
    idx_a    = 2 * d                                                    # a_new
    idx_b    = np.arange(2 * d + 1, 2 * d + 1 + N)                      # b_i
    idx_s1   = 2 * d + 1 + N                                            # s1
    idx_s2_1 = np.arange(2 * d + 2 + N, 2 * d + 2 + N + N)              # s3_1 (N)
    idx_s2_2 = np.arange(2 * d + 2 + N + N, 2 * d + 2 + N + 2 * N)      # s3_2 (N)

    n_vars = 2 * d + 1 + N + 1 + 2 * N 
    n_cons = 1 + 2 * N

    # === construct A (col-wise sparse: CSC) ===
    rows_coo = []
    cols_coo = []
    data_coo = []

    # sum(β_plus + β_minus) - a + s1 = -1
    for j in range(d):
        rows_coo.extend([0, 0])  # Cùng row 0
        cols_coo.extend([idx_bp[j], idx_bm[j]])
        data_coo.extend([1.0, 1.0])
    rows_coo.extend([0, 0])
    cols_coo.extend([idx_a, idx_s1])
    data_coo.extend([-1.0, 1.0])

    # Ràng buộc 2: |y_i - x_i^T β| <= b_i → 2 ràng buộc mỗi i
    for i in range(N):
        row_upper = 2 * i + 1  # upper: y_i - x_i^T β <= b_i + s3_1[i]
        row_lower = 2 * i + 2  # lower: x_i^T β - y_i <= b_i + s3_2[i]

        # upper: -X[i] * β_plus + X[i] * β_minus - b_i + s3_1 = -y_i
        for j in range(d):
            rows_coo.extend([row_upper, row_upper])
            cols_coo.extend([idx_bp[j], idx_bm[j]])
            data_coo.extend([-X[i, j], X[i, j]])
        rows_coo.extend([row_upper, row_upper])
        cols_coo.extend([idx_b[i], idx_s2_1[i]])
        data_coo.extend([-1.0, 1.0])

        # lower: X[i] * β_plus - X[i] * β_minus - b_i + s3_2 = y_i
        for j in range(d):
            rows_coo.extend([row_lower, row_lower])
            cols_coo.extend([idx_bp[j], idx_bm[j]])
            data_coo.extend([X[i, j], -X[i, j]])
        rows_coo.extend([row_lower, row_lower])
        cols_coo.extend([idx_b[i], idx_s2_2[i]])
        data_coo.extend([-1.0, 1.0])

    A_coo = sp.coo_matrix((data_coo, (rows_coo, cols_coo)), shape=(n_cons, n_vars)).tocsr()
    A_csc = A_coo.tocsc()  # Col-wise cho HiGHS

    # === col_cost ===
    col_cost = np.zeros(n_vars)
    col_cost[idx_a] = epsilon
    col_cost[idx_b] = 1.0 / N

    # === Bounds ===
    col_lower = np.zeros(n_vars)
    col_upper = np.full(n_vars, inf)

    row_lower = np.zeros(n_cons)
    row_upper = np.zeros(n_cons)
    row_lower[0] = row_upper[0] = -1.0  
    for i in range(N):
        row_lower[2 * i + 1] = row_upper[2 * i + 1] = -y[i][0] 
        row_lower[2 * i + 2] = row_upper[2 * i + 2] = y[i][0]

    # === HighsLp ===
    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_cons
    lp.col_cost_ = col_cost
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper

    # matrix A (col-wise)
    lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = A_csc.data

    h = highspy.Highs()
    
    h.setOptionValue("presolve", "off")

    h.setOptionValue("primal_feasibility_tolerance", 1e-10)
    h.setOptionValue("dual_feasibility_tolerance", 1e-10)
    h.setOptionValue("primal_residual_tolerance", 1e-10)
    h.setOptionValue("dual_residual_tolerance", 1e-10)
    h.setOptionValue("optimality_tolerance", 1e-10)

    h.setOptionValue("solver", "simplex")
    h.setOptionValue("simplex_strategy", 4)
    h.setOptionValue("simplex_scale_strategy", 4)
    h.setOptionValue("simplex_dual_edge_weight_strategy", 0)
    h.setOptionValue("simplex_primal_edge_weight_strategy", 0)
    h.setOptionValue("run_crossover", "off")

    h.setOptionValue("log_to_console", False)
    h.setOptionValue("output_flag", False)

    h.passModel(lp)  # Chỉ truyền lp object (1 tham số)

    # === solve ===
    h.run()

    # === check status ===
    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS failed: {status}")

    # === get solution ===
    sol = h.getSolution()
    x = np.array(sol.col_value)

    # === get basis ===
    basic_vars = h.getBasicVariables()[1]

    # === beta ===
    beta_hat = x[idx_bp] - x[idx_bm]

    idx_map = {
        "bp": idx_bp, "bm": idx_bm, "a": idx_a, "b": idx_b,
        "s1": idx_s1, "s2_1": idx_s2_1, "s2_2": idx_s2_2
    }
    
    A_eq_dense, v, c = construct_linf_A_v_c(X, y, idx_map)
    res = {
        "beta_hat": beta_hat,
        "basic_vars": basic_vars,
        "A": A_eq_dense,
        "v": v,
        "h": h,
        "x": x,
        "c": c
    }
    return res

def construct_l1_A_v_c(X, y, idx_map, epsilon=0.5, l=1.0):
    N, d = X.shape
    y = y.reshape(-1)

    # Variable: [β_plus (d), β_minus (d), a_new (1), b (N), s_1 (d), s_2 (d), s_3_1 (N), s_3_2 (N)]
    n_vars = (2 * d + 1 + N) + (2 * d + 2 * N)  # total number of variables: 4d + 3N + 1    

    idx_bp = idx_map["bp"]
    idx_bm = idx_map["bm"]
    idx_a = idx_map["a"]
    idx_b = idx_map["b"]
    idx_s1 = idx_map["s1"]
    idx_s2 = idx_map["s2"] 
    idx_s3_1 = idx_map["s3_1"]
    idx_s3_2 = idx_map["s3_2"]

    # Objective function: ε * a_old = ε * (a_new + 1) = ε * a_new + ε
    c = np.zeros(n_vars)
    c[idx_a] = epsilon          # a_new coefficient
    c[idx_b] = 1.0 / N          # b_i coefficient
    # Constant ε does not affect optimal solution

    # A_eq x = b_eq
    rows = []
    rhs = []

    # (1) a_old >= β_j => β_j_plus - β_j_minus - (a_new + 1) + s_1[j] = 0
    # => β_j_plus - β_j_minus - a_new + s_1[j] = 1
    for j in range(d):
        row = np.zeros(n_vars)
        row[idx_bp[j]] = 1.0
        row[idx_bm[j]] = -1.0
        row[idx_a] = -1.0
        row[idx_s1[j]] = 1.0
        rows.append(row)
        rhs.append(1.0)

    # (2) a_old >= -β_j => -β_j_plus + β_j_minus - (a_new + 1) + s_2[j] = 0
    # => -β_j_plus + β_j_minus - a_new + s_2[j] = 1
    for j in range(d):
        row = np.zeros(n_vars)
        row[idx_bp[j]] = -1.0
        row[idx_bm[j]] = 1.0
        row[idx_a] = -1.0
        row[idx_s2[j]] = 1.0
        rows.append(row)
        rhs.append(1.0)

    # (3) |y_i - x_i^T β| <= b_i
    for i in range(N):
        # y_i - x_i^T β <= b_i => -x_i^T β - b_i + s_3_1[i] = -y_i
        row1 = np.zeros(n_vars)
        row1[idx_bp] = -X[i]
        row1[idx_bm] = X[i]
        row1[idx_b[i]] = -1.0
        row1[idx_s3_1[i]] = 1.0
        rows.append(row1)
        rhs.append(-y[i])

    for i in range(N):
        # -(y_i - x_i^T β) <= b_i => x_i^T β - b_i + s_3_2[i] = y_i
        row2 = np.zeros(n_vars)
        row2[idx_bp] = X[i]
        row2[idx_bm] = -X[i]
        row2[idx_b[i]] = -1.0
        row2[idx_s3_2[i]] = 1.0
        rows.append(row2)
        rhs.append(y[i])

    A_eq = np.vstack(rows)
    b_eq = np.array(rhs)
    return A_eq, b_eq, c

def DRO_l1_linprog(X, y, epsilon=1.0, l=None, rs = True):
    N, d = X.shape

    # Variable: [β_plus (d), β_minus (d), a_new (1), b (N), s_1 (d), s_2 (d), s_3_1 (N), s_3_2 (N)]
    n_vars = (2 * d + 1 + N) + (2 * d + 2 * N)  # Tổng: 4d + 3N + 1    

    idx_map = {
        "bp": np.arange(0, d),
        "bm": np.arange(d, 2 * d),
        "a": 2 * d,
        "b": np.arange(2 * d + 1, 2 * d + 1 + N),
        "s1": np.arange(2 * d + 1 + N, 3 * d + 1 + N),
        "s2": np.arange(3 * d + 1 + N, 4 * d + 1 + N),
        "s3_1": np.arange(4 * d + 1 + N, 4 * d + 2 * N + 1),
        "s3_2": np.arange(4 * d + 2 * N + 1, 4 * d + 3 * N + 1)
    }
    idx_bp = idx_map["bp"]
    idx_bm = idx_map["bm"]
    A_eq, b_eq, c = construct_l1_A_v_c(X, y, idx_map, epsilon=epsilon, l=l)

    bounds = [(0, None)] * n_vars

    if rs:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='revised simplex', options={'maxiter': 100000})
    else:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='simplex', options={'maxiter': 100000})

    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")

    beta_hat = res.x[idx_bp] - res.x[idx_bm]
    B = res["basis"]
    B.sort()
    return beta_hat, B, A_eq, b_eq, res.x, c

def DRO_l1_highspy(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.5
) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    solve the DRO problem with L1 Wasserstein norm using HiGHS solver.
    """
    N, d = X.shape
    inf = highspy.kHighsInf

    # === variable index ===
    idx_bp     = np.arange(0, d)                                # β⁺
    idx_bm     = np.arange(d, 2*d)                              # β⁻
    idx_a      = 2*d                                            # a_new = a_old - 1 ≥ 0
    idx_b      = np.arange(2*d + 1, 2*d + 1 + N)                 # b_i
    idx_s1     = np.arange(2*d + 1 + N, 3*d + 1 + N)             # s1_j (d)
    idx_s2     = np.arange(3*d + 1 + N, 4*d + 1 + N)             # s2_j (d)
    idx_s3_1   = np.arange(4*d + 1 + N, 4*d + 1 + 2*N)           # s3_1[i]
    idx_s3_2   = np.arange(4*d + 1 + 2*N, 4*d + 1 + 3*N)         # s3_2[i]

    n_vars = 4*d + 3*N + 1
    n_cons = 2*d + 2*N                                 # 2d (cho a >= ±β_j) + 2N (cho residuals)

    # === construct A (col-wise CSC) ===
    rows_coo = []
    cols_coo = []
    data_coo = []

    y = y.reshape(-1)

    # (1) β_j - a_new + s1_j = 1    → a_old = a_new + 1 >= β_j
    for j in range(d):
        # β⁺_j - β⁻_j - a_new + s1_j = 1
        rows_coo.extend([j, j, j, j])
        cols_coo.extend([idx_bp[j], idx_bm[j], idx_a, idx_s1[j]])
        data_coo.extend([1.0, -1.0, -1.0, 1.0])

    # (2) -β_j - a_new + s2_j = 1   → a_old >= -β_j
    for j in range(d):
        row_idx = d + j
        rows_coo.extend([row_idx, row_idx, row_idx, row_idx])
        cols_coo.extend([idx_bp[j], idx_bm[j], idx_a, idx_s2[j]])
        data_coo.extend([-1.0, 1.0, -1.0, 1.0])

    # (3) Residual constraints
    base_row = 2*d
    for i in range(N):
        # y_i - x_i^T β <= b_i  →  -x_i^T β - b_i + s3_1[i] = -y_i
        row1 = base_row + 2*i
        # coef for β⁺: -X[i], β⁻: +X[i]
        for j in range(d):
            rows_coo.extend([row1, row1])
            cols_coo.extend([idx_bp[j], idx_bm[j]])
            data_coo.extend([-X[i, j], X[i, j]])
        rows_coo.extend([row1, row1])
        cols_coo.extend([idx_b[i], idx_s3_1[i]])
        data_coo.extend([-1.0, 1.0])

        # - (y_i - x_i^T β) <= b_i  →  x_i^T β - b_i + s3_2[i] = y_i
        row2 = base_row + 2*i + 1
        for j in range(d):
            rows_coo.extend([row2, row2])
            cols_coo.extend([idx_bp[j], idx_bm[j]])
            data_coo.extend([X[i, j], -X[i, j]])
        rows_coo.extend([row2, row2])
        cols_coo.extend([idx_b[i], idx_s3_2[i]])
        data_coo.extend([-1.0, 1.0])

    A_coo = sp.coo_matrix((data_coo, (rows_coo, cols_coo)), shape=(n_cons, n_vars))
    A_csc = A_coo.tocsc()

    # === Objective: ε * a_new + (1/N) * sum b_i ===
    c = np.zeros(n_vars)
    c[idx_a] = epsilon
    c[idx_b] = 1.0 / N

    # === Bounds: variables >= 0 ===
    col_lower = np.zeros(n_vars)
    col_upper = np.full(n_vars, inf)

    # === RHS (b_eq) ===
    b_eq = np.zeros(n_cons)
    b_eq[:2*d] = 1.0                    # 2*d constraints a_old >= ±β_j
    for i in range(N):
        b_eq[2*d + 2*i]     = -y[i]     # upper residual
        b_eq[2*d + 2*i + 1] =  y[i]     # lower residual

    # === HighsLp ===
    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = n_cons
    lp.col_cost_ = c
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = b_eq
    lp.row_upper_ = b_eq                    # equality constraints
    lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = A_csc.data

    # === Solver ===
    h = highspy.Highs()

    h.setOptionValue("presolve", "off")

    h.setOptionValue("primal_feasibility_tolerance", 1e-10)
    h.setOptionValue("dual_feasibility_tolerance", 1e-10)
    h.setOptionValue("primal_residual_tolerance", 1e-10)
    h.setOptionValue("dual_residual_tolerance", 1e-10)
    h.setOptionValue("optimality_tolerance", 1e-10)

    h.setOptionValue("solver", "simplex")
    h.setOptionValue("simplex_strategy", 4)
    h.setOptionValue("simplex_scale_strategy", 4)
    h.setOptionValue("simplex_dual_edge_weight_strategy", 0)
    h.setOptionValue("simplex_primal_edge_weight_strategy", 0)

    h.setOptionValue("run_crossover", "off")

    h.setOptionValue("log_to_console", False)
    h.setOptionValue("output_flag", False)

    h.passModel(lp)
    h.run()

    status = h.getModelStatus()
    if status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS failed: {status}")

    sol = h.getSolution()
    x = np.array(sol.col_value)

    basic_vars = h.getBasicVariables()[1]

    beta_hat = x[idx_bp] - x[idx_bm]

    idx_map = {
        "bp": idx_bp, "bm": idx_bm, "a": idx_a, "b": idx_b,
        "s1": idx_s1, "s2": idx_s2, "s3_1": idx_s3_1, "s3_2": idx_s3_2
    }
    A_eq_dense, v, c = construct_l1_A_v_c(X, y, idx_map, epsilon)
    
    res = {
        "beta_hat": beta_hat,
        "basic_vars": basic_vars,
        "A": A_eq_dense,
        "v": v,
        "h": h,
        "x": x,
        "c": c
    }
    return res


class DRO_AD:
    def __init__(self, epsilon, threshold, Wasserstein_norm = "1", l = None):
        self.epsilon = epsilon
        self.l = l
        self.threshold = threshold
        self.Wasserstein_norm = Wasserstein_norm
        self.A = None
        self.B = None
        self.A_B_inv = None
        self.v = None
        self.Beta = None
        self.X = None
        self.y = None
        # if Wasserstein_norm == "1":
        self.solver_success = True
        self.u = None

    def fit(self, X, y, linprog_only = False, rs = False, tol=1e-10):
        self.X = X
        self.y = y
        if linprog_only:
            if self.Wasserstein_norm == "1":
                self.Beta, B, A, v, x, c = DRO_l1_linprog(X, y, self.epsilon, self.l, rs=rs)
            elif self.Wasserstein_norm == "inf":
                self.Beta, B, A, v, x, c = DRO_linf_linprog(X, y, self.epsilon, self.l)
            else:
                raise(ValueError, "Wasserstein_norm has to be either 1 or inf")
            
            self.B, self.A, self.v, self.u = sorted(B), A, v, x
        else:
            if self.Wasserstein_norm == "1":
                res = DRO_l1_highspy(X, y, self.epsilon)
            elif self.Wasserstein_norm == "inf":
                res = DRO_linf_highspy(X, y, self.epsilon)
            else:
                raise(ValueError, "Wasserstein_norm has to be either 1 or inf")
            h, A, B, v, x, c = res["h"], res["A"], res["basic_vars"], res["v"], res["x"], res["c"]
            self.B, self.A, self.v, self.u = sorted(B), A, v, x
            self.Beta = res["beta_hat"]

        if check_LP_solution(B, c, A, x, v, tol=tol):
            self.solver_success = True
        else:
            self.solver_success = False
        return self
    
    def predict(self, X_test):
        return np.dot(X_test, self.Beta)
    
    def get_Beta_A_B_v(self):
        return self.Beta, self.A, self.B, self.v
    
    def get_AD(self):
        y_hat = np.dot(self.X, self.Beta).reshape(-1, 1)
        residual = abs(self.y - y_hat)
        Oobs = np.where(residual > self.threshold)[0]
        return Oobs.tolist()


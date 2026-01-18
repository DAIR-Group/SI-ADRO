import numpy as np
from mpmath import mp
from numpy.linalg import pinv
from DRO_regressor import DRO_AD

def calculate_SI_essentials(X, y_obs, Oobs, j, noise_Sigma):
    n = X.shape[0]
    p = X.shape[1]

    y_obs = np.reshape(y_obs, (n, 1))
    
    Sigma = noise_Sigma
    
    # construct eta
    ej = np.zeros((n, 1))
    ej[j][0] = 1
    xj = X[j].reshape((p, 1))
    
    I_minusOobs = np.zeros((n, n))
    for i in range(n):
        if i not in Oobs:
            I_minusOobs[i][i] = 1
    X_minusOobs = np.dot(I_minusOobs, X)
    eta = (ej.T - np.dot(np.dot(xj.T, pinv(X_minusOobs)), I_minusOobs)).T
        
    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)[0][0]  # variance of truncated normal distribution 
    etaT_yobs = np.dot((eta.T), y_obs)[0][0]  # test statistic
    
    b = np.dot(Sigma, eta) / etaT_Sigma_eta
    a = np.dot((np.identity(n) - np.dot(b, eta.T)), y_obs)
    return etaT_yobs, etaT_Sigma_eta, a, b

def compute_M_q(A, B, N, p, W_norm="1"):
    A_B = A[:, B]
    A_B_inv = np.linalg.inv(A_B)
    m = A_B_inv.shape[0]
    # print(A_B_inv.shape)
    # print(A_B_inv @ )

    if W_norm == "1":
        v0 = np.concatenate([np.ones(p), np.ones(p), np.zeros(N), np.zeros(N)])
        S = np.zeros((m, N))
        S[2*p : 2*p + N] = -np.identity(N)
        S[2*p + N : 2*p + 2*N] = np.identity(N)
    elif W_norm == "inf":
        v0 = np.concatenate([[-1.0], np.zeros(N), np.zeros(N)])
        S = np.zeros((m, N))
        S[1: N + 1] = -np.identity(N)
        S[N + 1: 2*N + 1] = np.identity(N)

    # M = A_B^{-1} S
    M = A_B_inv @ S

    # q = - A_B^{-1} v0
    q = - A_B_inv @ v0

    return M, q

def calc_sub_region(M, q, a, b):
    Ma = np.dot(M, a)
    Mb = np.dot(M, b)
    # print("Ma = ",Ma)
    # print("Mb = ",Mb)
    Vminus = np.NINF
    Vplus = np.Inf
    for j in range(len(q)):
        left = np.around(Mb[j][0], 15)
        right = np.around(q[j] - Ma[j][0], 15)

        if np.isclose(left, 0, 1e-12):
            if np.isclose(right, 0, 1e-12):
                continue
            if right > 0:
                print('Error')
                continue
        else:
            temp = right / left

            if left > 0:
                Vminus = max(temp, Vminus)
            else:
                Vplus = min(temp, Vplus)
    
    return [(Vminus, Vplus)]

def calc_f_g(X, a, b, M, q, B, etaT_y = None, Beta_hat = None):
    """
    u_B = (inv(A_B) @ v_0 + inv(A_B) @ S @ a ) + inv(A_B) @ S @ bz
    u_B = (-q + inv(A_B) @ S @ a ) + inv(A_B) @ S @ bz

    beta_i =    | 0 if idx_bp[i] and idx_bm[i] not in B
                | u[idx_bp[i]] if idx_bp[i] in B
                | -u[idx_bm[i]] if idx_bp[i] not in B

    beta = c + dz
    c in R^d
    d in R^d

    let temp_i = (inv(A_i) @ v_0 + inv(A_i) @ S @ a ) + inv(A_B) @ S @ bz

    c_i = 	| 0 if bp_idx[i] and idx_bm[i] not in B
            | (M[idx_bp[i]]@a + q[idx_bp[i]]) if idx_bp[i] in B
            | -(M[idx_bm[i]]@a + q[idx_bm[i]]) if bp_idx[i] not in B

    d_i =   | 0 if bp_idx[i] and idx_bm[i] not in B
            | M[bp_idx[i]]@b if bp_idx[i] in B
            | -M[idx_bm[i]]@b if bp_idx[i] not in B

    residual = y - X@beta = a + bz - X@c - X@dz = a - X@c + (b - X@d)z = f + gz
    f = a - X@c
    g = b - X@d
    """
    N, d = X.shape
    idx_bp = np.arange(0, d)
    idx_bm = np.arange(d, 2 * d)
    # index of all beta (plus and minus) in B
    # E.g. B = [1 3 4], d = 3, idx_bp = [0 1 2], idx_bm = [3 4 5] -> idx_bp_inB = [-1 0 -1], idx_bm_inB = [1 2 -1]
    idx_bp_inB = []
    idx_bm_inB = []
    cnt = 0
    for i in range(d):
        if idx_bp[i] in B:
            idx_bp_inB.append(cnt)
            cnt += 1
        elif idx_bp[i] not in B:
            idx_bp_inB.append(-1)

    for i in range(d):
        if idx_bm[i] in B:
            idx_bm_inB.append(cnt)
            cnt += 1
        elif idx_bm[i] not in B:
            idx_bm_inB.append(-1)

    # print(idx_bp_inB)
    # print(idx_bm_inB)

    ct = np.zeros((d, 1))
    dt = np.zeros((d, 1))
    for i in range(d):
        if idx_bp[i] not in B and idx_bm[i] not in B:
            ct[i][0] = dt[i][0] = 0
        elif idx_bp[i] in B:
            ct[i][0] = (M[idx_bp_inB[i]]@a - q[idx_bp_inB[i]])
            dt[i][0] = M[idx_bp_inB[i]]@b
            # print(ct[i][0] + dt[i][0]*etaT_y)
        else:
            ct[i][0] = -(M[idx_bm_inB[i]]@a - q[idx_bm_inB[i]])
            dt[i][0] = -(M[idx_bm_inB[i]]@b)
            # print(ct[i][0] + dt[i][0]*etaT_y)
    
    # print(B)
    # print(ct + dt * etaT_y)
    # print(Beta_hat)

    f = a - X@ct
    g = b - X@dt
    # print(f)
    # print(g)
    return f, g

def outlier_detection_region(interval, N, f, g, threshold, Oobs):
    V = [[] for _ in range(N)]
    for i in range(N):
        if g[i][0] == 0:
            if abs(f[i][0]) > threshold:
                V[i] = [(np.Inf, -np.Inf)]
        elif g[i][0] > 0:
            left = (-threshold - f[i][0])/g[i][0]
            right = (threshold - f[i][0])/g[i][0]
            V[i] = [(-np.Inf, left), (right, np.Inf)]
        elif g[i][0] < 0:
            left = (threshold - f[i][0])/g[i][0]
            right = (-threshold - f[i][0])/g[i][0]
            V[i] = [(-np.Inf, left), (right, np.Inf)]
    final_region = interval
    for i in range(N):
        if i in Oobs:
            final_region = getIntersection(final_region, V[i])
        else:
            final_region = getIntersection(final_region, getComplement(V[i]))

    return final_region

def OverConditioning_region(a, b, dro_reg):
    """
    pipeline for OC:
    calc_sub_region -> sub_region()
    f, g = calc_f_g()
    get intersect the sub_region with the outlier detection event:
    outlier_detection_region(beta_interval, N, f, g, threshold) - > new region with outlier detection event
    """
    X = dro_reg.X
    N, d = X.shape

    Beta, A, B, v = dro_reg.get_Beta_A_B_v() 
    B = np.sort(B)
    Oobs = dro_reg.get_AD()

    M, q = compute_M_q(A, B, N, d, W_norm=dro_reg.Wasserstein_norm)

    f, g = calc_f_g(X, a, b, M, q, B)
    beta_interval = calc_sub_region(M, q, a, b)
    # print(beta_interval)

    new_region = outlier_detection_region(beta_interval, N, f, g, dro_reg.threshold, Oobs)    

    return new_region

def identifying_truncated_region(a, b, dro_reg, limit, linprog_only = False):
    threshold = dro_reg.threshold
    epsilon = dro_reg.epsilon
    Wasserstein_norm = dro_reg.Wasserstein_norm
    X = dro_reg.X
    Oobs = dro_reg.get_AD()

    N, d = X.shape

    z_min = -limit
    z_max = limit
    # print(limit)
    z_cur = z_min

    z_last = - np.inf

    final_region = []
    polytope_count = 0
    while z_cur < z_max:
        polytope_count += 1
        y_z = a + b*z_cur

        dro_reg_z = DRO_AD(epsilon=epsilon, threshold=threshold, Wasserstein_norm=Wasserstein_norm)
        dro_reg_z.fit(X, y_z, linprog_only)
        if dro_reg_z.solver_success == False:
            return None, None, False
        Beta, A_z, B_z, v_z = dro_reg_z.get_Beta_A_B_v()
        # O_z = dro_reg_z.get_AD()

        M_z, q_z = compute_M_q(A_z, B_z, N, d, W_norm=dro_reg.Wasserstein_norm)

        f_z, g_z = calc_f_g(X, a, b, M_z, q_z, B_z)
        beta_interval = calc_sub_region(M_z, q_z, a, b)

        new_region = outlier_detection_region(beta_interval, N, f_z, g_z, threshold, Oobs)
        final_region = getUnion(final_region, new_region)

        # print(z_cur)
        # print(beta_interval[0])
        z_last = z_cur
        z_cur = beta_interval[0][1] + 0.001
        if z_last == z_cur:
            print("infinity loop detected")
            print(beta_interval)
            # basis = B_z
            np.savez("./results/inf_loop_test.npz", X=X, y=dro_reg.y)
            return None, None, False
            # break
    
    return final_region, polytope_count, True

def calculate_p_value(Regions, etaT_y, etaT_Sigma_eta):
    mp.dps = 1000
    numerator = 0
    denominator = 0
    mu = 0
    tn_sigma = np.sqrt(etaT_Sigma_eta)
    for i in Regions:
        left = i[0]
        right = i[1]
        denominator = denominator + mp.ncdf((right - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
        if etaT_y >= right:
            numerator = numerator + mp.ncdf((right - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
        elif (etaT_y >= left) and (etaT_y < right):
            numerator = numerator + mp.ncdf((etaT_y - mu)/tn_sigma) - mp.ncdf((left - mu)/tn_sigma)
    
    if denominator == 0:
        print("Cannot calculate p_value, maybe the region was too small")
        return None
    else:
        cdf = float(numerator/denominator) 
        pvalue = 2*min(cdf, 1 - cdf)
        return pvalue
    
def getIntersection(list1, list2):
    list1.sort()
    list2.sort()

    intersections = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        start1, end1 = list1[i]
        start2, end2 = list2[j]

        start_intersection = max(start1, start2)
        end_intersection = min(end1, end2)

        if start_intersection <= end_intersection:
            intersections.append((start_intersection, end_intersection))

        if end1 < end2:
            i += 1
        else:
            j += 1

    return intersections

def merge_ranges(ranges):
    if not ranges:
        return []
    
    ranges.sort()
    merged = [ranges[0]]

    for current in ranges[1:]:
        last_merged = merged[-1]
        
        if current[0] <= last_merged[1]:
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged.append(current)

    return merged

def getUnion(list1, list2):
    combined = list1 + list2
    return merge_ranges(combined)

def getComplement(intervals):
    result = []
    current_start = float('-inf')
    
    for interval in sorted(intervals):
        if current_start < interval[0]:
            result.append((current_start, interval[0]))
        current_start = max(current_start, interval[1])
    
    if current_start < float('inf'):
        result.append((current_start, float('inf')))
    
    return result
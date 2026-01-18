import numpy as np
from gen_data import generate_data
from DRO_regressor import DRO_AD
from utils import identifying_truncated_region, calculate_p_value, calculate_SI_essentials
from tqdm import tqdm

def run(): 
    true_beta = [i%2 + 1 for i in range(d)]
    # true_beta = [0 for i in range(d)]
    noise_sigma = np.identity(N)
    X, y, _ = generate_data(N, d, true_beta=true_beta, sigma=noise_sigma, q_prob=0)

    if Wasserstein_norm == "1":
        epsilon = 0.2
    else:
        epsilon = 0.05

    lin_prog_only = False # option to use scipy linprog only, default value is True, which uses highspy simplex solver 
    dro_reg = DRO_AD(epsilon = epsilon, threshold = threshold, Wasserstein_norm = Wasserstein_norm)
    dro_reg.fit(X, y, linprog_only=lin_prog_only)
    if dro_reg.solver_success == False:
        print("solver failed")
        return None
    Oobs = dro_reg.get_AD()

    j_selected = None
    if len(Oobs) == 0:
        # print("No outlier")
        return None
    j_selected = np.random.choice(Oobs)

    etaT_yobs, etaT_Sigma_eta, a, b = calculate_SI_essentials(X, y, Oobs, j_selected, noise_sigma)

    # The function identifying_truncated_region returns a tuple of truncation intervals, number of polytopes searched, and solver state 
    Truncation_Region, _, solver_state = identifying_truncated_region(a, b, dro_reg, 20*np.sqrt(etaT_Sigma_eta))
    if solver_state == False:
        print("solver failed")
        return None

    selective_pvalue = calculate_p_value(Truncation_Region, etaT_yobs, etaT_Sigma_eta)

    return selective_pvalue
    

if __name__ == "__main__":
    Wasserstein_norm = "1"
    N, d = 50, 10
    threshold = 2

    trials = 200
    alpha = 0.05
    
    reject = 0
    pvalue_list = []

    trial = 0
    pbar = tqdm(total=trials, desc="Trials", unit="trial")
    while trial < trials:
        res = run()
        if res is None:
            continue
        trial += 1
        
        pvalue_list.append(res)
        if res < alpha:
            reject += 1
        pbar.update(1)
    
    pbar.close()
    print(f"False Positive Rate: {reject/trials if trials>0 else 0}")

    import matplotlib.pyplot as plt
    plt.hist(pvalue_list, bins=20, range=(0,1))
    plt.savefig("results/ex2_fpr_hist.png")
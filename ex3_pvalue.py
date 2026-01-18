import numpy as np
from gen_data import generate_data
from DRO_regressor import DRO_AD
from utils import identifying_truncated_region, calculate_p_value, calculate_SI_essentials

def run(): 
    true_beta = [i%2 + 1 for i in range(d)]
    noise_sigma = np.identity(N)
    X, y, _ = generate_data(N, d, true_beta=true_beta, sigma=noise_sigma, q_prob=0)

    if Wasserstein_norm == "1":
        epsilon = 0.2
    else:
        epsilon = 0.05

    lin_prog_only = True # option to use scipy linprog only, default value is True, which uses highspy simplex solver 
    dro_reg = DRO_AD(epsilon = epsilon, threshold = threshold, Wasserstein_norm = Wasserstein_norm)
    dro_reg.fit(X, y, linprog_only=lin_prog_only)
    if dro_reg.solver_success == False:
        print("solver failed")
        return 
    Oobs = dro_reg.get_AD()

    for j in Oobs:
        etaT_yobs, etaT_Sigma_eta, a, b = calculate_SI_essentials(X, y, Oobs, j, noise_sigma)

        # The function identifying_truncated_region returns a tuple of truncation intervals, number of polytopes searched, and solver state 
        Truncation_Region, _, solver_state = identifying_truncated_region(a, b, dro_reg, 20*np.sqrt(etaT_Sigma_eta))
        if solver_state == False:
            print("solver failed")
            return 

        selective_pvalue = calculate_p_value(Truncation_Region, etaT_yobs, etaT_Sigma_eta)

        print(f"The p value of the {j}(th) instance is {selective_pvalue}")
    

if __name__ == "__main__":
    Wasserstein_norm = "1"
    N, d = 50, 10
    threshold = 2

    run()
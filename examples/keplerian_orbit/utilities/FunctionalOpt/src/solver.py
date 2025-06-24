import torch
import numpy as np
from .models import TorchGMM
from .train import train_model
from .helpers import check_pdf_integral, check_pdf_interval, integral_torchgmm


def solve_funcopt(problem):
    device = torch.device("cpu")  # or torch.device("cuda") if available.

    # --- Init model ---
    torch.manual_seed(0); np.random.seed(0)
    trained_model = TorchGMM(problem['constants']).to(device)
    # trained_model = RBFDensity(num_basis=256).to(device) # (default degree of basis)

    # --- Train model ---
    trained_model  = train_model(problem, trained_model, num_iterations=20000)

    # --- Load model ---
    # trained_model.load_state_dict(torch.load("data/app1/tar1/pdf_models/gmm_64_t{:.3f}.pth".format(t)))
    
    # --- Check constraints ---
    trained_model.eval()
    check_pdf_integral(problem, trained_model, method="analy")
    check_pdf_interval(problem, trained_model)
    
    # --- Evaluate Probability ---
    Pr_opt = integral_torchgmm(trained_model.get_gmm_paramters(), problem['target_bounds'])
    print("Integrated phat over target region: {:.4f}".format(problem['Pr_p0_est']))
    print("Prob. pGMM over target region: {:.4f}".format(Pr_opt))
    # visual_fo_rbf(t, p_net, trained_model, target_region)

    return Pr_opt, trained_model


# --- NOTE: not yet implemented, should be implemented in another module ---
    # (1.1) naive: integral (phat + B1) dx
    # p0_flat = p0.flatten()
    # pr = np.sum(p0_flat[mask_flat])*dV + sum(mask_flat)*dV*B
    # return pr

    # (2.0) linear program
    # p0_flat = p0.flatten()
    # # Decision variable: p_vec is now a vector of length N.
    # p_vec = cp.Variable(N)
    # # Constraints.
    # constraints = [
    #     cp.sum(p_vec) * dV == 1,  # total mass constraint.
    #     p_vec >= 0,                              # non-negativity.
    #     p_vec >= p0_flat - B,                      # lower bound.
    #     p_vec <= p0_flat + B                       # upper bound.
    # ]
    # # Objective: maximize probability mass in target region minus smoothness penalty.
    # objective = cp.Maximize(
    #     cp.sum(cp.multiply(p_vec, mask_flat)) * dV #- lambda_reg * smoothness_penalty
    # )
    # # Set up and solve the problem.
    # prob = cp.Problem(objective, constraints)
    # result = prob.solve()
    # pr = np.sum(p_vec.value[mask_flat]) * dV
    # print("NN+Error Upper Bound value (target region):", pr)
    # print("Total mass (should be 1):", np.sum(p_vec.value) * dV)
    # return pr
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

    # --- Load trained model ---
    # trained_model.load_state_dict(torch.load("data/app1/tar1/pdf_models/gmm_64_t{:.3f}.pth".format(t)))
    
    # --- Check constraints ---
    trained_model.eval()
    check_pdf_integral(problem, trained_model, method="analy")
    check_pdf_interval(problem, trained_model)
    
    # --- Evaluate Probability ---
    Pr_opt = integral_torchgmm(trained_model.get_gmm_paramters(), problem['target_bounds'])
    print(" [Solved] Pr_tar: {:.4f}".format(Pr_opt))

    return Pr_opt, trained_model
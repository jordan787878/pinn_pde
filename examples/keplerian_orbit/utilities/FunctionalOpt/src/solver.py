import torch
import numpy as np
from .models import TorchGMM
from .train import train_model, visual_loss
from .helpers import check_pdf_integral, check_pdf_interval, integral_torchgmm


def solve_funcopt(problem):
    device = torch.device("cpu")  # or torch.device("cuda") if available.

    # --- Init model ---
    torch.manual_seed(0); np.random.seed(0)
    trained_model = TorchGMM(problem['constants']).to(device)
    # trained_model = RBFDensity(num_basis=256).to(device) # (default degree of basis)

    # --- Train model ---
    if(problem['use_trained_gmm'] == False):
        trained_model  = train_model(problem, trained_model, num_iterations=20000)
    else:
        # --- Load trained model ---
        trained_model.load_state_dict(torch.load(problem["trained_gmm_path"]+"_t{:.3f}.pth".format(problem["time"])))
        if(problem['show_loss_landscape']):
            visual_loss(problem, trained_model)
    
    # --- Check constraints ---
    trained_model.eval()
    check_pdf_integral(problem, trained_model, method="analy")
    check_pdf_interval(problem, trained_model)
    
    # --- Evaluate Probability ---
    Pr_opt = integral_torchgmm(trained_model.get_gmm_paramters(), problem['target_bounds'])
    print(" [Solved] time: {:.3f}, Pr_tar: {:.4f}".format(problem['time'], Pr_opt))

    return Pr_opt, trained_model
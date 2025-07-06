import torch
import numpy as np
import gc
from .models import TorchGMM
from .train import train_model, visual_loss
from .helpers import check_pdf_integral, check_pdf_interval, integral_torchgmm


def solve_funcopt(problem):
    device = torch.device("cpu")  # or torch.device("cuda") if available.

    # --- Init model ---
    torch.manual_seed(0); np.random.seed(0)
    s = problem['label']
    num_components = int(s.split("gmmx",1)[1].split("_",1)[0])
    trained_model = TorchGMM(problem['constants'], num_components=num_components).to(device)

    # --- Inherit from previously-trained GMM ---
    if(problem["label_parent_gmm"] is not None):
        s = problem["label_parent_gmm"]
        num_parent_components = int(s.split("gmmx",1)[1].split("_",1)[0])
        gmm_parent = TorchGMM(problem['constants'], num_components=num_parent_components).to(device)
        gmm_parent.load_state_dict(torch.load(problem["path_pdf_models"]+problem["label_parent_gmm"]+"_t{:.3f}.pth".format(problem["time"])))
        with torch.no_grad():
            # 2) copy in the parent-component parameters
            trained_model.raw_weights.data[:num_parent_components] = gmm_parent.raw_weights.data
            trained_model.raw_weights.data[num_parent_components:] = torch.randn(num_components-num_parent_components)*1e-3
            trained_model.means.data[:num_parent_components, :]   = gmm_parent.means.data
            trained_model.raw_scales.data[:num_parent_components, :] = gmm_parent.raw_scales.data
        # release memory
        del gmm_parent
        gc.collect()

    # --- Train model ---
    if(problem['use_trained_gmm'] == False):
        trained_model = train_model(problem, trained_model, num_iterations=problem["num_iterations"])
    else:
        # --- Load trained model ---
        trained_model.load_state_dict(torch.load(problem["trained_gmm_path"]+"_t{:.3f}.pth".format(problem["time"])))
        if(problem['show_loss_landscape']):
            visual_loss(problem, trained_model)
    
    # --- Check constraints ---
    assert trained_model is not None, "Training didn’t return a valid model!"
    trained_model.eval()
    check_pdf_integral(problem, trained_model, method="analy")
    check_pdf_interval(problem, trained_model)
    
    # --- Evaluate Probability ---
    Pr_opt = integral_torchgmm(trained_model.get_gmm_paramters(), problem['target_bounds'])
    print(" [Solved] time: {:.3f}, Pr_tar: {:.8f}".format(problem['time'], Pr_opt))

    return Pr_opt, trained_model
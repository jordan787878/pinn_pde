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

    # with torch.no_grad():
    #     trained_model.logits.fill_(0.0)

    # --- Inherit from previously-trained GMM ---
    if(problem["label_parent_gmm"] is not None):
        print("load from parent: ", problem["label_parent_gmm"])
        s = problem["label_parent_gmm"]
        num_parent_components = int(s.split("gmmx",1)[1].split("_",1)[0])
        gmm_parent = TorchGMM(problem['constants'], num_components=num_parent_components).to(device)
        gmm_parent.load_state_dict(torch.load(problem["path_pdf_models"]+problem["label_parent_gmm"]+"_t{:.3f}.pth".format(problem["time"])))
        Pr_parent = integral_torchgmm(gmm_parent.get_gmm_paramters(), problem['target_bounds'])
        with torch.no_grad():
            # 2) copy in the parent-component parameters
            min_logit = min(gmm_parent.logits.data)
            if(min_logit < 0):
                min_logit = 2*min_logit
            else:
                min_logit =0.5*min_logit
            trained_model.logits.data[:num_parent_components] = gmm_parent.logits.data
            trained_model.logits.data[num_parent_components:] = min_logit
            trained_model.means.data[:num_parent_components, :] = gmm_parent.means.data
            trained_model.raw_scales.data[:num_parent_components, :] = gmm_parent.raw_scales.data
            p1, p2, p3 = gmm_parent.get_gmm_paramters()
            print("gmm parent weights", p1)
        # release memory
        del gmm_parent
        gc.collect()
    else:
        Pr_parent = 0.0
    p1, p2, p3 = trained_model.get_gmm_paramters()
    print("init param")
    print(p1)

    # --- Train model ---
    if(problem['use_trained_gmm'] == False):
        trained_model = train_model(problem, trained_model, num_iterations=problem["num_iterations"], lower_bound=Pr_parent)
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
    print("[Solved] time: {:.3f}, Pr_tar: {:.8f}".format(problem['time'], Pr_opt))
    p1, p2, p3 = trained_model.get_gmm_paramters()
    print("best param")
    print(p1)

    return Pr_opt, trained_model
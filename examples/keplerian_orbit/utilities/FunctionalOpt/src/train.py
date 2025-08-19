import torch
import numpy as np
from tqdm import tqdm
from .helpers import *


def train_model(problem, model, num_iterations=1000, batch_size=512, grad_threshold=1e-3, device=torch.device("cpu")):
    # --- Extract problem ----
    problem['p_net'], _, _ = problem['networks']
    target_bounds = problem["target_bounds"]
    dtype  = next(model.parameters()).dtype
    target_bounds_tensor = torch.tensor(target_bounds, dtype=dtype)

    # --- Set optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    
    # --- Select deterministic samples over domain and target ---
    x_dom, p0_dom, x_remains, p0_remains = get_samples_determin(problem, model, batch_size)

    # --- Run ---
    # mu           = torch.tensor(0.0, device=device)   # dual variable
    mu = 1000.0
    # rho          = 10.0                               # penalty weight
    best_model_state = None
    best_loss = np.inf
    # increment = 0.001 # 0.01
    increment_factor = 0.05
    patient = 0.0
    for it in tqdm(range(num_iterations), desc="training GMM"):
        optimizer.zero_grad()
        # --- Augment samples with random points --- 
        x_dom_train, p0_dom_train = aug_samples_random(problem, model, x_dom, p0_dom, batch_size)

        # --- hard constraint loss -- 
        loss_hc, vio_percent, _x_vio_topk, _p0_vio_topk = loss_hardconstraint(problem, model, x_dom_train, p0_dom_train)
        # if(vio_percent > 0.0 and best_model_state is not None):
        #     x_dom = torch.cat((x_dom, _x_vio_topk), dim=0)
        #     p0_dom = torch.cat((p0_dom, _p0_vio_topk), dim=0)
        #     print("[check] after adding loss_hc violation:", x_dom.shape[0], x_remains.shape[0])

        Pr = model.region_prob(target_bounds_tensor)
        loss_obj               = -Pr   # we *minimize* −Pr to *maximize* Pr

        # 3) augmented Lagrangian
        total_loss = (
            loss_obj
            + mu * loss_hc
            # + 0.5 * rho * loss_hc.pow(2)
        )
        total_loss.backward(retain_graph=True)
        patient += 1
        
        # --- Early stopping ---
        # gn = get_grad_norm(model)
        # if gn < grad_threshold:
        #     if(best_model_state is not None):
        #         print(f"[info] early stopping at iter {it}, grad_norm={gn:.2e} < {grad_threshold:.2e}")
        #         break
        if patient > 10000:
            if(best_model_state is not None):
                print(f"[info] early stoppint at iter {it}, exceeds patient")
                break

        # --- Print progress ---
        # if it % int(num_iterations/50) == 0:
        #     print(f"Iteration {it:4d}, loss: {total_loss.item()}, loss hc: {loss_hc.item()}, Pr: {Pr.item()}")
        #     print(f"   violation percent: {vio_percent:.4f}%")
        #     print(f"   mu: {mu:.4f}")
            # print(f"   grad_norm={gn:.2e}, {grad_threshold:.2e}")

        # --- Update the best model ---
        if(total_loss.item() < (1.+increment_factor)*best_loss and vio_percent <= 0.0):
            # --- Augment by checking violation on grid ---
            _x_vio, _p0_vio, x_remains, p0_remains = aug_samples_violate(
                problem, model, x_remains, p0_remains, batch_size)
            # if exists violation on grids, add it to deterministic samples ---
            if(_x_vio is not None):
                x_dom   = torch.cat([x_dom,   _x_vio], dim=0)         
                p0_dom  = torch.cat([p0_dom, _p0_vio], dim=0)
                # print("[check] after grid aug. sample size:", x_dom.shape[0], x_remains.shape[0])
            else:
                # --- Scenario-based Approach vertification ---
                _x_rand, _p0_rand = get_samples_random(problem, 100000) #10000
                _, vio_rand_hc_check, _x_vio_topk, _p0_vio_topk = loss_hardconstraint(problem, model, _x_rand, _p0_rand)
                # [testing...] --- RAR ---
                # print("[info] scenaro-based checking")
                if(vio_rand_hc_check > 0.0):
                    x_dom = torch.cat((x_dom, _x_vio_topk), dim=0)
                    p0_dom = torch.cat((p0_dom, _p0_vio_topk), dim=0)
                    # print("[check] after rand aug. sample size:", x_dom.shape[0], x_remains.shape[0])
                else:
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_loss = total_loss.item()
                    ws, _, _ = model.get_gmm_paramters()
                    if(True):
                        print(f"[Best] Iteration {it:4d}, loss: {total_loss.item()}, {loss_hc.item()}")
                        print(f"   violation percent: {vio_percent:.4f}%, Pr: {Pr:.4f}")
                        # print(np.sum(ws))
                        patient = 0
                del _x_rand, _p0_rand
            del _x_vio, _p0_vio
        optimizer.step()
        scheduler.step()
        # # 5) dual update: μ ← max(0, μ + ρ · loss_hc)
        # with torch.no_grad():
        #     mu.add_(rho * loss_hc)
        #     mu.clamp_(min=0.0)
            
    # --- Return best model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Returning best model with total loss: {best_loss}")
        return model
    else:
        return None


def get_grad_norm(model):
    grad_l2_norm = torch.sqrt(
        sum(p.grad.detach().pow(2).sum()
            for p in ([model.logits] if hasattr(model, 'logits') else [])
            + [model.raw_weights, model.means]
        )
    )
    return grad_l2_norm.item()


def get_samples_determin(problem, model, batch_size):
    # unpack
    p0_flat           = problem['p0']               # np.ndarray, shape [N]
    mask         = problem['mask']        # boolean np.ndarray, shape [N]
    grid_points_tensor = problem['x']               # torch.Tensor, shape [N, d]

    device = grid_points_tensor.device
    N, d   = grid_points_tensor.shape

    # to torch
    _p0_tensor = torch.from_numpy(p0_flat).to(device)

    # 1) Top‐k deviations
    p_xgrid   = model(grid_points_tensor).view(-1)          # [N]
    deviation = torch.abs(p_xgrid - _p0_tensor)             # [N]
    _, dev_idx = torch.topk(deviation, batch_size, largest=True)  # [batch_size]
    x_dev      = grid_points_tensor[dev_idx]                # [batch_size, d]
    p0_dev     = _p0_tensor[dev_idx]                        # [batch_size]

    # # 2) Top‐k p0 within mask
    # mask_tensor = torch.from_numpy(mask).to(device)    # [N], bool
    # masked_p0    = _p0_tensor[mask_tensor]                  # [M]
    # masked_x     = grid_points_tensor[mask_tensor]          # [M, d]
    # _, rel_idx   = torch.topk(masked_p0, batch_size, largest=True)  # [batch_size]
    # # map back to global indices
    # mask_idx     = torch.nonzero(mask_tensor, as_tuple=False).view(-1)  # [M]
    # tar_idx      = mask_idx[rel_idx]                        # [batch_size]
    # x_tar        = grid_points_tensor[tar_idx]              # [batch_size, d]
    # p0_tar       = _p0_tensor[tar_idx]                      # [batch_size]

    # 3) Combine “domain” selections
    # dom_idx = torch.cat([dev_idx, tar_idx], dim=0)          # [2*batch_size]
    # x_dom   = torch.cat([x_dev,   x_tar], dim=0)            # [2*batch_size, d]
    # p0_dom  = torch.cat([p0_dev, p0_tar], dim=0)            # [2*batch_size]
    dom_idx = dev_idx       # [2*batch_size]
    x_dom   = x_dev            # [2*batch_size, d]
    p0_dom  = p0_dev            # [2*batch_size]

    # 4) Compute “not selected” as complement of dom_idx
    all_idx       = torch.arange(N, device=device)          # [N]
    sel_mask      = torch.zeros(N, dtype=torch.bool, device=device)
    sel_mask[dom_idx] = True
    not_sel_idx   = all_idx[~sel_mask]                     # [N - 2*batch_size]
    x_not_select  = grid_points_tensor[not_sel_idx]         # [...]
    p0_not_select = _p0_tensor[not_sel_idx]                 # [...]

    return x_dom, p0_dom, x_not_select, p0_not_select


def get_samples_random(problem, batch_size):
    constants = problem['constants']
    t = problem['time']
    p_net = problem['p_net'] 
    domain_bounds = problem['domain_bounds']
    _x_dom_rand = constants.sample_points(batch_size, domain_bounds)
    _t_dom_rand = torch.full((batch_size, 1), t)
    _p0_dom_rand = p_net(_x_dom_rand, _t_dom_rand).view(-1,)
    return _x_dom_rand, _p0_dom_rand


def aug_samples_random(problem, model, x_dom, p0_dom, batch_size):
    constants = problem['constants']
    t = problem['time']
    p_net = problem['p_net'] 
    domain_bounds = problem['domain_bounds']
    # target_bounds = problem['target_bounds']

    _x_dom_rand = constants.sample_points(batch_size, domain_bounds)
    _t_dom_rand = torch.full((batch_size, 1), t)
    _p0_dom_rand = p_net(_x_dom_rand, _t_dom_rand).view(-1,)

    # _x_tar_rand = constants.sample_points(batch_size, target_bounds)
    # _t_tar_rand = torch.full((batch_size, 1), t)
    # _p0_dom_rand = p_net(_x_tar_rand, _t_tar_rand).view(-1,)

    # [test]
    _, mus, _ = model.get_gmm_paramters()
    x_at_mus = torch.tensor(mus, dtype=x_dom.dtype)
    t_at_mus = torch.full((x_at_mus.shape[0], 1), t)
    p0_at_mus = p_net(x_at_mus, t_at_mus).view(-1,)

    # x_dom_train = torch.cat((x_dom, _x_dom_rand, _x_tar_rand), dim=0)
    # p0_dom_train = torch.cat((p0_dom, _p0_dom_rand, _p0_dom_rand), dim=0)
    # x_tar_train = torch.cat((x_tar, _x_tar_rand), dim=0)
    x_dom_train = torch.cat((x_dom, _x_dom_rand, x_at_mus), dim=0)
    p0_dom_train = torch.cat((p0_dom, _p0_dom_rand, p0_at_mus), dim=0)
    # x_tar_train = torch.cat((x_tar), dim=0)

    return x_dom_train, p0_dom_train


def aug_samples_violate(problem, model, x_remains, p0_remains, batch_size):
    """
    x_remains:  Tensor [N, d]
    p0_remains: Tensor [N]
    B:          scalar float in problem['B']
    """
    device = x_remains.device
    B = problem['B']

    # 1) Evaluate model on all remains
    with torch.no_grad():
        p_opt = model(x_remains).view(-1)           # Tensor [N]

    # 2) Violation magnitude = relu(|p_opt - p0| - B)
    viol_mag = torch.relu((p_opt - p0_remains).abs() - B)

    # 3) All violating indices
    viol_idx_all = (viol_mag > 0).nonzero(as_tuple=False).view(-1)
    if viol_idx_all.numel() == 0:
        # print("No violations found.")
        return None, None, x_remains, p0_remains

    # 4) Pick top-k by descending violation mag
    k = min(batch_size, viol_idx_all.numel())
    topk_vals, topk_pos = torch.topk(viol_mag[viol_idx_all], k)
    topk_idx = viol_idx_all[topk_pos]             # global indices [k]

    # 5) Gather the violating samples
    x_vio  = x_remains[topk_idx]                   # [k, d]
    p0_vio = p0_remains[topk_idx]                  # [k]

    # 6) Remove those from remains
    N = x_remains.size(0)
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)
    keep_mask[topk_idx] = False

    x_rem_new   = x_remains[keep_mask]             # [N-k, d]
    p0_rem_new  = p0_remains[keep_mask]            # [N-k]

    # print(f"Selected {k} violations out of {N} remains.")
    return x_vio, p0_vio, x_rem_new, p0_rem_new


def visual_loss(problem, model):
    """
    ideally want to visualize a 3D surface plot where
        x,y axes represents the variation of the GMM parameters
        z height represents the loss
    """
    print("options[show_loss_landscape] has not been implemented, due to the lack of good representation of gmm paramters in 2D")
    # batch_size = 10
    # x_dom, p0_dom, x_tar, _, _ = get_samples_determin(problem, model, batch_size)
    # # --- loss (of given model) ---
    # loss, params = loss_custom(problem, model, x_dom, p0_dom, x_tar)
    # w, mu, cov = params
    # N = w.shape[0]
    # N_half = int(N/2)
    # d1 = np.concatenate((w[0:N_half].flatten(),mu[0:N_half,:].flatten(), cov[0:N_half,:].flatten()))
    # d2 = np.concatenate((w[N_half:].flatten(),mu[N_half:,:].flatten(), cov[N_half:,:].flatten()))
    # print(d1.shape)
    # print(loss)


def loss_hardconstraint(problem, model, x_dom, p0_dom):
    B = problem['B']
    domain_V = problem['domain_V']
    # target_V = problem['target_V']
    # --- loss (of given model) ---
    p_dom  = model(x_dom).view(-1,)
    hc = (p_dom >= (p0_dom - B)) & (p_dom <= (p0_dom + B))
    violation_mask = ~hc  # True for datapoints that do NOT satisfy the constraint
    vio_percent = 100.0*(violation_mask.sum()/p0_dom.shape[0])
    # loss_hc = torch.relu(p_dom - (p0_dom + B)).sum() + \
    #           torch.relu((p0_dom - B) - p_dom).sum()
    # loss_hc *= domain_V
    if(sum(violation_mask) > 0):
        # loss_hc = torch.mean(0.5*(p_dom[violation_mask] - p0_dom[violation_mask]) ** 2)*domain_V
        viol_mag = torch.relu((p_dom - p0_dom).abs() - B)
        _, max_index = torch.topk(torch.abs(viol_mag.squeeze()), 16)
        _x_top_vio = x_dom[max_index, :].clone()
        _p0_top_vio = p0_dom[max_index].clone()
    else:
        loss_hc = torch.zeros(1)
        _x_top_vio = None
        _p0_top_vio = None
    loss_hc = torch.sum(torch.relu(p_dom - (p0_dom + B))) + torch.sum(torch.relu((p0_dom - B) - p_dom))
    # loss_hc = torch.sum((torch.relu((p_dom - (p0_dom+B)).abs() - B))**2)*domain_V
    
    # --- maximize prob. over target loss ---
    # p_tar = model(x_tar).view(-1,)
    # target_mass_estimate = torch.mean(p_tar)*target_V
    # loss_tar = -target_mass_estimate
    # loss_tar = -integral_torchgmm(model.get_gmm_paramters(), problem['target_bounds'])
    # --- Combine loss and optimize ---
    # total_loss = loss_hc + loss_tar*1e-1

    return loss_hc, vio_percent, _x_top_vio, _p0_top_vio

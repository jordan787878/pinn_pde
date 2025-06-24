import torch
import numpy as np
from .helpers import *


def train_model(problem, model, num_iterations=1000, batch_size=256, device=torch.device("cpu")):
    # --- Extract problem ----
    B = problem['B']
    domain_bounds = problem['domain_bounds']
    target_bounds = problem['target_bounds']

    # --- Set optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    domain_V = compute_volume(domain_bounds)
    target_V = compute_volume(target_bounds)
    
    # --- Select deterministic samples over domain and target ---
    x_dom, p0_dom, x_tar, x_remains, p0_remains = get_samples_determin(problem, model, batch_size)

    # --- Run ---
    best_loss = np.inf
    for it in range(num_iterations):
        optimizer.zero_grad()

        # # --- Augment by checking violation on grid
        # if(it % 100 == 0):
        #     _x_vio, _p0_vio, x_remains, p0_remains = aug_samples_violate(problem, model, x_remains, p0_remains, batch_size)
        #     if(_x_vio is not None):
        #         x_dom   = torch.cat([x_dom,   _x_vio], dim=0)         
        #         p0_dom  = torch.cat([p0_dom, _p0_vio], dim=0)
        #         print("[check] sample size:", x_dom.shape[0], x_remains.shape[0])
            
        # --- Augment samples with random points --- 
        x_dom_train, p0_dom_train, x_tar_train = aug_samples_random(problem, x_dom, p0_dom, x_tar, batch_size)

        # --- hard constraint over domain loss ---
        p_dom  = model(x_dom_train).view(-1,)
        hc = (p_dom >= (p0_dom_train - B)) & (p_dom <= (p0_dom_train + B))
        violation_mask = ~hc  # True for datapoints that do NOT satisfy the constraint
        vio_percent = 100.0*(violation_mask.sum()/p0_dom_train.shape[0])
        if(sum(violation_mask) > 0):
            loss_full = torch.mean(0.5*(p_dom[violation_mask] - p0_dom_train[violation_mask]) ** 2)*domain_V
        else:
            loss_full = torch.zeros(1)

        # --- maximize prob. over target loss ---
        p_tar = model(x_tar_train).view(-1,)
        target_mass_estimate = torch.mean(p_tar)*target_V
        loss_region = -target_mass_estimate

        # --- Combine loss and optimize ---
        total_loss = loss_full + loss_region*1e-1
        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        # --- Print progress and update the best model ---
        if it % int(num_iterations/10) == 0:
            print(f"Iteration {it:4d}, loss: {total_loss.item()}, loss hc: {loss_full.item()}, loss reg: {loss_region.item()}")
            print(f"   violation percent: {vio_percent:.4f}%")
        if(total_loss.item() < best_loss):
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_loss = total_loss.item()
            print(f"[Saved] Iteration {it:4d}, loss: {total_loss.item()}, {loss_full.item()}, {loss_region.item()}")
            print(f"   violation percent: {vio_percent:.4f}%")
            
    # --- Return best model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Returning best model with total loss: {best_loss}")
    return model


def get_samples_determin(problem, model, batch_size):
    # unpack
    p0_flat           = problem['p0']               # np.ndarray, shape [N]
    mask_flat         = problem['mask_flat']        # boolean np.ndarray, shape [N]
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

    # 2) Top‐k p0 within mask_flat
    mask_tensor = torch.from_numpy(mask_flat).to(device)    # [N], bool
    masked_p0    = _p0_tensor[mask_tensor]                  # [M]
    masked_x     = grid_points_tensor[mask_tensor]          # [M, d]
    _, rel_idx   = torch.topk(masked_p0, batch_size, largest=True)  # [batch_size]
    # map back to global indices
    mask_idx     = torch.nonzero(mask_tensor, as_tuple=False).view(-1)  # [M]
    tar_idx      = mask_idx[rel_idx]                        # [batch_size]
    x_tar        = grid_points_tensor[tar_idx]              # [batch_size, d]
    p0_tar       = _p0_tensor[tar_idx]                      # [batch_size]

    # 3) Combine “domain” selections
    dom_idx = torch.cat([dev_idx, tar_idx], dim=0)          # [2*batch_size]
    x_dom   = torch.cat([x_dev,   x_tar], dim=0)            # [2*batch_size, d]
    p0_dom  = torch.cat([p0_dev, p0_tar], dim=0)            # [2*batch_size]

    # 4) Compute “not selected” as complement of dom_idx
    all_idx       = torch.arange(N, device=device)          # [N]
    sel_mask      = torch.zeros(N, dtype=torch.bool, device=device)
    sel_mask[dom_idx] = True
    not_sel_idx   = all_idx[~sel_mask]                     # [N - 2*batch_size]
    x_not_select  = grid_points_tensor[not_sel_idx]         # [...]
    p0_not_select = _p0_tensor[not_sel_idx]                 # [...]

    return x_dom, p0_dom, x_tar, x_not_select, p0_not_select


def aug_samples_random(problem, x_dom, p0_dom, x_tar, batch_size):
    constants = problem['constants']
    constants = problem['constants']
    t = problem['time']
    p_net = problem['p_net'] 
    domain_bounds = problem['domain_bounds']
    target_bounds = problem['target_bounds']

    _x_dom_rand = constants.sample_points(batch_size, domain_bounds)
    _t_dom_rand = torch.full((batch_size, 1), t)
    _p0_dom_rand = p_net(_x_dom_rand, _t_dom_rand).view(-1,)

    _x_tar_rand = constants.sample_points(batch_size, target_bounds)
    _t_tar_rand = torch.full((batch_size, 1), t)
    _p0_dom_rand = p_net(_x_tar_rand, _t_tar_rand).view(-1,)

    x_dom_train = torch.cat((x_dom, _x_dom_rand, _x_tar_rand), dim=0)
    p0_dom_train = torch.cat((p0_dom, _p0_dom_rand, _p0_dom_rand), dim=0)
    x_tar_train = torch.cat((x_tar, _x_tar_rand), dim=0)

    return x_dom_train, p0_dom_train, x_tar_train


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
        print("No violations found.")
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

    print(f"Selected {k} violations out of {N} remains.")
    return x_vio, p0_vio, x_rem_new, p0_rem_new
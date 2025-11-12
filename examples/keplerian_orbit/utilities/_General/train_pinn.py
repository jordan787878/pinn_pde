import numpy as np
import torch
import time
import copy
from torch.distributions import Categorical
torch.set_default_device("cpu")
device = torch.device("cpu")


### helper functions ###

@torch.no_grad()
def sample_gmm_per_time_multi(p_net, constants, N_times, S_per_time, device, 
                              return_reshaped=False, only_t0=False):
    """
    For each time t_i ~ Uniform[0, T_end), draw S_per_time samples from the *mixture* p(x | t_i).
    Vectorized: runs the model once on the N_times unique t_i, then repeats parameters S times.

    Returns:
      x: (N_times*S_per_time, D)  [or (N_times, S_per_time, D) if return_reshaped]
      t: (N_times*S_per_time, 1)  [or (N_times, S_per_time, 1)]
    """
    N = int(N_times)
    S = int(S_per_time)
    D = int(len(constants.N_MEAN_I))
    T_end = float(getattr(p_net, "T_end", float(constants.T_PRIME_SPAN[-1])))

    # 1) Sample unique times and normalize
    if only_t0:
        t   = torch.zeros(N, 1, device=device, dtype=torch.float32)
        tau = torch.zeros_like(t)                     # normalized time = 0
    else:
        t   = torch.rand(N, 1, device=device, dtype=torch.float32) * T_end
        tau = t / T_end

    # 2) Get GMM params for the N unique times (vectorized)
    #    means_x: (N,K,D), Ls_x: (N,K,D,D), logw: (N,K)
    means_z, Ls_z, logw = p_net.params(tau)
    means_x, Ls_x       = p_net._zparams_to_xparams(tau, means_z, Ls_z)

    # 3) Repeat each row S times so we can draw S samples per time without re-running the net
    means_x_rep = means_x.repeat_interleave(S, dim=0)   # (N*S, K, D)
    Ls_x_rep    = Ls_x.repeat_interleave(S, dim=0)      # (N*S, K, D, D)
    logw_rep    = logw.repeat_interleave(S, dim=0)      # (N*S, K)
    t_rep       = t.repeat_interleave(S, dim=0)         # (N*S, 1)

    # 4) Sample a component per (time,sample) using logits (stable)
    comp = Categorical(logits=logw_rep).sample()        # (N*S,)
    idx  = torch.arange(N*S, device=device)

    # 5) Pick μ_k and L_k for each chosen component, then sample x = μ + L ε
    mu = means_x_rep[idx, comp, :]                      # (N*S, D)
    L  = Ls_x_rep[idx, comp, :, :]                      # (N*S, D, D)

    eps = torch.randn(N*S, D, device=device, dtype=mu.dtype)
    x   = mu + (L @ eps.unsqueeze(-1)).squeeze(-1)      # (N*S, D)

    if return_reshaped:
        x = x.view(N, S, D)
        t_rep = t_rep.view(N, S, 1)

    return x, t_rep

@torch.no_grad()
def rar_candidate_pool_mixed(p_net,
                             constants,
                             N_cand,
                             x_range,
                             bias_fac=0.0,
                             sample_uniform_fn=None,   # function that returns (x_u, t_u) uniform
                             device="cpu",
                             only_t0=False):
    """
    Build a candidate pool for RAR by mixing:
      - (1-alpha) fraction from uniform sampler over domain/time
      - alpha fraction from the current model GMM in x-space

    Returns:
      x_cand: (N_cand, D)
      t_cand: (N_cand, 1)
    """
    N_model = int(round(bias_fac * N_cand))
    N_uni   = max(0, N_cand - N_model)

    xs, ts = [], []

    if N_uni > 0:
        assert sample_uniform_fn is not None, "Provide sample_uniform_fn to draw uniform residual points."
        x_u, t_u = sample_uniform_fn(N_uni)            # expect tensors on CPU/GPU already
        xs.append(x_u.to(device))
        ts.append(t_u.to(device))
        # print("sample from uniform")

    if N_model > 0:
        # x_m, t_m = sample_from_model_gmm_xspace(p_net, constants, N_model, device=device)
        S_per_time = 100
        N_model_time = int(N_model / S_per_time)
        x_m, t_m = sample_gmm_per_time_multi(p_net, constants, N_model_time, S_per_time, 
            device=device, only_t0=only_t0)
        
        lo = x_range[:, 0].view(1, -1)            # (1, D) for broadcasting
        hi = x_range[:, 1].view(1, -1)            # (1, D)
        viol = (x_m < lo) | (x_m > hi)                 # (N, D) True where out of range
        pct  = 100.0 * viol.float().mean().item()  # % of elements out of range
        # if(pct > 0.):
        #     print(f"[range] {pct:.2f}% out of bounds")
        
        x_sat = torch.maximum(torch.minimum(x_m, hi), lo)
        viol = (x_sat < lo) | (x_sat > hi)                 # (N, D) True where out of range
        pct  = 100.0 * viol.float().mean().item()  # % of elements out of range
        if(pct > 0.):
            print(f"[range] {pct:.2f}% out of bounds")

        xs.append(x_sat)
        ts.append(t_m)
        # print("sample from gmm")

    x_cand = torch.cat(xs, dim=0)
    t_cand = torch.cat(ts, dim=0)
    # np.testing.assert_equal(x_cand.shape[0], N_cand)

    # Random shuffle to avoid ordering bias
    perm = torch.randperm(x_cand.shape[0], device=device)
    return x_cand[perm], t_cand[perm]

def rar_append_with_cap(x_buf, t_buf, x_new, t_new, cap: int):
    """
    Append new rows to buffers; drop oldest if exceeding 'cap'.
    Buffers store VALUES ONLY (no grad history).
    """
    with torch.no_grad():
        x_cat = torch.cat([x_buf, x_new.detach()], dim=0)
        t_cat = torch.cat([t_buf, t_new.detach()], dim=0)
        excess = x_cat.shape[0] - int(cap)
        if excess > 0:
            x_cat = x_cat[excess:, :]
            t_cat = t_cat[excess:, :]
    return x_cat, t_cat

### End of helper functions ###


### Vanilla PINN Training ###

def train_pinn_vanilla(p_net, config):
    p_net.train()
    p_net.to(device)
    iterations = config["iterations"]
    sample_ic = config["sample_ic"]
    sample_res = config["sample_res"]
    p_init = config["p_ic"]
    res_func = config["res_func"]
    res_weight = config["res_weight"]
    save_path = config["save_path"]
    normalize = config["loss_normalize"]
    N0_samples_initial = config["N0_samples_initial"]
    Nr_samples_initial = config["Nr_samples_initial"]
    iterations_per_decay = config["iterations_per_decay"]
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    min_loss = np.inf
    loss_history = []
    best_state = None
    best_epoch = -1
    min_loss = float("inf")
    
    start_time = time.time(); train_time = start_time
    for epoch in range(1, iterations+1):
        optimizer.zero_grad(set_to_none=True)    
        
        # Sample new points at each epoch for better generalization
        x_bc_new, t_bc_new = sample_ic(N0_samples_initial)
        # np.testing.assert_equal(x_bc_new.shape[0], N0_samples_initial)
        x_res_new, t_res_new = sample_res(Nr_samples_initial)
        # np.testing.assert_equal(x_res_new.shape[0], Nr_samples_initial)
        
        # Combine with RAR points and ensure they are on the correct device
        x_bc = x_bc_new.detach()
        t_bc = t_bc_new.detach()
        x_res = x_res_new.detach().requires_grad_(True)
        t_res = t_res_new.detach().requires_grad_(True)
        del x_bc_new, t_bc_new, x_res_new, t_res_new
        
        # --- Loss based on initial conditions ---
        with torch.no_grad():
            p_i = p_init(x_bc)
        phat_i = p_net(x_bc, t_bc)
        mse_u = (phat_i/normalize - p_i/normalize).pow(2).mean()
        # --- Loss based on PDE ---
        res_p = res_func(x_res, t_res, p_net, beta=1.0)
        mse_res = (res_p / normalize).pow(2).mean()

        # --- Total Loss ---
        loss = mse_u + res_weight * mse_res

        # --- Optimize ---
        if not torch.isfinite(loss):
            print("[NaN] loss blew up; skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        loss_history.append(loss.item())
        # guard bad grads
        bad_grad = False
        for p in p_net.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True; break
        if bad_grad:
            print("[NaN] gradient non-finite; skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(p_net.parameters(), max_norm=1.0)
        optimizer.step()
        del p_i, phat_i, res_p
        
        # --- Learning rate decay ---
        # if (epoch % 100 == 0): print(epoch) #[debug]
        if (epoch) % iterations_per_decay == 0:
            scheduler.step()
            print(f"[info] Training epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Save min loss model and update beta ---
        if loss.data < min_loss:
            train_time = time.time() - start_time
            print(f"--- Best Epoch: {epoch}, Loss: {loss.item():.4e}, IC: {mse_u.item():.4e}, Res: {mse_res.item():.4e} ---")
            min_loss = loss.item()
            best_epoch = epoch
            best_state = copy.deepcopy(p_net.state_dict())
    if save_path is not None and best_state is not None:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_state,
            'loss_history': loss_history,
            'train_time': train_time,
        }, save_path + "/p_net.pth")

def train_pinn_error_vanilla(networks, config):
    """
    Improved training loop for the error network e1_net.
    Uses same structure as train_pinn_sol_v0_scaled_improved.
    """
    # --- Unpack ---
    p_net, e1_net = networks
    e1_net.train().to(device)
    p_net.eval().to(device)   # frozen reference
    # Freeze p_net parameters
    for p in p_net.parameters():
        p.requires_grad_(False)

    # -------- SAFETY CHECK 1: all p_net params frozen --------
    assert all(p.requires_grad is False for p in p_net.parameters()), "p_net has trainable params!"

    # --- Config ---
    iterations = config["iterations"]
    p_init   = config["p_ic"]
    diff_opt = config["res_func"]
    e1_path  = config["save_path"] + "/e1_net.pth"
    # e1_path_inter = configs.get("save_path_inter", None)
    sample_ic = config["sample_ic"]
    sample_res = config["sample_res"]
    normalize = config["loss_normalize"]
    N0_samples_initial = config["N0_samples_initial"]
    Nr_samples_initial = config["Nr_samples_initial"]
    iterations_per_decay = config["iterations_per_decay"]
    
    # --- Optimizer ---
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # -------- SAFETY CHECK 2: optimizer only sees e1_net params --------
    e1_param_ids = {id(p) for p in e1_net.parameters()}
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert opt_param_ids <= e1_param_ids, "Optimizer includes non-e1_net parameters!"
    
    # --- Inits ---
    min_loss = float("inf")
    loss_history = []
    beta = np.float32(1.0)
    # Best model dict
    best_model_dict = {
        "epoch": -1,
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "loss_history": None,
        "train_time": None,
    }

    start_time = time.time()
    for epoch in range(1, iterations+1):
        optimizer.zero_grad()
        
        # Sample new points at each epoch for better generalization
        x_bc_new, t_bc_new = sample_ic(N0_samples_initial)
        # np.testing.assert_equal(x_bc_new.shape[0], N0_samples_initial)
        x_res_new, t_res_new = sample_res(Nr_samples_initial)
        # np.testing.assert_equal(x_res_new.shape[0], Nr_samples_initial)
        
        x_bc = x_bc_new.detach()
        t_bc = t_bc_new.detach()
        x_res = x_res_new.detach().requires_grad_(True)
        t_res = t_res_new.detach().requires_grad_(True)
        del x_bc_new, t_bc_new, x_res_new, t_res_new
        
        # --- IC loss: ehat ~ (p_init - p_net) ---
        with torch.no_grad():
            p_i = p_init(x_bc)
            phat_i = p_net(x_bc, t_bc)
            e_i = p_i - phat_i
        ehat_i = e1_net(x_bc, t_bc)
        mse_ic = mse(ehat_i / normalize, e_i / normalize)
        
        # --- PDE loss: diff_opt[e1] ~ -diff_opt[p] ---
        res_p = diff_opt(x_res, t_res, p_net, beta=beta).detach()
        res_e = diff_opt(x_res, t_res, e1_net, beta=beta)
        mse_res = mse(res_e / normalize, (-res_p) / normalize)
        
        # --- Total loss ---
        loss = mse_ic + mse_res
        loss_history.append(loss.item())
        
        # --- Backward ---
        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(e1_net.parameters(), max_norm=1.0)
        optimizer.step()

        # # -------- SAFETY CHECK 5: verify p_net weights unchanged after step --------
        # _assert_pnet_unchanged(f"epoch {epoch+1}/post-step")
        
        # --- LR decay ---
        if (epoch) % iterations_per_decay == 0:
            scheduler.step()
            print(f"[info] epoch {epoch}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        # --- Save best ---
        if loss.item() < 0.95 * min_loss:
            train_time = time.time() - start_time
            print(f"--- Save Epoch {epoch}, Loss={loss.item():.4e}, "
                  f"IC={mse_ic.item():.4e}, Res={mse_res.item():.4e} ---")
            # Res={mse_res.item():.4f}
            
            best_model_dict.update({
                "epoch": epoch,
                "model_state_dict": e1_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                "train_time": train_time,
            })
            torch.save(best_model_dict, e1_path)
            min_loss = loss.item()

### Proposed Training ###

def train_pinn(p_net, config):
    p_net.train()
    p_net.to(device)
    x_dim = p_net.D
    constants = config["constants"]
    iterations = config["iterations"]
    sample_ic = config["sample_ic"]
    sample_res = config["sample_res"]
    p_init = config["p_ic"]
    res_func = config["res_func"]
    res_weight = config["res_weight"]
    save_path = config["save_path"]
    normalize = config["loss_normalize"]
    reg_tv = config["reg_tv"]
    N0_samples_initial = config["N0_samples_initial"]
    Nr_samples_initial = config["Nr_samples_initial"]
    iterations_per_decay = config["iterations_per_decay"]
    N_RAR = config["N_RAR"]
    N_RAR_TO_ADD = config["N_RAR_TO_ADD"]
    N_RAR_CAP = config["N_RAR_CAP"]
    iterations_per_rar = config["iterations_per_rar"]
    RAR_eps = config["RAR_eps"]
    beta_incre = config["beta_incre"]
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    min_loss = np.inf
    loss_history = []
    beta = np.float32(0.0)
    inter_count = 0
    x_bc_rar = torch.empty(0, x_dim, device=device)
    t_bc_rar = torch.empty(0, 1, device=device)
    x_res_rar = torch.empty(0, x_dim, device=device)
    t_res_rar = torch.empty(0, 1, device=device)
    FLAG = False
    beta = np.float32(0.0)

    best_state = None
    best_epoch = -1
    min_loss = float("inf")
    
    start_time = time.time()
    for epoch in range(1, iterations+1):
        optimizer.zero_grad(set_to_none=True)    
        
        # Sample new points at each epoch for better generalization
        x_bc_new, t_bc_new = sample_ic(N0_samples_initial)
        # np.testing.assert_equal(x_bc_new.shape[0], N0_samples_initial)
        x_res_new, t_res_new = sample_res(Nr_samples_initial)
        # np.testing.assert_equal(x_res_new.shape[0], Nr_samples_initial)
        
        # Combine with RAR points and ensure they are on the correct device
        x_bc = torch.cat((x_bc_new.to(device), x_bc_rar), dim=0).detach()
        t_bc = torch.cat((t_bc_new.to(device), t_bc_rar), dim=0).detach()
        x_res = torch.cat((x_res_new.to(device), x_res_rar), dim=0).detach().requires_grad_(True)
        t_res = torch.cat((t_res_new.to(device), t_res_rar), dim=0).detach().requires_grad_(True)
        del x_bc_new, t_bc_new, x_res_new, t_res_new
        
        # --- Loss based on initial conditions ---
        with torch.no_grad():
            # p_i_np = p_init(x_bc.detach().cpu().numpy())
            # p_i = torch.from_numpy(p_i_np).to(device=device, dtype=torch.float32)
            p_i = p_init(x_bc)
        phat_i = p_net(x_bc, t_bc)
        eps = torch.finfo(phat_i.dtype).tiny
        ph = phat_i.clamp_min(eps)
        p  = p_i.clamp_min(eps)
        # Standard Hellinger^2 often has a 1/2 factor; optional for training
        hellinger_u = 0.5 * (torch.sqrt(ph)/ normalize - torch.sqrt(p)/ normalize).pow(2).mean()
        mse_u = hellinger_u
        # mse_u = mse_cost_function(phat_i / normalize, p_i / normalize)

        # --- Loss based on PDE ---
        res_p = res_func(x_res, t_res, p_net, beta=beta)
        mse_res = (res_p / normalize).pow(2).mean()

        # --- Total Loss ---
        loss = mse_u + res_weight * mse_res
        if(reg_tv is not None):
            tv_t = torch.autograd.grad(res_p, t_res, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
            mse_tv = (tv_t / normalize).pow(2).mean()
            loss += reg_tv * mse_tv

        # --- Optimize ---
        if not torch.isfinite(loss):
            print("[NaN] loss blew up; skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        loss_history.append(loss.item())
        # guard bad grads
        bad_grad = False
        for p in p_net.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True; break
        if bad_grad:
            print("[NaN] gradient non-finite; skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(p_net.parameters(), max_norm=1.0)
        optimizer.step()
        del p_i, phat_i, res_p
        
        # --- Learning rate decay ---
        if (epoch % 100 == 0): print(epoch) #[debug]
        if (epoch) % iterations_per_decay == 0:
            scheduler.step()
            print(f"[info] Training epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Save min loss model and update beta ---
        if loss.data < min_loss:
            train_time = time.time() - start_time
            if(reg_tv is not None):
                print(f"--- Best Epoch: {epoch}, Loss: {loss.item():.4e}, IC: {mse_u.item():.4e}, Res: {mse_res.item():.4e} & {mse_tv.item():.4e}, Beta: {beta:.2f} ---")
            else:
                print(f"--- Best Epoch: {epoch}, Loss: {loss.item():.4e}, IC: {mse_u.item():.4e}, Res: {mse_res.item():.4e}, Beta: {beta:.2f} ---")
            min_loss = loss.item()
            beta = min(1.0, beta + beta_incre)
            FLAG = True
            best_epoch = epoch
            best_state = copy.deepcopy(p_net.state_dict())
            if save_path is not None and best_state is not None:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_state,
                    'loss_history': loss_history,
                    'train_time': train_time,
                }, save_path + "/p_net.pth")

        # --- RAR ---
        if (epoch % iterations_per_rar == 0 and FLAG) or (epoch == 1):
            FLAG = False # reset flag
            
            # Add IC points
            with torch.no_grad():
                x_bc_check, t_bc_check = sample_ic(N_RAR)                
                # p_i_np = p_init(x_bc_check.detach().cpu().numpy())
                # p_i = torch.from_numpy(p_i_np).to(device=device, dtype=torch.float32)
                p_i = p_init(x_bc_check)
                phat_i = p_net(x_bc_check, t_bc_check)
                ic_errors = torch.abs(p_i - phat_i) / normalize
                max_error_ic = ic_errors.max().item()
                if max_error_ic > RAR_eps:
                    # max_indices = torch.topk(ic_errors.squeeze(), 10).indices # NOTE: change this
                    # x_bc_rar = torch.cat((x_bc_rar, x_bc_check[max_indices, :]), dim=0)
                    # t_bc_rar = torch.cat((t_bc_rar, t_bc_check[max_indices]), dim=0)
                    topk  = torch.topk(ic_errors.squeeze(), k=N_RAR_TO_ADD)
                    idx   = topk.indices
                    x_bc_rar_new = x_bc_check.detach()[idx, :]
                    t_bc_rar_new = t_bc_check.detach()[idx, :]
                    x_bc_rar, t_bc_rar = rar_append_with_cap(x_bc_rar, t_bc_rar, x_bc_rar_new, t_bc_rar_new, cap=N_RAR_CAP)
                    print(f"... RAR IC , Max IC error: {max_error_ic:.4f}, t:", torch.max(t_bc_rar_new))
                    del topk, idx, x_bc_rar_new, t_bc_rar_new
                del p_i, phat_i, ic_errors, max_error_ic, x_bc_check, t_bc_check

            # Add Residual points
            x_cand, t_cand = sample_res(N_RAR)
            x_cand = x_cand.detach().requires_grad_(True)
            t_cand = t_cand.detach().requires_grad_(True)
            res_vals = res_func(x_cand, t_cand, p_net, beta=beta)
            res_err  = torch.abs(res_vals) / normalize
            max_err  = res_err.max().item()
            if max_err > RAR_eps:
                topk  = torch.topk(res_err.squeeze(), k=N_RAR_TO_ADD)
                idx   = topk.indices
                x_res_rar_new = x_cand.detach()[idx, :]
                t_res_rar_new = t_cand.detach()[idx, :]
                # 4) append with FIFO cap
                x_res_rar, t_res_rar = rar_append_with_cap(x_res_rar, t_res_rar, x_res_rar_new, t_res_rar_new, cap=N_RAR_CAP)
                print(f"... RAR RES, Max residual error: {max_err:.4f}, t:", t_res_rar_new[0:3].data)
                del topk, idx, x_res_rar_new, t_res_rar_new
            del res_vals, res_err, max_err, x_cand, t_cand
    
    # after training, write a single file
    if save_path is not None and best_state is not None:
        # torch.save({
        #     'epoch': best_epoch,
        #     'model_state_dict': best_state,
        #     'loss_history': loss_history,
        #     'train_time': train_time,
        # }, save_path + "/p_net.pth")
        np.savez(save_path+"/p_net-RARsamples.npz", 
            X_BC_RAR=x_bc_rar, X_RES_RAR=x_res_rar, T_RES_RAR=t_res_rar)
        
def train_pinngmm(p_net, config, beta_0=0.):
    p_net.train()
    p_net.to(device)
    x_dim = p_net.D
    constants = config["constants"]
    iterations = config["iterations"]
    sample_ic = config["sample_ic"]
    sample_res = config["sample_res"]
    p_init = config["p_ic"]
    res_func = config["res_func"]
    res_weight = config["res_weight"]
    save_path = config["save_path"]
    normalize = config["loss_normalize"]
    reg_tv = config["reg_tv"]
    N0_samples_initial = config["N0_samples_initial"]
    Nr_samples_initial = config["Nr_samples_initial"]
    iterations_per_decay = config["iterations_per_decay"]
    N_RAR = config["N_RAR"]
    N_RAR_TO_ADD = config["N_RAR_TO_ADD"]
    N_RAR_CAP = config["N_RAR_CAP"]
    iterations_per_rar = config["iterations_per_rar"]
    RAR_eps = config["RAR_eps"]
    beta_incre = config["beta_incre"]
    bias_fac = config["bias_fac"]
    x_range = config["x_range"]
    
    mean_head_params  = list(p_net.mean_head.parameters())
    tri_head_params   = list(p_net.tri_head.parameters())
    weight_head_params= list(p_net.weight_head.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": mean_head_params,   "lr": 1e-3},
            {"params": tri_head_params,    "lr": 1e-4},
            {"params": weight_head_params, "lr": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    sample_uniform_fn = lambda n: sample_res(n)

    best_state = None
    best_epoch = -1
    min_loss = np.inf
    loss_history = []
    x_bc_rar = torch.empty(0, x_dim, device=device)
    t_bc_rar = torch.empty(0, 1, device=device)
    x_res_rar = torch.empty(0, x_dim, device=device)
    t_res_rar = torch.empty(0, 1, device=device)
    FLAG = False
    beta = np.float32(beta_0)
    
    start_time = time.time()
    print("[check] bias_fac: ", bias_fac)
    for epoch in range(1, iterations+1):
        optimizer.zero_grad(set_to_none=True)    
        
        # Sample new points at each epoch for better generalization
        x_bc_new, t_bc_new = sample_ic(N0_samples_initial)
        # np.testing.assert_equal(x_bc_new.shape[0], N0_samples_initial)
        x_res_new, t_res_new = rar_candidate_pool_mixed(
                p_net, constants, N_cand=Nr_samples_initial, 
                x_range = x_range,
                bias_fac=bias_fac,
                sample_uniform_fn=sample_uniform_fn, 
                device=device)
        
        # Combine with RAR points and ensure they are on the correct device
        x_bc = torch.cat((x_bc_new.to(device), x_bc_rar), dim=0).detach()
        t_bc = torch.cat((t_bc_new.to(device), t_bc_rar), dim=0).detach()
        x_res = torch.cat((x_res_new.to(device), x_res_rar), dim=0).detach().requires_grad_(True)
        t_res = torch.cat((t_res_new.to(device), t_res_rar), dim=0).detach().requires_grad_(True)
        del x_bc_new, t_bc_new, x_res_new, t_res_new
        
        # --- Loss based on initial conditions ---
        with torch.no_grad():
            # p_i_np = p_init(x_bc.detach().cpu().numpy())
            # p_i = torch.from_numpy(p_i_np).to(device=device, dtype=torch.float32)
            p_i = p_init(x_bc)
        phat_i = p_net(x_bc, t_bc)
        eps = torch.finfo(phat_i.dtype).tiny
        ph = phat_i.clamp_min(eps)
        p  = p_i.clamp_min(eps)
        # Standard Hellinger^2 often has a 1/2 factor; optional for training
        hellinger_u = 0.5 * (torch.sqrt(ph)/ normalize - torch.sqrt(p)/ normalize).pow(2).mean()
        mse_u = hellinger_u
        # mse_u = mse_cost_function(phat_i / normalize, p_i / normalize)

        # --- Loss based on PDE ---
        res_p = res_func(x_res, t_res, p_net, beta=beta)
        mse_res = (res_p / normalize).pow(2).mean()

        # --- Total Loss ---
        loss = mse_u + res_weight * mse_res
        if(reg_tv is not None):
            tv_t = torch.autograd.grad(res_p, t_res, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
            mse_tv = (tv_t / normalize).pow(2).mean()
            loss += reg_tv * mse_tv

        # --- Optimize ---
        if not torch.isfinite(loss):
            print("[NaN] loss blew up; skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        loss_history.append(loss.item())
        # guard bad grads
        bad_grad = False
        for p in p_net.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True; break
        if bad_grad:
            print("[NaN] gradient non-finite; skipping update")
            optimizer.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(p_net.parameters(), max_norm=1.0)
        optimizer.step()
        del p_i, phat_i, res_p
        
        # --- Learning rate decay ---
        if (epoch) % iterations_per_decay == 0:
            scheduler.step()
            print(f"[info] Training epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Save min loss model and update beta ---
        if loss.data < min_loss:
            train_time = time.time() - start_time
            if(reg_tv is not None):
                print(f"--- Save Epoch: {epoch}, Loss: {loss.item():.4e}, IC: {mse_u.item():.4e}, Res: {mse_res.item():.4e} & {mse_tv.item():.4e}, Beta: {beta:.2f} ---")
            else:
                print(f"--- Save Epoch: {epoch}, Loss: {loss.item():.4e}, IC: {mse_u.item():.4e}, Res: {mse_res.item():.4e}, Beta: {beta:.2f} ---")
            best_epoch = epoch
            best_state = copy.deepcopy(p_net.state_dict())
            # if save_path is not None:
            #     torch.save({
            #         'epoch': epoch, 'model_state_dict': p_net.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss_history': loss_history, 'train_time': train_time,
            #     }, save_path+"/p_net.pth")
            #     np.savez(save_path+"/p_net-RARsamples.npz",
            #              X_BC_RAR=x_bc_rar,
            #              X_RES_RAR=x_res_rar,
            #              T_RES_RAR=t_res_rar)
            min_loss = loss.data
            beta = min(1.0, beta + beta_incre)
            FLAG = True

        # --- RAR ---
        if (epoch % iterations_per_rar == 0 and FLAG) or (epoch == 1):
            FLAG = False # reset flag
            
            # Add IC points
            with torch.no_grad():
                x_bc_check, t_bc_check = sample_ic(N_RAR)                
                # p_i_np = p_init(x_bc_check.detach().cpu().numpy())
                # p_i = torch.from_numpy(p_i_np).to(device=device, dtype=torch.float32)
                p_i = p_init(x_bc_check)
                phat_i = p_net(x_bc_check, t_bc_check)
                ic_errors = torch.abs(p_i - phat_i) / normalize
                max_error_ic = ic_errors.max().item()
                if max_error_ic > RAR_eps:
                    # max_indices = torch.topk(ic_errors.squeeze(), 10).indices # NOTE: change this
                    # x_bc_rar = torch.cat((x_bc_rar, x_bc_check[max_indices, :]), dim=0)
                    # t_bc_rar = torch.cat((t_bc_rar, t_bc_check[max_indices]), dim=0)
                    topk  = torch.topk(ic_errors.squeeze(), k=N_RAR_TO_ADD)
                    idx   = topk.indices
                    x_bc_rar_new = x_bc_check.detach()[idx, :]
                    t_bc_rar_new = t_bc_check.detach()[idx, :]
                    x_bc_rar, t_bc_rar = rar_append_with_cap(x_bc_rar, t_bc_rar, x_bc_rar_new, t_bc_rar_new, cap=N_RAR_CAP)
                    print(f"... RAR IC , Max IC error: {max_error_ic:.4f}, t:", torch.max(t_bc_rar_new))
                    del topk, idx, x_bc_rar_new, t_bc_rar_new
                del p_i, phat_i, ic_errors, max_error_ic, x_bc_check, t_bc_check

            # Add Residual points
            x_cand, t_cand = rar_candidate_pool_mixed(
                p_net, constants, N_cand=N_RAR, 
                x_range = x_range,
                bias_fac=bias_fac,
                sample_uniform_fn=sample_uniform_fn, device=device)
            x_cand = x_cand.detach().requires_grad_(True)
            t_cand = t_cand.detach().requires_grad_(True)
            res_vals = res_func(x_cand, t_cand, p_net, beta=beta)
            res_err  = torch.abs(res_vals) / normalize
            max_err  = res_err.max().item()
            if max_err > RAR_eps:
                topk  = torch.topk(res_err.squeeze(), k=N_RAR_TO_ADD)
                idx   = topk.indices
                x_res_rar_new = x_cand.detach()[idx, :]
                t_res_rar_new = t_cand.detach()[idx, :]
                # 4) append with FIFO cap
                x_res_rar, t_res_rar = rar_append_with_cap(x_res_rar, t_res_rar, x_res_rar_new, t_res_rar_new, cap=N_RAR_CAP)
                print(f"... RAR RES, Max residual error: {max_err:.4f}, t:", t_res_rar_new[0:3].data)
                del topk, idx, x_res_rar_new, t_res_rar_new
            del res_vals, res_err, max_err, x_cand, t_cand
    # After training
    if save_path is not None and best_state is not None:
        torch.save({
            'epoch': best_epoch, 'model_state_dict': best_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history, 'train_time': train_time,
        }, save_path+"/p_net.pth")
        np.savez(save_path+"/p_net-RARsamples.npz", 
            X_BC_RAR=x_bc_rar, X_RES_RAR=x_res_rar, T_RES_RAR=t_res_rar)
    
def train_pinn_error(networks, config):
    """
    Improved training loop for the error network e1_net.
    Uses same structure as train_pinn_sol_v0_scaled_improved.
    """
    # --- Unpack ---
    p_net, e1_net = networks
    e1_net.train().to(device)
    p_net.eval().to(device)   # frozen reference
    # Freeze p_net parameters
    for p in p_net.parameters():
        p.requires_grad_(False)
    x_dim = p_net.D

    # -------- SAFETY CHECK 1: all p_net params frozen --------
    assert all(p.requires_grad is False for p in p_net.parameters()), "p_net has trainable params!"

    # --- Config ---
    constants = config["constants"]
    iterations = config["iterations"]
    p_init   = config["p_ic"]
    diff_opt = config["res_func"]
    e1_path  = config["save_path"] + "/e1_net.pth"
    # e1_path_inter = configs.get("save_path_inter", None)
    sample_ic = config["sample_ic"]
    sample_res = config["sample_res"]
    normalize = config["loss_normalize"]
    N0_samples_initial = config["N0_samples_initial"]
    Nr_samples_initial = config["Nr_samples_initial"]
    iterations_per_decay = config["iterations_per_decay"]
    N_RAR = config["N_RAR"]
    N_RAR_TO_ADD = config["N_RAR_TO_ADD"]
    N_RAR_CAP = config["N_RAR_CAP"]
    iterations_per_rar = config["iterations_per_rar"]
    RAR_eps = config["RAR_eps"]
    beta_incre = config["beta_incre"]
    bias_fac = config["bias_fac"]
    x_range = config["x_range"]
    sample_uniform_fn = lambda n: sample_res(n)
    
    # --- Optimizer ---
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # -------- SAFETY CHECK 2: optimizer only sees e1_net params --------
    e1_param_ids = {id(p) for p in e1_net.parameters()}
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert opt_param_ids <= e1_param_ids, "Optimizer includes non-e1_net parameters!"
    
    # --- Inits ---
    min_loss = float("inf")
    loss_history = []
    FLAG = False
    # inter_count = 0
    beta = np.float32(0.0)
    # Buffers for RAR points
    x_bc_rar = torch.empty(0, x_dim, device=device)
    t_bc_rar = torch.empty(0, 1, device=device)
    x_res_rar = torch.empty(0, x_dim, device=device)
    t_res_rar = torch.empty(0, 1, device=device)
    # Best model dict
    best_model_dict = {
        "epoch": -1,
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "loss_history": None,
        "train_time": None,
    }

    # # -------- SAFETY CHECK 3: snapshot p_net weights --------
    # with torch.no_grad():
    #     pnet_ref = {k: v.detach().cpu().clone() for k, v in p_net.state_dict().items()}
    # def _assert_pnet_unchanged(when: str, atol=0.0, rtol=0.0):
    #     """Verifies p_net.state_dict matches the frozen snapshot exactly (float-exact)."""
    #     cur = p_net.state_dict()
    #     for k, v0 in pnet_ref.items():
    #         v = cur[k].detach().cpu()
    #         if v.dtype.is_floating_point:
    #             if not torch.allclose(v, v0, atol=atol, rtol=rtol):
    #                 raise RuntimeError(f"p_net changed at {when}: param {k} drift detected.")
    #         else:
    #             if not torch.equal(v, v0):
    #                 raise RuntimeError(f"p_net changed at {when}: param {k} drift detected.")
    # # First check before training
    # _assert_pnet_unchanged("start")
    
    start_time = time.time()
    for epoch in range(1, iterations+1):
        optimizer.zero_grad()
        
        # --- New samples ---
        x_bc_new, t_bc_new = sample_ic(N0_samples_initial)
        # x_res_new, t_res_new = constants.sample_res_points_scaled(Nr_samples_initial)
        x_res_new, t_res_new = rar_candidate_pool_mixed(
            p_net, constants, N_cand=Nr_samples_initial, 
            x_range = x_range,
            bias_fac=bias_fac,
            sample_uniform_fn=sample_uniform_fn, 
            device=device)
        
        x_bc = torch.cat((x_bc_new.to(device), x_bc_rar), dim=0).detach()
        t_bc = torch.cat((t_bc_new.to(device), t_bc_rar), dim=0).detach()
        x_res = torch.cat((x_res_new.to(device), x_res_rar), dim=0).detach().requires_grad_(True)
        t_res = torch.cat((t_res_new.to(device), t_res_rar), dim=0).detach().requires_grad_(True)
        del x_bc_new, t_bc_new, x_res_new, t_res_new
        
        # --- IC loss: ehat ~ (p_init - p_net) ---
        with torch.no_grad():
            # p_i_np = p_init(constants, x_bc.detach().cpu().numpy())
            # p_i = torch.tensor(p_i_np, dtype=torch.float32, device=device)
            p_i = p_init(x_bc)
            phat_i = p_net(x_bc, t_bc)
            e_i = p_i - phat_i
        ehat_i = e1_net(x_bc, t_bc)
        mse_ic = mse(ehat_i / normalize, e_i / normalize)
        
        # --- PDE loss: diff_opt[e1] ~ -diff_opt[p] ---
        res_p = diff_opt(x_res, t_res, p_net, beta=beta).detach()
        res_e = diff_opt(x_res, t_res, e1_net, beta=beta)
        mse_res = mse(res_e / normalize, (-res_p) / normalize)
        
        # --- Total loss ---
        loss = mse_ic + mse_res
        loss_history.append(loss.item())
        
        # --- Backward ---
        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(e1_net.parameters(), max_norm=1.0)
        optimizer.step()

        # # -------- SAFETY CHECK 5: verify p_net weights unchanged after step --------
        # _assert_pnet_unchanged(f"epoch {epoch+1}/post-step")
        
        # --- LR decay ---
        if (epoch) % iterations_per_decay == 0:
            scheduler.step()
            print(f"[info] epoch {epoch}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        # --- Save best ---
        if loss.item() < 0.95 * min_loss:
            train_time = time.time() - start_time
            print(f"--- Save Epoch {epoch}, Loss={loss.item():.4e}, "
                  f"IC={mse_ic.item():.4e}, Res={mse_res.item():.4e}, Beta={beta:.2f} ---")
            # Res={mse_res.item():.4f}
            
            best_model_dict.update({
                "epoch": epoch,
                "model_state_dict": e1_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                "train_time": train_time,
            })
            torch.save(best_model_dict, e1_path)
            
            min_loss = loss.item()
            beta = min(1.0, beta + beta_incre)
            FLAG = True

        # --- RAR ---
        if (epoch % iterations_per_rar == 0 and FLAG) or (epoch == 1):
            FLAG = False
            # IC RAR
            with torch.no_grad():
                xb_chk, tb_chk = sample_ic(N_RAR)
                # p_i_np = p_init(constants, xb_chk.detach().cpu().numpy())
                # p_i = torch.tensor(p_i_np, dtype=torch.float32, device=device)
                p_i = p_init(xb_chk)
                phat_i = p_net(xb_chk, tb_chk)
                e_i = p_i - phat_i
                ehat_i = e1_net(xb_chk, tb_chk)
                ic_errors = torch.abs(e_i - ehat_i) / normalize
                max_error_ic = ic_errors.max().item()
                if max_error_ic > RAR_eps:
                    topk  = torch.topk(ic_errors.squeeze(), k=N_RAR_TO_ADD)
                    idx   = topk.indices
                    x_bc_rar_new = xb_chk.detach()[idx, :]
                    t_bc_rar_new = tb_chk.detach()[idx, :]
                    x_bc_rar, t_bc_rar = rar_append_with_cap(x_bc_rar, t_bc_rar, x_bc_rar_new, t_bc_rar_new, cap=N_RAR_CAP)
                    print(f"... RAR IC , Max IC error: {max_error_ic:.4f}, t:", torch.max(t_bc_rar_new))
            
            # Residual RAR
            # xr_chk, tr_chk = constants.sample_res_points_scaled(N_RAR)
            xr_chk, tr_chk = rar_candidate_pool_mixed(
                p_net, constants, N_cand=N_RAR, 
                x_range = x_range,
                bias_fac=bias_fac,
                sample_uniform_fn=sample_uniform_fn, device=device)
            xr_chk = xr_chk.detach().requires_grad_(True)
            tr_chk = tr_chk.detach().requires_grad_(True)
            res_p = diff_opt(xr_chk, tr_chk, p_net, beta=beta)
            res_e = diff_opt(xr_chk, tr_chk, e1_net, beta=beta)
            res_err = torch.abs(res_e + res_p) / normalize
            max_err = res_err.max().item()
            if max_err > RAR_eps:
                topk  = torch.topk(res_err.squeeze(), k=N_RAR_TO_ADD)
                idx   = topk.indices
                x_res_rar_new = xr_chk.detach()[idx, :]
                t_res_rar_new = tr_chk.detach()[idx, :]
                # 4) append with FIFO cap
                x_res_rar, t_res_rar = rar_append_with_cap(x_res_rar, t_res_rar, x_res_rar_new, t_res_rar_new, cap=N_RAR_CAP)
                print(f"... RAR RES, Max residual error: {max_err:.4f}, t:", t_res_rar_new[0:3].data)
    # End of training

### End of Standard Training ###

import torch
import numpy as np
import time
device = "cpu"


# --- Base PNet training ---
def train_pnet(p_net, configuration):
    # --- get configuration ---
    iterations = configuration["iterations"]
    train_helper_sample_ic = configuration["sample_ic"]
    train_helper_sample_res = configuration["sample_res"]
    p_init = configuration["p_ic"]
    res_func = configuration["res_func"]
    res_weight = configuration["res_weight"]
    save_path = configuration["save_path"]

    # --- setup ---
    p_net.train()
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    batch_size = 500
    min_loss = np.inf
    loss_history = []
    normalize = p_net.scale
    iterations_per_decay = 1000
    start_time = time.time()

    for epoch in range(iterations):
        optimizer.zero_grad() # to make the gradients zero

        # Loss based on boundary conditions
        x_bc, t_bc = train_helper_sample_ic(batch_size)
        u_bc = p_init(x_bc).detach()
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)

        # Loss based on PDE
        x, t = train_helper_sample_res(batch_size)
        all_zeros = torch.zeros((batch_size,1), dtype=torch.float32, requires_grad=False).to(device)
        res_out = res_func(x, t, p_net)
        mse_res = mse_cost_function(res_out/normalize, all_zeros)

        loss = mse_u + res_weight*mse_res
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.9*min_loss):
            print("save epoch:", epoch, ", loss:", loss.data, 
                ", ic: ", mse_u.data,
                ", res: ", mse_res.data)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(), 
                    'train_time': time.time() - start_time
                    }, save_path)
            min_loss = loss.item()

        loss.backward() # This is for computing gradients using backward propagation
        optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


# --- V0 PNet training ---
def train_pnet_v0(p_net, configuration):
    # --- get configuration ---
    iterations = configuration["iterations"]
    train_helper_sample_ic = configuration["sample_ic"]
    train_helper_sample_res = configuration["sample_res"]
    p_init = configuration["p_ic"]
    res_func = configuration["res_func"]
    res_weight = configuration["res_weight"]
    save_path = configuration["save_path"]

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    min_loss = np.inf
    loss_history = []
    normalize = p_net.scale
    iterations_per_decay = 1000
    
    N0_samples_initial = 1000
    Nr_samples_initial = 1000
    N_RAR = 30000
    x_bc_rar = torch.empty(0, 1, device=device)
    t_bc_rar = torch.empty(0, 1, device=device)
    x_res_rar = torch.empty(0, 1, device=device)
    t_res_rar = torch.empty(0, 1, device=device)
    FLAG = False

    beta = np.float32(0.0)
    beta_incre = np.float32(0.02)
    RAR_eps = 0.05
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()
        
        # Sample new points at each epoch for better generalization
        x_bc_new, t_bc_new = train_helper_sample_ic(N0_samples_initial)
        x_res_new, t_res_new = train_helper_sample_res(Nr_samples_initial)
        
        # Combine with RAR points and ensure they are on the correct device
        x_bc = torch.cat((x_bc_new.to(device), x_bc_rar), dim=0)
        t_bc = torch.cat((t_bc_new.to(device), t_bc_rar), dim=0)
        x_res = torch.cat((x_res_new.to(device), x_res_rar), dim=0)
        t_res = torch.cat((t_res_new.to(device), t_res_rar), dim=0)

        # FIX: Set requires_grad=True for x_res before using it in diff_opt_scaled
        x_res.requires_grad_(True)
        
        # --- Loss based on initial conditions ---
        u_bc = p_init(x_bc).detach()
        net_bc_out = p_net(x_bc, t_bc).to(device) # output of u(x,t)
        mse_u = mse_cost_function(net_bc_out/normalize, u_bc/normalize)

        # --- Loss based on PDE ---
        # The create_graph=True is necessary for the nested gradients.
        res_p = res_func(x_res, t_res, p_net, beta=beta)
        all_zeros = torch.zeros_like(res_p)
        mse_res = mse_cost_function(res_p / normalize, all_zeros)

        # --- Total Loss ---
        loss = mse_u + res_weight*mse_res
        loss_history.append(loss.item())
        
        # --- Backpropagation ---
        # `retain_graph=True` is needed for the nested gradient calls inside diff_opt_scaled
        # This will be handled implicitly if create_graph=True is used in the inner grad calls
        # and not explicitly set for loss.backward().
        loss.backward(retain_graph=True)
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(p_net.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # --- Learning rate decay ---
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
            print(f"[info] Training epoch: {epoch+1}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Save min loss model and update beta ---
        if loss.data < 0.95 * min_loss:
            train_time = time.time() - start_time
            print(f"--- Save Epoch: {epoch+1}, Loss: {loss.item():.4f}, IC: {mse_u.item():.4f}, Res: {mse_res.item():.4f}, Beta: {beta:.2f} ---")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(), 
                    'train_time': time.time() - start_time
                    }, save_path)
            min_loss = loss.data
            beta = min(1.0, beta + beta_incre)
            FLAG = True

        # --- RAR ---
        if epoch % 100 == 0 and FLAG:
            FLAG = False

            # Add IC points
            with torch.no_grad():
                x_bc_check, t_bc_check = train_helper_sample_ic(N_RAR)
                p_i = p_init(x_bc_check).detach()
                phat_i = p_net(x_bc_check, t_bc_check)
                ic_errors = torch.abs(p_i - phat_i) / normalize
                
                max_error_ic = ic_errors.max().item()
                if max_error_ic > RAR_eps:
                    max_indices = torch.topk(ic_errors.squeeze(), 10).indices
                    x_bc_rar = torch.cat((x_bc_rar, x_bc_check[max_indices, :]), dim=0)
                    t_bc_rar = torch.cat((t_bc_rar, t_bc_check[max_indices]), dim=0)
                    print(f"... RAR IC, added {len(max_indices)} points. Max IC error: {max_error_ic:.4f}")

            # Add Residual points
            x_res_check, t_res_check = train_helper_sample_res(N_RAR)
            res_p = res_func(x_res_check, t_res_check, p_net, beta=beta)
            res_errors = torch.abs(res_p) / normalize
            
            max_error_res = res_errors.max().item()
            if max_error_res > RAR_eps:
                max_indices = torch.topk(res_errors.squeeze(), 10).indices
                x_res_rar = torch.cat((x_res_rar, x_res_check[max_indices, :]), dim=0)
                t_res_rar = torch.cat((t_res_rar, t_res_check[max_indices]), dim=0)
                print(f"... RAR Res, added {len(max_indices)} points. Max Res error: {max_error_res:.4f}")
        
    end_time = time.time()
    print(f"Total training time: {(end_time - start_time):.2f} seconds.")
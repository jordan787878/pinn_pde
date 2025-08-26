import numpy as np
import torch
import time


device = "cpu"


def train_pinn_sol_base(constants, p_net, configurations, iterations=50000, save_model=False, beta_incre=0.02):
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    p_init = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    pnet_path = configurations["pnet_path"]
    pnet_path_inter = configurations["pnet_path_inter"]

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale

    N0_samples = 2000
    Nr_samples = 2000
    x_bc, t_bc = constants.sample_init_points(N0_samples)
    x, t = constants.sample_res_points(Nr_samples)

    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(constants, x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc)
        mse_u = mse_cost_function(phat_i/normalize, p_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt(x, t, p_net, beta=beta)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_p/normalize, all_zeros)

        # Loss Function
        loss = mse_u + mse_res
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data, ",res:", mse_res.data,
                  ",beta: ", beta
                   )
            if(save_model):
                torch.save({
                        'epoch': epoch, 'model_state_dict': p_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history, 'train_time': train_time,
                        }, pnet_path)
            min_loss = loss.data
            FLAG = True
            
            beta = beta + np.float32(beta_incre)
            if(beta > 1.0):
                beta = np.float32(1.0)

            # FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
            # if(FLAG_SAVE_INTER >= 10):
            #     INTER_COUNT = INTER_COUNT + 1
            #     if(save_model):
            #         torch.save({
            #             'epoch': epoch, 'model_state_dict': p_net.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'loss_history': loss_history, 'train_time': train_time,
            #             }, pnet_path_inter+str(INTER_COUNT)+".pth")
            #         FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG):
            x_bc_rar, t_bc_rar = constants.sample_init_points(S)
            x_rar, t_rar = constants.sample_res_points(S)
            # add initial points
            p_i = p_init(constants, x_bc_rar.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(p_i - phat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(p_i.squeeze() - phat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # add residual points
            res_p = diff_opt(x_rar, t_rar, p_net)/normalize
            max_error= torch.max(torch.abs(res_p))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res_p.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
            print("[info] training epoch: ", epoch)


def train_pinn_sol_v0(constants, p_net, configurations, iterations=50000, save_model=False, beta_incre=0.02):
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    p_init = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    pnet_path = configurations["pnet_path"]
    pnet_path_inter = configurations["pnet_path_inter"]

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale

    N0_samples = 2000
    Nr_samples = 2000
    x_bc, t_bc = constants.sample_init_points(N0_samples)
    x, t = constants.sample_res_points(Nr_samples)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0) # beta = np.float32(1.0) # beta set to 1.0 means no curriculur training is employed
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(constants, x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc)
        mse_u = mse_cost_function(phat_i/normalize, p_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt(x, t, p_net, beta=beta)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_p/normalize, all_zeros)
        
        # TV_loss by random perturbation (epsilon)
        # res_x = torch.autograd.grad(res_p, x, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
        res_t = torch.autograd.grad(res_p/normalize, t, grad_outputs=torch.ones_like(res_p/normalize), create_graph=True)[0]
        # tv_x = (res_x ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        tv_t = (res_t ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        tv_loss = torch.mean(tv_t)

        # Loss Function
        loss = mse_u + mse_res + 1e-2 * tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data, 
                  ",res:", mse_res.data,
                  ",tv:", tv_loss.data,
                  ",beta: ", beta
                   )
            if(save_model):
                torch.save({
                        'epoch': epoch, 'model_state_dict': p_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history, 'train_time': train_time,
                        }, pnet_path)
            min_loss = loss.data
            FLAG = True
            beta = beta + np.float32(beta_incre)
            if(beta > 1.0):
                beta = np.float32(1.0)

            # FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
            # if(FLAG_SAVE_INTER >= 10):
            #     INTER_COUNT = INTER_COUNT + 1
            #     if(save_model):
            #         torch.save({
            #             'epoch': epoch, 'model_state_dict': p_net.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'loss_history': loss_history, 'train_time': train_time,
            #             }, pnet_path_inter+str(INTER_COUNT)+".pth")
            #         FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG):
            # add initial points
            x_bc_rar, t_bc_rar = constants.sample_init_points(S)
            p_i = p_init(constants, x_bc_rar.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(p_i - phat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(p_i.squeeze() - phat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # add residual points
            x_rar, t_rar = constants.sample_res_points(S)
            res_p = diff_opt(x_rar, t_rar, p_net)/normalize
            max_error= torch.max(torch.abs(res_p))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res_p.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            FLAG = False # reset flag

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
            print("[info] training epoch: ", epoch)


def train_pinn_sol_v0_scaled(constants, p_net, configurations, iterations=50000, save_model=False, beta_incre=0.02):
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    p_init = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    pnet_path = configurations["pnet_path"]
    pnet_path_inter = configurations["pnet_path_inter"]

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale

    N0_samples = 2000
    Nr_samples = 2000
    x_bc, t_bc = constants.sample_init_points_scaled(N0_samples)
    x, t = constants.sample_res_points_scaled(Nr_samples)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0) # beta = np.float32(1.0) # beta set to 1.0 means no curriculur training is employed
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(constants, x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc)
        mse_u = mse_cost_function(phat_i/normalize, p_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt(x, t, p_net, beta=beta)
        all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_p/normalize, all_zeros)
        
        # TV_loss by random perturbation (epsilon)
        # res_x = torch.autograd.grad(res_p, x, grad_outputs=torch.ones_like(res_p), create_graph=True)[0]
        # res_t = torch.autograd.grad(res_p/normalize, t, grad_outputs=torch.ones_like(res_p/normalize), create_graph=True)[0]
        # # tv_x = (res_x ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        # tv_t = (res_t ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        # tv_loss = torch.mean(tv_t)

        # Loss Function
        loss = mse_u + mse_res #+ 1e-2 * tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data, 
                  ",res:", mse_res.data,
                #   ",tv:", tv_loss.data,
                  ",beta: ", beta
                   )
            if(save_model):
                torch.save({
                        'epoch': epoch, 'model_state_dict': p_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history, 'train_time': train_time,
                        }, pnet_path)
            min_loss = loss.data
            FLAG = True
            beta = beta + np.float32(beta_incre)
            if(beta > 1.0):
                beta = np.float32(1.0)

            # FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
            # if(FLAG_SAVE_INTER >= 10):
            #     INTER_COUNT = INTER_COUNT + 1
            #     if(save_model):
            #         torch.save({
            #             'epoch': epoch, 'model_state_dict': p_net.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'loss_history': loss_history, 'train_time': train_time,
            #             }, pnet_path_inter+str(INTER_COUNT)+".pth")
            #         FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG):
            # add initial points
            x_bc_rar, t_bc_rar = constants.sample_init_points_scaled(S)
            p_i = p_init(constants, x_bc_rar.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(p_i - phat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(p_i.squeeze() - phat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].data, max_error.data, x_max[0,:].data)

            # add residual points
            x_rar, t_rar = constants.sample_res_points_scaled(S)
            res_p = diff_opt(x_rar, t_rar, p_net)/normalize
            max_error= torch.max(torch.abs(res_p))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res_p.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].data, max_error.data, x_max[0,:].data)

            FLAG = False # reset flag

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
            print("[info] training epoch: ", epoch)


def train_pinn_sol_v0_scaled_improved(constants, p_net, configurations, 
                                      iterations=50000, save_model=False, beta_incre=0.02):
    """
    Improved training loop for the physics-informed neural network.
    """
    p_net.train()
    p_net.to(device)
    
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(p_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    p_init = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    pnet_path = configurations["pnet_path"]
    pnet_path_inter = configurations["pnet_path_inter"]

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = p_net.scale
    
    N0_samples_initial = 2000
    Nr_samples_initial = 2000
    
    N_RAR = 30000
    x_bc_rar = torch.empty(0, 6, device=device)
    t_bc_rar = torch.empty(0, 1, device=device)
    x_res_rar = torch.empty(0, 6, device=device)
    t_res_rar = torch.empty(0, 1, device=device)
    FLAG = False

    beta = np.float32(0.0)
    RAR_eps = 0.05
    inter_count = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()
        
        # Sample new points at each epoch for better generalization
        x_bc_new, t_bc_new = constants.sample_init_points_scaled(N0_samples_initial)
        x_res_new, t_res_new = constants.sample_res_points_scaled(Nr_samples_initial)
        
        # Combine with RAR points and ensure they are on the correct device
        x_bc = torch.cat((x_bc_new.to(device), x_bc_rar), dim=0)
        t_bc = torch.cat((t_bc_new.to(device), t_bc_rar), dim=0)
        x_res = torch.cat((x_res_new.to(device), x_res_rar), dim=0)
        t_res = torch.cat((t_res_new.to(device), t_res_rar), dim=0)

        # FIX: Set requires_grad=True for x_res before using it in diff_opt_scaled
        x_res.requires_grad_(True)
        
        # --- Loss based on initial conditions ---
        p_i = p_init(constants, x_bc.detach().cpu().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, device=device)
        phat_i = p_net(x_bc, t_bc)
        mse_u = mse_cost_function(phat_i / normalize, p_i / normalize)

        # --- Loss based on PDE ---
        # The create_graph=True is necessary for the nested gradients.
        res_p = diff_opt(x_res, t_res, p_net, beta=beta)
        all_zeros = torch.zeros_like(res_p)
        mse_res = mse_cost_function(res_p / normalize, all_zeros)

        # --- Total Loss ---
        loss = mse_u + mse_res
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
            if save_model:
                torch.save({
                    'epoch': epoch, 'model_state_dict': p_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_history': loss_history, 'train_time': train_time,
                }, pnet_path)
            min_loss = loss.data
            beta = min(1.0, beta + beta_incre)
            FLAG = True

        # --- RAR ---
        if epoch % 100 == 0 and FLAG:
            FLAG = False

            # Add IC points
            with torch.no_grad():
                x_bc_check, t_bc_check = constants.sample_init_points_scaled(N_RAR)
                x_bc_check, t_bc_check = x_bc_check.to(device), t_bc_check.to(device)
                p_i = p_init(constants, x_bc_check.detach().cpu().numpy())
                p_i = torch.tensor(p_i, dtype=torch.float32, device=device)
                phat_i = p_net(x_bc_check, t_bc_check)
                ic_errors = torch.abs(p_i - phat_i) / normalize
                
                max_error_ic = ic_errors.max().item()
                if max_error_ic > RAR_eps:
                    max_indices = torch.topk(ic_errors.squeeze(), 10).indices
                    x_bc_rar = torch.cat((x_bc_rar, x_bc_check[max_indices, :]), dim=0)
                    t_bc_rar = torch.cat((t_bc_rar, t_bc_check[max_indices]), dim=0)
                    print(f"... RAR IC, added {len(max_indices)} points. Max IC error: {max_error_ic:.4f}")

            # Add Residual points
            x_res_check, t_res_check = constants.sample_res_points_scaled(N_RAR)
            x_res_check, t_res_check = x_res_check.to(device), t_res_check.to(device)
            res_p = diff_opt(x_res_check, t_res_check, p_net, beta=beta)
            res_errors = torch.abs(res_p) / normalize
            
            max_error_res = res_errors.max().item()
            if max_error_res > RAR_eps:
                max_indices = torch.topk(res_errors.squeeze(), 10).indices
                x_res_rar = torch.cat((x_res_rar, x_res_check[max_indices, :]), dim=0)
                t_res_rar = torch.cat((t_res_rar, t_res_check[max_indices]), dim=0)
                print(f"... RAR Res, added {len(max_indices)} points. Max Res error: {max_error_res:.4f}")
        

    end_time = time.time()
    print(f"Total training time: {(end_time - start_time):.2f} seconds.")


#################


def train_pinn_e1_v0(constants, networks, configurations, iterations=50000, save_model=False, beta_incre=0.005):
    """
    no TV loss
    """
    p_init = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    save_path = configurations["save_path"]
    save_path_inter = configurations["save_path_inter"]
    p_net, e1_net = networks

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    print("normalize: ", normalize)

    N0_samples = 1
    Nr_samples = 1
    x_bc, t_bc = constants.sample_init_points(N0_samples)
    x, t = constants.sample_res_points(Nr_samples)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0

    keys = ['epoch', 'model_state_dict', 'loss_history', 'train_time']
    best_model_dict = dict.fromkeys(keys)
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        _x_bc, _t_bc = constants.sample_init_points(500)
        # x_bc_train = x_bc 
        x_bc_train = torch.cat((x_bc, _x_bc), dim=0)
        # t_bc_train = t_bc 
        t_bc_train = torch.cat((t_bc, _t_bc), dim=0)

        p_i = p_init(constants, x_bc_train.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc_train, t_bc_train).to(device)
        e_i = p_i - phat_i
        ehat_i = e1_net(x_bc_train, t_bc_train).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        # Loss based on PDE
        _x, _t = constants.sample_res_points(500)
        x_train = torch.cat((x, _x), dim=0)
        t_train = torch.cat((t, _t), dim=0)
        res_p = diff_opt(x_train, t_train, p_net, beta=beta)
        res_e = diff_opt(x_train, t_train, e1_net, beta=beta)
        # res = res_e + res_p
        mse_res = mse_cost_function(res_e/normalize, -res_p.detach()/normalize)
        # res_t = torch.autograd.grad(res/normalize, t, 
        #                             grad_outputs=torch.ones_like(res), create_graph=True)[0]
        # tv_t = (res_t ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        # tv_loss = torch.mean(tv_t)

        loss = mse_u + mse_res #+ 1e-3*tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("[best model] epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data,
                  ",res:", mse_res.data,
                #   ",tv:", tv_loss.data,
                  ",beta: {:.3f}".format(beta)
                   )
            train_time = time.time() - start_time
            best_model_dict['epoch'] = epoch
            best_model_dict['model_state_dict'] = e1_net.state_dict()
            best_model_dict['loss_history'] = loss_history
            best_model_dict['train_time'] = train_time

            min_loss = loss.data
            FLAG = True
            beta = beta + np.float32(beta_incre)
            if(beta > 1.0):
                beta = np.float32(1.0)

            # --- Save intermediate model for debug/plot ---
            # if(save_model):
            #     FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
            #     if(FLAG_SAVE_INTER >= 10):
            #         INTER_COUNT = INTER_COUNT + 1
            #         torch.save({
            #             'epoch': epoch, 'model_state_dict': e1_net.state_dict(),
            #             'loss_history': loss_history, 'train_time': train_time,
            #             }, save_path_inter+str(INTER_COUNT)+".pth")
            #         FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG and epoch > 0):
            # add initial points
            # x_bc_rar, t_bc_rar = constants.sample_init_points(S)
            # p_i = p_init(constants, x_bc_rar.detach().numpy())
            # p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            # phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            # e_i = p_i - phat_i
            # ehat_i = e1_net(x_bc_rar, t_bc_rar).to(device)
            # max_error = torch.max(torch.abs(e_i - ehat_i))/normalize
            # if(max_error > RAR_eps):
            #     max_value, max_index = torch.topk(torch.abs(e_i.squeeze() - ehat_i.squeeze()), 50)
            #     x_max = x_bc_rar[max_index,:].clone().detach()
            #     t_max = t_bc_rar[max_index].clone().detach()
            #     # x_bc = torch.cat((x_bc, x_max), dim=0)
            #     # t_bc = torch.cat((t_bc, t_max), dim=0)
            #     x_bc = x_max
            #     t_bc = t_max
            #     print("... RAR IC, add: ", t_max[0].item(), max_error.item())

            # # add residual points
            # x_rar, t_rar = constants.sample_res_points(S)
            # res_p = diff_opt(x_rar, t_rar, p_net, beta=beta)
            # res_e = diff_opt(x_rar, t_rar, e1_net, beta=beta)
            # res = (res_e + res_p)/normalize
            # max_error= torch.max(torch.abs(res))
            # if(max_error > RAR_eps):
            #     max_value, max_index = torch.topk(torch.abs(res.squeeze()), 50)
            #     x_max = x_rar[max_index,:].clone()
            #     t_max = t_rar[max_index].clone()
            #     # x = torch.cat((x, x_max), dim=0)
            #     # t = torch.cat((t, t_max), dim=0)
            #     x = x_max
            #     t = t_max
            #     print("... RAR Res, add: ", t_max[0].item(), max_error.item())

            # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
            print("[debug] training epoch: ", epoch)

    # --- Save best model ---
    if(save_model):
        torch.save(best_model_dict, save_path)


def train_pinn_e1_v0_scaled(constants, networks, configurations, iterations=50000, save_model=False, beta_incre=0.005):
    """
    no TV loss
    """
    p_init = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    save_path = configurations["save_path"]
    p_net, e1_net = networks
    p_net.eval()
    # Freeze p_net parameters
    for p in p_net.parameters():
        p.requires_grad_(False)

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    print("normalize: ", normalize)

    N0_samples = 1
    Nr_samples = 1
    x_bc, t_bc = constants.sample_init_points_scaled(N0_samples)
    x, t = constants.sample_res_points_scaled(Nr_samples)

    # RAR
    S = 30000
    beta = np.float32(0.0)

    keys = ['epoch', 'model_state_dict', 'loss_history', 'train_time']
    best_model_dict = dict.fromkeys(keys)
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        _x_bc, _t_bc = constants.sample_init_points_scaled(500)
        # x_bc_train = x_bc 
        x_bc_train = torch.cat((x_bc, _x_bc), dim=0)
        # t_bc_train = t_bc 
        t_bc_train = torch.cat((t_bc, _t_bc), dim=0)

        p_i = p_init(constants, x_bc_train.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc_train, t_bc_train).to(device)
        e_i = p_i - phat_i
        ehat_i = e1_net(x_bc_train, t_bc_train).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        alpha_i = torch.max(torch.abs(ehat_i - e_i)) / torch.max(torch.abs(ehat_i))
        max_e1 = torch.max(torch.abs(e_i)).item()
        max_e1_pinn = torch.max(torch.abs(ehat_i)).item()

        # Loss based on PDE
        # _x, _t = constants.sample_res_points_scaled(500)
        # x_train = torch.cat((x, _x), dim=0)
        # t_train = torch.cat((t, _t), dim=0)
        # res_p = diff_opt(x_train, t_train, p_net, beta=beta)
        # res_e = diff_opt(x_train, t_train, e1_net, beta=beta)
        # # res = res_e + res_p
        # mse_res = mse_cost_function(res_e/normalize, -res_p.detach()/normalize)
        # res_t = torch.autograd.grad(res/normalize, t, 
        #                             grad_outputs=torch.ones_like(res), create_graph=True)[0]
        # tv_t = (res_t ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        # tv_loss = torch.mean(tv_t)

        loss = mse_u #+ mse_res #+ 1e-3*tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("[best model] epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data,
                  ",alpha_i:", alpha_i.item(), max_e1, max_e1_pinn,
                #   ",res:", mse_res.data,
                #   ",tv:", tv_loss.data,
                  ",beta: {:.3f}".format(beta)
                   )
            train_time = time.time() - start_time
            best_model_dict['epoch'] = epoch
            best_model_dict['model_state_dict'] = e1_net.state_dict()
            best_model_dict['loss_history'] = loss_history
            best_model_dict['train_time'] = train_time
            min_loss = loss.data
            beta = beta + np.float32(beta_incre)
            if(beta > 1.0):
                beta = np.float32(1.0)
            # --- Save best model ---
            if(save_model):
                torch.save(best_model_dict, save_path)

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
            print("[debug] training epoch: ", epoch)


def train_pinn_e1_v0_scaled_improved(
    constants,
    networks,                          # (p_net, e1_net)
    configurations,
    iterations=50000,
    save_model=False,
    beta_incre=0.02
):
    """
    Improved training loop for the error network e1_net.
    Uses same structure as train_pinn_sol_v0_scaled_improved.
    """
    import time
    
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
    p_init   = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    e1_path  = configurations["save_path"]
    # e1_path_inter = configurations.get("save_path_inter", None)
    
    # --- Optimizer ---
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # -------- SAFETY CHECK 2: optimizer only sees e1_net params --------
    e1_param_ids = {id(p) for p in e1_net.parameters()}
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert opt_param_ids <= e1_param_ids, "Optimizer includes non-e1_net parameters!"
    
    # --- Settings ---
    min_loss = float("inf")
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.normalize
    
    N0_samples_initial = 2000
    Nr_samples_initial = 2000
    N_RAR = 30000
    RAR_eps = 1e-1
    FLAG = False
    # inter_count = 0
    beta = np.float32(0.0)
    
    # Buffers for RAR points
    x_bc_rar = torch.empty(0, 6, device=device)
    t_bc_rar = torch.empty(0, 1, device=device)
    x_res_rar = torch.empty(0, 6, device=device)
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
    for epoch in range(iterations):
        optimizer.zero_grad()
        
        # --- New samples ---
        x_bc_new, t_bc_new = constants.sample_init_points_scaled(N0_samples_initial)
        x_res_new, t_res_new = constants.sample_res_points_scaled(Nr_samples_initial)
        
        x_bc = torch.cat((x_bc_new.to(device), x_bc_rar), dim=0)
        t_bc = torch.cat((t_bc_new.to(device), t_bc_rar), dim=0)
        x_res = torch.cat((x_res_new.to(device), x_res_rar), dim=0)
        t_res = torch.cat((t_res_new.to(device), t_res_rar), dim=0)
        x_res.requires_grad_(True)
        
        # --- IC loss: ehat ~ (p_init - p_net) ---
        with torch.no_grad():
            p_i_np = p_init(constants, x_bc.detach().cpu().numpy())
            p_i = torch.tensor(p_i_np, dtype=torch.float32, device=device)
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
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(e1_net.parameters(), max_norm=1.0)
        optimizer.step()

        # # -------- SAFETY CHECK 5: verify p_net weights unchanged after step --------
        # _assert_pnet_unchanged(f"epoch {epoch+1}/post-step")
        
        # --- LR decay ---
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
            print(f"[info] epoch {epoch+1}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        # --- Save best ---
        if loss.item() < 0.95 * min_loss:
            train_time = time.time() - start_time
            print(f"--- Save Epoch {epoch+1}, Loss={loss.item():.4f}, "
                  f"IC={mse_ic.item():.4f}, Res={mse_res.item():.4f}, Beta={beta:.2f} ---")
            # Res={mse_res.item():.4f}
            
            best_model_dict.update({
                "epoch": epoch,
                "model_state_dict": e1_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                "train_time": train_time,
            })
            
            if save_model:
                torch.save(best_model_dict, e1_path)
            
            min_loss = loss.item()
            beta = min(1.0, beta + beta_incre)
            FLAG = True

        # --- RAR ---
        if epoch % 100 == 0 and FLAG:
            FLAG = False
            # IC RAR
            with torch.no_grad():
                xb_chk, tb_chk = constants.sample_init_points_scaled(N_RAR)
                xb_chk, tb_chk = xb_chk.to(device), tb_chk.to(device)
                p_i_np = p_init(constants, xb_chk.detach().cpu().numpy())
                p_i = torch.tensor(p_i_np, dtype=torch.float32, device=device)
                phat_i = p_net(xb_chk, tb_chk)
                e_i = p_i - phat_i
                ehat_i = e1_net(xb_chk, tb_chk)
                ic_err = torch.abs(e_i - ehat_i) / normalize
                max_error_ic = ic_err.max().item()
                if max_error_ic > RAR_eps:
                    idx = torch.topk(ic_err.squeeze(), 10).indices
                    x_bc_rar = torch.cat((x_bc_rar, xb_chk[idx, :]), dim=0)
                    t_bc_rar = torch.cat((t_bc_rar, tb_chk[idx]), dim=0)
                    print(f"... RAR IC, added {len(idx)} points. Max IC error: {max_error_ic:.4f}")
            
            # Residual RAR
            xr_chk, tr_chk = constants.sample_res_points_scaled(N_RAR)
            xr_chk, tr_chk = xr_chk.to(device), tr_chk.to(device)
            res_p = diff_opt(xr_chk, tr_chk, p_net, beta=beta)
            res_e = diff_opt(xr_chk, tr_chk, e1_net, beta=beta)
            res_err = torch.abs(res_e + res_p) / normalize
            max_error_res = res_err.max().item()
            if max_error_res > RAR_eps:
                idx = torch.topk(res_err.squeeze(), 10).indices
                x_res_rar = torch.cat((x_res_rar, xr_chk[idx, :]), dim=0)
                t_res_rar = torch.cat((t_res_rar, tr_chk[idx]), dim=0)
                print(f"... RAR Res, added {len(idx)} points. Max Res error: {max_error_res:.4f}")

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f}s")


def train_pinn_e1seq1_base(constants, networks, configurations, iterations=50000, save_model=False):
    """
    base: no TV loss
    """
    p_init = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    save_path = configurations["save_path"]
    save_path_inter = configurations["save_path_inter"]
    p_net, e1_net = networks

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    print("normalize: ", normalize)
    t_seq = np.float32(np.array([constants.TI, constants.TF*1.0]))

    N0_samples = 2000
    Nr_samples = 2000
    # x_bc, t_bc = constants.sample_init_points(N0_samples)
    # x, t = constants.sample_res_points(Nr_samples)
    x_bc, t_bc = constants.sample_init_points_seq(N0_samples, t_seq)
    x, t = constants.sample_res_points_seq(Nr_samples, t_seq)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0

    keys = ['epoch', 'model_state_dict', 'loss_history', 'train_time']
    best_model_dict = dict.fromkeys(keys)
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(constants, x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc).to(device)
        e_i = p_i - phat_i
        ehat_i = e1_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt(x, t, p_net, beta=beta)
        res_e = diff_opt(x, t, e1_net, beta=beta)
        res = res_e + res_p
        mse_res = mse_cost_function(res_e/normalize, -res_p.detach()/normalize)

        # res_t = torch.autograd.grad(res/normalize, t, 
        #                             grad_outputs=torch.ones_like(res), create_graph=True)[0]
        # tv_t = (res_t ** 2).sum(dim=1, keepdim=True)  # shape: (N, 1)
        # tv_loss = torch.mean(tv_t)

        loss = mse_u + mse_res #+ 1e-3*tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("[best model] epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data,   
                  ",res:", mse_res.data,
                #   ",tv:", tv_loss.data,
                  ",beta: {:.3f}".format(beta)
                   )
            train_time = time.time() - start_time
            best_model_dict['epoch'] = epoch
            best_model_dict['model_state_dict'] = e1_net.state_dict()
            best_model_dict['loss_history'] = loss_history
            best_model_dict['train_time'] = train_time

            min_loss = loss.data
            FLAG = True
            beta = beta + np.float32(0.005)
            if(beta > 1.0):
                beta = np.float32(1.0)

            # --- Save intermediate model for debug/plot ---
            # if(save_model):
            #     FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
            #     if(FLAG_SAVE_INTER >= 10):
            #         INTER_COUNT = INTER_COUNT + 1
            #         torch.save({
            #             'epoch': epoch, 'model_state_dict': e1_net.state_dict(),
            #             'loss_history': loss_history, 'train_time': train_time,
            #             }, save_path_inter+str(INTER_COUNT)+".pth")
            #         FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG and epoch > 0):
            x_bc_rar, t_bc_rar = constants.sample_init_points_seq(S, t_seq, sobol_seed=epoch)
            x_rar, t_rar = constants.sample_res_points_seq(S, t_seq, sobol_seed=epoch)

            # add initial points
            p_i = p_init(constants, x_bc_rar.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            e_i = p_i - phat_i
            ehat_i = e1_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(e_i - ehat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(e_i.squeeze() - ehat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # # add residual points
            res_p = diff_opt(x_rar, t_rar, p_net, beta=beta)
            res_e = diff_opt(x_rar, t_rar, e1_net, beta=beta)
            res = (res_e + res_p)/normalize
            max_error= torch.max(torch.abs(res))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            # # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()
    # --- Save best model ---
    if(save_model):
        torch.save(best_model_dict, save_path)


def train_pinn_e1seq1_v0(constants, networks, configurations, iterations=50000, save_model=False):
    p_init = configurations["ic_fcn"]
    diff_opt = configurations["diff_opt_fcn"]
    save_path = configurations["save_path"]
    save_path_inter = configurations["save_path_inter"]
    p_net, e1_net = networks

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    print("normalize: ", normalize)
    t_seq = np.float32(np.array([constants.TI, constants.TF*0.5]))

    N0_samples = 2000
    Nr_samples = 2000
    # x_bc, t_bc = constants.sample_init_points(N0_samples)
    # x, t = constants.sample_res_points(Nr_samples)
    x_bc, t_bc = constants.sample_init_points_seq(N0_samples, t_seq)
    x, t = constants.sample_res_points_seq(Nr_samples, t_seq)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0)
    # beta = np.float32(1.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        p_i = p_init(constants, x_bc.detach().numpy())
        p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
        phat_i = p_net(x_bc, t_bc).to(device)
        e_i = p_i - phat_i
        ehat_i = e1_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt(x, t, p_net, beta=beta)
        res_e = diff_opt(x, t, e1_net, beta=beta)
        res = res_e + res_p
        # all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
        mse_res = mse_cost_function(res_e/normalize, -res_p.detach()/normalize)

        # # tv
        tau = t + torch.randn_like(t) * np.float32(0.05)
        res_tau = diff_opt(x, tau, e1_net, beta=beta) + diff_opt(x, tau, p_net, beta=beta)
        tv_loss = mse_cost_function(res_tau/normalize, res/normalize)

        loss = mse_u + mse_res + 1e-2*tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data,   
                  ",res:", mse_res.data,
                  ",tv:",tv_loss.data,
                  ",beta: {:.3f}".format(beta)
                   )
            if(save_model):
                train_time = time.time() - start_time
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': e1_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history,
                        'train_time': train_time,
                        }, save_path)
            min_loss = loss.data
            FLAG = True
            beta = beta + np.float32(0.005)
            if(beta > 1.0):
                beta = np.float32(1.0)

            if(save_model):
                FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
                if(FLAG_SAVE_INTER >= 10):
                    INTER_COUNT = INTER_COUNT + 1
                    torch.save({
                        'epoch': epoch, 'model_state_dict': e1_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history, 'train_time': train_time,
                        }, save_path_inter+str(INTER_COUNT)+".pth")
                    FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG and epoch > 0):
            x_bc_rar, t_bc_rar = constants.sample_init_points_seq(S, t_seq, sobol_seed=epoch)
            x_rar, t_rar = constants.sample_res_points_seq(S, t_seq, sobol_seed=epoch)

            # add initial points
            p_i = p_init(constants, x_bc_rar.detach().numpy())
            p_i = torch.tensor(p_i, dtype=torch.float32, requires_grad=False)
            phat_i = p_net(x_bc_rar, t_bc_rar).to(device)
            e_i = p_i - phat_i
            ehat_i = e1_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(e_i - ehat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(e_i.squeeze() - ehat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # # add residual points
            res_p = diff_opt(x_rar, t_rar, p_net, beta=beta)
            res_e = diff_opt(x_rar, t_rar, e1_net, beta=beta)
            res = (res_e + res_p)/normalize
            max_error= torch.max(torch.abs(res))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            # # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def train_pinn_e1seq2_base(constants, networks, configurations, iterations=50000, save_model=False):
    diff_opt = configurations["diff_opt_fcn"]
    save_path = configurations["save_path"]
    save_path_inter = configurations["save_path_inter"]
    p_net, e1_net_seq1, e1_net = networks

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    print("normalize: ", normalize)
    t_seq = np.float32(np.array([constants.TF*0.5, constants.TF]))

    N0_samples = 2000
    Nr_samples = 2000
    # x_bc, t_bc = constants.sample_init_points(N0_samples)
    # x, t = constants.sample_res_points(Nr_samples)
    x_bc, t_bc = constants.sample_init_points_seq(N0_samples, t_seq)
    x, t = constants.sample_res_points_seq(Nr_samples, t_seq)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        e_i = e1_net_seq1(x_bc, t_bc).to(device)
        ehat_i = e1_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt(x, t, p_net, beta=beta)
        res_e = diff_opt(x, t, e1_net, beta=beta)
        res = res_e + res_p
        mse_res = mse_cost_function(res_e/normalize, -res_p.detach()/normalize)

        loss = mse_u + mse_res
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data,   
                  ",res:", mse_res.data,
                  ",beta:", beta
                   )
            if(save_model):
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': e1_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history,
                        'train_time': train_time,
                        }, save_path)
            min_loss = loss.data
            FLAG = True
            beta = beta + np.float32(0.005)
            if(beta > 1.0):
                beta = np.float32(1.0)

            if(save_model):
                FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
                if(FLAG_SAVE_INTER >= 10):
                    INTER_COUNT = INTER_COUNT + 1
                    torch.save({
                        'epoch': epoch, 'model_state_dict': e1_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history, 'train_time': train_time,
                        }, save_path_inter+str(INTER_COUNT)+".pth")
                    FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG and epoch > 0):
            x_bc_rar, t_bc_rar = constants.sample_init_points_seq(S, t_seq, sobol_seed=epoch)
            x_rar, t_rar = constants.sample_res_points_seq(S, t_seq, sobol_seed=epoch)

            # add initial points
            e_i = e1_net_seq1(x_bc_rar, t_bc_rar).to(device)
            ehat_i = e1_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(e_i - ehat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(e_i.squeeze() - ehat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # # add residual points
            res_p = diff_opt(x_rar, t_rar, p_net, beta=beta)
            res_e = diff_opt(x_rar, t_rar, e1_net, beta=beta)
            res = (res_e + res_p)/normalize
            max_error= torch.max(torch.abs(res))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            # # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


def train_pinn_e1seq2_v0(constants, networks, configurations, iterations=50000, save_model=False):
    diff_opt = configurations["diff_opt_fcn"]
    save_path = configurations["save_path"]
    save_path_inter = configurations["save_path_inter"]
    p_net, e1_net_seq1, e1_net = networks

    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(e1_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    min_loss = np.inf
    iterations_per_decay = 1000
    loss_history = []
    normalize = e1_net.scale
    print("normalize: ", normalize)
    t_seq = np.float32(np.array([constants.TF*0.5, constants.TF]))

    N0_samples = 2000
    Nr_samples = 2000
    # x_bc, t_bc = constants.sample_init_points(N0_samples)
    # x, t = constants.sample_res_points(Nr_samples)
    x_bc, t_bc = constants.sample_init_points_seq(N0_samples, t_seq)
    x, t = constants.sample_res_points_seq(Nr_samples, t_seq)

    # RAR
    S = 30000
    RAR_eps = 1e-1
    FLAG = False
    beta = np.float32(0.0)
    FLAG_SAVE_INTER = 0
    INTER_COUNT = 0
    
    start_time = time.time()
    for epoch in range(iterations):
        optimizer.zero_grad()

        # Loss based on boundary conditions
        e_i = e1_net_seq1(x_bc, t_bc).to(device)
        ehat_i = e1_net(x_bc, t_bc).to(device)
        mse_u = mse_cost_function(ehat_i/normalize, e_i.detach()/normalize)

        # Loss based on PDE
        res_p = diff_opt(x, t, p_net, beta=beta)
        res_e = diff_opt(x, t, e1_net, beta=beta)
        res = res_e + res_p
        mse_res = mse_cost_function(res_e/normalize, -res_p.detach()/normalize)

        # # tv
        tau = t + torch.randn_like(t) * np.float32(0.05)
        res_tau = diff_opt(x, tau, e1_net, beta=beta) + diff_opt(x, tau, p_net, beta=beta)
        tv_loss = mse_cost_function(res_tau/normalize, res/normalize)

        loss = mse_u + mse_res + 1e-2*tv_loss
        loss_history.append(loss.item())

        # Save the min loss model
        if(loss.data < 0.95*min_loss):
            train_time = time.time() - start_time
            print("save epoch:", epoch, ",loss:", loss.data, 
                  ",ic:", mse_u.data,   
                  ",res:", mse_res.data,
                  ",tv:",tv_loss.data,
                  ",beta:", beta
                   )
            if(save_model):
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': e1_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history,
                        'train_time': train_time,
                        }, save_path)
            min_loss = loss.data
            FLAG = True
            beta = beta + np.float32(0.005)
            if(beta > 1.0):
                beta = np.float32(1.0)

            if(save_model):
                FLAG_SAVE_INTER = FLAG_SAVE_INTER + 1
                if(FLAG_SAVE_INTER >= 10):
                    INTER_COUNT = INTER_COUNT + 1
                    torch.save({
                        'epoch': epoch, 'model_state_dict': e1_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_history': loss_history, 'train_time': train_time,
                        }, save_path_inter+str(INTER_COUNT)+".pth")
                    FLAG_SAVE_INTER = 0

        # RAR
        if(epoch % 100 == 0 and FLAG and epoch > 0):
            x_bc_rar, t_bc_rar = constants.sample_init_points_seq(S, t_seq, sobol_seed=epoch)
            x_rar, t_rar = constants.sample_res_points_seq(S, t_seq, sobol_seed=epoch)

            # add initial points
            e_i = e1_net_seq1(x_bc_rar, t_bc_rar).to(device)
            ehat_i = e1_net(x_bc_rar, t_bc_rar).to(device)
            max_error = torch.max(torch.abs(e_i - ehat_i))/normalize
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(e_i.squeeze() - ehat_i.squeeze()), 10)
                x_max = x_bc_rar[max_index,:].clone().detach()
                t_max = t_bc_rar[max_index].clone().detach()
                x_bc = torch.cat((x_bc, x_max), dim=0)
                t_bc = torch.cat((t_bc, t_max), dim=0)
                print("... RAR IC, add: ", t_max[0].item(), max_error.item())
            # # add residual points
            res_p = diff_opt(x_rar, t_rar, p_net, beta=beta)
            res_e = diff_opt(x_rar, t_rar, e1_net, beta=beta)
            res = (res_e + res_p)/normalize
            max_error= torch.max(torch.abs(res))
            if(max_error > RAR_eps):
                max_value, max_index = torch.topk(torch.abs(res.squeeze()), 10)
                x_max = x_rar[max_index,:].clone()
                t_max = t_rar[max_index].clone()
                x = torch.cat((x, x_max), dim=0)
                t = torch.cat((t, t_max), dim=0)
                print("... RAR Res, add: ", t_max[0].item(), max_error.item())
            # # reset flag
            FLAG = False

        loss.backward(retain_graph=True) 
        optimizer.step()
        # Exponential learning rate decay
        if (epoch + 1) % iterations_per_decay == 0:
            scheduler.step()


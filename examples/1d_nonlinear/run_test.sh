#!/bin/bash

# Define the random seed
i=4

# # Construct the script name
# script="main_train_pnet.py"
# # Run the script
# python3 $script --seed=${i}
# # Check if the script was successful
# if [ $? -eq 0 ]; then
#   echo "$script executed successfully."
# else
#   echo "$script failed to execute."
#   exit 1
# fi

# Construct the script name
script="main_train_enet.py"
# Run the script
python3 $script --seed=${i}
# Check if the script was successful
if [ $? -eq 0 ]; then
  echo "$script executed successfully."
else
  echo "$script failed to execute."
  exit 1
fi


# def train_enet_model(p_net, e1_net, optimizer, scheduler, mse_cost_function, iterations=40000):
#     global x_low, x_hig, t0, T_end
#     ti = t0; tf = T_end
#     min_loss = np.inf
#     loss_history = []
#     iterations_per_decay = 10000
#     PATH = FOLDER+"output/e1_net.pth"
#     x_mar = 0.0

#     _x = np.linspace(x_low, x_hig, num=30, endpoint=True)
#     _t = np.linspace(ti, tf, num=30, endpoint=True)
    
#     # space-time points for BC
#     print(np.min(_x), np.max(_x))
#     x_bc_fix = Variable(torch.from_numpy(_x.reshape(-1,1)).float(), requires_grad=True).to(device)
#     x_bc = (torch.rand(500, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
#     x_bc = torch.cat((x_bc, x_bc_fix), dim=0)
#     t_bc = (torch.ones(len(x_bc), 1, requires_grad=True) * ti).to(device)
    
#     # space-time points for RES
#     # x = (torch.rand(3500, n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
#     # t = (torch.rand(2500, 1, requires_grad=True)   * (tf - ti) + ti).to(device)
#     # _t_init = (torch.ones(500, 1, requires_grad=True) * ti).to(device)
#     # _t_end =  (torch.ones(500, 1, requires_grad=True) * tf).to(device)
#     # t = torch.cat((t, _t_init, _t_end), dim=0)
#     _xx, _tt = np.meshgrid(_x, _t)
#     x = Variable(torch.from_numpy(_xx.reshape(-1,1)).float(), requires_grad=True).to(device)
#     t = Variable(torch.from_numpy(_tt.reshape(-1,1)).float(), requires_grad=True).to(device)
#     x_rand = (torch.rand(1000, n_d, requires_grad=True) * (x_hig - x_low) + x_low).to(device)
#     t_rand = (torch.rand(1000, 1  , requires_grad=True) * (tf - ti) + ti).to(device)
#     x = torch.cat((x, x_rand), dim=0)
#     t = torch.cat((t, t_rand), dim=0)
    
#     weight_reg = 1.0
#     FLAG = 0
#     S = 10000
#     normalize = e1_net.scale

#     # Store alpha data (prepare)
#     Nsample_list = []
#     Alpha_list = []
#     Alpha_mean_list = []
#     Loss_1_list = []
#     Loss_2_list = []
#     E1_list = []
#     x_data = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
#     pt_x_data = Variable(torch.from_numpy(x_data).float(), requires_grad=False).to(device)
#     for t1 in t1s:
#         p_data = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
#         pt_t_data = Variable(torch.from_numpy(x_data*0+t1).float(), requires_grad=True).to(device)
#         phat = p_net(pt_x_data, pt_t_data).data.cpu().numpy() # change tensor to numpy
#         e1 = p_data - phat
#         E1_list.append(e1)

#     start_time = time.time()
#     for epoch in range(iterations):
#         optimizer.zero_grad() # to make the gradients zero

#         p0 = p_init_torch(x_bc)
#         p0_hat  =  p_net(x_bc, t_bc)
#         e10     = p0 - p0_hat
#         # e10_target = e10.clone().detach().numpy()
#         # e10_target = Variable(torch.from_numpy(e10_target).float(), requires_grad=False).to(device)
#         e10_target = e10.detach()
#         e10_hat = e1_net(x_bc, t_bc)[:,0].view(-1,1)
#         mse_e1_ic = mse_cost_function(e10_hat/normalize, e10_target/normalize)
        
#         # using detached p_net
#         diff_e = Diff_e_func(x, t, e1_net)
#         # diff_e_target = -p_res_func(x, t, p_net).detach().numpy()
#         # diff_e_target = Variable(torch.from_numpy(diff_e_target).float(), requires_grad=False).to(device)
#         diff_e_target = -p_res_func(x, t, p_net).detach()
#         mse_e1_res = mse_cost_function(diff_e/normalize, diff_e_target/normalize)
#         # using no detached p_net
#         # all_zeros = torch.zeros((len(t),1), dtype=torch.float32, requires_grad=False).to(device)
#         # e1_res_out = e_res_func(x, t, e1_net, p_net)/normalize
#         # mse_e1_res = mse_cost_function(e1_res_out, all_zeros)

#         e1_res_out = e_res_func(x, t, e1_net, p_net)/normalize
#         res_x = torch.autograd.grad(e1_res_out, x, grad_outputs=torch.ones_like(e1_res_out), create_graph=True)[0]
#         res_t = torch.autograd.grad(e1_res_out, t, grad_outputs=torch.ones_like(e1_res_out), create_graph=True)[0]
#         mse_res_grad = torch.mean(res_x**2+res_t**2)
    
#         # Combining the loss functions
#         loss = mse_e1_ic + mse_e1_res + weight_reg*mse_res_grad
#         loss_history.append(loss.data)

#         # Save the min loss model
#         if(loss.data < 0.95*min_loss):
#             min_loss = loss.data
#             FLAG = FLAG + 1
#             training_time = time.time() - start_time
#             print("e1net best epoch:", epoch, ", loss:", loss.data, 
#                   "ic:", mse_e1_ic.data,
#                   "res:", mse_e1_res.data,
#                   "grad:", mse_res_grad.data,
#                   )
#             torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': e1_net.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': loss.data,
#                     'label': "e1_net",
#                     'train_time': training_time,
#                     }, PATH)
#             np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))
#             # Store alpha data (calculate data)
#             Nsample_i = x_bc.shape[0] + x.shape[0]
#             Loss_1_i    = (loss).data.cpu().numpy().item()
#             Loss_2_i    = (0.0*loss).data.cpu().numpy().item() ###
#             alpha_over_time = []
#             for i in range(len(t1s)):
#                 t1 = t1s[i]
#                 pt_t_data = Variable(torch.from_numpy(x_data*0+t1).float(), requires_grad=True).to(device)
#                 ehat = e1_net(pt_x_data, pt_t_data).data.cpu().numpy()
#                 e1 = E1_list[i]
#                 alpha = np.max(np.abs(e1-ehat))/ np.max(np.abs(ehat))
#                 alpha = np.round(alpha, 3)
#                 alpha_over_time.append(alpha)
#             max_alpha = np.max(alpha_over_time)
#             Nsample_list.append(Nsample_i)
#             Loss_1_list.append(Loss_1_i)
#             Loss_2_list.append(Loss_2_i)
#             Alpha_list.append(max_alpha)
#             Alpha_mean_list.append(np.mean(alpha_over_time))
#             # RAR
#             if (FLAG >= 3 and epoch > 1000):
#                 quad_number = random.randint(10,30)
#                 _x = np.linspace(x_low, x_hig, num=quad_number, endpoint=True)
#                 _t = np.linspace(ti, tf, num=quad_number, endpoint=True)
#                 _xx, _tt = np.meshgrid(_x, _t)
#                 x_quad = Variable(torch.from_numpy(_xx.reshape(-1,1)).float(), requires_grad=True).to(device)
#                 t_quad = Variable(torch.from_numpy(_tt.reshape(-1,1)).float(), requires_grad=True).to(device)
#                 t_RAR = (torch.rand(S, 1, requires_grad=True) *   (tf - ti) + ti).to(device)
#                 # _t_init_RAR = (torch.ones(500, 1, requires_grad=True) * ti).to(device)
#                 # _t_end_RAR = (torch.ones(500, 1, requires_grad=True) * tf).to(device)
#                 # t_RAR = torch.cat((t_RAR, _t_init_RAR, _t_end_RAR), dim=0)
#                 x_RAR = (torch.rand(len(t_RAR), n_d, requires_grad=True) * (x_hig - x_low + 2*x_mar) + x_low-x_mar).to(device)
#                 t_RAR = torch.cat((t_RAR, t_quad), dim=0)
#                 x_RAR = torch.cat((x_RAR, x_quad), dim=0)
#                 # x_RAR = torch.clamp(x_RAR, min=x_low, max=x_hig)
#                 # t0_RAR = 0.0*t_RAR.clone() + t0
#                 # x0_RAR = x_RAR.clone()
#                 # p0_RAR = p_init_torch(x0_RAR)
#                 # p0_hat_RAR = p_net(x0_RAR, t0_RAR)
#                 # e10_RAR = p0_RAR - p0_hat_RAR
#                 # e10_hat_RAR = e1_net(x0_RAR, t0_RAR)
#                 # mean_e0_RAR = torch.mean(torch.abs(e10_RAR/normalize-e10_hat_RAR/normalize))
#                 # if(mean_e0_RAR > 0.0):
#                 #     max_abs_e0, max_index = torch.max(torch.abs(e10_RAR/normalize-e10_hat_RAR/normalize), dim=0)
#                 #     x_max = x0_RAR[max_index]
#                 #     t_max = t0_RAR[max_index]
#                 #     x_bc = torch.cat((x_bc, x_max), dim=0)
#                 #     t_bc = torch.cat((t_bc, t_max), dim=0)
#                 #     print("... Ic add [x,t]:", x_max.data, t_max.data, max_abs_e0.data)
#                 res_RAR = e_res_func(x_RAR, t_RAR, e1_net, p_net)/normalize
#                 mean_res_error = torch.mean(torch.abs(res_RAR))
#                 print("RAR mean res: ", mean_res_error.data)
#                 if(mean_res_error > 0.0):
#                     max_abs_res, max_index = torch.max(res_RAR**2, dim=0)
#                     x_max = x_RAR[max_index]
#                     t_max = t_RAR[max_index]
#                     x = torch.cat((x, x_max), dim=0)
#                     t = torch.cat((t, t_max), dim=0)
#                     debug_value = (e_res_func(x_max, t_max, e1_net, p_net)/normalize)**2
#                     print("... Res add [x,t]:", x_max.data, t_max.data, max_abs_res.data, debug_value.data)
#                 # res_x_RAR = torch.autograd.grad(res_RAR, x_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
#                 # res_t_RAR = torch.autograd.grad(res_RAR, t_RAR, grad_outputs=torch.ones_like(res_RAR), create_graph=True)[0]
#                 # mean_res_grad = torch.mean(torch.abs(res_x_RAR)+torch.abs(res_t_RAR))
#                 # print("RAR mean res input: ", mean_res_grad.data)
#                 # if(mean_res_grad > 0.0):
#                 #     max_abs_res_input, max_index = torch.max(res_x_RAR**2+ res_t_RAR**2, dim=0)
#                 #     x_max = x_RAR[max_index].view(-1,1)
#                 #     t_max = t_RAR[max_index].view(-1,1)
#                 #     x = torch.cat((x, x_max), dim=0)
#                 #     t = torch.cat((t, t_max), dim=0)
#                 #     print("... Res Grad add [x,t]:", x_max.data, t_max.data, ". max res value: ", max_abs_res_input.data)
#                 FLAG = 0
#             # Store alpha data (write data)
#             np.save(FOLDER+"output/e1_Nsample_list.npy", np.array(Nsample_list))
#             np.save(FOLDER+"output/e1_Loss_1_list.npy", np.array(Loss_1_list))
#             np.save(FOLDER+"output/e1_Loss_2_list.npy", np.array(Loss_2_list))
#             np.save(FOLDER+"output/e1_Alpha_list.npy", np.array(Alpha_list))
#             np.save(FOLDER+"output/e1_Alpha_mean_list.npy", np.array(Alpha_mean_list))
#             np.save(FOLDER+"output/e1_xsamples.npy", np.array(x.clone().detach().numpy()))
#             np.save(FOLDER+"output/e1_tsamples.npy", np.array(t.clone().detach().numpy()))

#         # Terminate training 
#         if(loss.data < enet_terminate):
#             training_time = time.time() - start_time
#             print("e1net best epoch:", epoch, ", loss:", loss.data, 
#                   )
#             torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': e1_net.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': loss.data,
#                     'label': "e1_net",
#                     'train_time': training_time,
#                     }, PATH)
#             np.save(FOLDER+"output/e1_net_train_loss.npy", np.array(loss_history))
#             # Store alpha data (calculate data)
#             Nsample_i = x_bc.shape[0] + x.shape[0]
#             Loss_1_i    = (loss).data.cpu().numpy().item()
#             Loss_2_i    = (0.0*loss).data.cpu().numpy().item() ###
#             alpha_over_time = []
#             for i in range(len(t1s)):
#                 t1 = t1s[i]
#                 pt_t_data = Variable(torch.from_numpy(x_data*0+t1).float(), requires_grad=True).to(device)
#                 ehat = e1_net(pt_x_data, pt_t_data).data.cpu().numpy()
#                 e1 = E1_list[i]
#                 alpha = np.max(np.abs(e1-ehat))/ np.max(np.abs(ehat))
#                 alpha = np.round(alpha, 3)
#                 alpha_over_time.append(alpha)
#             max_alpha = np.max(alpha_over_time)
#             Nsample_list.append(Nsample_i)
#             Loss_1_list.append(Loss_1_i)
#             Loss_2_list.append(Loss_2_i)
#             Alpha_list.append(max_alpha)
#             Alpha_mean_list.append(np.mean(alpha_over_time))
#             # Store alpha data (write data)
#             np.save(FOLDER+"output/e1_Nsample_list.npy", np.array(Nsample_list))
#             np.save(FOLDER+"output/e1_Loss_1_list.npy", np.array(Loss_1_list))
#             np.save(FOLDER+"output/e1_Loss_2_list.npy", np.array(Loss_2_list))
#             np.save(FOLDER+"output/e1_Alpha_list.npy", np.array(Alpha_list))
#             np.save(FOLDER+"output/e1_Alpha_mean_list.npy", np.array(Alpha_mean_list))
#             return

#         if (epoch) % 1000 == 0:
#             print(epoch, " loss: ", loss.data)
#         loss.backward(retain_graph=True) 
#         optimizer.step()
#         if (epoch + 1) % iterations_per_decay == 0:
#             scheduler.step()

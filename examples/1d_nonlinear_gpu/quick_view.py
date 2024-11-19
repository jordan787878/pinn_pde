import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter, ScalarFormatter
import random
from tqdm import tqdm
import warnings
import time
from scipy.optimize import curve_fit
import argparse

from main_train_pnet import PNet
from main_train_enet import ENet


FOLDER = "exp1/main_alphas/"
DATA_FOLDER = "exp1/data/"

device = "cpu"
print(device)

n_d = 1
mu = -2.0
std = 0.5
const_a = -0.1
const_b = 0.1
const_c = 0.5
const_d = 0.5
const_e = 0.8
x_low = -6
x_hig = 6

t0 = 0.0
T_end = 5.0
t1s = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

datas = ["data1/"]


def p_init(x):
    return np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi))


def p_init_torch(x):
    # Ensure x is a torch tensor
    x = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Compute the Gaussian function
    exponent = -0.5 * ((x - mu) / std) ** 2
    normalization = std * torch.sqrt(torch.tensor(2 * torch.pi))
    result = torch.exp(exponent) / normalization

    return result


def p_res_func(x, t, pnet, verbose=False):
    p = pnet(x,t)
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_t = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x), create_graph=True)[0]
    residual = p_t + (3*const_a*x*x + 2*const_b*x + const_c)*p \
                   + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*p_x \
                   - 0.5*const_e*const_e*p_xx
    if(verbose):
        print(p_xx[0:10,:]) #; print(p_x.shape, p_t.shape, p_xx.shape, residual.shape)
    return residual


def e_res_func(x, t, e_net, p_net, verbose=False):
    e1_out = e_net(x, t)[:, 0].view(-1, 1)
    p_res = p_res_func(x, t, p_net)
    e_x = torch.autograd.grad(e1_out, x, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_t = torch.autograd.grad(e1_out, t, grad_outputs=torch.ones_like(e1_out), create_graph=True)[0]
    e_xx = torch.autograd.grad(e_x,  x, grad_outputs=torch.ones_like(e_x), create_graph=True)[0]
    residual = p_res + e_t + (3*const_a*x*x + 2*const_b*x + const_c)*e1_out \
                     + (const_a*x*x*x + const_b*x*x + const_c*x + const_d)*e_x \
                     - 0.5*const_e*const_e*e_xx
    if(verbose):
        print(e_x.shape, e_t.shape, e_xx.shape, residual.shape)
    return residual
 

def get_p_normalize():
    x = np.linspace(x_low, x_hig, num=200, endpoint=True)
    p0_true = p_init(x)
    return np.max(np.abs(p0_true))


def get_e1_normalize(pnet):
    x = np.linspace(x_low, x_hig, num=200, endpoint=True).reshape(-1,1)
    p0_true = p_init(x)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_t = pt_x*0.0 + t0
    p0_hat = pnet(pt_x, pt_t).data.cpu().numpy()
    e0_true = p0_true - p0_hat
    return np.max(np.abs(e0_true))


def load_trained_model(net, PATH, PATH_LOSS):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(checkpoint['label'] + " best epoch   : ", epoch, ", loss:", loss.data)
    print(checkpoint['label'] + " training time: ", checkpoint['train_time'])
    loss_history = np.load(PATH_LOSS)
    min_loss = min(loss_history)
    plt.figure()
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.ylim([min_loss, 10*min_loss])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(FOLDER+"figs/"+checkpoint['label']+"_loss_history.png")
    plt.close()
    return net


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"


def show_results(pnet, enet):
    plt.rcParams['font.size'] = 18
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    limit_margin = 0.1

    # Create a ScalarFormatter object
    formatter = ScalarFormatter()
    formatter.set_scientific(True)

    p_monte_list = []
    p_hat_list = []
    e1_list = []
    e1_hat_list = []
    e2_list = []
    e2_hat_list = []
    p_res_list = []
    e_res_list = []
    e2_res_list = []
    t1s = [1.0, 3.0, 5.0]
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        # if(t1 == 0.0): p_monte = p_init(x)
        p_monte_list.append(p_monte)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        phat = pnet(pt_x, pt_t1)
        ehat = enet(pt_x, pt_t1)[:,0].view(-1,1)
        #e2hat = enet(pt_x, pt_t1)[:,1].view(-1,1)
        pres = p_res_func(pt_x, pt_t1, pnet).data.cpu().numpy()
        eres = e_res_func(pt_x, pt_t1, enet, pnet).data.cpu().numpy()
        # e2res = e2_res_func(pt_x, pt_t1, enet, pnet).data.cpu().numpy()

        phat = phat.data.cpu().numpy() # change tensor to numpy
        ehat = ehat.data.cpu().numpy() # change tensor to numpy
        #e2hat = e2hat.data.cpu().numpy()
        p_hat_list.append(phat)
        e1 = p_monte - phat
        e1_list.append(e1)
        e1_hat_list.append(ehat)
        # e2 = e1 - ehat
        # e2_list.append(e2)
        # e2_hat_list.append(e2hat)

        p_res_list.append(pres)
        e_res_list.append(eres)
        # e2_res_list.append(e2res)

    global_max = float('-inf')
    for i, (phat) in enumerate(zip(p_hat_list)):
        max_value = np.max(np.abs(phat))
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
    for i, (p_monte, p_hat, e1_hat) in enumerate(zip(p_monte_list, p_hat_list, e1_hat_list)):
        # if i == 0:
        #     ax1 = axs[0,0]
        #     ax1.set_ylabel("PDF")
        # if i == 1:
        #     ax1 = axs[1,0]
        #     ax1.set_ylabel("PDF")
        # if i == 2:
        #     ax1 = axs[2,0]
        #     ax1.set_xlabel("x")
        #     ax1.set_ylabel("PDF")
        # if i == 3:
        #     ax1 = axs[0,1]
        # if i == 4:
        #     ax1 = axs[1,1]
        # if i == 5:
        #     ax1 = axs[2,1]
        #     ax1.set_xlabel("x")
        eL = 2.0 * np.max(np.abs(e1_hat)); eL = np.round(eL, 3)
        ax1 = axs[i]
        ax1.plot(x, p_monte, "black", linewidth = 1.0, label=r"$p$")
        ax1.plot(x, p_hat, "red", linewidth = 1.0, linestyle="--", label=r"$\hat{p}$")
        ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+eL, y2=p_hat.reshape(-1)-eL,
                            color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")
        if i < 2:
            ax1.set_xticks([])
        if i == 2:
            ax1.set_xlabel("x")
        ax1.set_xlim([x_low, x_hig])
        ax1.set_ylim(0.0, global_max+limit_margin)
        ax1.set_xlabel("PDF")
        # ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $e_S:$ "+str(eL),
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
    plt.tight_layout(pad=0.7, h_pad=0.1)
    fig.savefig(FOLDER+'figs/phat_eS.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        max_value = max(np.max(np.abs(e1_true)), np.max(np.abs(e1_hat)))
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 1, figsize=(5, 6))
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        # if i == 0:
        #     ax1 = axs[0,0]
        #     ax1.set_ylabel("Error")
        # if i == 1:
        #     ax1 = axs[1,0]
        #     ax1.set_ylabel("Error")
        # if i == 2:
        #     ax1 = axs[2,0]
        #     ax1.set_xlabel("x")
        #     ax1.set_ylabel("Error")
        # if i == 3:
        #     ax1 = axs[0,1]
        # if i == 4:
        #     ax1 = axs[1,1]
        # if i == 5:
        #     ax1 = axs[2,1]
        #     ax1.set_xlabel("x")
        eL = 2.0 * np.max(np.abs(e1_hat))
        eL = np.round(eL, 3)
        alpha = np.max(np.abs(e1_true-e1_hat))/ np.max(np.abs(e1_hat))
        alpha = np.round(alpha, 3)
        print("eL: ", np.round(eL, 3), "\t alpha: ", np.round(alpha,3))
        ax1 = axs[i]
        ax1.plot(x, e1_true, "black", linewidth=1.0, label=r"$e_1$")
        ax1.plot(x, e1_hat,  "red", linewidth=1.0, linestyle="--", label=r"$\hat{e}_1$")
        ax1.fill_between(x.reshape(-1), y1=0.0*p_hat.reshape(-1)+eL, y2=0.0*p_hat.reshape(-1)-eL,
                         color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")
        if i < 2:
            ax1.set_xticks([])
        if i == 2:
            ax1.set_xlabel("x")

        ax1.set_xlim([x_low, x_hig])
        ax1.set_ylim(-(1.5*eL), 1.5*eL)
        ax1.set_ylabel("e")
        # ax1.set_ylim(-(global_max), global_max)
        # ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $\alpha_1:$ "+str(alpha),
                 transform=axs[i].transAxes, verticalalignment='top', fontsize=18)
        # Set y-axis to scientific notation
        yScalarFormatter = ScalarFormatterClass(useMathText=True)
        yScalarFormatter.set_powerlimits((0,0))
        ax1.yaxis.set_major_formatter(yScalarFormatter)
    plt.tight_layout(pad=0.3, h_pad=0.1)
    fig.savefig(FOLDER+'figs/e1hat_eS.pdf', format='pdf', dpi=300)
    plt.close()

    # global_max = float('-inf')
    # for i, (eres, pres) in enumerate(zip(e_res_list, p_res_list)):
    #     max_value = max(np.max(np.abs(pres/pnet.scale)), 0.0)
    #     global_max = max(global_max, max_value)
    # fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    # for i in range(0,6):
    #     pres = p_res_list[i]
    #     if i == 0:
    #         ax1 = axs[0,0]
    #         ax1.set_ylabel("Error")
    #     if i == 1:
    #         ax1 = axs[1,0]
    #         ax1.set_ylabel("Error")
    #     if i == 2:
    #         ax1 = axs[2,0]
    #         ax1.set_xlabel("x")
    #         ax1.set_ylabel("Error")
    #     if i == 3:
    #         ax1 = axs[0,1]
    #     if i == 4:
    #         ax1 = axs[1,1]
    #     if i == 5:
    #         ax1 = axs[2,1]
    #         ax1.set_xlabel("x")
    #     if i == 0:
    #         ax1.plot(x, pres/pnet.scale, "red", linewidth=1.0, linestyle="--", label=r"$D[\hat{e}_1]$")
    #         #ax1.plot(x, pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
    #         ax1.legend()  # Add legend only to the first subplot
    #     else:
    #         ax1.plot(x, pres/pnet.scale, "red", linewidth=1.0, linestyle="--")
    #         #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
    #     ax1.set_ylim([-global_max, global_max])
    #     ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    # plt.tight_layout()
    # # plt.savefig(FOLDER+"figs/enet_res.png")
    # fig.savefig(FOLDER+'figs/p_res.pdf', format='pdf', dpi=300)
    # plt.close()

    # global_max = float('-inf')
    # for i, (e1hat, eres, pres) in enumerate(zip(e1_hat_list, e_res_list, p_res_list)):
    #     max_value = max(np.max(np.abs(eres/enet.scale)), 0.0)**2
    #     global_max = max(global_max, max_value)
    # fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    # for i in range(0,6):
    #     pres = p_res_list[i]
    #     eres = e_res_list[i]/enet.scale
    #     if i == 0:
    #         ax1 = axs[0,0]
    #         ax1.set_ylabel("Error")
    #     if i == 1:
    #         ax1 = axs[1,0]
    #         ax1.set_ylabel("Error")
    #     if i == 2:
    #         ax1 = axs[2,0]
    #         ax1.set_xlabel("x")
    #         ax1.set_ylabel("Error")
    #     if i == 3:
    #         ax1 = axs[0,1]
    #     if i == 4:
    #         ax1 = axs[1,1]
    #     if i == 5:
    #         ax1 = axs[2,1]
    #         ax1.set_xlabel("x")
    #     if i == 0:
    #         ax1.plot(x, eres**2, "red", linewidth=1.0, linestyle="--", label=r"$r_2^2$")
    #         #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
    #         ax1.legend()  # Add legend only to the first subplot
    #     else:
    #         ax1.plot(x, eres**2, "red", linewidth=1.0, linestyle="--")
    #         #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
    #     # ax1.set_ylim([0, global_max])
    #     ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    # plt.tight_layout()
    # # plt.savefig(FOLDER+"figs/enet_res.png")
    # fig.savefig(FOLDER+'figs/e1_res.pdf', format='pdf', dpi=300)
    # plt.close()


def show_results_old(pnet, enet):
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    limit_margin = 0.0

    # # Create a ScalarFormatter object
    # formatter = ScalarFormatter()
    # formatter.set_scientific(True)

    p_monte_list = []
    p_hat_list = []
    e1_list = []
    e1_hat_list = []
    e2_list = []
    e2_hat_list = []
    p_res_list = []
    e_res_list = []
    e2_res_list = []
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        # if(t1 == 0.0): p_monte = p_init(x)
        p_monte_list.append(p_monte)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        phat = pnet(pt_x, pt_t1)
        ehat = enet(pt_x, pt_t1)[:,0].view(-1,1)
        #e2hat = enet(pt_x, pt_t1)[:,1].view(-1,1)
        pres = p_res_func(pt_x, pt_t1, pnet).data.cpu().numpy()
        eres = e_res_func(pt_x, pt_t1, enet, pnet).data.cpu().numpy()
        # e2res = e2_res_func(pt_x, pt_t1, enet, pnet).data.cpu().numpy()
        
        phat = phat.data.cpu().numpy() # change tensor to numpy
        ehat = ehat.data.cpu().numpy() # change tensor to numpy
        #e2hat = e2hat.data.cpu().numpy()
        p_hat_list.append(phat)
        e1 = p_monte - phat
        e1_list.append(e1)
        e1_hat_list.append(ehat)
        # e2 = e1 - ehat
        # e2_list.append(e2)
        # e2_hat_list.append(e2hat)

        p_res_list.append(pres)
        e_res_list.append(eres)
        # e2_res_list.append(e2res)

    global_max = float('-inf')
    for i, (phat) in enumerate(zip(p_hat_list)):
        max_value = np.max(np.abs(phat))
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for i, (p_monte, p_hat, e1_hat) in enumerate(zip(p_monte_list, p_hat_list, e1_hat_list)):
        if i == 0:
            ax1 = axs[0,0]
            ax1.set_ylabel("PDF")
        if i == 1:
            ax1 = axs[1,0]
            ax1.set_ylabel("PDF")
        if i == 2:
            ax1 = axs[2,0]
            ax1.set_xlabel("x")
            ax1.set_ylabel("PDF")
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
            ax1.set_xlabel("x")
        eL = 2.0 * np.max(np.abs(e1_hat)); eL = np.round(eL, 3)

        ax1.plot(x, p_monte, "black", linewidth = 1.0, label=r"$p$")
        ax1.plot(x, p_hat, "red", linewidth = 1.0, linestyle="--", label=r"$\hat{p}$")
        ax1.fill_between(x.reshape(-1), y1=p_hat.reshape(-1)+eL, y2=p_hat.reshape(-1)-eL, 
                            color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")

        ax1.set_xlim([x_low, x_hig])
        ax1.set_ylim(0.0, global_max+limit_margin)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $e_S:$ "+str(eL), 
                  transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/phat_eS.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        max_value = max(np.max(np.abs(e1_true)), np.max(np.abs(0.0)))
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for i, (e1_true, e1_hat) in enumerate(zip(e1_list, e1_hat_list)):
        if i == 0:
            ax1 = axs[0,0]
            ax1.set_ylabel("Error")
        if i == 1:
            ax1 = axs[1,0]
            ax1.set_ylabel("Error")
        if i == 2:
            ax1 = axs[2,0]
            ax1.set_xlabel("x")
            ax1.set_ylabel("Error")
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
            ax1.set_xlabel("x")
        eL = 2.0 * np.max(np.abs(e1_hat))
        eL = np.round(eL, 3)
        alpha = np.max(np.abs(e1_true-e1_hat))/ np.max(np.abs(e1_hat))
        alpha = np.round(alpha, 3)
        print("eL: ", np.round(eL, 3), "\t alpha: ", np.round(alpha,3))

        ax1.plot(x, e1_true, "black", linewidth=1.0, label=r"$e_1$")
        ax1.plot(x, e1_hat,  "red", linewidth=1.0, linestyle="--", label=r"$\hat{e}_1$")
        ax1.fill_between(x.reshape(-1), y1=0.0*p_hat.reshape(-1)+eL, y2=0.0*p_hat.reshape(-1)-eL, 
                         color="green", alpha=0.3, label=r"$e_S$")
        if i == 0:
            ax1.legend(loc="upper right")

        ax1.set_xlim([x_low, x_hig])
        ax1.set_xticks([-6, -4, -2, 0, 2, 4, 6])
        # ax1.set_ylim(-(1.5*eL), 1.5*eL)
        # ax1.set_ylim(-(global_max), global_max)
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
        ax1.text(0.01, 0.98, r"$t:$ "+str(t1s[i]) + r", $\alpha_1:$ "+str(alpha), 
                  transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        # Set y-axis to scientific notation
        # yScalarFormatter = ScalarFormatterClass(useMathText=True)
        # yScalarFormatter.set_powerlimits((0,0))
        # ax1.yaxis.set_major_formatter(yScalarFormatter)
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/e1hat_eS.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (eres, pres) in enumerate(zip(e_res_list, p_res_list)):
        max_value = max(np.max(np.abs(pres/pnet.scale)), 0.0)**2
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for i in range(0,6):
        pres = p_res_list[i]
        if i == 0:
            ax1 = axs[0,0]
            ax1.set_ylabel("Error")
        if i == 1:
            ax1 = axs[1,0]
            ax1.set_ylabel("Error")
        if i == 2:
            ax1 = axs[2,0]
            ax1.set_xlabel("x")
            ax1.set_ylabel("Error")
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
            ax1.set_xlabel("x")
        if i == 0:
            ax1.plot(x, (pres/pnet.scale)**2, "red", linewidth=1.0, linestyle="--", label=r"$r_1^2$")
            #ax1.plot(x, pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, (pres/pnet.scale)**2, "red", linewidth=1.0, linestyle="--")
            #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
        ax1.set_ylim([0, global_max])
        ax1.set_xlim([x_low, x_hig])
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(FOLDER+"figs/enet_res.png")
    fig.savefig(FOLDER+'figs/p_res.pdf', format='pdf', dpi=300)
    plt.close()

    global_max = float('-inf')
    for i, (e1hat, eres, pres) in enumerate(zip(e1_hat_list, e_res_list, p_res_list)):
        max_value = max(np.max(np.abs(eres/enet.scale)), 0.0)**2
        global_max = max(global_max, max_value)
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for i in range(0,6):
        pres = p_res_list[i]
        eres = e_res_list[i]/enet.scale
        if i == 0:
            ax1 = axs[0,0]
        if i == 1:
            ax1 = axs[1,0]
        if i == 2:
            ax1 = axs[2,0]
            ax1.set_xlabel("x")
        if i == 3:
            ax1 = axs[0,1]
        if i == 4:
            ax1 = axs[1,1]
        if i == 5:
            ax1 = axs[2,1]
            ax1.set_xlabel("x")
        if i == 0:
            ax1.plot(x, eres**2, "red", linewidth=1.0, linestyle="--", label=r"$r_2^2$")
            #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-", label=r"$-D[\hat{p}]$")
            ax1.legend()  # Add legend only to the first subplot
        else:
            ax1.plot(x, eres**2, "red", linewidth=1.0, linestyle="--")
            #ax1.plot(x, -pres, "black", linewidth=1.0, linestyle="-")
        ax1.set_ylim([0, global_max])
        ax1.set_xlim([x_low, x_hig])
        ax1.set_xticks([-6, -4, -2, 0, 2, 4, 6])
        ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    # plt.savefig(FOLDER+"figs/enet_res.png")
    fig.savefig(FOLDER+'figs/e1_res.pdf', format='pdf', dpi=300)
    plt.close()


def plot_p_monte():
    markers = ['o', 's', 'D', '^', 'v', '<']
    plt.figure(figsize=(8,6))
    for i in range(len(t1s)):
        t1 = t1s[i]
        x_sim = np.load(DATA_FOLDER+ datas[0] + "xsim.npy")
        p_sim = np.load(DATA_FOLDER+ datas[0] + "psim_t"+str(t1)+".npy")
        line, = plt.plot(x_sim, p_sim, linewidth=1.0)
        
        # Plot markers at specific intervals
        interval = 5  # Marker interval
        plt.plot(x_sim[::interval], p_sim[::interval], markers[i], markersize=3, color=line.get_color(), label="t="+str(t1))  # 'o' marker style


    plt.grid(linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER+"figs/p_sol_monte.png")
    print("save fig to "+FOLDER+"figs/p_sol_monte.png")
    plt.close()


def plot_p_surface(p_net, num=100):
    # x_samples = np.load(FOLDER+"output/p_xsamples.npy")
    # t_samples = np.load(FOLDER+"output/p_tsamples.npy")
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    phat = p_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)

    p_list = []
    x_monte = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x_monte = Variable(torch.from_numpy(x_monte).float(), requires_grad=True).to(device)
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        p_list.append(p_monte)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, phat, cmap='viridis', alpha=0.8, label=r"$\hat{p}$")
    # z_max = 1.5*np.max(np.abs(phat))
    # ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="x", color="black", s=0.02, label='Data Points')
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x_monte*0 + t1
        if i == 0:
            ax.plot(x_monte, t1_monte, p_list[i], color="black", label=r"$p_s$")
        else:
            ax.plot(x_monte, t1_monte, p_list[i], color="black")

    ax.set_xlabel("x")
    ax.set_ylabel("t"); 
    ax.set_zlabel('PDF')
    ax.legend()
    ax.view_init(20, -60)
    # y_ticks = np.array([1, 2, 3])  # Example y-tick positions
    # ax.set_yticks(y_ticks)  # Set the positions of the y-ticks
    # Adjust layout manually
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/phat_surface_plot.pdf', format='pdf', dpi=300)


def plot_e1_surface(p_net, e1_net, num=100):
    # x_samples = np.load(FOLDER+"output/e1_xsamples.npy")
    # t_samples = np.load(FOLDER+"output/e1_tsamples.npy")
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    e1hat = e1_net(pt_x, pt_t).data.cpu().numpy().reshape(num, -1)
    
    e1_list = []
    x_monte = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x_monte = Variable(torch.from_numpy(x_monte).float(), requires_grad=True).to(device)
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        pt_t1_monte = Variable(torch.from_numpy(x_monte*0+t1).float(), requires_grad=True).to(device)
        p_hat = p_net(pt_x_monte, pt_t1_monte).data.cpu().numpy()
        e1 = p_monte - p_hat
        e1_list.append(e1)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, e1hat, cmap='viridis', alpha=0.6, label=r"$\hat{e}_1$")
    # z_max = 1.2*np.max(np.abs(e1hat))
    # ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="x", color="black", s=0.02, label='Data Points')
    for i in range(len(t1s)):
        t1 = t1s[i]
        t1_monte = x_monte*0 + t1
        if(i == 0):
            ax.plot(x_monte, t1_monte, e1_list[i], color="black", label=r"$e_1$")
        else:
            ax.plot(x_monte, t1_monte, e1_list[i], color="black")

    # Set z-ticks to scientific notation
    ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.zaxis.get_major_formatter().set_powerlimits((-2, 2))  # Use scientific notation if value is outside this range

    # ax.set_zlim([-z_max, z_max])
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("Error")
    ax.view_init(20, -60)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/e1hat_surface_plot.pdf', format='pdf', dpi=300)


def plot_pres_surface(p_net, num=100):
    x = np.linspace(x_low, x_hig, num=num)
    t = np.linspace(t0, T_end, num=num)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pres = p_res_func(pt_x, pt_t, p_net).data.cpu().numpy().reshape(num, -1)/p_net.scale

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, pres**2, cmap="Greys")
    x_samples = np.load(FOLDER+"output/p_xsamples.npy")
    t_samples = np.load(FOLDER+"output/p_tsamples.npy")
    z_max = 1.0*np.max(pres**2)
    ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="x", color="blue", s=0.05, alpha=0.2, label='Data Points')
    # ax.plot_surface(x_mesh, t_mesh, pres**2, cmap='viridis', alpha=0.8, label=r"$r_1^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("t"); 
    ax.set_zlabel('r')
    # ax.legend()
    ax.view_init(40, -60)
    ax.grid(False)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/pres_surface_plot.pdf', format='pdf', dpi=300)


def plot_e1res_surface(p_net, e1_net, num=100):
    x = np.linspace(x_low, x_hig, num=num, endpoint=True)
    t = np.linspace(t0, T_end, num=num, endpoint=True)
    x_mesh, t_mesh = np.meshgrid(x,t)
    pt_x = Variable(torch.from_numpy(x_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_mesh.reshape(-1,1)).float(), requires_grad=True).to(device)
    e1res = (e_res_func(pt_x, pt_t, e1_net, p_net)/e1_net.scale).detach().cpu().numpy().reshape(num, -1)

    # index = np.argmax(np.abs(e1res.flatten()))
    # max_e1_res = np.round(np.max(np.abs(e1res.flatten())),4)
    # x_max = np.round(x_mesh.flatten()[index],2)
    # t_max = np.round(t_mesh.flatten()[index],2)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, t_mesh, e1res**2, cmap="Greys")
    print(np.max(e1res**2))
    x_samples = np.load(FOLDER+"output/e1_xsamples.npy")
    t_samples = np.load(FOLDER+"output/e1_tsamples.npy")
    z_max = 1.0*np.max(e1res**2)
    ax.scatter(x_samples, t_samples, t_samples*0+z_max, marker="x", color="blue", s=0.05, alpha=0.2, label='Data Points')
    ax.set_xlabel("x")
    ax.set_ylabel("t"); 
    ax.set_zlabel('r')
    # ax.legend()
    ax.view_init(40, -60)
    ax.grid(False)
    # zScalarFormatter = ScalarFormatterClass(useMathText=True)
    # zScalarFormatter.set_powerlimits((0,0))
    # ax.zaxis.set_major_formatter(zScalarFormatter)
    # plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    fig.savefig(FOLDER+'figs/e1res_surface_plot.pdf', format='pdf', dpi=300)


def plot_train_loss(path_1, path_2):
    loss_history_1 = np.load(path_1)
    min_loss_1 = min(loss_history_1)
    loss_history_2 = np.load(path_2)
    min_loss_2 = min(loss_history_2)
    print(loss_history_2)
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    axs[0].plot(np.arange(len(loss_history_1)), loss_history_1, "black", linewidth=1.0)
    axs[0].set_ylim([min_loss_1, 10*min_loss_1])
    axs[1].plot(np.arange(len(loss_history_2)), loss_history_2, "black", linewidth=1.0)
    axs[1].set_ylim([min_loss_2, 10*min_loss_2])
    axs[0].grid(linewidth=0.5)
    axs[1].grid(linewidth=0.5)
    axs[1].set_xlabel("epochs")
    axs[0].set_ylabel("train loss: "+r"$\hat{p}$")
    axs[1].set_ylabel("train loss: "+r"$\hat{e}_1$")
    plt.tight_layout()
    fig.savefig(FOLDER+'figs/train_loss.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def fit_function(x, C1, C2):
    return C1 * np.sqrt(x) + C2


def plot_alpha_data():
    plt.rcParams['font.size'] = 18
    data_folder = "exp1/main_alphas/"
    max_alpha_to_fit = 300.0
    max_total_loss_to_fit = 100.0
    num_runs = 6
    plt.figure(figsize=(8, 6))
    plt.axhline(y=1, color="black", linestyle='-', linewidth=2)
    X = []
    Y = []
    X1 = []
    X2 = []
    for i in range(0, num_runs):
        data_folder_seedi = data_folder  + "seed" + str(i) + "/"
        # data_folder_seedi = FOLDER
        Nsample_array = np.load(data_folder_seedi+"output/e1_Nsample_list.npy")
        Loss_1_array = np.load(data_folder_seedi+"output/e1_Loss_1_list.npy")
        Alpha_array = np.load(data_folder_seedi+"output/e1_Alpha_list.npy")

        mask_1 = Alpha_array < max_alpha_to_fit
        mask_2 = (Loss_1_array) < max_total_loss_to_fit
        mask = mask_1 & mask_2

        Loss_1_array = Loss_1_array[mask]
        Alpha_array = Alpha_array[mask]
        x_data = Loss_1_array
        y_data = Alpha_array
        plt.plot(x_data, y_data, marker='o', linestyle='--', linewidth=0.5, markersize=1, label="seed"+str(i)) # , label="run seed"+str(i))
        plt.yscale('log')
        plt.xscale('log')
        X.append(x_data)
        Y.append(y_data)

    # Fit the model
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    lower_bounds = [0, 0]
    upper_bounds = [np.inf, np.inf]
    # popt, pcov = curve_fit(fit_function, X, Y, bounds=(lower_bounds, upper_bounds))
    # C1, C2 = popt
    # print("fit line constants:", C1, C2)
    # xs = np.linspace(np.min(X), np.max(X), num=100)
    # ys = C1 * np.sqrt(xs) + C2
    # plt.plot(xs, ys, color="black", linewidth=0.5, linestyle="--", label="fit line")
    plt.gca().invert_xaxis()
    plt.grid(linewidth=0.5)
    plt.xlabel(r"$\hat{e}_1$" + " training loss")
    plt.ylabel(r"$\max_{\tau} (\alpha_1(t))$")
    plt.legend(loc="upper right", fontsize=16)
    plt.tight_layout(pad=0.1)
    plt.savefig(FOLDER+'figs/alpha_vs_loss.pdf', format='pdf', dpi=300)
    plt.close()


def plot_training_loss_data():
    data_folder = "exp1/main_alphas/"
    plt.figure()
    for i in range(0,1):
        data_folder_seedi = data_folder + "seed" + str(i) + "/"
        loss_data = np.load(data_folder_seedi+"output/e1_net_train_loss.npy")
        plt.plot(np.arange(len(loss_data)), loss_data, linewidth=0.5, label="run seed"+str(i))
    plt.xlim([15000, 50000])
    plt.ylim([0, 0.01])
    plt.legend()
    plt.tight_layout()
    plt.savefig(FOLDER+'figs/loss_history_compare.pdf', format='pdf', dpi=300)
    plt.close()


def show_table(pnet, enet):
    x = np.load(DATA_FOLDER + datas[0] + "xsim.npy").reshape(-1,1)
    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)

    gap_list = []
    a1_list = []
    eS_ratio_list = []
    t1s = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    for t1 in t1s:
        p_monte = np.load(DATA_FOLDER + datas[0] + "psim_t" + str(t1) + ".npy").reshape(-1, 1)
        pt_t1 = Variable(torch.from_numpy(x*0+t1).float(), requires_grad=True).to(device)
        phat = pnet(pt_x, pt_t1)
        ehat = enet(pt_x, pt_t1)[:,0].view(-1,1)
        phat = phat.data.cpu().numpy() # change tensor to numpy
        ehat = ehat.data.cpu().numpy() # change tensor to numpy
        e1 = p_monte - phat
        a1 = np.max(np.abs(e1 - ehat))/np.max(np.abs(ehat))
        a1_list.append(a1)
        eS = 2.0*np.max(np.abs(ehat))
        eS_ratio = eS/np.max(np.abs(p_monte))
        gap = (eS - np.max(np.abs(e1)))/np.max(np.abs(p_monte))
        gap_list.append(gap)
        eS_ratio_list.append(eS_ratio)
    print("[info] max a1: " +str(np.max(np.array(a1_list))) + ", var a1:" + str(np.var(np.array(a1_list))))
    print("[info] max gap: " +str(np.max(np.array(gap_list))) + ", avg a1:" + str(np.mean(np.array(gap_list))))
    print("[info] max eS_ratio: " +str(np.max(np.array(eS_ratio_list))) + ", avg a1:" + str(np.mean(np.array(eS_ratio_list))))


def main():
    global seed
    global FOLDER
    FOLDER = FOLDER + "seed" + str(seed) + "/"
    print("seed: ", seed, ", folder:", FOLDER)
    torch.cuda.empty_cache()

    # plot_p_monte()
    
    p_model = PNet().to(device)
    e_model = ENet().to(device) 
    
    p_model.normalize = get_p_normalize()
    p_model = load_trained_model(p_model, PATH=FOLDER+"output/p_net.pth", PATH_LOSS=FOLDER+"output/p_net_train_loss.npy")
    p_model.eval()

    e_model.scale = get_e1_normalize(p_model)
    print("enet scale: ", e_model.scale)
    e_model = load_trained_model(e_model, PATH=FOLDER+"output/e1_net.pth", PATH_LOSS=FOLDER+"output/e1_net_train_loss.npy")
    e_model.eval()
    
    show_table(p_model, e_model)
    show_results(p_model, e_model)
    torch.cuda.empty_cache()

    # plot_p_surface(p_model)
    # plot_pres_surface(p_model)
    # plot_e1res_surface(p_model, e_model)
    # plot_train_loss(FOLDER+"output/p_net_train_loss.npy", 
    #                 FOLDER+"output/e1_net_train_loss.npy")
    plot_alpha_data()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass seed as a command-line argument")
    parser.add_argument("--seed", type=int, required=True, help="Seed value")
    args = parser.parse_args()
    # Modify the global variable
    seed = args.seed
    main()
    

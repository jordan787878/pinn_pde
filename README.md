# PINN_PDE_ERROR

Provide the 1D Nonlinear SDE example, using GeLU MLP, trained on GPU by cuda

## Prerequisites

The project is built on Linux desktop

Make sure you have the following on your system:
- NVIDIA GPU
- Python 3.12.2
- pip3 (python package manager)

## Installation

Follow these steps to set up your environment:

1. **create python virtual environment** run: python3 -m venv venv
2. **activate python virtual environment** run: source venv/bin/activate
3. **install packages** run: pip3 install -r requirements.txt

## 1D Nonlinear Dynamics
The pre-trained models are stored in the folder: examples/1d_nonlinear_gpu/exp1/main_alphas/[seed number]/output/

The data (of the Monte-Carlo simulation for "true" solution) is in the folder: examples/1d_nonlinear_gpu/exp1/data/

### To see the results of the pre-trained models
- cd examples/1d_nonlinear_gpu
- python quick_view.py --seed=[seed number from 0 to 5]
- This will generate plots in the corresponding folder: examples/1d_nonlinear_gpu/exp1/main_alphas/[seed number]/figs/


### To train new models
- cd examples/1d_nonlinear_gpu
- python main_train_pnet.py --seed=[seed number you want]
- after pnet training is complete, python main_train_enet.py --seed=[seed number you want]
- after enet training is complete, the models will be saved to: examples/1d_nonlinear_gpu/exp1/main_alphas/seed-test/output/
- and the figures will be saved to: examples/1d_nonlinear_gpu/exp1/main_alphas/seed-test/figs/



# Solving Stochastic Differential Equations with Error Bounds: Application for Orbit Uncertainty Propagation with Worst-Case Guarantees

Solve PDE with error bounds using physics-informed learning
- propagated probability density function (uncertainties) over time for orbital problems
- construct associated worst-case error bound of the approximate PDF function

See [this release](https://github.com/aria-systems-group/pinn_pde/tree/release/uai2025) for other applications.

## Prerequisites

The project is built on MacbookPro M2 chip. 
Make sure you have the following installed on your system:

- Python 3.12.2
- pip3 (python package manager)

## Installation

Follow these steps to set up your environment:

1. **create python virtual environment** run: python3 -m venv venv
2. **activate python virtual environment** run: source venv/bin/activate
3. **install required packages** run: pip3 install -r requirements.txt

## Run Examples

In each example, you can run:

- python train_p.py--train=0 (this will generate plots using pretrained models)
- python train_e1.py--train=0 (this will generate plots using pretrained models)
- python train_p.py --train=1 (this will train the pdf models and generate plots)
- python train_e1.py --train=1 (this will train the error models and generate plots)
- python compare_methods.py to generate plots in the paper that compare our approach with tested baselines
- python event_prob.py to generate plots in the paper that illustrate application of PDF + error bounds to upper/lower bound event probabilities

Saved Structure:
- The models are stored in the folder: output/
- The plots are saved in the folder: figs/

### 6D Keplerian Orbit in Spherical Coordinates
See exp_case1/

This shows the scalability of our method to 6D while providing tight error bounds

### 6D Keplerian Orbit in Equinoctial Orbital Elements
See exp_case1_equin/

This shows that our method is agnostic to state coordinates

### 4D peturbed Keplerian Orbit in Spherical Coordinates
See exp_case2_j2/

This is a plannar orbit under J2 perturbation and random process noise. This is the case where the golden standard is expensive Monte-Carlo (MC) simulation. And we compare our results to the MC simulation.

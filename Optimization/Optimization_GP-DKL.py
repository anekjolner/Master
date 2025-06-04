#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:06:31 2025

@author: anekristinekjolner
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mat73
from scipy.io import loadmat, savemat
from MOFE_PSO import MOFE_PSO
from functools import partial
import gpytorch

# ---------------------------------------------------------------------
# 1) LOADING GLOBAL DATA
# ---------------------------------------------------------------------
Input_global = loadmat('../Data/Input_global.mat')['Input_global']
Output_global = loadmat('../Data/Output_global.mat')['Output_global']

Input_Train = mat73.loadmat('../Data/Input_train.mat')['Input_train']
Output_Train = mat73.loadmat('../Data/Output_train.mat')['Output_train']
Input_Valid = mat73.loadmat('../Data/Input_valid.mat')['Input_valid']
Output_Valid = mat73.loadmat('../Data/Output_valid.mat')['Output_valid']
Input_Test  = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']


# ---------------------------------------------------------------------
# 2) RE-CREATE YOUR DKL CLASSES (FeatureExtractor + DKL_GPModel)
# ---------------------------------------------------------------------
# Recreate the FeatureExtractor with a configurable constructor:
class FeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_units=[512, 256, 128, 64],
                 dropout_prob=0.1, activation='relu'):
        super().__init__()
        act_functions = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh()
        }
        chosen_act = act_functions[activation.lower()]
        layers = []
        current_dim = input_dim
        for hdim in hidden_units:
            layers.append(torch.nn.Linear(current_dim, hdim))
            layers.append(chosen_act)
            layers.append(torch.nn.Dropout(dropout_prob))
            current_dim = hdim
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# DKL_GPModel remains as before:
class DKL_GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        proj_x = self.feature_extractor(x)
        mean_x = self.mean_module(proj_x)
        covar_x = self.covar_module(proj_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# For example, assuming the target is in column 1:
train_x = torch.tensor(Input_Train, dtype=torch.float32)
train_y_recovery = torch.tensor(Output_Train[:, 1], dtype=torch.float32)
train_y_purity = torch.tensor(Output_Train[:, 0], dtype=torch.float32)

def load_model(model_name, train_x, train_y):
    """
    Load a trained DKL+GP model using the config stored in the checkpoint.
    """
    # Load checkpoint and extract configuration
    checkpoint = torch.load(model_name)
    config = checkpoint['config']
    
    # Recreate likelihood (should match training)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(1e-3)
    
    # Create the feature extractor using the saved configuration
    feature_extractor = FeatureExtractor(input_dim=config['input_dim'],
                                         hidden_units=config['hidden_units'],
                                         dropout_prob=config.get('dropout_prob', 0.0),
                                         activation=config['activation'])
    
    # Initialize the DKL_GPModel using the original training data
    model = DKL_GPModel(train_x, train_y, likelihood, feature_extractor)
    
    # Load the state dictionary from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    likelihood.eval()
    
    return model, likelihood

# ---------------------------------------------------------------------
# 3) LOAD BOTH MODELS (ONE FOR RECOVERY, ONE FOR PURITY)
#     (Assuming you have saved them similarly.)
# ---------------------------------------------------------------------
model_recovery, likelihood_recovery = load_model('../Models/model_GNN_recovery.pth', train_x, train_y_recovery)
model_purity,   likelihood_purity   = load_model('../Models/model_GNN_purity.pth', train_x, train_y_purity)

# ---------------------------------------------------------------------
# 4) NORMALIZE / DENORMALIZE FUNCTIONS
# ---------------------------------------------------------------------
def des_normalize(Mat_ref, Mat_dados):
    """Denormalize the matrix using reference data."""
    result = np.zeros_like(Mat_dados)
    cols = min(Mat_ref.shape[1], Mat_dados.shape[1])
    for i in range(cols):
        max_value, min_value = Mat_ref[:, i].max(), Mat_ref[:, i].min()
        result[:, i] = (Mat_dados[:, i]) * (max_value - min_value) + min_value
    return result

def normalize(Mat_ref, Mat_dados):
    """Normalize the matrix using reference data."""
    result = np.zeros_like(Mat_dados)
    cols = min(Mat_ref.shape[1], Mat_dados.shape[1])
    for i in range(cols):
        max_value, min_value = Mat_ref[:, i].max(), Mat_ref[:, i].min()
        result[:, i] = (Mat_dados[:, i] - min_value) / (max_value - min_value)
    return result

# ---------------------------------------------------------------------
# 5) PREDICTION FUNCTION USING THE DKL MODELS
# ---------------------------------------------------------------------
def make_predictions(x, model, likelihood):
    """
    Make a prediction using the trained DKL model.
    Returns the mean prediction as a numpy scalar or array (depending on x shape).
    """
    # Convert input to a float tensor and ensure shape [N, 5]
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 5)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = model(x_tensor)
        pred_mean = pred_dist.mean
    return pred_mean.numpy().squeeze()

# ---------------------------------------------------------------------
# 6) CONSTRAINT FUNCTION
# ---------------------------------------------------------------------
def ConsFunc(x):
    """Compute constraints based on predicted Purity and Recovery."""
    x_p = np.reshape(x, (1, 5))
    # Use the correct likelihood for each model.
    purity   = make_predictions(x_p, model_purity, likelihood_purity)
    recovery = make_predictions(x_p, model_recovery, likelihood_recovery)
    # Example: require purity > 0 and recovery > 0.
    # To use constraints in the optimizer, return -purity and -recovery.
    consFunc = np.array([-purity, -recovery])
    return consFunc

# ---------------------------------------------------------------------
# 7) OBJECTIVE FUNCTION
# ---------------------------------------------------------------------
def ObjFun(x):
    """
    Compute the objective function:
    We want to maximize purity and recovery; therefore, return the negatives
    (so that MOFE_PSO, which minimizes, can solve the problem).
    """
    x_p = np.reshape(x, (1, 5))
    purity   = make_predictions(x_p, model_purity, likelihood_purity)
    recovery = make_predictions(x_p, model_recovery, likelihood_recovery)
    objFun = np.array([-purity, -recovery])  # shape (2,)
    return objFun

# ---------------------------------------------------------------------
# 8) MOFE_PSO CONFIGURATION & RUN
# ---------------------------------------------------------------------
options = type('', (), {})()
GlobalVars = type('', (), {})()

options.Nvar  = 5  # Number of decision variables
options.Nobj  = 2  # Number of objective functions
options.nCons = 2  # Number of constraints

options.ObjFun   = ObjFun
options.ConsFunc = ConsFunc

# Basic PSO parameters (adjust as needed)
options.swarmSize = 100
options.maxIter   = 100
options.useInputDecoder   = False
options.transferVariables = False
options.velocityInitializationFactor = np.array([0.3])
options.incrementFactor = np.array([0.3])
options.boundaryTolerance = np.array([0.01])
options.initialC0 = np.array([0.9])
options.finalC0   = np.array([0.4])
options.C1        = np.array([2])
options.C2        = np.array([2])
options.verbose   = 2

GlobalVars.nConsEvals = np.array([0])
GlobalVars.nObjEvals  = np.array([0])

# Bounds for your 5 inputs
options.lBound = np.array([0, 0, 0, 0, 0])
options.uBound = np.array([1, 1, 1, 1, 1])

(results, GlobalVars, options) = MOFE_PSO.RUN(options, GlobalVars)

# ---------------------------------------------------------------------
# 9) EXTRACT RESULTS
# ---------------------------------------------------------------------
Global_results = []
Global_results.append(results)

Num_nomDom = 0
for i in range(len(Global_results)):
    Num_nomDom += Global_results[i].nonDom.shape[0]

XX = np.zeros([Num_nomDom, options.Nvar])
YY = np.zeros([Num_nomDom, options.Nobj])

idx = 0
for j in range(len(Global_results)):
    for i in range(Global_results[j].nonDom.shape[0]):
        XX[idx, :] = Global_results[j].nonDom[i].x
        YY[idx, :] = Global_results[j].nonDom[i].obj
        idx += 1

# Gather info about all feasible (non-dominated) particles
Posic = []
Fval = []
for j in range(len(Global_results)):
    for i in range(options.swarmSize):
        for k in range(len(Global_results[j].Particle.nonDom[i])):
            Posic.append(Global_results[j].Particle.nonDom[i][k].x)
            Fval.append(Global_results[j].Particle.nonDom[i][k].obj)

Posic = np.array(Posic)
Fval = np.array(Fval)

# Collect all visited particles
Num_FisPar = 0
for i in range(len(Global_results)):
    Num_FisPar += Global_results[i].AllFisPart.shape[0]

AllPartXX = np.zeros([Num_FisPar, options.Nvar])
AllPartYY = np.zeros([Num_FisPar, options.Nobj])

jj = 0
for j in range(len(Global_results)):
    for i in range(len(Global_results[j].AllFisPart)):
        AllPartXX[jj, :] = Global_results[j].AllFisPart[i].x
        AllPartYY[jj, :] = Global_results[j].AllFisPart[i].obj
        jj += 1

# ---------------------------------------------------------------------
# 10) DENORMALIZE & PLOT
# ---------------------------------------------------------------------
YY_desnorm_1 = des_normalize(Output_global[:, 0:1], -YY[:, 0:1])
YY_desnorm_2 = des_normalize(Output_global[:, 1:2], -YY[:, 1:2])

Fval_desnorm_1 = des_normalize(Output_global[:, 0:1], -Fval[:, 0:1])
Fval_desnorm_2 = des_normalize(Output_global[:, 1:2], -Fval[:, 1:2])

AllPartYY_desnorm_1 = des_normalize(Output_global[:, 0:1], -AllPartYY[:, 0:1])
AllPartYY_desnorm_2 = des_normalize(Output_global[:, 1:2], -AllPartYY[:, 1:2])

XX_desnorm     = des_normalize(Input_global, XX)
Posic_desnorm  = des_normalize(Input_global, Posic)
AllPartXX_desnorm = des_normalize(Input_global, AllPartXX)

plt.figure()
plt.plot(YY_desnorm_1, YY_desnorm_2, 'o')
plt.xlabel('Purity (denormalized)')
plt.ylabel('Recovery (denormalized)')
plt.title('Non-dominated solutions found by MOFE_PSO')
plt.show()

YY = np.hstack((YY_desnorm_1, YY_desnorm_2))
Fval = np.hstack((Fval_desnorm_1, Fval_desnorm_2))
AllPartYY = np.hstack((AllPartYY_desnorm_1, AllPartYY_desnorm_2))

# ---------------------------------------------------------------------
# 11) SAVE RESULTS TO .mat
# ---------------------------------------------------------------------
mydict = {
    'Tempo_redes':  results.PartTimer,
    'XX_redes':     XX_desnorm,
    'Fval_redes':   Fval,
    'Posic_redes':  Posic_desnorm,
    'AllPartYY_redes': AllPartYY,
    'AllPartXX_redes': AllPartXX_desnorm,
    'YY_redes': YY,
}

savemat('PSO_GNN_100p_100i_purity_recovery.mat', mydict)

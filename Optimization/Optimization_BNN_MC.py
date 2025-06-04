#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:19:48 2025

@author: anekristinekjolner
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:26:05 2025

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
import torch.nn as nn

# ---------------------------------------------------------------------
# 1) LOADING GLOBAL AND TRAIN/VALID/TEST DATA
# ---------------------------------------------------------------------
Input_global = loadmat('../Data/Input_global.mat')['Input_global']
Output_global = loadmat('../Data/Output_global.mat')['Output_global']

Input_Train = mat73.loadmat('../Data/Input_train.mat')['Input_train']
Output_Train = mat73.loadmat('../Data/Output_train.mat')['Output_train']
Input_Valid = mat73.loadmat('../Data/Input_valid.mat')['Input_valid']
Output_Valid = mat73.loadmat('../Data/Output_valid.mat')['Output_valid']
Input_Test  = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']

# ---------------------------------------------------------------------------------------------------------------------------------------
# =============================================================================
# Define MC Dropout Model
# =============================================================================
class BNNModel(nn.Module):
    def __init__(self, input_dim=5, hidden_units=[300, 250, 200, 150, 100], output_dim=1, activation='relu', dropout_rate=0.1):
        super(BNNModel, self).__init__()
        layers = []
        act_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()
        prev_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn)
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = BNNModel(input_dim=5)

# ---------------------------------------------------------------------
# 3) FUNCTION TO LOAD A TRAINED MODEL
# ---------------------------------------------------------------------
def load_model(path):
    model = BNNModel()
    model.load_state_dict(torch.load(path))
    model.train()  # Keep dropout active
    return model

# ---------------------------------------------------------------------
# 4) LOAD BOTH MODELS (ONE FOR PURITY AND ONE FOR RECOVERY)
# ---------------------------------------------------------------------
model_recovery = load_model('../Models/model_BNN_MC_Recovery.pth')
model_purity   = load_model('../Models/model_BNN_MC_Purity.pth')

# ---------------------------------------------------------------------
# 5) NORMALIZATION/ DENORMALIZATION FUNCTIONS
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
    for i in range(Mat_ref.shape[1]):
        max_value, min_value = Mat_ref[:, i].max(), Mat_ref[:, i].min()
        result[:, i] = (Mat_dados[:, i] - min_value) / (max_value - min_value)
    return result

# ---------------------------------------------------------------------
# 6) PREDICTION FUNCTION USING THE TRAINED MODELS
# ---------------------------------------------------------------------
def make_predictions(x, model, samples=100):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            preds.append(model(x_tensor).squeeze().cpu().numpy())
    return np.mean(np.stack(preds), axis=0)

# ---------------------------------------------------------------------
# 7) CONSTRAINT FUNCTION
# ---------------------------------------------------------------------
def ConsFunc(x):
    """Compute the constraint function using predicted Purity and Recovery."""
    x_p = np.reshape(x, (1, 5))
    purity = make_predictions(x_p, model_purity)
    recovery = make_predictions(x_p, model_recovery)
    consFunc = np.array([-purity, -recovery])  # If using constraints of the form cons <= 0
    return consFunc

# ---------------------------------------------------------------------
# 8) OBJECTIVE FUNCTION
# ---------------------------------------------------------------------
def ObjFun(x):
    """Compute the objective function for maximizing Purity and Recovery."""
    x_p = np.reshape(x, (1, 5))
    purity = make_predictions(x_p, model_purity)
    recovery = make_predictions(x_p, model_recovery)
    # Returning negatives because MOFE_PSO minimizes the objective.
    objFun = np.array([-purity, -recovery])
    return objFun

# ---------------------------------------------------------------------
# 9) MOFE_PSO CONFIGURATION & INITIALIZATION
# ---------------------------------------------------------------------
options = type('', (), {})()
GlobalVars = type('', (), {})()

# Decision space and functions
options.Nvar = int(5)  # Number of decision variables (inputs)
options.Nobj = int(2)  # Two objectives: Purity and Recovery
options.nCons = int(2) # Two constraint functions

options.ObjFun = ObjFun
options.ConsFunc = ConsFunc

# MOFE_PSO parameters
options.swarmSize = int(100)  # Size of swarm
options.maxIter = int(100)     # Maximum number of iterations
options.useInputDecoder = False
options.transferVariables = False
options.velocityInitializationFactor = np.array([0.3])
options.incrementFactor = np.array([0.3])
options.boundaryTolerance = np.array([0.01])
options.initialC0 = np.array([0.9])
options.finalC0 = np.array([0.4])
options.C1 = np.array([2])
options.C2 = np.array([2])
options.verbose = int(2)

GlobalVars.nConsEvals = np.array([0])
GlobalVars.nObjEvals  = np.array([0])

# Set decision variable bounds
options.lBound = np.array([0, 0, 0, 0, 0])
options.uBound = np.array([1, 1, 1, 1, 1])

# ---------------------------------------------------------------------
# 10) RUN THE PSO OPTIMIZATION
# ---------------------------------------------------------------------
Global_results = []
(results, GlobalVars, options) = MOFE_PSO.RUN(options, GlobalVars)
Global_results.append(results)

# ---------------------------------------------------------------------
# 11) EXTRACT AND ORGANIZE RESULTS FROM THE PSO RUN
# ---------------------------------------------------------------------
Num_nomDom = 0
for res in Global_results:
    Num_nomDom += res.nonDom.shape[0]

XX = np.zeros([Num_nomDom, options.Nvar])
YY = np.zeros([Num_nomDom, options.Nobj])
idx = 0
for res in Global_results:
    for i in range(res.nonDom.shape[0]):
        XX[idx, :] = res.nonDom[i].x
        YY[idx, :] = res.nonDom[i].obj
        idx += 1

Posic = []
Fval = []
for res in Global_results:
    for i in range(options.swarmSize):
        for part in res.Particle.nonDom[i]:
            Posic.append(part.x)
            Fval.append(part.obj)
Posic = np.array(Posic)
Fval = np.array(Fval)

Num_FisPar = 0
for res in Global_results:
    Num_FisPar += res.AllFisPart.shape[0]

AllPartXX = np.zeros([Num_FisPar, options.Nvar])
AllPartYY = np.zeros([Num_FisPar, options.Nobj])
jj = 0
for res in Global_results:
    for i in range(res.AllFisPart.shape[0]):
        AllPartXX[jj, :] = res.AllFisPart[i].x
        AllPartYY[jj, :] = res.AllFisPart[i].obj
        jj += 1

# ---------------------------------------------------------------------
# 12) DENORMALIZE RESULTS & PLOT THE NON-DOMINATED SOLUTIONS
# ---------------------------------------------------------------------
YY_desnorm_1 = des_normalize(Output_global[:, 0:1], -YY[:, 0:1])
YY_desnorm_2 = des_normalize(Output_global[:, 1:2], -YY[:, 1:2])

Fval_desnorm_1 = des_normalize(Output_global[:, 0:1], -Fval[:, 0:1])
Fval_desnorm_2 = des_normalize(Output_global[:, 1:2], -Fval[:, 1:2])

AllPartYY_desnorm_1 = des_normalize(Output_global[:, 0:1], -AllPartYY[:, 0:1])
AllPartYY_desnorm_2 = des_normalize(Output_global[:, 1:2], -AllPartYY[:, 1:2])

XX_desnorm      = des_normalize(Input_global, XX)
Posic_desnorm   = des_normalize(Input_global, Posic)
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
# 13) SAVE THE RESULTS TO A MAT-FILE
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

savemat('PSO_BNN_100p_100i_purity_recovery_MC.mat', mydict)

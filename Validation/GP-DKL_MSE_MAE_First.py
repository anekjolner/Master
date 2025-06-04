#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:32:49 2025
@author: anekristinekjolner
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mat73
import torch
import torch.nn as nn
from functools import partial
import gpytorch

# ---------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------
Input_global = loadmat('../Data/Input_global.mat')['Input_global']
Output_global = loadmat('../Data/Output_global.mat')['Output_global']

# Load training, validation, and test data
Input_Train = mat73.loadmat('../Data/Input_train.mat')['Input_train']
Output_Train = mat73.loadmat('../Data/Output_train.mat')['Output_train']
Input_Valid = mat73.loadmat('../Data/Input_valid.mat')['Input_valid']
Output_Valid = mat73.loadmat('../Data/Output_valid.mat')['Output_valid']
Input_Test  = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']

# ---------------------------------------------------------------------
# DKL Model Definitions
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Load Trained Models
# ---------------------------------------------------------------------
def load_model(model_name, train_x, train_y):
    checkpoint = torch.load(model_name)
    config = checkpoint['config']
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(1e-3)
    feature_extractor = FeatureExtractor(
        input_dim=config['input_dim'],
        hidden_units=config['hidden_units'],
        dropout_prob=config.get('dropout_prob', 0.0),
        activation=config['activation']
    )
    model = DKL_GPModel(train_x, train_y, likelihood, feature_extractor)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    likelihood.eval()
    return model, likelihood

train_x = torch.tensor(Input_Train, dtype=torch.float32)
train_y_recovery = torch.tensor(Output_Train[:, 1], dtype=torch.float32)
train_y_purity = torch.tensor(Output_Train[:, 0], dtype=torch.float32)

model_recovery, likelihood_recovery = load_model('../Models/model_GNN_recovery.pth', train_x, train_y_recovery)
model_purity,   likelihood_purity   = load_model('../Models/model_GNN_purity.pth', train_x, train_y_purity)

# ---------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------
def predict_dkl(model, likelihood, input_data):
    model.eval()
    likelihood.eval()
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(input_tensor))
        return preds.mean.numpy().squeeze()

DKL_pred_purity = predict_dkl(model_purity, likelihood_purity, Input_Test)
DKL_pred_recovery = predict_dkl(model_recovery, likelihood_recovery, Input_Test)

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
# Load ground truth from MATLAB
YY_matlab = loadmat('Results_GNN_first_principle_model.mat')
xx = np.array(YY_matlab['x'])
yy = np.array(YY_matlab['y'])

matlab_model_purity = yy[:, 0]
matlab_model_recovery = yy[:, 1]

def adjust_prediction(pred, expected_length):
    pred = np.array(pred).flatten()
    pred_shape = pred.shape[0]
    if pred_shape > expected_length:
        pred = pred[:expected_length]
    elif pred_shape < expected_length:
        print(f"Warning: Predicted data is shorter than expected. Padding with zeros.")
        pred = np.pad(pred, (0, expected_length - pred_shape), 'constant')
    return pred

expected_length = len(matlab_model_purity)
DKL_pred_purity = adjust_prediction(DKL_pred_purity, expected_length)
DKL_pred_recovery = adjust_prediction(DKL_pred_recovery, expected_length)

# Filter out bad targets (e.g., corrupted 100000 values)
valid_mask = (matlab_model_purity <= 1.0) & (matlab_model_recovery <= 1.0)
matlab_model_purity = matlab_model_purity[valid_mask]
matlab_model_recovery = matlab_model_recovery[valid_mask]
DKL_pred_purity = DKL_pred_purity[valid_mask]
DKL_pred_recovery = DKL_pred_recovery[valid_mask]

# Compute errors
error_purity = np.abs(matlab_model_purity - DKL_pred_purity)
error_recovery = np.abs(matlab_model_recovery - DKL_pred_recovery)
squared_error_purity = error_purity**2
squared_error_recovery = error_recovery**2

mae_vector = [np.mean(error_purity), np.mean(error_recovery)]
mse_vector = [np.mean(squared_error_purity), np.mean(squared_error_recovery)]

print("MAE Vector:", mae_vector)
print("MSE Vector:", mse_vector)

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
import seaborn as sns
colors = sns.color_palette("Set1", 9)  # 2 variables: purity & recovery
color_purity = colors[4]   # orange
color_recovery = colors[3] # purple

# Plot MAE histograms
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.hist(error_purity, bins=30, edgecolor='black', alpha=0.5, color=color_purity)
plt.title('Histogram of Purity Point-wise MAE')
plt.xlabel('Error (MAE)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(error_recovery, bins=30, edgecolor='black', alpha=0.6, color=color_recovery)
plt.title('Histogram of Recovery Point-wise MAE')
plt.xlabel('Error (MAE)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Plot MSE histograms
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.hist(squared_error_purity, bins=30, edgecolor='black', alpha=0.5, color=color_purity)
plt.title('Histogram of Purity Point-wise MSE')
plt.xlabel('Error (MSE)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(squared_error_recovery, bins=30, edgecolor='black', alpha=0.6, color=color_recovery)
plt.title('Histogram of Recovery Point-wise MSE')
plt.xlabel('Error (MSE)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

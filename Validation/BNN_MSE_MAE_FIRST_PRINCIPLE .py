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

# Load data
Input_Test = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']
Input_Train = mat73.loadmat('../Data/Input_train.mat')
Output_Train = mat73.loadmat('../Data/Output_train.mat')

# Ensure data compatibility
Input_Test = np.array(Input_Test)
Output_Test = np.array(Output_Test)

# Define BNN model
class BNNModel(nn.Module):
    def __init__(self, input_dim=5, hidden_units=[300, 250, 200, 150, 100], output_dim=1, activation='relu'):
        super(BNNModel, self).__init__()
        layers = []
        previous_units = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(previous_units, units))
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            previous_units = units
        layers.append(nn.Linear(previous_units, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Load model function
def load_model(model_name):
    model = BNNModel(input_dim=5, hidden_units=[300, 250, 200, 150, 100], output_dim=1, activation='relu')
    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model

# Load models
model_recovery = load_model('../Models/model_BNN_MC_Recovery.pth')
model_purity   = load_model('../Models/model_BNN_MC_Purity.pth')

# Prediction function
def make_predictions(x, model):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return model(x_tensor).numpy().squeeze()

BNN_pred_purity = make_predictions(Input_Test, model_purity)
BNN_pred_recovery = make_predictions(Input_Test, model_recovery)

# Load MATLAB results
YY_matlab = loadmat('Results_BNN_first_principle_model.mat')
xx = np.array(YY_matlab['x'])
yy = np.array(YY_matlab['y'])

matlab_model_purity = yy[:, 0]
matlab_model_recovery = yy[:, 1]

# Trim predictions to match expected length
def adjust_prediction(pred, expected_length):
    pred = np.array(pred).flatten()
    pred_shape = pred.shape[0]
    if pred_shape > expected_length:
        pred = pred[:expected_length]
    elif pred_shape < expected_length:
        print(f"Warning: Predicted data is shorter than expected. Padding with zeros.")
        pred = np.pad(pred, (0, expected_length - pred_shape), 'constant')
    return pred

# Apply adjustment
expected_length = len(matlab_model_purity)
BNN_pred_purity = adjust_prediction(BNN_pred_purity, expected_length)
BNN_pred_recovery = adjust_prediction(BNN_pred_recovery, expected_length)

# ---- Filter out corrupted points (e.g., > 1.0) ----
valid_mask = (matlab_model_purity <= 1.0) & (matlab_model_recovery <= 1.0)
matlab_model_purity = matlab_model_purity[valid_mask]
matlab_model_recovery = matlab_model_recovery[valid_mask]
BNN_pred_purity = BNN_pred_purity[valid_mask]
BNN_pred_recovery = BNN_pred_recovery[valid_mask]

# Error calculations
error_purity = np.abs(matlab_model_purity - BNN_pred_purity)
error_recovery = np.abs(matlab_model_recovery - BNN_pred_recovery)
squared_error_purity = (matlab_model_purity - BNN_pred_purity)**2
squared_error_recovery = (matlab_model_recovery - BNN_pred_recovery)**2

# Summary stats
mae_vector = [np.mean(error_purity), np.mean(error_recovery)]
mse_vector = [np.mean(squared_error_purity), np.mean(squared_error_recovery)]

print("MAE Vector:", mae_vector)
print("MSE Vector:", mse_vector)

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parity plots and error metrics using PySR symbolic models
"""

from pysr import PySRRegressor
from scipy.io import loadmat
import mat73
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------
# Load training and test datasets
# -------------------------------
Input_Test = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']
Input_Train = mat73.loadmat('../Data/Input_train.mat')
Output_Train = mat73.loadmat('../Data/Output_train.mat')

# Extract from dicts if necessary
if isinstance(Input_Train, dict):
    Input_Train = Input_Train['Input_train']
if isinstance(Output_Train, dict):
    Output_Train = Output_Train['Output_train']

# Ensure arrays
Input_Train = np.array(Input_Train)
Output_Train = np.array(Output_Train)
Input_Test = np.array(Input_Test)
Output_Test = np.array(Output_Test)

# -------------------------------
# Initialize PySR models
# -------------------------------
symbolic_model_purity = PySRRegressor(
    niterations=500,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["sin","cos", "exp", "inv(x) = 1/x", "p1(x) = x^2", "p2(x) = x^3", "p3(x) = x^4"],
    extra_sympy_mappings={"inv": lambda x: 1 / x, "p1": lambda x: x**2, "p2": lambda x: x**3, "p3": lambda x: x**4},
    loss="loss(prediction, target) = (abs(prediction - target))"
)

symbolic_model_recovery = PySRRegressor(
    niterations=500,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["cos", "sin", "exp", "inv(x) = 1/x", "p1(x) = x^2", "p2(x) = x^3", "p3(x) = x^4"],
    extra_sympy_mappings={"inv": lambda x: 1 / x, "p1": lambda x: x**2, "p2": lambda x: x**3, "p3": lambda x: x**4},
    loss="loss(prediction, target) = (abs(prediction - target))"
)

# -------------------------------
# Train symbolic models
# -------------------------------
symbolic_model_purity.fit(Input_Train, Output_Train[:, 0])     # Purity
symbolic_model_recovery.fit(Input_Train, Output_Train[:, 1])   # Recovery

# -------------------------------
# Predict on test data
# -------------------------------
y_pred_purity = symbolic_model_purity.predict(Input_Test)
y_pred_recovery = symbolic_model_recovery.predict(Input_Test)

y_true_purity = Output_Test[:, 0]
y_true_recovery = Output_Test[:, 1]

# -------------------------------
# Evaluate model performance
# -------------------------------
mse_purity = mean_squared_error(y_true_purity, y_pred_purity)
mae_purity = mean_absolute_error(y_true_purity, y_pred_purity)

mse_recovery = mean_squared_error(y_true_recovery, y_pred_recovery)
mae_recovery = mean_absolute_error(y_true_recovery, y_pred_recovery)

print("=== PySR Symbolic Model Evaluation ===")
print(f"Purity   - MAE: {mae_purity:.4f}, MSE: {mse_purity:.4f}")
print(f"Recovery - MAE: {mae_recovery:.4f}, MSE: {mse_recovery:.4f}")

# -------------------------------
# Plot: Parity Plots
# -------------------------------
sns.set_theme(style="ticks")
colors = sns.color_palette("Set1", 9)
color_purity = colors[4]     # Orange
color_recovery = colors[3]   # Purple

# --- Purity Plot ---
plt.figure(figsize=(6, 6))
plt.scatter(y_true_purity, y_pred_purity, s=25, alpha=0.6, color=color_purity, label='Predicted')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Ideal (y = x)')
plt.xlabel('Test Data', fontsize=14)
plt.ylabel('Predicted Purity', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# --- Recovery Plot ---
plt.figure(figsize=(6, 6))
plt.scatter(y_true_recovery, y_pred_recovery, s=25, alpha=0.6, color=color_recovery, label='Predicted')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Ideal (y = x)')
plt.xlabel('Test Data', fontsize=14)
plt.ylabel('Predicted Recovery', fontsize=14)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

print(symbolic_model_purity.latex())
print(symbolic_model_recovery.latex())


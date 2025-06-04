#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated training script with improved kernel stability and regularization.
@author: anekristinekjolner
"""

# Uncomment the next line to use double precision, which might help with numerical stability
# torch.set_default_dtype(torch.float64)

import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
import mat73
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------------------
# 1) DATA LOADING
# -------------------------------------------------------------------------------------
Input_Train = mat73.loadmat('../Data/Input_train.mat')['Input_train']
Output_Train = mat73.loadmat('../Data/Output_train.mat')['Output_train']
Input_Valid = mat73.loadmat('../Data/Input_valid.mat')['Input_valid']
Output_Valid = mat73.loadmat('../Data/Output_valid.mat')['Output_valid']
Input_Test  = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']

train_x = torch.tensor(Input_Train, dtype=torch.float32)
# We predict only the second column (index 1) as during training
train_y = torch.tensor(Output_Train[:, 1], dtype=torch.float32)
valid_x = torch.tensor(Input_Valid, dtype=torch.float32)
valid_y = torch.tensor(Output_Valid[:, 1], dtype=torch.float32)
test_x  = torch.tensor(Input_Test,  dtype=torch.float32)
test_y  = torch.tensor(Output_Test[:, 1], dtype=torch.float32)

# -------------------------------------------------------------------------------------
# 2) CUSTOMIZABLE FEATURE EXTRACTOR (NEURAL NETWORK) FOR DKL
# -------------------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """
    A configurable feed-forward network that outputs latent features.
    Uses 4 hidden layers with units [512, 256, 128, 64] and adds dropout after each activation.
    """
    def __init__(self, input_dim=5, hidden_units=[512, 256, 128, 64],
                 dropout_prob=0.1, activation='relu'):
        super().__init__()
        act_functions = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
        }
        chosen_act = act_functions[activation.lower()]
        layers = []
        current_dim = input_dim
        for hdim in hidden_units:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(chosen_act)
            layers.append(nn.Dropout(dropout_prob))   # Dropout for regularization
            current_dim = hdim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------------------------------
# 3) DKL MODEL DEFINITION: EXACT GAUSSIAN PROCESS
# -------------------------------------------------------------------------------------
class DKL_GPModel(gpytorch.models.ExactGP):
    """
    Combines the neural-network-based feature extractor with a standard ExactGP.
    Here we use a Matern kernel (nu=2.5) for improved robustness.
    """
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ZeroMean()
        # Use a Matern kernel, which can be more stable than RBF in some cases.
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        # Constrain the lengthscale to avoid numerical issues.
        self.covar_module.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.Interval(1e-2, 1e2)
        )

    def forward(self, x):
        proj_x = self.feature_extractor(x)
        mean_x = self.mean_module(proj_x)
        covar_x = self.covar_module(proj_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# -------------------------------------------------------------------------------------
# 4) MODEL & LIKELIHOOD INSTANTIATION
# -------------------------------------------------------------------------------------
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# Set the likelihood noise manually for stability.
likelihood.noise = torch.tensor(1e-3)
# (Removed constraint registration since it was causing errors.)

feature_extractor = FeatureExtractor(
    input_dim= 5,
    hidden_units=[512, 256, 128, 64],
    dropout_prob=0.1,
    activation='relu'
)

model = DKL_GPModel(train_x, train_y, likelihood, feature_extractor)

# -------------------------------------------------------------------------------------
# 5) TRAINING SETUP (FULL-BATCH)
# -------------------------------------------------------------------------------------
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Use weight decay for the feature extractor for additional regularization.
optimizer = optim.Adam([
    {'params': model.feature_extractor.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': model.covar_module.parameters(),        'lr': 1e-2},
    {'params': model.mean_module.parameters(),         'lr': 1e-2},
    {'params': model.likelihood.parameters(),          'lr': 1e-2},
])

def train_one_epoch():
    """Perform one epoch of full-batch training."""
    model.train()
    likelihood.train()
    optimizer.zero_grad()
    output_dist = model(train_x)
    loss = -mll(output_dist, train_y)  # Negative log marginal likelihood
    loss.backward()
    optimizer.step()
    return loss.item()


def validation_loss():
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        with gpytorch.settings.cholesky_jitter(1.0), gpytorch.settings.max_cholesky_size(0):
            pred_dist = model(valid_x)
            val_loss = -mll(pred_dist, valid_y).item()
    return val_loss
# -------------------------------------------------------------------------------------
# 6) TRAIN LOOP WITH EARLY STOPPING & CHECKPOINT SAVING WHEN VALIDATION IMPROVES
# -------------------------------------------------------------------------------------
best_val = float('inf')
patience = 40
no_improve = 0
max_epochs = 500  # Increased epochs for deeper network training

for epoch in range(max_epochs):
    train_l = train_one_epoch()
    val_l = validation_loss()

    print(f"Epoch [{epoch+1}/{max_epochs}] - Train (neg. MLL): {train_l:.6f} | Val (neg. MLL): {val_l:.6f}")

    # Save checkpoint only when validation improves.
    if val_l < best_val:
        best_val = val_l
        no_improve = 0
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_l,
            'val_loss': val_l
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}!")
    else:
        no_improve += 1

    if no_improve >= patience:
        print("Early stopping!")
        break

# -------------------------------------------------------------------------------------
# 7) TEST EVALUATION & PLOT
# -------------------------------------------------------------------------------------
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred_dist = model(test_x)
    pred_mean = pred_dist.mean
    pred_std  = pred_dist.stddev

y_true = test_y.numpy()
y_hat  = pred_mean.numpy()

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_hat, s=25, alpha=0.8, label='Predicted vs True')
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Ideal (y = x)')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('True', fontsize=14)
plt.ylabel('Predicted', fontsize=14)
plt.title('GNN Predictions vs. True', fontsize=16)
plt.legend()
plt.show()

mse_test = ((y_hat - y_true)**2).mean()
mae_test = np.abs(y_hat - y_true).mean()
print("Test MSE:", mse_test)
print("Test MAE:", mae_test)

# -------------------------------------------------------------------------------------
# 8) SAVE FINAL MODEL CONFIGURATION & STATE
# -------------------------------------------------------------------------------------
config = {
    'input_dim': 5,
    'hidden_units': [512, 256, 128, 64],
    'activation': 'relu',
    'dropout_prob': 0.1
}

torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, 'model_GNN_recovery.pth')



import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mat73
import tensorflow as tf
import seaborn as sns

# Load data
Input_Test = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']
Input_Train = mat73.loadmat('../Data/Input_train.mat')
Output_Train = mat73.loadmat('../Data/Output_train.mat')

# Ensure data compatibility
Input_Test = np.array(Input_Test)
Output_Test = np.array(Output_Test)

# Load MATLAB results for comparison
YY_matlab = loadmat('Results_reg_first_principle_model.mat')  # change for test data
xx = np.array(YY_matlab['x'])
yy = np.array(YY_matlab['y'])

matlab_model_purity = yy[:, 0]
matlab_model_recovery = yy[:, 1]

# Define regression functions for purity and recovery
def purityFunc(x):
      
    # Purity-beregning
    purity_denominator = (x[2] / 0.421) + 0.605
    model_purity = np.sin((x[1] / purity_denominator) + 0.0963)

    return model_purity

def recoveryFunc(x):
    # Recovery-beregning
    inner_fraction = np.sin(x[1]) / (0.124 - np.exp((9 * x[2]) / x[1]))
    model_recovery = (np.cos(inner_fraction) ** 4) / 1.10

    return model_recovery

# Apply the functions to all rows of xx
reg_pred_purity = np.array([purityFunc(x) for x in xx])
reg_pred_recovery = np.array([recoveryFunc(x) for x in xx])

# Lag en maske som kun tar med verdier mellom 0 og 1
valid_mask = (reg_pred_purity >= 0) & (reg_pred_purity <= 1.08)

# Bruk masken til å filtrere
filtered_purity = reg_pred_purity[valid_mask]

reg_pred_purity = reg_pred_purity[valid_mask]
reg_pred_recovery = reg_pred_recovery[valid_mask]
matlab_model_purity = matlab_model_purity[valid_mask]
matlab_model_recovery = matlab_model_recovery[valid_mask]

# Calculate point-wise errors (MAE for each point)
error_purity = np.abs(matlab_model_purity - reg_pred_purity)
error_recovery = np.abs(matlab_model_recovery - reg_pred_recovery)

# Calculate squared errors for MSE (point-wise)
squared_error_purity = (matlab_model_purity - reg_pred_purity) ** 2
squared_error_recovery = (matlab_model_recovery - reg_pred_recovery) ** 2

# Store MAE and MSE in vectors
mae_vector = [np.mean(error_purity), np.mean(error_recovery)]
mse_vector = [np.mean(squared_error_purity), np.mean(squared_error_recovery)]

# Print the vectors for MAE and MSE
print("MAE Vector:", mae_vector)
print("MSE Vector:", mse_vector)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Visualization settings
import seaborn as sns
colors = sns.color_palette("Set1", 9)
color_purity = colors[4]   # orange
color_recovery = colors[3] # purple

# === MAE Histogrammer ===
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
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))  # Begrens x-ticks

plt.tight_layout()
plt.show()


# === MSE Histogrammer ===
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
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))  # Begrens x-ticks

plt.tight_layout()
plt.show()


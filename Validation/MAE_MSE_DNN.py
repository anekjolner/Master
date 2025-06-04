import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mat73
import tensorflow as tf

# Load data
Input_Test = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']
Input_Train = mat73.loadmat('../Data/Input_train.mat')
Output_Train = mat73.loadmat('../Data/Output_train.mat')

# Load the NN models
model_purity = tf.keras.models.load_model('../Models/model_Purity.h5')
model_recovery = tf.keras.models.load_model('../Models/model_Recovery.h5')

# Ensure data compatibility
Input_Test = np.array(Input_Test)
Output_Test = np.array(Output_Test)

# Load MATLAB results for comparison
YY_matlab = loadmat('Results.mat')
yy = np.array(YY_matlab['y'])

matlab_model_purity = yy[:, 0]  # Actual values from MATLAB model for purity
matlab_model_recovery = yy[:, 1]  # Actual values from MATLAB model for recovery

# NN Predictions
NN_pred_purity = model_purity.predict(Input_Test)
NN_pred_recovery = model_recovery.predict(Input_Test)

# Check the shapes of the arrays and make sure they match the length of the MATLAB data
print(f"Shape of matlab_model_purity: {matlab_model_purity.shape}")
print(f"Shape of NN_pred_purity: {NN_pred_purity.shape}")

# Ensure that both arrays have the same length by trimming or adjusting
expected_length = len(matlab_model_purity)

# Trim or pad NN predictions to match the expected length of the MATLAB data
def adjust_prediction(pred, expected_length):
    pred_shape = pred.shape[0]
    if pred_shape > expected_length:
        # If the predicted values have more data points than the expected, slice it
        pred = pred[:expected_length].flatten()  # Flatten to 1D if needed
    elif pred_shape < expected_length:
        # If the predicted values are shorter, pad them with zeros
        print(f"Warning: Predicted data is shorter than expected. Padding with zeros.")
        pred = np.pad(pred, ((0, expected_length - pred_shape), (0, 0)), 'constant').flatten()
    return pred

# Apply adjustment function to all predictions
NN_pred_purity = adjust_prediction(NN_pred_purity, expected_length)
NN_pred_recovery = adjust_prediction(NN_pred_recovery, expected_length)

# Calculate point-wise errors (MAE for each point)
error_purity = np.abs(matlab_model_purity - NN_pred_purity)  # MAE for each point (Purity)
error_recovery = np.abs(matlab_model_recovery[:expected_length] - NN_pred_recovery)  # MAE for each point (Recovery)

# Calculate squared errors for MSE (point-wise)
squared_error_purity = (matlab_model_purity - NN_pred_purity)**2  # MSE for each point (Purity)
squared_error_recovery = (matlab_model_recovery[:expected_length] - NN_pred_recovery)**2  # MSE for each point (Recovery)

# Store MAE and MSE in vectors
mae_vector = [np.mean(error_purity), np.mean(error_recovery)]
mse_vector = [np.mean(squared_error_purity), np.mean(squared_error_recovery)]

# Print the vectors for MAE and MSE
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

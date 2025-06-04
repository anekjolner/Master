# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:20:33 2023
@author: Carine Rebello
"""

# --- Imports ---
from scipy.io import loadmat, savemat
import mat73
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import os
import IPython

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# --- Load datasets from .mat files ---
# These datasets are used for training, validation, and testing a recovery prediction model

# Global dataset (used for normalization or reference)
Input_global = loadmat('../Data/Input_global.mat')['Input_global']
Output_global = loadmat('../Data/Output_global.mat')['Output_global']

# Training data
Input_Train = mat73.loadmat('../Data/Input_train.mat')['Input_train']
Output_Train = mat73.loadmat('../Data/Output_train.mat')['Output_train']

# Validation data
Input_Valid = mat73.loadmat('../Data/Input_valid.mat')['Input_valid']
Output_Valid = mat73.loadmat('../Data/Output_valid.mat')['Output_valid']

# Test data
Input_Test = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']

# (Optional) Preprocessing placeholder if normalization was needed
Input_Train_norm = Input_Train
Input_Valid_norm = Input_Valid
Input_Test_norm = Input_Test
Output_Train_norm = Output_Train
Output_Valid_norm = Output_Valid
Output_Test_norm = Output_Test

# --- Define model architecture for Keras Tuner hyperparameter optimization ---
def hyp_model(hp):
    model = Sequential()
    # Add variable number of dense layers
    for i in range(hp.Int('num_layers', 1, 6)):
        hp_units = hp.Int(f'units_{i}', min_value=50, max_value=300, step=10)
        model.add(Dense(units=hp_units,
                        activation=hp.Choice('activation_dense', values=['relu', 'tanh'])))

    model.add(Dense(units=1))  # Output layer

    # Compile the model with a tunable learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='mse',
                  metrics=['mae'])
    return model

# --- Initialize Keras Tuner with Hyperband search strategy ---
tuner = kt.Hyperband(hyp_model,
                     objective='val_mae',   # Minimize validation MAE
                     max_epochs=100,
                     factor=3,
                     directory='PSA',
                     project_name='Recovery')#change when trained for other metrics

# Early stopping to prevent overfitting
earlystop_hp = EarlyStopping(monitor='val_mae', patience=70, restore_best_weights=True)

# Optional: Clear output after training
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)

# --- Hyperparameter search ---
# Trains models using different hyperparameter combinations
tuner.search(x=Input_Train_norm,
             y=Output_Train_norm[:, 1],  # Only second column (Recovery) change everything to 0 for purity, 1 for recovery, 2 for productivity and 3 for energy
             validation_data=(Input_Valid_norm, Output_Valid_norm[:, 1]),
             batch_size=8,
             shuffle=True,
             callbacks=[earlystop_hp])

# --- Train final model with best hyperparameters ---
best_hps = tuner.get_best_hyperparameters()[0]
model_Rho_Ar = tuner.hypermodel.build(best_hps)

model_Rho_Ar.fit(x=Input_Train_norm,
                 y=Output_Train_norm[:, 1],
                 validation_data=(Input_Valid_norm, Output_Valid_norm[:, 1]),
                 batch_size=8,
                 shuffle=True,
                 epochs=200,
                 callbacks=[earlystop_hp])

# Evaluate model on test data
model_Rho_Ar.evaluate(Input_Test_norm, Output_Test_norm[:, 1])

# Predict recovery on test set
rho_Ar_norm = model_Rho_Ar.predict(Input_Test_norm)

# --- Parity Plot ---
# Compare true test values vs predicted values visually
x = Output_Test_norm[:, 1:2]  # True values
y = rho_Ar_norm               # Predicted values

sns.set_theme(style="ticks")
plt.figure(figsize=(8, 6))

# Scatter plot with regression line and reference diagonal
sns.regplot(x=x[:, 0], y=y[:, 0],
            scatter_kws={'color': '#1E90FF', 's': 30},
            line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2})

# Add perfect fit reference line
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=4)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('True Recovery (Test Set)', fontsize=20)
plt.ylabel('Predicted Recovery', fontsize=20)
plt.title('Parity Plot - Recovery', fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Save the trained model ---
model_Rho_Ar.save('model_Recovery.h5')#change name if trained for other metrics

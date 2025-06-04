#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:17:38 2025
@author: anekristinekjolner
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mat73
import datetime
import pandas as pd
import numpy.matlib
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional, Input, Masking
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.losses import Huber
from keras.models import clone_model
import seaborn as sns

# --- Load data ---

Input_Test = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']
Input_Train = mat73.loadmat('../Data/Input_train.mat')
Output_Train = mat73.loadmat('../Data/Output_train.mat')

Input_Test = np.array(Input_Test)
Output_Test = np.array(Output_Test)

# Load DNN model
Input_global = loadmat('../Optimization/PSO_NN_100p_100i_purity_recovery.mat')
XX_DNN = np.array(Input_global['XX_redes'])

# Load MATLAB-based model
YY_matlab = loadmat('Results.mat')
XX_matlab = np.array(YY_matlab['x'])

# Load GNN
Input_global_GNN = loadmat('../Optimization/PSO_GNN_100p_100i_purity_recovery.mat')
XX_GNN = np.array(Input_global_GNN['XX_redes'])

# Load BNN
Input_global_BNN = loadmat('../Optimization/PSO_BNN_100p_100i_purity_recovery_MC.mat')
XX_BNN = np.array(Input_global_BNN['XX_redes'])

# Load regression
Input_global_reg = loadmat('../Optimization/PSO_reg_100p_100i_purity_recovery.mat')
XX_reg = np.array(Input_global_reg['XX_redes'])

# Load first-principles model
Optimization_data = mat73.loadmat('Optimization.mat')
input_pheno = np.array(Optimization_data['Optimization'])

# --- Prepare data for plotting ---

input_labels = [r'$t_{pres}$', r'$t_{depres}$', r'$t_{ads}$', r'$t_{LR}$', r'$t_{HR}$']

df_dnn = pd.DataFrame(XX_DNN, columns=input_labels)
df_dnn['Model'] = 'DNN'

df_bnn = pd.DataFrame(XX_BNN, columns=input_labels)
df_bnn['Model'] = 'BNN'

df_gnn = pd.DataFrame(XX_GNN, columns=input_labels)
df_gnn['Model'] = 'GP-DKL'

df_reg = pd.DataFrame(XX_reg, columns=input_labels)
df_reg['Model'] = 'SR'

df_matlab = pd.DataFrame(XX_matlab, columns=input_labels)
df_matlab['Model'] = 'First-principles model'

# Sample regression model for clarity
df_reg_sampled = df_reg.sample(n=1000, random_state=42)

# Combine all models
df_all_sampled = pd.concat([df_dnn, df_bnn, df_gnn, df_reg_sampled, df_matlab], ignore_index=True)

# --- Create pairplot ---

pairplot = sns.pairplot(df_all_sampled,
                        hue='Model',
                        corner=True,
                        palette='Set1',
                        plot_kws={'alpha': 0.6, 's': 10},
                        diag_kws={'common_norm': False},
                        height=2.5,
                        aspect=1.0)

# Move legend to upper right (outside)
pairplot._legend.set_bbox_to_anchor((0.85, 0.85))
pairplot._legend.set_loc("upper left")

# Style legend text
pairplot._legend.set_title("Model")
plt.setp(pairplot._legend.get_title(), fontsize=26, weight='bold')
plt.setp(pairplot._legend.get_texts(), fontsize=24)

# Make legend dots larger
for handle in pairplot._legend.legend_handles:
    handle.set_markersize(14)

# Set larger axis labels and tick sizes
for ax in pairplot.axes.flatten():
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontsize=30)
        ax.set_ylabel(ax.get_ylabel(), fontsize=30)
        ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.show()

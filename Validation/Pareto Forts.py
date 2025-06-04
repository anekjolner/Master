import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mat73

from scipy.io import loadmat
from scipy.io import savemat
import datetime
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
from plotly.subplots import make_subplots
#Importing Tensorflow and other modules
import tensorflow as tf
#Importing Libraries
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,SimpleRNN,GRU,TimeDistributed,Bidirectional,Input, Masking
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.losses import Huber
from keras.models import clone_model


# Load data
Input_Test = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']
Input_Train = mat73.loadmat('../Data/Input_train.mat')
Output_Train = mat73.loadmat('../Data/Output_train.mat')



# Ensure data compatibility
Input_Test = np.array(Input_Test)
Output_Test = np.array(Output_Test)


# Load data for the original model DNN
Input_global = loadmat('../Optimization/PSO_NN_100p_100i_purity_recovery.mat')
Input_YY = np.array(Input_global['AllPartYY_redes'])  # Scatter points (swarm plot)
YY = np.array(Input_global['YY_redes'])  # Pareto front
XX = np.array(Input_global['XX_redes']) #paste into the matlab


# Load data for the NN model
#Input_global_BNN = loadmat('PSO_NN_100p_100i_purity_recovery_BNN.mat')
Input_global_GNN = loadmat('../Optimization/PSO_GNN_100p_100i_purity_recovery.mat')
Input_YY_GNN = np.array(Input_global_GNN['AllPartYY_redes'])  # Scatter points
YY_GNN = np.array(Input_global_GNN['YY_redes'])  # Pareto front

#Load data for BNN
Input_global_BNN = loadmat('../Optimization/PSO_BNN_100p_100i_purity_recovery_MC.mat')
Input_YY_BNN = np.array(Input_global_BNN['AllPartYY_redes'])  # Scatter points
YY_BNN = np.array(Input_global_BNN['YY_redes'])  # Pareto front


#load data for regression
#Input_global_reg = loadmat('PSO_NN_100p_100i_purity_recovery_regression.mat',)
Input_global_reg = loadmat('../Optimization/PSO_reg_100p_100i_purity_recovery.mat',)
Input_YY_reg = np.array(Input_global_reg['AllPartYY_redes'])  # Scatter points
YY_reg = np.array(Input_global_reg['YY_redes'])  # Pareto front


# Load data for the phenomenological model
Optimization_data = mat73.loadmat('Optimization.mat')  # Ensure correct variable extraction
input_pheno = np.array(Optimization_data['Optimization'])  # Extract the correct variable

import seaborn as sns
palette = sns.color_palette('Set1', n_colors=9)

# Assign colors from Set1 palette
color_dnn = palette[0]
color_bnn = palette[1]
color_gnn = palette[2]
color_regression = palette[3]
color_first_principles = palette[4]

# Create the plot
plt.figure(figsize=(10, 8))

# DNN
plt.scatter(YY[:, 0], YY[:, 1], color=color_dnn, s=30, label='DNN Point Cloud', alpha=0.6)

# GP-DKL (GNN)
plt.scatter(YY_GNN[:, 0], YY_GNN[:, 1], color=color_gnn, s=30, label='GP-DKL Point Cloud', alpha=0.6)

# BNN
plt.scatter(YY_BNN[:, 0], YY_BNN[:, 1], color=color_bnn, s=30, label='BNN Point Cloud', alpha=0.6)

# Regression
plt.scatter(YY_reg[:, 0], YY_reg[:, 1], color=color_regression, s=30, label='Regression Point Cloud', alpha=0.6)

# First-principles model
plt.scatter(input_pheno[:, 0], input_pheno[:, 1], color=color_first_principles, s=30, label='First-Principles Model', alpha=0.8)

# Add labels, legend, and title
plt.xlabel('Purity', fontsize=18)
plt.ylabel('Recovery', fontsize=18)
#plt.title('Comparison of Pareto Fronts: DNN vs BNN vs Regression vs GP-DKL vs First-Principles Model', fontsize=18)

# Make axis tick labels bigger
plt.tick_params(axis='both', labelsize=16)

# Improved legend visibility
plt.legend(fontsize=14, markerscale=2)

# Grid and layout improvements
plt.grid(False)
plt.tight_layout()

# Show plot
plt.show()


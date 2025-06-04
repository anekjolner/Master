#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 10 09:42:03 2025

@author: anekristinekjolner
"""

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
Input_Test = mat73.loadmat('Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('Output_test.mat')['Output_test']
Input_Train = mat73.loadmat('Input_train.mat')
Output_Train = mat73.loadmat('Output_train.mat')


# Load data for the scatterplot
Input_global = loadmat('PSO_BNN_100p_100i_purity_recovery_MC.mat')
Input_YY = np.array(Input_global['AllPartYY_redes'])  # Scatter points (swarm plot)
Input_YY = Input_global['AllPartYY_redes']  # Points for the objective function (swarm plot)
YY = Input_global['YY_redes']  # Pareto front (solution for the objective variables)

# Extracting the first and second elements for plotting
purity = Input_YY[:, 0]  # First element (e.g., purity)
recovery = Input_YY[:, 1]  # Second element (e.g., recovery)
pareto_purity = YY[:, 0]  # First element of the Pareto front
pareto_recovery = YY[:, 1]  # Second element of the Pareto front

import matplotlib.pyplot as plt
import seaborn as sns

# Get colors from Seaborn Set1 palette
colors = sns.color_palette("Set1", 9)
color_cloud = colors[4]   # orange
color_pareto = colors[3]  # purple

# Create the main plot
plt.figure(figsize=(10, 6))

# Scatter plot for the full point cloud
plt.scatter(purity, recovery, color=color_cloud, s=10, label='Point Cloud')

# Plotting the Pareto front
plt.scatter(pareto_purity, pareto_recovery, color=color_pareto, s=20, label='Pareto Front')

# Labels and styling
plt.xlabel('Purity', fontsize=30)
plt.ylabel('Recovery', fontsize=30)
#plt.title('Pareto Front', fontsize=40)

plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=16, loc='best')  # larger legend

plt.grid(False)
plt.tight_layout()
plt.show()

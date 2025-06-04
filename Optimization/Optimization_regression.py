#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:45:36 2025

@author: anekristinekjolner
"""


from scipy.io import loadmat
from scipy.io import savemat
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
import mat73

from plotly.subplots import make_subplots
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from MOFE_PSO import MOFE_PSO



# Base de dados global
Input_global = loadmat('../Data/Input_global.mat')
Output_global = loadmat('../Data/Output_global.mat')


Input_global = Input_global['Input_global']
Output_global = Output_global['Output_global']




"""#Otimização"""

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
import os
import distutils
import random
import copy
#from OLFCN import OLFCN

#import matlab.engine
import time
import math
import tensorflow as tf
import copy
from keras.models import clone_model
from pymcmcstat.plotting.MCMCPlotting import Plot
import mat73
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import time
from keras.models import clone_model
from pymcmcstat.MCMC import MCMC


# # --- Normalization Utilities ---
def des_normalize(Mat_ref,Mat_dados):
    result = 0*Mat_dados.copy()
    for i in range(Mat_ref.shape[1]):
        max_value = Mat_ref[:,i].max()
        min_value = Mat_ref[:,i].min()
        result[:,i] = (Mat_dados[:,i]) * (max_value - min_value) + min_value
    return result

def normalize(Mat_ref,Mat_dados):
    result = 0*Mat_dados.copy()
    for i in range(Mat_ref.shape[1]):
        max_value = Mat_ref[:,i].max()
        min_value = Mat_ref[:,i].min()
        result[:,i] = (Mat_dados[:,i] - min_value) / (max_value - min_value)
    return result
    
 
# Constraints function
def ConsFunc(x):
  
    # Purity-beregning
    purity_denominator = (x[2] / 0.421) + 0.605
    model_purity = np.sin((x[1] / purity_denominator) + 0.0963)

    # Recovery-beregning
    inner_fraction = np.sin(x[1]) / (0.124 - np.exp((9 * x[2]) / x[1]))
    model_recovery = (np.cos(inner_fraction) ** 4) / 1.10

    x_p = np.reshape(model_purity,(1,1))
    x_r = np.reshape(model_recovery, (1,1))

    consFunc = np.array([[x_p-1],[x_r-1]])
    return consFunc
#END 


# Objective function
def ObjFun(x):
   
    # Purity-beregning
    purity_denominator = (x[2] / 0.421) + 0.605
    model_purity = np.sin((x[1] / purity_denominator) + 0.0963)

    # Recovery-beregning
    inner_fraction = np.sin(x[1]) / (0.124 - np.exp((9 * x[2]) / x[1]))
    model_recovery = (np.cos(inner_fraction) ** 4) / 1.10

    x_p = np.reshape(model_recovery,(1,1))
    x_e = np.reshape(model_purity, (1,1))

    
    objFun = np.array([[-x_e],[-x_p]]) #Maximise
    return objFun
  
    
options = type('',(),{})()
GlobalVars = type('',(),{})()

# Options of MOFEPSO
options.Nvar = int(5) # Number of decision variables
options.Nobj = int(2) # Number of objective functions
options.nCons = int(2) # Number of constraints
options.ObjFun = ObjFun
options.ConsFunc = ConsFunc

# Assign default values
options.swarmSize = int(100) # Number of particles in swarm
options.maxIter = int(100)    # Maximum iterations
options.useInputDecoder = False
options.transferVariables = False
options.velocityInitializationFactor = np.array([0.3]) #Variables for the algorythme
options.incrementFactor = np.array([0.3])
options.boundaryTolerance = np.array([0.01])
options.initialC0 = np.array([0.9])
options.finalC0 = np.array([0.4])
options.C1 = np.array([2])
options.C2 = np.array([2])
options.verbose = int(2)
GlobalVars.nConsEvals = np.array([0])
GlobalVars.nObjEvals = np.array([0])

Xmin=[0, 0, 0, 0, 0]
Xmax=[1, 1, 1, 1, 1];
Vmax=Xmax #max velocities
Xvmax = []
Xvmin = []
# N_malha = 2
# x = np.linspace(Xmin[0], Xmax[0], N_malha)
# y = np.linspace(Xmin[1], Xmax[1], N_malha)
# xx, yy = np.meshgrid(x, y)
#print(xx)
#print(yy)

Global_results = []


# for i in range(N_malha-1):
#   for j in range(N_malha-1):
options.lBound = np.array([0, 0, 0, 0, 0])   # Lower bound of input variables
options.uBound = np.array([1, 1, 1, 1, 1])   # Upper bound of input variables
    #print(options.lBound)
    #print(options.uBound)
    #print('--------')
    
(results, GlobalVars, options) = MOFE_PSO.RUN(options,GlobalVars)
Global_results.append(results)
    
# Separando os resultados em matrizes
Num_nomDom = 0
for i in range(len(Global_results)):
  Num_nomDom = Num_nomDom + Global_results[i].nonDom.shape[0]
print(Num_nomDom)
XX = np.zeros([Num_nomDom,options.Nvar])
YY = np.zeros([Num_nomDom,options.Nobj])
for j in range(len(Global_results)):
  for i in range(Global_results[j].nonDom.shape[0]):
      for k in range(options.Nvar):
          XX[i,k] = Global_results[j].nonDom[i].x[k]

# print(len(range(options.Nobj)))

for j in range(len(Global_results)):
    for i in range(Global_results[j].nonDom.shape[0]):
        for k in range(options.Nobj):
            YY[i, k] = Global_results[j].nonDom[i].obj[k]


# # Commented out IPython magic to ensure Python compatibility.

Posic = np.array([0,0,0,0,0]) #Because of the desicion variable
Fval = np.array([0,0]) # because of the objective function

Num_Fval = 0
for j in range(len(Global_results)):
    for i in range(options.swarmSize):
        Num_Fval = Num_Fval + len(Global_results[j].Particle.nonDom[i])
Posic = np.zeros([Num_Fval,options.Nvar])
jj = 0
for j in range(len(Global_results)):
  for i in range(options.swarmSize):
      for k in range(len(Global_results[j].Particle.nonDom[i])):
          Posic[jj,:] = Global_results[j].Particle.nonDom[i][k].x
          jj = jj+1
          

Fval = np.zeros([Num_Fval,options.Nobj])
jj = 0
for j in range(len(Global_results)):
  for i in range(options.swarmSize):
      for k in range(len(Global_results[j].Particle.nonDom[i])):
          for kk in range(options.Nobj):
              #print(Global_results[j].Particle.nonDom[i][k].obj[0][0])
              Fval[jj,kk] = Global_results[j].Particle.nonDom[i][k].obj[kk]
              #End For
          jj = jj+1
          
          
Num_FisPar = 0
for i in range(len(Global_results)):
  Num_FisPar = Num_FisPar + Global_results[i].AllFisPart.shape[0]


AllPartXX = np.zeros([Num_FisPar,options.Nvar])

jj=0
for j in range(len(Global_results)):
    for i in range(len(Global_results[j].AllFisPart)):
            AllPartXX[jj,:] = Global_results[j].AllFisPart[i].x
            jj=jj+1
            
AllPartYY = np.zeros([Num_FisPar, options.Nobj])  # Inicialize com as dimensões corretas
jj = 0
for j in range(len(Global_results)):
    for i in range(len(Global_results[j].AllFisPart)):
        for kk in range(options.Nobj):
            value = Global_results[j].AllFisPart[i].obj[kk]
            if isinstance(value, np.ndarray):  # Se for um array, extraia o escalar
                value = value.item()
            AllPartYY[jj, kk] = value
        jj += 1

        
##Plots
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

#ax = plt.axes(projection='2d')

YY_desnorm_1=des_normalize(Output_global[:,0:1],-YY[:,0:1])
YY_desnorm_2=des_normalize(Output_global[:,1:2],-YY[:,1:2])

Fval_desnorm_1=des_normalize(Output_global[:,0:1],-Fval[:,0:1])
Fval_desnorm_2=des_normalize(Output_global[:,1:2],-Fval[:,1:2])
AllPartYY_desnorm_1=des_normalize(Output_global[:,0:1],-AllPartYY[:,0:1])
AllPartYY_desnorm_2=des_normalize(Output_global[:,1:2],-AllPartYY[:,1:2])


XX_desnorm =des_normalize(Input_global,XX)
Posic_desnorm =des_normalize(Input_global,Posic)
AllPartXX_desnorm =des_normalize(Input_global,AllPartXX)

plt.plot(YY_desnorm_1,YY_desnorm_2,'o')

YY = np.hstack((YY_desnorm_1,YY_desnorm_2))
Fval = np.hstack((Fval_desnorm_1,Fval_desnorm_2))
AllPartYY = np.hstack((AllPartYY_desnorm_1,AllPartYY_desnorm_2))


mydict = {}
mydict['Tempo_redes'] = results.PartTimer
mydict['XX_redes'] = XX_desnorm
mydict['Fval_redes'] = Fval
mydict['Posic_redes'] = Posic_desnorm

mydict['AllPartYY_redes'] = AllPartYY
mydict['AllPartXX_redes'] = AllPartXX_desnorm
mydict['YY_redes'] = YY

Save = True
if Save:
      savemat('PSO_reg_100p_100i_purity_recovery.mat',mydict)
      
      
      
      
      
      
      
      
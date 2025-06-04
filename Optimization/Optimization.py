"""
Created  

@author: Carine Rebello
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
from tensorflow.keras.models import load_model

# Load full/global dataset (used for general reference or unsupervised training)
Input_global = loadmat('../Data/Input_global.mat')
Output_global = loadmat('../Data/Output_global.mat')

# Load training dataset (used to train the model)
Input_Train = mat73.loadmat('../Data/Input_train.mat')
Output_Train = mat73.loadmat('../Data/Output_train.mat')

# Load validation dataset (used to tune hyperparameters and prevent overfitting)
Input_Valid = mat73.loadmat('../Data/Input_valid.mat')
Output_Valid = mat73.loadmat('../Data/Output_valid.mat')

# Load testing dataset (used to evaluate final model performance)
Input_Test = mat73.loadmat('../Data/Input_test.mat')
Output_Test = mat73.loadmat('../Data/Output_test.mat')


Input_Train = Input_Train['Input_train']
Output_Train = Output_Train['Output_train']
Input_Valid = Input_Valid['Input_valid']
Output_Valid = Output_Valid['Output_valid']
Input_Test = Input_Test['Input_test']
Output_Test = Output_Test['Output_test']
Input_global = Input_global['Input_global']
Output_global = Output_global['Output_global']

# --- Normalization Utilities ---

def des_normalize(Mat_ref, Mat_dados):
    """Denormalizes data based on reference matrix min/max."""
    result = np.zeros_like(Mat_dados)
    for i in range(Mat_ref.shape[1]):
        max_value = Mat_ref[:, i].max()
        min_value = Mat_ref[:, i].min()
        result[:, i] = Mat_dados[:, i] * (max_value - min_value) + min_value
    return result

def normalize(Mat_ref, Mat_dados):
    """Normalizes data between 0 and 1 based on reference matrix min/max."""
    result = np.zeros_like(Mat_dados)
    for i in range(Mat_ref.shape[1]):
        max_value = Mat_ref[:, i].max()
        min_value = Mat_ref[:, i].min()
        result[:, i] = (Mat_dados[:, i] - min_value) / (max_value - min_value)
    return result

# --- Load pre-trained models ---
model_purity = load_model('../Models/model_Purity.h5')
model_recovery = load_model('../Models/model_Recovery.h5')


# Constraints function
def ConsFunc(x):
    x_p = np.reshape(x,(1,5))
    purity = model_purity.predict(x_p,verbose = 0)
    recovery = model_recovery.predict(x_p, verbose = 0)
    

    consFunc = np.array([[-purity],[-recovery]])
    return consFunc
#END 


# Objective function
def ObjFun(x):
    
    x_p = np.reshape(x,(1,5))
    purity = model_purity.predict(x_p,verbose = 0)
    recovery = model_recovery.predict(x_p, verbose = 0)

    
    objFun = np.array([[-purity],[-recovery]])
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
options.velocityInitializationFactor = np.array([0.3])
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
Vmax=Xmax # velocidade máxima das partículas em cada direção d
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
          YY[i,k] = Global_results[j].nonDom[i].obj[k]

# # Commented out IPython magic to ensure Python compatibility.

Posic = np.array([0,0,0,0,0])
Fval = np.array([0,0])

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
            
AllPartYY = np.zeros([Num_FisPar,options.Nobj])
jj=0
for j in range(len(Global_results)):
    for i in range(len(results.AllFisPart)):
        for kk in range(options.Nobj):
            AllPartYY[jj,kk] = Global_results[j].AllFisPart[i].obj[kk]
            #End For
        jj=jj+1
        
## Gráficos
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

#ax = plt.axes(projection='2d')

YY_desnorm_1 = des_normalize(Output_global[:, 0:1], -YY[:, 0:1])
YY_desnorm_2 = des_normalize(Output_global[:, 1:2], -YY[:, 1:2])

Fval_desnorm_1 = des_normalize(Output_global[:, 0:1], -Fval[:, 0:1])
Fval_desnorm_2 = des_normalize(Output_global[:, 1:2], -Fval[:, 1:2])

AllPartYY_desnorm_1 = des_normalize(Output_global[:, 0:1], -AllPartYY[:, 0:1])
AllPartYY_desnorm_2 = des_normalize(Output_global[:, 1:2], -AllPartYY[:, 1:2])

XX_desnorm     = des_normalize(Input_global, XX)
Posic_desnorm  = des_normalize(Input_global, Posic)
AllPartXX_desnorm = des_normalize(Input_global, AllPartXX)

# Fval_desnorm_1=des_normalize(Output_Global,Fval[:])

# AllPartYY_desnorm_1 = des_normalize(Output_Global,AllPartYY[:])

# Data for a three-dimensional line

# fig, ax=plt.subplots(figsize=(8, 6))
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#   label.set_fontsize(20)
# ax.scatter(AllPartYY_desnorm_1[:,0],AllPartYY_desnorm_2[:,0], alpha=0.5, linewidth=5)
# ax.scatter(Fval_desnorm_1[:,0], Fval_desnorm_2[:,0], alpha=0.5, c='#4682B4',linewidth=5)   
# ax.scatter(YY_desnorm_1[:,0], YY_desnorm_2[:,0], alpha=0.5, c='#DC143C',linewidth=6)
# plt.xlabel('F$_{1}$(x$_1$,x$_2$)',fontsize=20)
# plt.ylabel('F$_{2}$(x$_{1}$,x$_2$)',fontsize=20)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(AllPartXX[:,0],AllPartXX[:,1],AllPartYY_desnorm_1[:,0])

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
      savemat('PSO_NN_100p_100i_purity_recovery.mat',mydict)
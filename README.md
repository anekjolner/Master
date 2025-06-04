# Master Thesis: Data-Driven Optimization of PSA Processes for CO₂ Capture

This repository contains the code, models, and data used in my master's thesis project focused on developing, evaluating, and optimizing surrogate models for Pressure Swing Adsorption (PSA) systems using deep learning and symbolic regression.

## 📁 Project Structure

<details>
<summary>Click to expand</summary>

Master thesis final codes/
├── Data/                      # All datasets in MATLAB .mat format
│   ├── Input_global.mat
│   ├── Input_train.mat
│   ├── Input_valid.mat
│   ├── Input_test.mat
│   ├── Output_global.mat
│   ├── Output_train.mat
│   ├── Output_valid.mat
│   └── Output_test.mat

├── Models/                    # Model training scripts and saved models
│   ├── BNN_MC.py
│   ├── GP-DKL.py
│   ├── Hyperband_and_training.py
│   ├── PySRegressor.py
│   ├── model_BNN_MC_Purity.pth
│   ├── model_BNN_MC_Recovery.pth
│   ├── model_GNN_purity.pth
│   ├── model_GNN_recovery.pth
│   ├── model_Purity.h5
│   ├── model_Recovery.h5
│   └── PSA/                    # Hyperparameter tuning logs

├── Optimization/              # Multi-objective optimization scripts and results
│   ├── MOFE_PSO.py
│   ├── Optimization.py
│   ├── Optimization_BNN_MC.py
│   ├── Optimization_GP-DKL.py
│   ├── Optimization_regression.py
│   ├── PSO_BNN_100p_100i_purity_recovery_MC.mat
│   ├── PSO_GNN_100p_100i_purity_recovery.mat
│   ├── PSO_NN_100p_100i_purity_recovery.mat
│   ├── PSO_reg_100p_100i_purity_recovery.mat
│   └── __pycache__/

├── Validation/                # Evaluation, metrics and visualizations
│   ├── BNN_MSE_MAE_FIRST_PRINCIPLE.py   #histograms of BNNs vs first-principles model
│   ├── GP-DKL_MSE_MAE_First.py          #histograms of GP-DKLs vs first-principles model
│   ├── MAE_MSE_DNN.py                   #histograms of DNNs vs first-principles model
│   ├── MAE_MSE_reg.py                   #histograms of SR vs first-principles model
│   ├── Pareto Forts.py                  #the optimizations from all surrogate models vs optimization first-principles model
│   ├── Pareto front BNN.py
│   ├── Results_BNN_first_principle_model.mat
│   ├── Results_GNN_first_principle_model.mat
│   ├── Results_reg_first_principle_model.mat
│   ├── Results.mat
│   ├── Optimization.mat
│   └── Inputs all.py                     #Inputs form all surrogate models and first-principles model plottet together



## 📊 Description

- **Data Preparation**: MATLAB `.mat` files in `/Data/` contain preprocessed inputs and outputs for different stages (train/valid/test/global).
- **Model Training**: Surrogate models are trained using a variety of architectures including:
  - Deep Neural Networks (DNN)
  - Bayesian Neural Networks (BNN)
  - Deep Kernel Learning (GP-DKL)
  - Symbolic Regression via PySR
- **Optimization**: Multi-objective Particle Swarm Optimization (PSO) is used to find optimal operating conditions that balance CO₂ purity and recovery.
- **Validation**: Includes parity plots and metrics (MAE, MSE) for comparison between model predictions and ground truth.

## ⚙️ Requirements

This project uses:

- Python 3.8+
- TensorFlow / Keras
- PyTorch
- NumPy, SciPy, Matplotlib, Seaborn
- Plotly (for visualization)
- `mat73` (for loading MATLAB v7.3 files)
- `keras-tuner`
- `PySR` (for symbolic regression)
- `pymcmcstat` (if applicable for Bayesian analysis)

Install dependencies:

```bash
pip install -r requirements.txt

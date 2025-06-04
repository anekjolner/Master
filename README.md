# Master Thesis: Data-Driven Optimization of PSA Processes for CO₂ Capture

This repository contains the code, models, and data used in my master's thesis project focused on developing, evaluating, and optimizing surrogate models for Pressure Swing Adsorption (PSA) systems using deep learning and symbolic regression.

## 📁 Project Structure

- **Data Preparation**: MATLAB `.mat` files in `/Data/` contain preprocessed inputs and outputs for different stages:
  - `Input_train.mat`, `Input_valid.mat`, `Input_test.mat`
  - `Output_train.mat`, `Output_valid.mat`, `Output_test.mat`
  - `*_global.mat` files for full dataset reference

- **Model Training**: Surrogate models are trained using various architectures in `/Models/`:
  - **DNN**: Standard deep neural networks
  - **BNN**: Bayesian neural networks with Monte Carlo dropout
  - **GP-DKL**: Gaussian Process with Deep Kernel Learning
  - **PySR**: Symbolic regression using PySR
  - Trained models are saved as `.pth` and `.h5` files

- **Optimization**: Located in `/Optimization/`, this includes:
  - Multi-objective Particle Swarm Optimization (PSO)
  - Custom scripts for optimizing each surrogate model
  - `.mat` files with optimization results

- **Validation**: The `/Validation/` folder includes:
  - Parity plots comparing surrogate models to first-principles model
  - Evaluation metrics (MAE, MSE)
  - Pareto front comparisons and result summaries

---

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

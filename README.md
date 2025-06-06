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

---

## 📊 Description

- **Data Preparation**: MATLAB `.mat` files in `/Data/` contain preprocessed inputs and outputs for different stages (train/valid/test/global).
- **Model Training**: Surrogate models are trained using a variety of architectures including:
  - Deep Neural Networks (DNN) (only the results due to other work related to this code)
  - Bayesian Neural Networks (BNN)
  - Deep Kernel Learning (GP-DKL)
  - Symbolic Regression via PySR
--
## ⚠️ Disclaimer

This repository is meant as an archive where I can store my code. I hope that even suboptimal code can serve as inspiration for students who also want to explore machine learning.  
There are many potential improvements in my code, so please be critical and don't assume everything is correct 😉.  
I hope you find something useful here, and if you have any questions, please don't hesitate to contact me.

[LinkedIn](https://www.linkedin.com/in/ane-kristine-kjølner-a2a42a251/) · [E-mail](mailto:anekjolner@gmail.com)



## ⚙️ Requirements

To install the required dependencies, use the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate master


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mat73
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =============================================================================
# Load Data
# =============================================================================
Input_Train = mat73.loadmat('../Data/Input_train.mat')['Input_train']
Output_Train = mat73.loadmat('../Data/Output_train.mat')['Output_train']
Input_Valid = mat73.loadmat('../Data/Input_valid.mat')['Input_valid']
Output_Valid = mat73.loadmat('../Data/Output_valid.mat')['Output_valid']
Input_Test = mat73.loadmat('../Data/Input_test.mat')['Input_test']
Output_Test = mat73.loadmat('../Data/Output_test.mat')['Output_test']

Input_Train_norm = torch.tensor(Input_Train, dtype=torch.float32)
Input_Valid_norm = torch.tensor(Input_Valid, dtype=torch.float32)
Input_Test_norm  = torch.tensor(Input_Test, dtype=torch.float32)
Output_Train_norm = torch.tensor(Output_Train[:, 1], dtype=torch.float32)
Output_Valid_norm = torch.tensor(Output_Valid[:, 1], dtype=torch.float32)
Output_Test_norm  = torch.tensor(Output_Test[:, 1], dtype=torch.float32)

train_data = TensorDataset(Input_Train_norm, Output_Train_norm)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

valid_data = TensorDataset(Input_Valid_norm, Output_Valid_norm)
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False)

# =============================================================================
# Define MC Dropout Model
# =============================================================================
class BNNModel(nn.Module):
    def __init__(self, input_dim=5, hidden_units=[300, 250, 200, 150, 100], output_dim=1, activation='relu', dropout_rate=0.1):
        super(BNNModel, self).__init__()
        layers = []
        act_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()
        prev_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn)
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = BNNModel(input_dim=5)

# =============================================================================
# Training Loop
# =============================================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, valid_loader, epochs=200, patience=70):
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print("Early stopping")
            break

train_model(model, train_loader, valid_loader)

# =============================================================================
# MC Dropout Prediction (mean only)
# =============================================================================
def mc_dropout_predict(model, inputs, num_samples=100):
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(inputs).squeeze()
            preds.append(pred.cpu().numpy())
    preds = np.stack(preds)
    mean_preds = preds.mean(axis=0)
    return mean_preds

mean_pred = mc_dropout_predict(model, Input_Test_norm)
x_true = Output_Test_norm.numpy()

# =============================================================================
# Compute MAE and MSE
# =============================================================================
mae = mean_absolute_error(x_true, mean_pred)
mse = mean_squared_error(x_true, mean_pred)

print(f"\nTest MAE: {mae:.6f}")
print(f"Test MSE: {mse:.6f}")

# =============================================================================
# Plot predicted vs. true values (no uncertainty)
# =============================================================================
plt.figure(figsize=(6, 6))
plt.scatter(x_true, mean_pred, color='#1E90FF', s=30, label='Predicted vs True')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='y = x')
plt.title('Recovery Prediction vs True Values (MC Dropout)', fontsize=14)
plt.xlabel('True Recovery', fontsize=12)
plt.ylabel('Predicted Recovery', fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# Save the Model
# =============================================================================
torch.save(model.state_dict(), 'model_BNN_MC_Recovery.pth')

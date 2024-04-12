import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from model import MLP


# Load data from NPZ file
data = np.load(r'dataset\water_configurations_and_potentials.npz')
configurations = data['configurations'].reshape(-1, 9)  # Reshape configurations from N*3*3 to N*9
potentials = data['potentials']

# Convert numpy arrays to torch tensors
X = torch.tensor(configurations, dtype=torch.float32)
y = torch.tensor(potentials, dtype=torch.float32)

# load model with weights
model = MLP()
model.load_state_dict(torch.load('weight\mlp_model_weights.pth'))

# Predict the potential energies
model.eval()
y_pred = model(X)

# Plot the results
plot_y = y_pred.detach().cpu().numpy()
plot_y_true = y.detach().cpu().numpy()

# create a linear regression model and acquire R^2 score
# implement it inside plot_results
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Plot function
def plot_results(X, y_true, y_pred):
    # Create a linear regression model
    lr = LinearRegression()
    lr.fit(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Plot the results and linear fit line
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(y_true, lr.predict(y_true), color='red', label=f'R^2 Score: {r2}')
    plt.xlabel('True Potential Energy')
    plt.ylabel('Predicted Potential Energy')
    plt.legend()
    plt.show()

plot_results(X, plot_y, plot_y_true)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from model_MLP import MLP
#from PES_model import PESNetwork
from dotenv import load_dotenv
import os
import torchani

load_dotenv()
MLP_WEIGHTPATH = os.getenv('MLP_WEIGHTPATH')
PES_WEIGHTPATH = os.getenv('PES_WEIGHTPATH')
ANI_WEIGHTPATH = os.getenv('ANI_WEIGHTPATH')
DATAPATH = os.getenv('DATAPATH')


# Load data from NPZ file
data = np.load(DATAPATH)
configurations = data['configurations'].reshape(-1, 9)  # Reshape configurations from N*3*3 to N*9
potentials = data['potentials']

# Convert numpy arrays to torch tensors
X = torch.tensor(configurations, dtype=torch.float32)#.reshape(-1, 3, 3) if use ANI
y = torch.tensor(potentials, dtype=torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load model with weights
model = MLP()
#model=PESNetwork(3)
#model = torchani.models.ANI1x(periodic_table_index=True).to(device).double()
model.load_state_dict(torch.load(r"weight\mlp_model_weights.pth"))

# Predict the potential energies
model.eval()
batch_size = X.shape[0]
species = torch.tensor([[8, 1, 1]], device=device).repeat(batch_size, 1)
#y_pred = model((species,X)).energies*2625.5
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
    #lr = LinearRegression()
    #lr.fit(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Plot the results and linear fit line
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true/100, y_pred/100, alpha=0.5)
    #plt.plot(y_true, lr.predict(y_true), color='red', label=f'R^2 Score: {r2}')
    plt.xlabel('True Potential Energy eV')
    plt.ylabel('Predicted Potential Energy eV')
    plt.legend()
    #save
    plt.savefig('potential.png')
    #plt.show()

plot_results(X, plot_y, plot_y_true)
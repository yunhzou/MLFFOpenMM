#for each force of atom, plot the force
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from model_MLP import MLP
from PES_model import PESNetwork
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
forces_true = data['forces']

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
X.requires_grad = True
y_pred = model(X)
# Calculate the gradients of the potential energy with respect to the input coordinates
y_pred.backward(torch.ones_like(y_pred))
forces = -X.grad.reshape(-1, 3, 3)/100
forces_true = forces_true/100

# Plot the results of force vs true force for each atom
plt.figure(figsize=(8, 8))
for i in range(3):
    plt.scatter(forces_true[:, i], forces[:, i], alpha=0.5, label=f'Atom {i+1}')
plt.xlabel('True Force eV/Angstrom')
plt.ylabel('Predicted Force eV/Angstrom')
#save
plt.savefig('force.png')
#plt.show()




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from model_MLP import *
from reference_scripts.PES_model import PESNetwork
from dotenv import load_dotenv
import os
import torchani
model_name = 'mlp' # or ani
weight_path = r"weight\mlp_model_weights.pth"
data_path = r"dataset\water_configurations_and_potentials.npz"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1000
data_loader,val_loader,test_loader = load_data(batch_size)
if model_name == 'mlp':
    model = MLP().to(device)
    model.load_state_dict(torch.load(weight_path))
elif model_name == 'ani':
    model = torchani.models.ANI1x(periodic_table_index=True).to(device).double()
    model.load_state_dict(torch.load(weight_path))
    species = torch.tensor([[8, 1, 1]], device=device).repeat(batch_size, 1) 
# Predict the potential energies
model.eval()
for i,(batch_X, batch_y, batch_f) in enumerate(data_loader):
    batch_X.requires_grad = True  
    batch_X, batch_y, batch_f = batch_X.to(device), batch_y.to(device), batch_f.to(device)
    batch_X.retain_grad()
    y_pred = model(batch_X)
    if model_name == 'ani':
        y_pred = y_pred.energies*2625.5 # Convert from Hartree to kj/mol
    y_pred.sum().backward(retain_graph=True)
    forces = -batch_X.grad # in the unit of kj/mol/angstrom
    batch_X.grad.zero_()
    # Initialize y_pred_total if it doesn't exist, else concatenate
    if i == 0:
        y_pred_total = y_pred
        force_total_pred = forces
    else:
        y_pred_total = torch.cat((y_pred_total, y_pred), 0)
        force_total_pred = torch.cat((force_total_pred, forces), 0)
X = torch.tensor(data_loader.dataset.tensors[0], dtype=torch.float32).to(device)
y_pred_total_1 = model(X)
# y_pred_total shape: (N, 1), force_total shape: (N, 3, 3)
energies_pred = y_pred_total.detach().cpu().numpy()
#acquire all the potential energies from dataloader
energies_true = data_loader.dataset.tensors[1].detach().cpu().numpy() 

# create a linear regression model and acquire R^2 score
# implement it inside plot_results
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Plot function
def plot_results(X, y_true, y_pred):
    lr = LinearRegression()
    lr.fit(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(y_true, lr.predict(y_true), color='red', label=f'R^2 Score: {r2}')
    plt.xlabel('True Potential Energy eV')
    plt.ylabel('Predicted Potential Energy eV')
    plt.legend()
    #save
    plt.savefig('potential.png')
    #plt.show()

plot_results(1,energies_true, energies_pred)
force_total_pred = force_total_pred.detach().cpu().numpy()
force_total_true = data_loader.dataset.tensors[2].detach().cpu().numpy()

plt.figure(figsize=(8, 8))
for i in range(3):
    plt.scatter(force_total_pred[:, i], force_total_true[:, i], alpha=0.5, label=f'Atom {i+1}')
plt.xlabel('True Force eV/Angstrom')
plt.ylabel('Predicted Force eV/Angstrom')
#save
plt.savefig('force.png')
#plt.show()
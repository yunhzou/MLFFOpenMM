import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from model_MLP import train, load_data

load_dotenv()
MLP_WEIGHTPATH = os.getenv('MLP_WEIGHTPATH')
PES_WEIGHTPATH = os.getenv('PES_WEIGHTPATH')
DATAPATH = os.getenv('DATAPATH')

class AtomSubnet(nn.Module):
    def __init__(self):
        super(AtomSubnet, self).__init__()
        # The number of hidden units for each atom's subnet can be adjusted
        self.layers = nn.Sequential(
            nn.Linear(3, 50),  # 3 for 3D coordinates input
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)   # Output a single energy term
        )

    def forward(self, x):
        return self.layers(x)

# Define the main PES network
class PESNetwork(nn.Module):
    def __init__(self, num_atoms):
        super(PESNetwork, self).__init__()
        self.num_atoms = num_atoms
        self.atom_subnets = nn.ModuleList([AtomSubnet() for _ in range(num_atoms)])

    def forward(self, x):
        # Ensure x is of shape (batch_size, num_atoms, 3) before transposing
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_atoms, -1)  # Reshape to (batch_size, num_atoms, 3)
        
        # Transpose and then apply each subnet to the corresponding atom coordinates
        energies = [subnet(atom_coords) for subnet, atom_coords in zip(self.atom_subnets, x.transpose(0, 1))]
        energies = torch.stack(energies, dim=1)
        
        # Sum energies from each subnetwork for total energy
        total_energy = energies.sum(dim=1, keepdim=True).flatten()
        return total_energy
    
# Main script
def PES_main(load=True,epochs=20,batch_size=256,num_atoms=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load data
    data_loader = load_data(batch_size)

    if load==True:
        model = PESNetwork(num_atoms)
        model.load_state_dict(torch.load(PES_WEIGHTPATH))
    else:
        model = PESNetwork(num_atoms)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train(model, criterion, optimizer, data_loader, epochs, device)

    # Save the model weights
    torch.save(model.state_dict(), PES_WEIGHTPATH)

if __name__ == '__main__':
    PES_main(load=False,
             epochs=300,
             batch_size=256)
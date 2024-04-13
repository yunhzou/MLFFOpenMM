import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()
MLP_WEIGHTPATH = os.getenv('MLP_WEIGHTPATH')
PES_WEIGHTPATH = os.getenv('PES_WEIGHTPATH')
DATAPATH = os.getenv('DATAPATH')

# Define the MLP model tailored to the water molecule
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 50),  # Input layer adjusted to take 9 features (3 atoms * 3 coordinates)
            nn.ReLU(),
            nn.Linear(50, 50), # Intermediate layer
            nn.ReLU(),
            nn.Linear(50, 50),   # Output layer adjusted to produce 1 output (potential energy)
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        return self.layers(x)


# Training function
def train(model, optimizer, data_loader, epochs, device, eta):
    model.to(device)
    for epoch in range(epochs):
        for batch_X, batch_y, batch_f in data_loader:  # Assuming the loader now also provides forces
            batch_X.requires_grad = True
            batch_f=batch_f.reshape(-1, 9)
            batch_X, batch_y, batch_f = batch_X.to(device), batch_y.to(device), batch_f.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = custom_loss(output, batch_y, batch_X, batch_f, eta)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#load data 
def load_data(batch_size):
    data = np.load(DATAPATH)
    configurations = data['configurations'].reshape(-1, 9)  # Reshape configurations from N*3*3 to N*9
    potentials = data['potentials']
    forces = data['forces']  # You will need to add force data to your dataset
    X = torch.tensor(configurations, dtype=torch.float32)
    y = torch.tensor(potentials, dtype=torch.float32).reshape(-1, 1)  # Ensure y is the correct shape
    f = torch.tensor(forces, dtype=torch.float32)  # Assuming forces have the correct shape
    dataset = TensorDataset(X, y, f)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def custom_loss(output_energies, target_energies, input_coordinates, target_forces, eta):
    # Calculate MSE for the energies
    energy_loss = nn.functional.mse_loss(output_energies, target_energies)

    # Calculate gradients of energies to get predicted forces
    # Requires running model in backward mode to calculate gradients
    # Since we cannot call backward in the loss function, we use torch.autograd.grad
    predicted_forces = -torch.autograd.grad(outputs=output_energies,
                                            inputs=input_coordinates,
                                            grad_outputs=torch.ones_like(output_energies),
                                            create_graph=True)[0]
    
    # Calculate MSE for the forces
    force_loss = nn.functional.mse_loss(predicted_forces, target_forces)

    # Combine the losses
    total_loss = force_loss + eta * energy_loss
    total_loss = energy_loss    # For now, we will only use the energy loss
    return total_loss


# Main script
def MLP_main(load=True):
    epochs = 1000
    batch_size = 256
    eta = 0.2
    load = input("Do you want to load the weights? (y/n): ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load data
    data_loader = load_data(batch_size)

    if load=='y':  
        model = MLP()
        model.load_state_dict(torch.load(MLP_WEIGHTPATH))
    else:
        model = MLP()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    train(model, optimizer, data_loader, epochs, device, eta)

    # Save the model weights
    torch.save(model.state_dict(), MLP_WEIGHTPATH)

if __name__ == '__main__':
    MLP_main(load = True)
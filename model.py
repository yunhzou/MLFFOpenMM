import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Define the MLP model tailored to the water molecule
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 50),  # Input layer adjusted to take 9 features (3 atoms * 3 coordinates)
            nn.ReLU(),
            nn.Linear(50, 50), # Intermediate layer
            nn.ReLU(),
            nn.Linear(50, 1),   # Output layer adjusted to produce 1 output (potential energy)
            #nn.ReLU(),
            #nn.Linear(50, 1),
        )

    def forward(self, x):
        return self.layers(x)

# Training function
def train(model, criterion, optimizer, data_loader, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#load data 
def load_data(batch_size):
    data = np.load('water_configurations_and_potentials.npz')
    configurations = data['configurations'].reshape(-1, 9)  # Reshape configurations from N*3*3 to N*9
    potentials = data['potentials']
    X = torch.tensor(configurations, dtype=torch.float32)
    y = torch.tensor(potentials, dtype=torch.float32).reshape(-1, 1)  # Ensure y is the correct shape
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# Main script
if __name__ == '__main__':
    epochs = 1300
    batch_size = 128
    load = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load data
    data_loader = load_data(batch_size)

    if load==True:
        model = MLP()
        model.load_state_dict(torch.load(r'weight\mlp_model_weights.pth'))
    else:
        model = MLP()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train(model, criterion, optimizer, data_loader, epochs, device)

    # Save the model weights
    torch.save(model.state_dict(), r'weight\mlp_model_weights.pth')


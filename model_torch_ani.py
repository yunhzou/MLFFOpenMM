import torch
import torchani
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from dotenv import load_dotenv
import torchani

load_dotenv()
MLP_WEIGHTPATH = os.getenv('MLP_WEIGHTPATH')
PES_WEIGHTPATH = os.getenv('PES_WEIGHTPATH')
ANI_WEIGHTPATH = os.getenv('ANI_WEIGHTPATH')

load = input("Do you want to load the weights? (y/n): ")
# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
# Load dataset
data_path = r'C:\Users\Lenovo\Desktop\MachineLearningForceField\MLLP\dataset\water_configurations_and_potentials.npz'
data = np.load(data_path)
configurations = torch.tensor(data['configurations'], dtype=torch.float64, device=device)  # Ensuring double precision
potentials = torch.tensor(data['potentials'], dtype=torch.float64, device=device).reshape(-1, 1)  # Ensuring double precision

# Create DataLoader for batch processing
dataset = TensorDataset(configurations, potentials)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model using TorchANI and ensure it is in double precision
ani_model = torchani.models.ANI1x(periodic_table_index=True).to(device).double()
if load == 'y':
    ani_model.load_state_dict(torch.load(r"weight\ani_model_weights.pth"))
# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ani_model.parameters(), lr=0.001)

# Species tensor for water molecule (Oxygen, Hydrogen, Hydrogen)
species_input = torch.tensor([[8, 1, 1]], device=device).repeat(batch_size, 1)  # Pre-replicated for each batch

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        # Ensure batch size matches
        batch_size = inputs.size(0)
        species = species_input[:batch_size]  # Adjust species size to current batch size

        targets = targets / 2625.5   # Convert from kj/mol to Hartree
        optimizer.zero_grad()
        outputs = ani_model((species, inputs)).energies
        outputs = outputs.reshape(-1, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(ani_model.state_dict(), r'weight\ani_model_weights.pth')
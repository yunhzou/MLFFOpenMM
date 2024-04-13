import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
import torchani

#Rule:
# Configuration: shape (N, 3, 3) where N is the number of configurations
# Potential energy: shape (N, 1)
# Forces: shape (N, 3, 3)

load_dotenv()
ANI_WEIGHTPATH = os.getenv('ANI_WEIGHTPATH')
DATAPATH = os.getenv('DATAPATH')


#load data 
def load_data(batch_size):
    data = np.load(DATAPATH)
    configurations = data['configurations']  # Reshape configurations from N*3*3 to N*9
    potentials = data['potentials']
    forces = data['forces']

    # Convert to tensors
    X = torch.tensor(configurations, dtype=torch.float64)
    y = torch.tensor(potentials, dtype=torch.float64).reshape(-1, 1)
    f = torch.tensor(forces, dtype=torch.float64)

    # Split the data into training and test+validation
    X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
        X, y, f, test_size=0.3, random_state=42)

    # Split the test+validation into test and validation
    X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
        X_temp, y_temp, f_temp, test_size=0.5, random_state=42)

    # Creating TensorDatasets
    train_dataset = TensorDataset(X_train, y_train, f_train)
    val_dataset = TensorDataset(X_val, y_val, f_val)
    test_dataset = TensorDataset(X_test, y_test, f_test)

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def custom_loss(output_energies, target_energies, input_coordinates, target_forces, eta_potential, eta_force):
    # output_energies: shape (batch_size, 1)
    # Calculate MSE for the energies
    input_coordinates.retain_grad()
    energy_loss = nn.functional.mse_loss(output_energies, target_energies)

    #output_energies.sum().backward(retain_graph=True)
    predicted_forces = -torch.autograd.grad(output_energies.sum(), input_coordinates, create_graph=True)[0]
    #predicted_forces = -input_coordinates.grad
    
    # Calculate MSE for the forces
    force_loss = nn.functional.mse_loss(predicted_forces, target_forces)

    # Combine the losses
    total_loss = eta_force*force_loss + eta_potential * energy_loss
    input_coordinates.grad.zero_()
    return total_loss


# Training function
def train(model, optimizer, train_loader, val_loader, epochs, device, eta_potential, eta_force):
    model.to(device)
    best_loss = np.inf

    try:
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y, batch_f in train_loader:
                batch_X.requires_grad = True
                batch_X, batch_y, batch_f = batch_X.to(device), batch_y.to(device), batch_f.to(device)
                batch_X.retain_grad()
                optimizer.zero_grad()
                species = torch.tensor([[8, 1, 1,8,1,1]], device=device).repeat(batch_X.size(0), 1)
                output = model((species,batch_X)).energies
                output = output.reshape(-1,1)
                loss = custom_loss(output, batch_y, batch_X, batch_f, eta_potential, eta_force)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0
            val_batches = 0
            for batch_X, batch_y, batch_f in val_loader:
                batch_X.requires_grad = True
                batch_X, batch_y, batch_f = batch_X.to(device), batch_y.to(device), batch_f.to(device)
                batch_X.retain_grad()
                species = torch.tensor([[8, 1, 1,8,1,1]], device=device).repeat(batch_X.size(0), 1)
                output = model((species,batch_X)).energies
                output = output.reshape(-1,1)
                loss = custom_loss(output, batch_y, batch_X, batch_f, eta_potential, eta_force)
                val_loss += loss.item()
                val_batches += 1
            average_val_loss = val_loss / val_batches

            # Save the model if it's the best so far
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                torch.save(model.state_dict(), ANI_WEIGHTPATH)  # Save best model state

            print(f'Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {average_val_loss}')

    except KeyboardInterrupt:
        print("Training stopped manually via keyboard interrupt. Saving current model state...")
        torch.save(model.state_dict(), ANI_WEIGHTPATH)  # Save the current model state
        print("Current model weights saved.")

# Main script
def ANI_main():
    epochs = 1000
    batch_size = 512
    eta_potential = 0.4
    eta_force = 1
    lr = 0.00008
    load = input("Do you want to load the weights? (y/n): ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size)

    model = torchani.models.ANI1x(periodic_table_index=True).to(device).double()
    if load=='y':
        model.load_state_dict(torch.load(ANI_WEIGHTPATH))

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train(model, optimizer, train_loader, val_loader, epochs, device, eta_potential, eta_force)
    # Save the model weights
    torch.save(model.state_dict(), ANI_WEIGHTPATH)

if __name__ == '__main__':
    ANI_main()
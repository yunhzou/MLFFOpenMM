import torch
import torch.nn as nn
import torch.optim as optim

# Define a model with two layers and one ReLU activation
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # First layer
        self.relu = nn.ReLU()        # ReLU activation
        self.fc2 = nn.Linear(10, 1)  # Second layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate synthetic data
x = torch.linspace(-2, 2, 100).reshape(-1, 1)
y = x ** 2  # potential energy
true_forces = 2 * x  # true forces

# Model, optimizer, and criterion
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # reduce number of epochs for testing
    optimizer.zero_grad()
    x_tensor = x.clone().requires_grad_(True)  # Clone x and allow gradient computation
    predictions = model(x_tensor)
    energy_loss = nn.functional.mse_loss(predictions, y)
    # Force calculation with gradients
    #predictions.sum().backward(create_graph=True)
    forces = torch.autograd.grad(predictions.sum(), x_tensor, create_graph=True)[0]
    #forces = x_tensor.grad

    # Compute the force loss directly using the forces
    force_loss = nn.functional.mse_loss(forces, true_forces)

    # Compute gradients for the model parameters based on force loss
    force_loss.backward()
    #energy_loss.backward()
    
    optimizer.step()

    # Print losses to monitor training
    if (epoch + 1) % 1 == 0:  # changed to print every epoch for testing
        print(f'Epoch {epoch+1}: Force Loss: {force_loss.item():.4f}')
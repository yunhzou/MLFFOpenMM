from model import *
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP()
model.load_state_dict(torch.load('mlp_model_weights.pth'))
#model.to(device)


#acquire the bottom 3 input gradient with repect to the output y instead of the loss

X, y = load_data(1)

model.eval()
X.requires_grad = True
y.requires_grad = True
output = model(X)
output.backward(torch.ones_like(output))
print(X.grad) 
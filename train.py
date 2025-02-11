import torch
import torch.nn as nn
import torch.optim as optim
from model import GCN
from smiles_to_graph import smiles_to_graph
import pandas as pd

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)




# Example dataset (replace with your own dataset)
# smiles_list = ['CCO', 'CCN']  # Example SMILES
# pka_values = [5.1, 7.2]  # Corresponding pKa values


# Define the model
model = GCN(in_feats=4, hidden_size=64, out_feats=1, num_layers=3)  # 4 feature input, 1 feature output

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(200):
    model.train()
    total_loss = 0
    for smiles, pka in zip(smiles_list, pka_values):
        # Convert SMILES to graph
        g = smiles_to_graph(smiles)
        features = g.ndata['feat']

        # Forward pass
        predictions = model(g, features)
        
        # Compute loss
        loss = loss_fn(predictions, torch.tensor([pka], dtype=torch.float32))
        total_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {total_loss}")

# After training, you can save the model if needed
torch.save(model.state_dict(), 'pka_gcn_model.pth')

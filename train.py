import torch
import torch.nn as nn
import torch.optim as optim
from model import GCN
from smiles_to_graph import smiles_to_graph

# Example dataset (replace with your own dataset)
smiles_list = ['CCO', 'CCN']  # Example SMILES
pka_values = [5.1, 7.2]  # Corresponding pKa values

# Define the model
model = GCN(in_feats=4, hidden_size=64, out_feats=1)  # 1 feature input, 1 feature output

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    for smiles, pka in zip(smiles_list, pka_values):
        # Convert SMILES to graph
        g = smiles_to_graph(smiles)
        features = g.ndata['feat']
        # print(features)
        # Forward pass
        predictions = model(g, features).squeeze()
        
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

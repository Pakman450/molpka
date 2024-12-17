import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        """
        Initializes the GCN model.

        Args:
        - in_feats (int): Number of input features (e.g., 1 for atomic number).
        - hidden_size (int): Size of the hidden layer.
        - out_feats (int): Number of output features (1 for regression).
        """
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, out_feats)

    def forward(self, g, features):
        """
        Forward pass of the GCN.

        Args:
        - g (DGLGraph): The graph of atoms and bonds.
        - features (Tensor): Node features (atomic number or other properties).

        Returns:
        - Tensor: Predicted output (pKa value).
        """
        x = F.relu(self.conv1(g, features))
        x = self.conv2(g, x)
        return x

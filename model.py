import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_feats: int, hidden_size: int, out_feats: int, num_layers: int):
        """
        Initializes the GCN model.

        Args:
        - in_feats (int): Number of input features (e.g., 1 for atomic number).
        - hidden_size (int): Size of the hidden layer.
        - out_feats (int): Number of output features (1 for regression).
        """
        super(GCN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.GraphConv(in_feats, hidden_size))

        for _ in range(num_layers - 2):
            self.layers.append(dgl.nn.GraphConv(hidden_size,hidden_size))


        self.layers.append(dgl.nn.GraphConv(hidden_size, out_feats))


    def forward(self, g, features):
        """
        Forward pass of the GCN.

        Args:
        - g (DGLGraph): The graph of atoms and bonds.
        - features (Tensor): Node features (atomic number or other properties).

        Returns:
        - Tensor: Predicted output (pKa value).
        """

        x = features
       
        for layer in self.layers[:-1]:  # Apply activation to all but last layer
            x = F.relu(layer(g, x))


        x = self.layers[-1](g, x)  # No activation in final layer

        g.ndata['out_feat'] = x

        # x = F.relu(self.conv1(g, features))
        # x = F.relu(self.conv2(g, x))
        # # You could add dropout like so:
        # # x = self.dropout(F.relu(self.conv2(g, x)))
        # # to avoide overfitting
        # x = self.conv2(g, x)

        # Now we pool the node features into a single scalar value (mean, sum, or max)
        pooled_x = dgl.mean_nodes(g, 'out_feat')  # Mean pooling over all nodes

        return pooled_x

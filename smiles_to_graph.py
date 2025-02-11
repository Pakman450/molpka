import rdkit
from rdkit import Chem
import torch
import dgl

def smiles_to_graph(smiles):
    """
    Converts a SMILES string to a DGL graph.

    Args:
    - smiles (str): SMILES string of the molecule.

    Returns:
    - DGLGraph: DGL graph representing the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    
    node_features=[]

    for atom in atoms:
        # Hybridization (new feature)
        hybridization = atom.GetHybridization()
        hybridization = {'SP': 0, 'SP2': 1, 'SP3': 2, 'SP3D': 3, 'SP3D2': 4}.get(str(hybridization), -1)

        # Aromaticity (new feature)
        aromatic = 1 if atom.GetIsAromatic() else 0

        node_features.append([
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            hybridization,
            aromatic
            ])

    node_features = torch.tensor(node_features, dtype=torch.float32)
    # # Create nodes (atoms)
    # node_features = [atom.GetAtomicNum() for atom in atoms]
    # print(node_features)
    # Create edges (bonds)
    edges = []
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append((i, j))
        edges.append((j, i))  # undirected edges
    
    # Create a DGL graph
    g = dgl.graph(([e[0] for e in edges], [e[1] for e in edges]))

    # Assign node features to the graph
    g.ndata['feat'] = node_features
    # g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32).view(-1, 1)
    
    return g

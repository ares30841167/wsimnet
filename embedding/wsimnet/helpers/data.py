import os
import torch
import numpy as np
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


ROOT = '0'


# Load the graphs
def load_graphs(dataset_path: str, dataset_type: str = 'train') -> dict[str, nx.DiGraph]:
    # If option invalid, raise an error
    if (dataset_type not in ['train', 'validate', 'test', 'overall']):
        raise Exception('Unknown dataset type')

    if (dataset_type == 'overall'):
        all_types = ['train', 'validate', 'test']
        graphs = {}

        for type in all_types:
            graphs.update(load_graphs(dataset_path, type))

        return graphs

    # Init variables
    graphs = {}

    # Loop through all files in the folder and load graph
    dataset_folder_path = f'{dataset_path}/{dataset_type}'
    for filename in os.listdir(dataset_folder_path):
        if filename.endswith('.gexf'):
            file_path = os.path.join(dataset_folder_path, filename)
            site_name = '.'.join(filename.split('.')[:-1])
            graphs[site_name] = nx.read_gexf(file_path)

    return graphs


# Load the node features
def load_features(dataset_path: str) -> dict[str, np.ndarray]:
    """Load all the website graphs."""
    FEATURES_PATH = f'{dataset_path}/node_features.npy'

    # Import features from the npy file
    features = np.load(FEATURES_PATH, allow_pickle=True)
    
    return features.item()


# Generate the mapping between the ground-truth and the numeric.
def gen_label_map(graphs: dict[str, nx.DiGraph]) -> dict[str, int]:
    # Collect all label type
    unique_label = set()
    for graph in graphs.values():
        unique_label.add(graph.nodes[ROOT]['company_label'])

    # Create a mapping of string categories to integer labels
    return {label: idx for idx, label in enumerate(unique_label)}


# Convert the graph into PyTorch Geometric Data format
def convert_to_pyg(name: str, graph: nx.DiGraph, node_features: np.ndarray, label_map: dict[str, int]) -> Data:
    """Convert a NetworkX graph to PyTorch Geometric Data format."""

    # Match node indices (ensure consistent mapping between features and graph nodes)
    node_mapping = {node: i for i, node in enumerate(graph.nodes)}

    if (len(graph.edges) > 0):
        edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in graph.edges], dtype=torch.long).t()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=len(graph.nodes))

    # Extract features based on mapping
    node_features = np.array(node_features, dtype=np.float32)
    x = torch.tensor(node_features[list(node_mapping.values())], dtype=torch.float32)

    # Assign graph-level label (only one label per graph)
    y = torch.tensor([label_map[graph.nodes[ROOT]['company_label']]] if label_map != None else -1, dtype=torch.long)

    # All nodes in the graph have the same batch index (needed for pooling)
    batch = torch.zeros(x.shape[0], dtype=torch.long)  # All nodes belong to the same graph

    return Data(x=x, edge_index=edge_index, y=y, batch=batch, site_name=str(name), gt_label=str(graph.nodes[ROOT]['company_label']))

import networkx as nx
import numpy as np
import torch
import os
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
project_dir = os.path.dirname(__file__)
datasets_dir = os.path.join(project_dir, '../data', 'poi.data')

class S2VGraph(object):
    def __init__(self, g, label, num_nodes=None, edge_mat=None, node_features=None):
        """
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        """
        self.label = label
        self.g = g
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = edge_mat
        self.num_nodes = num_nodes
        self.max_neighbor = 0

def load_data(path, dataset):
    """
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    """
    print('loading data')
    poi_content = os.path.join(datasets_dir, f'poi.data.graphcontent.txt')
    idx_features_labels = np.genfromtxt(poi_content, dtype=np.dtype(str))
    all_features = idx_features_labels[:, 1:-2].astype(np.float32)

    poi_zoneTypes = os.path.join(datasets_dir, f'poi.data.ZoneTypes.txt')
    idx_ZoneTypes_labels = np.genfromtxt(poi_zoneTypes, dtype=np.dtype(str))
    all_labels = idx_ZoneTypes_labels[:, 1]

    g_list = []
    label_dict = {}
    id_to_row = {int(row[0]): row for row in idx_features_labels}

    for i in range(2411):
        try:
            file_path = os.path.join(datasets_dir, f'poi.data.poi_graph_zone{[i]}.txt')
            edges_unordered = np.genfromtxt(file_path.format(i), dtype=np.int32)
            l = all_labels[i]
            unique_nodes = np.unique(edges_unordered)
            n = len(unique_nodes)
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            g.add_nodes_from(range(n))
            source_features_set = set()
            # Create a dictionary to map source_id and target_id to their rows in idx_features_labels
            for edge in edges_unordered:
                source_id, target_id = edge
                source_row = id_to_row[source_id]
                target_row = id_to_row[target_id]
                source_features_set.add(tuple(source_row[0:-2]))
                source_features_set.add(tuple(target_row[0:-2]))

            sorted_data = sorted(source_features_set, key=lambda x: x[0])
            sorted_data = np.array(sorted_data)[:, 1:]
            node_features = sorted_data.astype(float)
            node_features = torch.tensor(node_features)
            unique_nodes = np.unique(edges_unordered)
            mapping = {node: idx for idx, node in enumerate(unique_nodes)}
            edges_index_mapped = np.vectorize(mapping.get)(edges_unordered)
            g.add_edges_from(edges_index_mapped)
            edge_mat = torch.LongTensor(edges_index_mapped).transpose(0,1)
            g_list.append(S2VGraph(g, int(l), n, edge_mat, node_features))
        except FileNotFoundError:
            continue

    print('# classes: %d' % len(label_dict))
    print("# data: %d" % len(g_list))
    return g_list, len(label_dict)

def S2V_to_PyG(data):
    new_data = Data()
    setattr(new_data, 'edge_index', data.edge_mat)
    setattr(new_data, 'x', data.node_features)
    setattr(new_data, 'num_nodes', data.num_nodes)
    setattr(new_data, 'y', torch.tensor(data.label).unsqueeze(0).long())
    return new_data

def get_indices(complex_list, seed, test_size=0.3):
    labels = [complex.y.item() for complex in complex_list]
    train_idx, test_idx = train_test_split(range(len(complex_list)), test_size=test_size, stratify=labels, random_state=seed)
    train_idx.sort()
    test_idx.sort()
    return train_idx, test_idx


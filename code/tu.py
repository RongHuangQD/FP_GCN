import os
import torch
import pickle
import numpy as np
from tu_utils import load_data, S2V_to_PyG, get_indices
project_dir = os.path.dirname(__file__)
root = os.path.join(project_dir, '../data')
def load_tu_graph_dataset(name, root, seed=0):
    raw_dir = os.path.join(root, name, 'raw')
    load_from = os.path.join(raw_dir, '{}_graph_list.pkl'.format(name))
    if os.path.isfile(load_from):
        with open(load_from, 'rb') as handle:
            graph_list = pickle.load(handle)
    else:
        data, num_classes = load_data(raw_dir, name)
        print('Converting graph data into PyG format...')
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(load_from, 'wb') as handle:
            pickle.dump(graph_list, handle)
    train_ids, test_ids = get_indices(graph_list, seed, test_size=0.3)
    return graph_list, train_ids, test_ids
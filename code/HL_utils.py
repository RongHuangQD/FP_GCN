import torch
import numpy as np
import networkx as nx
from torch_sparse import SparseTensor, fill_diag, matmul, mul
import os
import yaml
import itertools
import pickle
import time

device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

def gen_HL(name, root, graph_list, petal_dim):

    HL_path = os.path.join(root, name, name+'_HL.pt')
    max_petal_dim = 6
    if os.path.exists(HL_path):
        HL = torch.load(HL_path)
        print("Load HL!")
    else:
        HL = []
        tot_petal_num = np.zeros(max_petal_dim + 1, dtype=int)
        for idx, graph in enumerate(graph_list):
            print(f"generate HL for graph {idx}({graph.x.shape[0]} nodes and {graph.edge_index.shape[1]/2} edges)")
            src, dst = graph.edge_index
            edges = zip(src.tolist(), dst.tolist())
            G = nx.from_edgelist(edges)
            G.add_nodes_from(range(graph.x.shape[0]))
            L, petalNum = gen_HL_samll_space(G, max_petal_dim=max_petal_dim)
            tot_petal_num = tot_petal_num + petalNum
            HL.append(L)

        torch.save(HL, HL_path)

        with open(os.path.join(root, name, name+'_simplex_statics.yaml'), "w") as f:
            statistics = {'graph_num': idx+1, 'n_simplex': tot_petal_num.tolist()}
            yaml.dump(statistics, f)

    for idx in range(len(graph_list)):
        graph_list[idx].HL = {order: HL[idx][order] for order in range(1, petal_dim+1)}

    if isinstance(graph_list[0].x, torch.LongTensor):
        for idx in range(len(graph_list)):
            graph_list[idx].x = graph_list[idx].x.float()

    return graph_list

def gen_HL_samll_space(G: nx.graph, max_petal_dim=2):

    def get_simplex(G: nx.graph, max_petal_dim):

        itClique = nx.enumerate_all_cliques(G)
        nextClique = next(itClique)
        while len(nextClique) <= 1:
            nextClique = next(itClique)
        while len(nextClique) <= max_petal_dim + 1:
            yield sorted(nextClique)

            try:
                nextClique = next(itClique)
            except StopIteration:
                break

    petalNum = np.zeros(max_petal_dim + 1, dtype=int)
    petalNum[0] = n_core = G.number_of_nodes()

    HL, _D, _delta = {}, {}, {}
    for order in range(1, max_petal_dim + 1):
        HL[order] = torch.zeros(n_core, n_core)
        _D[order] = torch.zeros(n_core)
        _delta[order] = torch.ones((order + 1) ** 2)

    for cell in get_simplex(G, max_petal_dim):
        order = len(cell) - 1
        petalNum[order] += 1
        _D[order][cell] += 1
        comb = torch.LongTensor([[a, b] for a, b in itertools.combinations(cell, 2)])
        rows = torch.cat([torch.LongTensor(cell), comb[:, 0], comb[:, 1]], dim=0)
        cols = torch.cat([torch.LongTensor(cell), comb[:, 1], comb[:, 0]], dim=0)
        HL[order][rows, cols] += 1

    for order in range(1, max_petal_dim + 1):
        d_inv = _D[order]
        d_inv[d_inv == 0] = 1
        d_inv = d_inv.pow_(-0.5)
        HL[order] = SparseTensor.from_dense(HL[order])
        HL[order] = mul(HL[order], d_inv.view(1, -1))
        HL[order] = mul(HL[order], d_inv.view(-1, 1) / (order + 1))
    return HL, petalNum
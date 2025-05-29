import copy
import os
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from HL_utils import gen_HL
from tu import  load_tu_graph_dataset

project_dir = os.path.dirname(__file__)
root_dir = os.path.join(project_dir, '../data')

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], **kwargs):

        self.follow_batch = follow_batch
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        batch = copy.deepcopy(batch)
        if 'HL' not in batch[0]:
            return Batch.from_data_list(batch, self.follow_batch)

        order = list(batch[0].HL.keys())
        n = {_: 0 for _ in order}
        rows = {_: torch.LongTensor([]) for _ in order}
        cols = {_: torch.LongTensor([]) for _ in order}
        values = {_: torch.LongTensor([]) for _ in order}

        for id in range(len(batch)):
            d = batch[id]
            for o in order:
                rows[o] = torch.cat([rows[o], d.HL[o].storage.row() + n[o]], dim=0)
                cols[o] = torch.cat([cols[o], d.HL[o].storage.col() + n[o]], dim=0)
                values[o] = torch.cat([values[o], d.HL[o].storage.value()], dim=0)
                n[o] += d.HL[o].sparse_sizes()[0]
            del batch[id].HL

        dataBatch = Batch.from_data_list(batch, self.follow_batch)
        dataBatch.HL = {o: SparseTensor(row=rows[o], col=cols[o], value=values[o], sparse_sizes=(n[o], n[o])) for o in order}
        return dataBatch
def load_graph_dataset(name, root=root_dir, **kwargs):
    graph_list, train_ids, test_ids = load_tu_graph_dataset(name, root=root, seed=0)
    graph_list = gen_HL(name, root, graph_list, petal_dim=kwargs['max_petal_dim'])
    data = (graph_list, train_ids, test_ids, 11)
    return data

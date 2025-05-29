import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import global_add_pool
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
from torch_sparse import  matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Linear, Sequential, BatchNorm1d as BN

class HiGCN_prop(MessagePassing):

    def __init__(self, K, alpha, Order=2, bias=True, **kwargs):
        super(HiGCN_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.Order = Order
        self.fW = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fW)
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k
        self.fW.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, HL):
        hidden = x * (self.fW[0])
        for k in range(self.K):
            x = matmul(HL, x, reduce=self.aggr)
            gamma = self.fW[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def __repr__(self):
        return '{}(Order={}, K={}, filterWeights={})'.format(self.__class__.__name__, self.Order, self.K, self.fW)

class HiGCNConv(MessagePassing):

    def __init__(self, num_features, order, hidden, num_classess, dropout_rate, nn, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.dropout = dropout_rate
        self.dprate = dropout_rate
        self.Order = order

        self.lin_in = torch.nn.ModuleList()
        self.hgc = torch.nn.ModuleList()
        for i in range(self.Order):
            self.lin_in.append(Linear(num_features, hidden, dtype=torch.float))
            self.hgc.append(HiGCN_prop(K=10, alpha=0.5, Order=self.Order))
        self.lin_out = Linear(hidden * self.Order, num_classess, dtype=torch.float)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_out.reset_parameters()
        for conv in self.lin_in:
            conv.reset_parameters()
        for conv in self.hgc:
            conv.reset_parameters()

    def forward(self, x, HL) :
        x_concat = torch.tensor([]).to(device)
        x = x.to(torch.float)
        for i in range(self.Order):
            xx = F.dropout(x, p=self.dropout, training=self.training)
            xx = self.lin_in[i](xx)
            if self.dprate > 0.0:
                xx = F.dropout(xx, p=self.dprate, training=self.training)
            xx = self.hgc[i](xx, HL[i + 1])
            x_concat = torch.concat((x_concat, xx), 1)

        x_concat = self.lin_out(x_concat)
        return self.nn(x_concat)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

class HiGCN(torch.nn.Module):
    def __init__(self, max_petal_dim, num_features, num_layers, hidden, num_classes,
                 dropout_rate=0.5):
        super(HiGCN, self).__init__()
        self.order = max_petal_dim
        self.dropout_rate = dropout_rate
        self.conv1 = HiGCNConv(num_features, self.order, hidden, hidden, dropout_rate,
            Sequential(
                Linear(hidden, hidden, dtype=torch.float),
                BN(hidden),
                torch.nn.ReLU(),
                Linear(hidden, hidden, dtype=torch.float),
                BN(hidden),
                torch.nn.ReLU(),
            ))
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                HiGCNConv(hidden, self.order, hidden, hidden, dropout_rate,
                    Sequential(
                        Linear(hidden, hidden, dtype=torch.float),
                        BN(hidden),
                        torch.nn.ReLU(),
                        Linear(hidden, hidden, dtype=torch.float),
                        BN(hidden),
                        torch.nn.ReLU(),
                    ))
            )
        self.lin1 = Linear(hidden, hidden, dtype=torch.float)
        self.lin2 = Linear(hidden, num_classes, dtype=torch.float)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, HL, batch = data.x, data.HL, data.batch
        x = self.conv1(x, HL)
        for conv in self.convs:
            x = conv(x, HL)
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
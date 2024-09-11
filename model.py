# from gincov import GINConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, k_hop_subgraph
from bondedgeconstruction import smiles_to_data, collate_with_circle_index
import copy
import random
# random.seed(42)
from KAN import KAN
import torch
from torch.nn import Module, Parameter, Linear, BatchNorm1d, ReLU, Linear
class SetRep(Module):
    def __init__(
        self,
        n_hidden_sets: int,
        n_elements: int,
        d: int,
        n_out_channels: int = 32,
    ):
        super(SetRep, self).__init__()

        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        self.d = d
        self.n_out_channels = n_out_channels

        # Ensure self.d == n_hidden_sets * n_elements for consistency
        self.Wc = Parameter(
            torch.FloatTensor(self.d, self.n_hidden_sets * self.n_elements)
        )

        self.bn = BatchNorm1d(self.n_hidden_sets)
        self.fc1 = Linear(self.n_hidden_sets, self.n_out_channels)
        self.relu = ReLU()

        # Init weights
        self.Wc.data.normal_()

    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc))

        # Ensure correct dimensions
        batch_size = t.size(0)
        feature_dim = t.size(1)
        assert feature_dim == self.n_hidden_sets * self.n_elements, "Feature dimension mismatch"

        t = t.view(batch_size, self.n_elements, self.n_hidden_sets)
        t, _ = torch.max(t, dim=1)
        t = self.bn(t)
        t = self.fc1(t)
        out = self.relu(t)

        return out




from typing import List, Optional
import torch
import lightning.pytorch as pl


from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch_geometric.nn import GIN
from torch_geometric.nn.pool import global_mean_pool

def set_seed(seed_value=42):
    random.seed(seed_value)
    # np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GINConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GINConv

import pooling as Pooling

# from your_module import make_gin_conv, GConv_OGSN
from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d, Dropout
from torch_geometric.nn import GINEConv, MLP, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN


class GINE(BasicGNN):
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(
        self, in_channels: int, out_channels: int, edge_dim: int, **kwargs
    ) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, train_eps=True, edge_dim=edge_dim, **kwargs)


class MLP(Module):
    def __init__(self, n_input_channels, n_hidden_channels, n_out_channels):
        super().__init__()

        self.layers = Sequential(
            BatchNorm1d(n_input_channels),
            Linear(n_input_channels, n_hidden_channels),
            ReLU(),
            # Linear(n_hidden_channels, n_hidden_channels),
            # ReLU(),
            Linear(n_hidden_channels, n_out_channels),
        )

    def forward(self, x):
        return self.layers(x)
class GNNRegressor(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_layers,
        in_edge_channels,
        n_hidden_channels: Optional[List] = None,
        gnn: Optional[torch.nn.Module] = None,
    ):
        super(GNNRegressor, self).__init__()

        if n_hidden_channels is None or len(n_hidden_channels) < 2:
            n_hidden_channels = [32, 16]

        self.in_edge_channels = in_edge_channels
        self.gnn = gnn

        if self.gnn is None:
            if self.in_edge_channels > 0:
                self.gnn = GINE(
                    in_channels,
                    n_hidden_channels[0],
                    num_layers,
                    edge_dim=in_edge_channels,
                    jk="cat",
                )
            else:
                self.gnn = GIN(in_channels, n_hidden_channels[0], num_layers)

        self.mlp = MLP(n_hidden_channels[0], n_hidden_channels[1], 1)

    def forward(self, batch):
        if self.in_edge_channels > 0:
            out = self.gnn(
                batch.x.float(), batch.edge_index, edge_attr=batch.edge_attr.float()
            )
        else:
            out = self.gnn(batch.x.float(), batch.edge_index)

        t = global_mean_pool(out, batch.batch)
        out = self.mlp(t)
        return out.squeeze(1)
class CCT(nn.Module):
    def __init__(self, num_features_xd=93, dropout=0.5, aug_ratio=0.4):
        super(CCT, self).__init__()

        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=100)

        self.fc = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.linear = nn.Sequential(
            nn.Linear(200, 512),
            nn.Linear(512, 256)
        )

        self.fc_g = nn.Sequential(
            nn.Linear(num_features_xd * 10 , 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        self.fc_g1 = nn.Sequential(
            nn.Linear(43 * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256 * 1, 1)
        )
        self.fc_final1 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256 * 1, 1)
        )
        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.conv3 = GINConv(nn.Linear(43, 43))
        self.conv4 = GINConv(nn.Linear(43, 43 * 10))
        self.relu = nn.ReLU()
        self.aug_ratio = aug_ratio
        self.linear2 = nn.Linear(300, 256)

        r_prime = feature_num = embed_dim = 256
        self.max_walk_len = 3
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.W = torch.nn.Parameter(torch.randn(embed_dim, feature_num), requires_grad=True)
        self.Wv = torch.nn.Parameter(torch.randn(r_prime, embed_dim), requires_grad=True)
        self.Ww = torch.nn.Parameter(torch.randn(r_prime, r_prime), requires_grad=True)
        self.Wg = torch.nn.Parameter(torch.randn(r_prime, r_prime), requires_grad=True)
        self.linea1 = nn.Linear(57, 93)
        self.linea2 = nn.Linear(43, 93)

        # 实例化 GConv_OGSN
        #self.gconv_ogsn = GConv_OGSN(id_dim=num_features_xd, input_dim=num_features_xd, hidden_dim=256, num_layers=3)


        self.additional_linear = nn.Linear(1 * 3, 1)


        self.pooling = Pooling.UOTPooling(
            dim=num_features_xd * 10,
            num=4,
            rho=None,
            same_para=False,
            p0="fixed",
            q0="fixed",
            eps=1e-18,
            a1=None,
            a2=None,
            a3=None,
            f_method='badmm-e',
        )

        self.pooling1 = Pooling.UOTPooling(
            dim=num_features_xd * 10,
            num=4,
            rho=None,
            same_para=False,
            p0="fixed",
            q0="fixed",
            eps=1e-18,
            a1=None,
            a2=None,
            a3=None,
            f_method=str,
        )
        self.pooling2 = Pooling.UOTPooling(
            dim=43 * 10,
            num=4,
            rho=None,
            same_para=False,
            p0="fixed",
            q0="fixed",
            eps=1e-18,
            a1=None,
            a2=None,
            a3=None,
            f_method=str,
        )





    def forward(self, data, x, edge_index, batch, a, edge, c, smi_em=None):
        x_g = self.relu(self.conv1(x, edge_index))
        x_g = self.relu(self.conv2(x_g, edge_index))

        x_g = self.pooling(x_g, batch)

        # print(gmp(x_g, batch).shape)
        # print(gap(x_g, batch).shape)


        x_g = self.fc_g(x_g)

        z = self.fc_final(x_g)

        x_g1 = self.relu(self.conv3(a, edge))
        x_g1 = self.relu(self.conv4(x_g1, edge))
        #x_g1=self.pooling2(x_g1,c)
        x_g1 = torch.cat([gmp(x_g1, c), gap(x_g1, c)], dim=1)
        x_g1 = self.fc_g1(x_g1)
        z1 = self.fc_final1(x_g1)


        #gconv_z, gconv_g = self.gconv_ogsn(x, edge_index, batch)


        inp_size = x.size(1)
        k = 3
        out_size = 7

        #z_grid_transformed = self.KAN(z, k, inp_size, out_size)



        #z1_grid_transformed = self.KAN(z1, k, inp_size, out_size)




        return z,  x_g, x_g1  ,z1

    @staticmethod
    def softmax(input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        soft_max_2d = F.softmax(trans_input.contiguous().view(-1, trans_input.size()[-1]), dim=1)
        return soft_max_2d.view(*trans_input.size()).transpose(axis, len(input_size) - 1)








"""
New module for GNN models.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn
import torchvision.models

import graph_nets as gn

from graph_utils import data_from_graph_maker
from graph_utils import cross_graph_ei_maker
from graph_utils import data_to_graph_simple, data_to_graph_double


class GraphModel(torch.nn.Module):
    def __init__(self, f_dict):
        super().__init__()
        # maybe define different attributes for a simple-input GM and a double-
        # input GM.
        f_e, f_x, f_u, h, f_out = self.get_features(f_dict)
        self.fe = f_e
        self.fx = f_x
        self.fu = f_u
        self.h = h
        self.fout = f_out
        self.GPU = False

        self.data_from_graph = data_from_graph_maker()

    def get_features(self, f_dict):
        """
        Gets the input and output features for graph processing.
        """
        f_e = f_dict['f_e']
        f_x = f_dict['f_x']
        f_u = f_dict['f_u']
        h = f_dict['h']
        f_out = f_dict['f_out']
        return f_e, f_x, f_u, h, f_out

    def cuda(self):
        super().cuda()
        self.GPU = True
        self.data_from_graph = data_from_graph_maker(cuda=True)

    def cpu(self):
        super().cpu()
        self.GPU = False
        self.data_from_graph = data_from_graph_maker(cuda=False)

class GraphModelSimple(GraphModel):
    """Single-input graph model"""
    def __init__(self, f_dict):
        
        super().__init__(f_dict)

    def forward(self, data):

        return data_to_graph_simple(data)

class GraphModelDouble(GraphModel):
    """Double-input graph model"""
    def __init__(self, f_dict):
        
        super().__init__(f_dict)

    def forward(self, data):

        return data_to_graph_double(data)

# deep sets

class DeepSet(GraphModelSimple):
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super(DeepSet, self).__init__(f_dict)
        mlp_fn = gn.mlp_fn(mlp_layers)
        self.deepset = gn.DeepSet(mlp_fn, self.fx, self.h, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, _, _, _, batch = self.data_from_graph(graph)
        return self.deepset(x, batch)

class DeepSet_modU(GraphModelSimple):
    def __init__(self,
                 mlp_layers,
                 N,
                 udim,
                 f_dict):
        
        super().__init__(f_dict)
        self.udim = udim
        mlp_fn = gn.mlp_fn(mlp_layers)
        self.deepset = gn.DeepSet_modU(mlp_fn,
                                  self.fx,
                                  self.h,
                                  self.udim,
                                  self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, _, _, _, batch = self.data_from_graph(graph)
        return self.deepset(x, batch)

class DeepSetPlus(GraphModelSimple):
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N # we allow multiple rounds
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.deepset = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, self.fu, mlp_fn, self.fx),
            gn.DS_GlobalModel(self.fx, self.fu, mlp_fn, self.fu))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, _, _, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, u = self.deepset(x, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class DeepSetPlus_modU(GraphModelSimple):

    def __init__(self,
                 mlp_layers,
                 N,
                 udim,
                 f_dict):

        super().__init__(f_dict)
        self.N = N # we allow multiple rounds
        self.udim = udim
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.proj_u = torch.nn.Linear(self.fu, self.udim)

        self.deepset = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, self.udim, mlp_fn, self.fx),
            gn.DS_GlobalModel_modU(self.fx, self.udim, mlp_fn, self.udim))

        self.mlp = mlp_fn(self.udim, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, _, _, u, batch = self.data_from_graph(graph)
        u = self.proj_u(u)
        out_list = []
        for i in range(self.N):
            x, u = self.deepset(x, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class DeepSetPlus_A(GraphModelSimple):
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N # we allow multiple rounds
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.deepset = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, self.fu, mlp_fn, self.fx),
            gn.DS_GlobalModel_A(self.fx, self.fu, self.h, mlp_fn, self.fu))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, _, _, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, u = self.deepset(x, u, batch)
            out_list.append(self.mlp(u))
        return out_list

# GNNs

# Node only

class N_GNN(GraphModelSimple):
    """
    Node-GNN. (No edge features)
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.N_GNN(
            gn.EdgeModel_NoMem(self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, edge_index, _, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, u = self.gnn(x, edge_index, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class N_GNN_A(GraphModelSimple):
    """
    Node-GNN, with attention in node and global aggregation.
    """
    def __init__(self, 
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.N_GNN(
            gn.EdgeModel_NoMem(self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel_A(self.fe, self.fx, self.fu, self.h, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly_A(self.fx, self.fu, self.h, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, edge_index, _, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, u = self.gnn(x, edge_index, u, batch)
            out_list.append(self.mlp(u))
        return out_list

# Edge models

# NGI

class GNN_NAgg_NGI(GraphModelSimple):
    """
    MPGNN with No Global Information.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):

        super().__init__(f_dict)
        self.N = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN_NGI(
            gn.EdgeModel_NGI(self.fe, self.fx, mlp_fn, self.fe),
            gn.NodeModel_NGI(self.fe, self.fx, mlp_fn, self.fx),
            mlp_fn(self.fx, self.fx)
            )
        self.mlp = mlp_fn(self.fu, self.fout)
    
    def forward(self, data):
        (graph,) = super().forward(data)
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, batch)
            out_list.append(self.mlp(u))
        return out_list

class GNN_NAgg(GraphModelSimple):
    """
    Edge-feature GNN, with node aggregation in the global model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list

# version with expanded u vector

class GNN_NAgg_modU(GraphModelSimple):
    """
    Same as previous one, but u is embedded in a higher-dimensional vector.
    x is also embedded in a higher-dimensional vector.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 udim,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        self.udim = udim
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.proj_u = torch.nn.Linear(self.fu, self.udim)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.udim, mlp_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.udim, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly_modU(self.fx,
                                         self.udim,
                                         mlp_fn,
                                         self.udim))

        self.mlp = mlp_fn(self.udim, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)

        x, edge_index, e, u, batch = self.data_from_graph(graph)
        u = self.proj_u(u) # project u
        out_list = []
        
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list


class GNN_NAgg_A(GraphModelSimple):
    """
    Edge-feature GNN, with node aggregation in the global model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel_A(self.fe, self.fx, self.fu, self.h, mlp_fn, self.fx),
            gn.GlobalModel_NodeOnly_A(self.fx, self.fu, self.h, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class GNN_NEAgg(GraphModelSimple):
    """
    Edge-feature GNN, with node aggregation in the global model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu, mlp_fn, self.fx),
            gn.GlobalModel(self.fe, self.fx, self.fu, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list

class GNN_NEAgg_A(GraphModelSimple):
    """
    Edge-feature GNN, with node aggregation in the global model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        mlp_fn = gn.mlp_fn(mlp_layers)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu, mlp_fn, self.fe),
            gn.NodeModel_A(self.fe, self.fx, self.fu, self.h, mlp_fn, self.fx),
            gn.GlobalModel_A(self.fe, self.fx, self.fu, self.h, mlp_fn, self.fx))
        self.mlp = mlp_fn(self.fu, self.fout)

    def forward(self, data):

        (graph,) = super().forward(data)
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        out_list = []
        for i in range(self.N):
            x, e, u = self.gnn(x, edge_index, e, u, batch)
            out_list.append(self.mlp(u))
        return out_list


class GCN(GraphModelSimple):
    def __init__(self, mlp_layers, N, f_dict):

        super().__init__(f_dict)
        self.proj = nn.Linear(self.fx, self.h)
        self.gnn = torch_geometric.nn.GCNConv(
            in_channels=self.h,
            out_channels=self.h,
        )
        mlp_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.mlp = mlp_fn(self.h, self.fout)

    def forward(self, data):
        (graph,) = super().forward(data)
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        x = self.proj(x)
        out_list = []
        for i in range(self.N):
            x = self.gnn(x, edge_index)
            u = gn.scatter_add(x, batch, dim=0)
            out_list.append(self.mlp(u))
        return out_list


# other models

class TGNN(GraphModelSimple):
    """
    Transformer-GNN, the nodes do a transformer-style aggregation on their
    neighbours.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N  = N
        mlp_fn = gn.mlp_fn(mlp_layers)
        self.tgnn = gn.MultiHeadAttention(self.fx, 8, self.h)        
        self.agg = gn.SumAggreg()

    def forward(self, data):

        (graph,) = super().forward(data)
        x, edge_index, e, u, batch = self.data_from_graph(graph)
        # out list ?
        for _ in range(self.N):
            x = self.tgnn(x, edge_index, batch)
        return self.agg(x, batch)

# Double-input graph models

class Parallel(GraphModelDouble):
    """
    Parallel processing of inputs.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):

        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'MPGNN'

        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu,model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu,model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu,model_fn, self.fx))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu,model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu,model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu,model_fn, self.fx))
        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, u1, batch1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, u2, batch2)
            out_list.append(self.mlp(torch.cat([u1, u2], 1)))
        return out_list

class Parallel_NGI(GraphModelDouble):
    """
    Parallel processing of inputs, No Global Information version.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):

        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'MPGNN'

        self.gnn1 = gn.GNN_NGI(
            gn.EdgeModel_NGI(self.fe, self.fx, model_fn, self.fe),
            gn.NodeModel_NGI(self.fe, self.fx, model_fn, self.fx),
            model_fn(self.fx, self.fx))
        self.gnn2 = gn.GNN_NGI(
            gn.EdgeModel_NGI(self.fe, self.fx, model_fn, self.fe),
            gn.NodeModel_NGI(self.fe, self.fx, model_fn, self.fx),
            model_fn(self.fx, self.fx))
        self.mlp = model_fn(2 * self.fx, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, batch1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, batch2)
            out_list.append(self.mlp(torch.cat([u1, u2], 1)))
        return out_list

class ParallelRDS(GraphModelDouble):
    """
    RDS version of Parallel model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'RDS'

        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, self.fu, model_fn, self.fu))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, self.fu, model_fn, self.fu))
        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, u1 = self.gnn1(x1, u1, batch1)
            x2, u2 = self.gnn2(x2, u2, batch2)
            out_list.append(self.mlp(torch.cat([u1, u2], 1)))
        return out_list

class ParallelGCN(GraphModelDouble):
    """
    Parallel processing of inputs.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):

        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'MPGNN'
        self.proj1 = nn.Linear(self.fx, self.h)
        self.proj2 = nn.Linear(self.fx, self.h)

        self.gnn1 = torch_geometric.nn.GCNConv(
            in_channels=self.h,
            out_channels=self.h,
        )
        self.gnn2 = torch_geometric.nn.GCNConv(
            in_channels=self.h,
            out_channels=self.h,
        )
        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        x1 = self.proj1(x1)
        x2 = self.proj1(x2)
        for _ in range(self.N):
            x1 = self.gnn1(x1, ei1)
            u1 = gn.scatter_add(x1, batch1, dim=0)
            x2 = self.gnn2(x2, ei2)
            u2 = gn.scatter_add(x2, batch2, dim=0)
            out_list.append(self.mlp(torch.cat([u1, u2], 1)))
        return out_list

class ParallelDS(GraphModelDouble):
    """
    DS version of Parallel model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'DS'

        self.ds1 = gn.DeepSet(model_fn, self.fx, self.h, self.fx)
        self.ds2 = gn.DeepSet(model_fn, self.fx, self.h, self.fx)

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []
        u1 = self.ds1(x1, batch1)
        u2 = self.ds2(x2, batch2)
        out_list.append(self.mlp(torch.cat([u1, u2], 1)))
        
        return out_list

class RecurrentGraphEmbedding(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'MPGNN'

        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, self.fu,model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, self.fu,model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, self.fu,model_fn, self.fx))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, 2 * self.fu,model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, 2 * self.fu,model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, 2 * self.fu,model_fn, self.fx))
        self.mlp = model_fn(self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class RecurrentGraphEmbeddingGCN(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):

        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'GCN'

        self.gnn1 = torch_geometric.nn.GCN(
            in_channels=self.fx,
            out_channels=self.fx,
        )
        self.gnn2 = torch_geometric.nn.GCN(
            in_channels=self.fx,
            out_channels=self.fx,
        )
        self.mlp = model_fn(self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1 = self.gnn1(x1, ei1)
            x2 = self.gnn2(x2, ei2)
            u2 = gn.scatter_add(x2, batch2, dim=0)
            out_list.append(self.mlp(u2))
        return out_list


class AlternatingSimple(GraphModelDouble):
    """
    Simple version of the Alternating model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        """
        Simpler version of the alternating model. In this model there is no
        encoder network, we only have 1 layer of GNN on each processing step.

        We condition on the output global embedding from the processing on the
        previous graph, and we only condition the node computations since there
        are less nodes than edges (this is a choice that can be discussed).

        We aggregate nodes with attention in the global model.

        We use the same gnn for processing both inputs.
        In this model, since we may want to chain the passes, we let the number
        of input features unchanged.
        """
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        """
        Forward pass. We alternate computing on 1 graph and then on the other.
        We initialize the conditioning vector at 0.
        At each step we concatenate the global vectors to the node vectors.
        """
        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            # we can do N passes of this
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDouble(GraphModelDouble):
    """
    Different gnns inside.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.NodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.GlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn1(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingSimpleRDS(GraphModelDouble):
    """
    RDS layer inside.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'RDS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, 2 * self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, u1 = self.gnn(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn(x2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingSimplev2(GraphModelDouble):
    """
    Projects the input features into a higher-dimensional space.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn = gn.GNN(
            gn.EdgeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, 2 * self.h, model_fn, self.h))
        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoublev2(GraphModelDouble):
    """
    Projects the input features into a higher-dimensional space.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, 2 * self.h, model_fn, self.h))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, 2 * self.h, model_fn, self.h))
        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn1(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoubleRDS(GraphModelDouble):
    """
    Recurrent DeepSet version of the AlternatingDouble model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'RDS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, 2 * self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, 2 * self.fu, model_fn, self.fu))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, 2 * self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, u1 = self.gnn1(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn2(x2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoubleGCN(GraphModelDouble):
    """
    Different gnns inside.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):

        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn1 = torch_geometric.nn.GCNConv(
            in_channels=self.fx,
            out_channels=self.fx,
        )
        self.gnn2 = torch_geometric.nn.GCNConv(
            in_channels=self.fx,
            out_channels=self.fx,
        )

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = gn.scatter_add(x1, batch1, dim=0)
            x1 = self.gnn1(x1, edge_index1)
            u2 = gn.scatter_add(x2, batch2, dim=0)
            x2 = self.gnn2(x2, edge_index2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoubleRDSv2(GraphModelDouble):
    """
    Recurrent DeepSet version of the AlternatingDouble model with linear
    projection on h dimensions.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'RDS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.h, 2 * self.h, model_fn, self.h),
            gn.DS_GlobalModel(self.h, 2 * self.h, model_fn, self.h))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.h, 2 * self.h, model_fn, self.h),
            gn.DS_GlobalModel(self.h, 2 * self.h, model_fn, self.h))

        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, u1 = self.gnn1(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn2(x2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoubleGCNv2(GraphModelDouble):
    """
    Projects the input features into a higher-dimensional space.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):

        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'GCN'

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn1 = torch_geometric.nn.GCNConv(
            in_channels=self.fx,
            out_channels=self.fx,
        )
        self.gnn2 = torch_geometric.nn.GCNConv(
            in_channels=self.fx,
            out_channels=self.fx,
        )
        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            # u1 = torch.cat([u1, u2], 1)
            x1 = self.gnn1(x1, edge_index1)
            # u2 = torch.cat([u2, u1], 1)
            x2 = self.gnn2(x2, edge_index2)
            u1 = gn.scatter_add(x1, batch1, dim=0)
            u2 = gn.scatter_add(x2, batch2, dim=0)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class RecurrentGraphEmbeddingv2(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        self.component = 'MPGNN'
        model_fn = gn.mlp_fn(mlp_layers)

        self.proj = torch.nn.Linear(self.fx, self.h)

        self.gnn1 = gn.GNN(
            gn.EdgeModel(self.h, self.h, self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, self.h, model_fn, self.h))
        self.gnn2 = gn.GNN(
            gn.EdgeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.NodeModel(self.h, self.h, 2 * self.h, model_fn, self.h),
            gn.GlobalModel_NodeOnly(self.h, 2 * self.h, model_fn, self.h))
        self.mlp = model_fn(self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class RecurrentGraphEmbeddingRDS(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'RDS'

        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, self.fu, model_fn, self.fu))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.fx, 2 * self.fu, model_fn, self.fx),
            gn.DS_GlobalModel(self.fx, 2 * self.fu, model_fn, self.fu))
        self.mlp = model_fn(self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, u1 = self.gnn1(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn2(x2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class RecurrentGraphEmbeddingRDSv2(GraphModelDouble):
    """
    Cast to h dimensions, and use RDS layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'RDS'

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.gnn1 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.h, self.h, model_fn, self.h),
            gn.DS_GlobalModel(self.h, self.h, model_fn, self.h))
        self.gnn2 = gn.DeepSetPlus(
            gn.DS_NodeModel(self.h, 2 * self.h, model_fn, self.h),
            gn.DS_GlobalModel(self.h, 2 * self.h, model_fn, self.h))
        self.mlp = model_fn(self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)

        # project everything in self.h dimensions
        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []
        for _ in range(self.N):
            x1, u1 = self.gnn1(x1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, u2 = self.gnn2(x2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class ResAlternatingDouble(GraphModelDouble):
    """
    Different gnns inside.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'MPGNN'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.gnn1 = gn.GNN(
            gn.ResEdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.ResNodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.ResGlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))
        self.gnn2 = gn.GNN(
            gn.ResEdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.ResNodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.ResGlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fu))

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            u1 = torch.cat([u1, u2], 1)
            x1, e1, u1 = self.gnn1(x1, edge_index1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            print('u {}'.format(u2.shape))
            print('e {}'.format(e2.shape))
            print('x {}'.format(x2.shape))
            x2, e2, u2 = self.gnn2(x2, edge_index2, e2, u2, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class ResRecurrentGraphEmbedding(GraphModelDouble):
    """
    Simplest double input graph model.
    We use the full GNN with node aggreg as a GNN layer.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        self.component = 'MPGNN'
        model_fn = gn.mlp_fn(mlp_layers)

        self.gnn1 = gn.GNN(
            gn.ResEdgeModel(self.fe, self.fx, self.fu, model_fn, self.fe),
            gn.ResNodeModel(self.fe, self.fx, self.fu, model_fn, self.fx),
            gn.ResGlobalModel_NodeOnly(self.fx, self.fu, model_fn, self.fx))
        self.gnn2 = gn.GNN(
            gn.ResEdgeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fe),
            gn.ResNodeModel(self.fe, self.fx, 2 * self.fu, model_fn, self.fx),
            gn.ResGlobalModel_NodeOnly(self.fx, 2 * self.fu, model_fn, self.fx))
        self.mlp = model_fn(self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        for _ in range(self.N):
            x1, e1, u1 = self.gnn1(x1, ei1, e1, u1, batch1)
            u2 = torch.cat([u2, u1], 1)
            x2, e2, u2 = self.gnn2(x2, ei2, e2, u2, batch2)
            out_list.append(self.mlp(u2))
        return out_list

class AlternatingDoubleDS(GraphModelDouble):
    """
    Recurrent DeepSet version of the AlternatingDouble model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'DS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.ds1 = gn.DeepSet(model_fn, self.fu + self.fx, self.h, self.fx)
        self.ds2 = gn.DeepSet(model_fn, self.fu + self.fx, self.h, self.fx)

        self.mlp = model_fn(2 * self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        out_list = []

        for _ in range(self.N):
            x1_ = torch.cat([x1, u2[batch2]], 1)
            u1 = self.ds1(x1_, batch1)
            x2_ = torch.cat([x2, u1[batch1]], 1)
            u2 = self.ds2(x2_, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class AlternatingDoubleRSv2(GraphModelDouble):
    """
    Recurrent DeepSet version of the AlternatingDouble model.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        model_fn = gn.mlp_fn(mlp_layers)
        self.N = N
        self.component = 'DS'
        # f_e, f_x, f_u, f_out = self.get_features(f_dict)

        self.proj = torch.nn.Linear(self.fx, self.h)
        self.ds1 = gn.DeepSet(model_fn, 2 * self.h, self.h, self.h)
        self.ds2 = gn.DeepSet(model_fn, 2 * self.h, self.h, self.h)

        self.mlp = model_fn(2 * self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, edge_index1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, edge_index2, e2, u2, batch2 = self.data_from_graph(graph2)

        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []

        for _ in range(self.N):
            x1_ = torch.cat([x1, u2[batch2]], 1)
            u1 = self.ds1(x1_, batch1)
            x2_ = torch.cat([x2, u1[batch1]], 1)
            u2 = self.ds2(x2_, batch2)

            out_list.append(self.mlp(torch.cat([u1, u2], 1)))

        return out_list

class RecurrentGraphEmbeddingDS(GraphModelDouble):
    """
    baseline.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'DS'

        self.ds1 = gn.DeepSet(model_fn, self.fx, self.h, self.fu)
        self.ds2 = gn.DeepSet(model_fn, self.fu + self.fx, self.h, self.fu)
        self.mlp = model_fn(self.fu, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)
        out_list = []
        u1 = self.ds1(x1, batch1)
        x2 = torch.cat([x2, u1[batch1]], 1)
        u2 = self.ds2(x2, batch2)
        out_list.append(self.mlp(u2))
        return out_list

class RecurrentGraphEmbeddingDSv2(GraphModelDouble):
    """
    baseline.
    """
    def __init__(self,
                 mlp_layers,
                 N,
                 f_dict):
        
        super().__init__(f_dict)
        self.N = N
        model_fn = gn.mlp_fn(mlp_layers)
        self.component = 'DS'

        self.proj = torch.nn.Linear(self.fx, self.h)

        self.ds1 = gn.DeepSet(model_fn, self.h, self.h, self.h)
        self.ds2 = gn.DeepSet(model_fn, 2 * self.h, self.h, self.h)
        self.mlp = model_fn(self.h, self.fout)

    def forward(self, data):

        graph1, graph2 = super().forward(data)

        x1, ei1, e1, u1, batch1 = self.data_from_graph(graph1)
        x2, ei2, e2, u2, batch2 = self.data_from_graph(graph2)

        x1 = self.proj(x1)
        e1 = self.proj(e1)
        u1 = self.proj(u1)
        x2 = self.proj(x2)
        e2 = self.proj(e2)
        u2 = self.proj(u2)

        out_list = []
        u1 = self.ds1(x1, batch1)
        x2 = torch.cat([x2, u1[batch1]], 1)
        u2 = self.ds2(x2, batch2)
        out_list.append(self.mlp(u2))
        return out_list

# Image models

class Resnet18(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = torchvision.models.resnet18()
        self.proj = nn.Linear(1000, 2)

    def forward(self, data):
        target, _ = data
        emb = self.net(target)
        return self.proj(emb)

class SiameseResnet18(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = torchvision.models.resnet18()
        self.proj = nn.Linear(2000, 2)

    def forward(self, data):
        target, ref, _ = data
        emb = torch.cat([self.net(target), self.net(ref)], dim=-1)
        return self.proj(emb)

# Graph model utilities

# model_list = [
#     DeepSetPlus,
#     GNN_NAgg,
#     DeepSet]

model_list = [
    DeepSet,
    DeepSetPlus,
    GCN,
    GNN_NAgg
]

model_list_double = [
    Parallel,
    ParallelRDS,
    ParallelGCN,
    ParallelDS,
]

model_list_imgs_simple = [
    'Resnet18',
]

model_list_imgs_double = [
    'SiameseResnet18',
]

model_names = [
    'Deep Set++ (0)',
    'Deep Set++, attention (1)',
    'Node GNN (2)',
    'Node GNN, attention (3)',
    'GNN, node aggreg (4)',
    'GNN, node aggreg, attention (5)',
    'GNN, node-edge aggreg (6)',
    'GNN, node-edge aggreg, attention (7)',
    'TGNN (8)',
    'Deep Set (9)'
]
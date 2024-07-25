from typing import List
import torch
import torch_geometric

import torch, torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import matmul
import scipy
import numpy as np

import ot

def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9, warmstart=None, **kwargs):
    # Convert input to torch tensors if they aren't already
    if len(a) == 0:
        a = torch.full((M.shape[0],), 1.0 / M.shape[0], dtype=M.dtype, device=M.device)
    if len(b) == 0:
        b = torch.full((M.shape[1],), 1.0 / M.shape[1], dtype=M.dtype, device=M.device)

    # Initialize data
    dim_a = len(a)
    dim_b = b.shape[0]

    n_hists = 0

    # Initialize u and v
    if warmstart is None:
        u = torch.ones(dim_a, dtype=M.dtype, device=M.device) / dim_a
        v = torch.ones(dim_b, dtype=M.dtype, device=M.device) / dim_b
    else:
        u, v = torch.exp(warmstart[0]), torch.exp(warmstart[1])

    K = torch.exp(M / (-reg))
    Kp = (1 / a).reshape(-1, 1) * K
    for ii in range(numItermax):
        uprev = u.clone()
        vprev = v.clone()
        KtransposeU = torch.matmul(K.T, u)
        v = b / KtransposeU
        u = 1.0 / torch.matmul(Kp, v)
        if torch.any(KtransposeU == 0) or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            # Numerical errors encountered
            u = uprev
            v = vprev
            break

        if ii % 10 == 0:
            tmp2 = torch.einsum('i,ij,j->j', u, K, v)
            err = torch.norm(tmp2 - b)  # Violation of marginal
            if err < stopThr:
                break
    
    return u.reshape((-1, 1)) * K * v.reshape((1, -1))

def sinkhorn_knopp_batch(a, b, M, reg, numItermax=1000, stopThr=1e-6, warmstart=None):
    n_batch, dim_a, dim_b = M.shape

    if not warmstart:
        u = torch.ones((n_batch, dim_a), dtype=M.dtype, device=M.device) / dim_a
        v = torch.ones((n_batch, dim_b), dtype=M.dtype, device=M.device) / dim_b
    else:
        u, v = torch.sqrt(a), torch.sqrt(b)

    K = torch.exp(M / (-reg))
    Kp = (1 / a.unsqueeze(-1)) * K

    for ii in range(numItermax):
        KtransposeU = torch.matmul(K.transpose(1,2), u.unsqueeze(2))
        v = b.unsqueeze(2) / KtransposeU
        u = 1.0 / torch.matmul(Kp, v)
        u = u.squeeze(2)
        v = v.squeeze(2)
        
        if ii % 10 == 0:
            tmp2 = torch.einsum('bi,bij,bj->bj', u, K, v)
            err = torch.norm(tmp2 - b, dim=-1)
            if torch.all(err < stopThr):
                break
    gamma = u.unsqueeze(-1) * K * v.unsqueeze(1)
    
    return gamma









class DistanceNet(torch.nn.Module):
    def __init__(self, num_input_features, gnn_num_layers=0, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.num_input_features = num_input_features
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embed_transform = nn.Sequential(
            nn.Linear(num_input_features+gnn_num_layers*hidden_dim, self.embedding_dim)
        )
        self.virtual_embedding = nn.Parameter(torch.rand(self.embedding_dim))

        self.conv = nn.ModuleList([GraphConv(in_channels=self.num_input_features, out_channels=self.hidden_dim)]+[GraphConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim) for _ in range(gnn_num_layers-1)])
        



    def forward(self, data):
        x_s, edge_index_s, edge_attr_s = data.x_s, data.edge_index_s, data.edge_attr_s
        x_t, edge_index_t, edge_attr_t = data.x_t, data.edge_index_t, data.edge_attr_t

        embedding_s = [x_s] 
        embedding_t = [x_t]
        for l in range(self.gnn_num_layers):
            embedding_s.append( self.conv[l](embedding_s[-1], edge_index_s.long()).relu() ) 
            embedding_t.append( self.conv[l](embedding_t[-1], edge_index_t.long()).relu() ) 
        
        embedding_s = torch.cat(embedding_s, dim=1)
        embedding_t = torch.cat(embedding_t, dim=1)
        

        embedding_s = torch_geometric.nn.global_add_pool(embedding_s, data.x_s_batch)
        embedding_t = torch_geometric.nn.global_add_pool(embedding_t, data.x_t_batch)

        embedding_s = self.embed_transform(embedding_s)
        embedding_t = self.embed_transform(embedding_t)
        # place them on hypersphere
        embedding_s = F.normalize(embedding_s, dim=-1)
        embedding_t = F.normalize(embedding_t, dim=-1)
        virtual = F.normalize(self.virtual_embedding, dim=-1)
        

        # compute edit cost for pairs of source and targets nodes
        npairs = data.x_s_batch[-1]+1
        edit_costs = [None for _ in range(npairs)]
        lin_nodemaps = [None for _ in range(npairs)]
        geds = torch.norm(embedding_s - embedding_t, dim=-1)
        
        return lin_nodemaps, edit_costs, geds









class GEDNet(torch.nn.Module):
    def __init__(self, num_input_features, gnn_num_layers=0, embedding_dim=64, hidden_dim=64, reg=0.1):
        super().__init__()
        self.num_input_features = num_input_features
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.reg = reg

        self.embed_transform = nn.Sequential(
            nn.Linear(num_input_features+gnn_num_layers*hidden_dim, self.embedding_dim)
        )
        self.virtual_embedding = nn.Parameter(torch.rand(self.embedding_dim))

        self.conv = nn.ModuleList([GraphConv(in_channels=self.num_input_features, out_channels=self.hidden_dim)]+[GraphConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim) for _ in range(gnn_num_layers-1)])
        



    def forward(self, data):
        reg = self.reg 

        x_s, edge_index_s, edge_attr_s = data.x_s, data.edge_index_s, data.edge_attr_s
        x_t, edge_index_t, edge_attr_t = data.x_t, data.edge_index_t, data.edge_attr_t

        embedding_s = [x_s] 
        embedding_t = [x_t]
        for l in range(self.gnn_num_layers):
            embedding_s.append( self.conv[l](embedding_s[-1], edge_index_s.long()).relu() ) 
            embedding_t.append( self.conv[l](embedding_t[-1], edge_index_t.long()).relu() ) 
        
        embedding_s = torch.cat(embedding_s, dim=1)
        embedding_t = torch.cat(embedding_t, dim=1)

        embedding_s = self.embed_transform(embedding_s)
        embedding_t = self.embed_transform(embedding_t)
        # place them on hypersphere
        embedding_s = F.normalize(embedding_s, dim=-1)
        embedding_t = F.normalize(embedding_t, dim=-1)
        virtual = F.normalize(self.virtual_embedding, dim=-1)
        

        # compute edit cost for pairs of source and targets nodes
        npairs = data.x_s_batch[-1]+1
        edit_costs = [None for _ in range(npairs)]
        lin_nodemaps = [None for _ in range(npairs)]
        geds = torch.zeros(npairs, device=x_s.device)
        
        maxn = torch.max(data.len_s)

        edit_costs2 = torch.ones((npairs, maxn, maxn), device=embedding_s.device)*1000.0
        hist1 = torch.zeros((npairs, maxn), device=embedding_s.device)
        hist2 = torch.zeros((npairs, maxn), device=embedding_s.device)

        pos_s_st = 0
        pos_t_st = 0
        for npair in range(npairs):
            # compute distances
            edit_costs[npair] = torch.cdist(
                embedding_s[pos_s_st:pos_s_st+data.len_s[npair]], 
                torch.cat( (embedding_t[pos_t_st:pos_t_st+data.len_t[npair]], virtual.unsqueeze(0)))
            )
            n, m = edit_costs[npair].shape
            m-=1
            assert n >= m
            edit_costs2[npair, :n, :m] = edit_costs[npair][:, :m]
            edit_costs2[npair, :n, m:n] = edit_costs[npair][:, m:]
            edit_costs2[npair, n:, n:] = 0.0

            hist1[npair, :] = 1 
            hist2[npair, :] = 1

            if reg == 0.0:
                with torch.no_grad():
                    lin_nodemaps[npair] = ot.emd(torch.ones(n, device=edit_costs2.device), torch.ones(n, device=edit_costs2.device), edit_costs2[npair, :n, :n]).detach()
                geds[npair] = torch.sum(lin_nodemaps[npair] * edit_costs2[npair, :n, :n])
                
            

            pos_s_st += data.len_s[npair]
            pos_t_st += data.len_t[npair]

        if reg > 0.0:
            with torch.no_grad():
                lin_nodemaps2 = sinkhorn_knopp_batch(hist1, hist2, edit_costs2, reg=reg, numItermax=8).detach()
            geds = torch.sum(lin_nodemaps2 * edit_costs2, dim=[1,2])

        geds2 = geds / (data.len_s + data.len_t)
        return lin_nodemaps, edit_costs, geds2









class GraphConv(MessagePassing):
    def __init__(
            self,
            in_channels,
            out_channels,
            aggr: str = 'add',
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_rel = torch_geometric.nn.dense.linear.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = torch_geometric.nn.dense.linear.Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = self.lin_rel(out)

        x_r = x
        out += self.lin_root(x_r)

        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)



import os, sys
import numpy as np
from timeit import default_timer as timer
import argparse
import scipy
import math

import torch, torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, WLConv, GCNConv, GATConv, global_mean_pool

import torch.autograd.profiler as profiler


INF = 10000.0





class GEDModel(torch.nn.Module):
    def __init__(self, num_input_features, noise=0.0, gnn_num_layers=0, embedding_dim=64, hidden_dim=0):
        super(GEDModel, self).__init__()
        self.num_input_features = num_input_features+1 #add 1 for dummy
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = self.num_input_features 
        if hidden_dim > 0:
            self.hidden_dim = hidden_dim
        self.noise = noise
        self.embedding_dim = embedding_dim


        self.enlarge = nn.Parameter(torch.eye(num_input_features, num_input_features+1))

        self.embed_transform = nn.Sequential(
            nn.Linear(self.num_input_features+self.hidden_dim, self.embedding_dim)
        )
        self.virtual_embedding = nn.Parameter(0.01*torch.rand(self.num_input_features+self.hidden_dim))
        with torch.no_grad():
            tmp = 0.01*torch.rand((self.embedding_dim, self.num_input_features+self.hidden_dim))
            tmp[:,0:self.hidden_dim] = tmp[:,0:self.hidden_dim] + torch.eye(self.embedding_dim, self.hidden_dim)
            self.embed_transform[0].weight = torch.nn.Parameter( tmp )
            self.embed_transform[0].bias.fill_(0.0)
            self.virtual_embedding[self.hidden_dim-1] += 1.0
        

        self.conv = nn.ModuleList([GATConv(in_channels=self.num_input_features, out_channels=self.hidden_dim, dropout=0.1)]+[GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, dropout=0.1) for _ in range(gnn_num_layers-1)])
        

    
    def lsape(self, edit_costs):
        n, m = edit_costs.shape
        n-=1; m-=1
        
        noise = self.noise 
        if not self.training: noise = 0

        # python implementation of lsape
        C = torch.zeros((n+m, n+m)) 
        C[:n, :m] = edit_costs[:n, :m]
        C[n:, :m] = INF+0.0
        C[:n, m:] = INF+0.0
        C[(torch.arange(n,n+m), torch.arange(0,m))] = edit_costs[n, 0:m]
        C[(torch.arange(0,n), torch.arange(m,m+n))] = edit_costs[0:n, m]
        C[(torch.isfinite(C) == False)] = INF
        if self.noise > 0.0: C = C + torch.randn(C.shape) * noise
        q, p = scipy.optimize.linear_sum_assignment(C.detach().numpy())

        alignment = torch.zeros((n+1, m+1), dtype=int)
        q1 = np.minimum(q, n)
        p1 = np.minimum(p, m)
        alignment[q1, p1] = 1
        alignment[-1, -1] = 0
        return alignment.detach()


    def forward(self, data):
        x_s, edge_index_s, edge_attr_s = data.x_s, data.edge_index_s, data.edge_attr_s
        x_t, edge_index_t, edge_attr_t = data.x_t, data.edge_index_t, data.edge_attr_t

        embedding_s = [x_s @ self.enlarge]
        embedding_t = [x_t @ self.enlarge]
        for l in range(self.gnn_num_layers):
            embedding_s.append( self.conv[l](embedding_s[-1], edge_index_s.long(), edge_attr=edge_attr_s) )
            embedding_t.append( self.conv[l](embedding_t[-1], edge_index_t.long(), edge_attr=edge_attr_t) )
        
        # take first and last
        embedding_s = torch.cat([embedding_s[0], embedding_s[-1]], dim=-1)
        embedding_t = torch.cat([embedding_t[0], embedding_t[-1]], dim=-1)

        embedding_s = self.embed_transform(embedding_s)
        embedding_t = self.embed_transform(embedding_t)
        virtual = self.embed_transform(self.virtual_embedding)
        # place them on hypersphere
        embedding_s = F.normalize(embedding_s, dim=-1)
        embedding_t = F.normalize(embedding_t, dim=-1)
        virtual = F.normalize(virtual, dim=-1)
        

        # compute edit cost for pairs of source and targets nodes
        npairs = data.x_s_batch[-1]+1
        edit_costs = [None for _ in range(npairs)]
        lin_nodemaps = [None for _ in range(npairs)]
        geds = torch.zeros(npairs, device=x_s.device)


        

        pos_s_st = 0
        pos_t_st = 0
        for npair in range(npairs):
            edit_costs[npair] = torch.cdist(
                torch.cat( (embedding_s[pos_s_st:pos_s_st+data.len_s[npair]], virtual.unsqueeze(0))), 
                torch.cat( (embedding_t[pos_t_st:pos_t_st+data.len_t[npair]], virtual.unsqueeze(0)))
            )
            
            pos_s_st += data.len_s[npair]
            pos_t_st += data.len_t[npair]



        for npair in range(npairs):
            # compute optimal linear assignment given costs 
            lin_nodemap = self.lsape(edit_costs[npair].cpu()).to(edit_costs[npair].device)
            ged = torch.sum(lin_nodemap * edit_costs[npair])

            lin_nodemaps[npair] = lin_nodemap
            geds[npair] = ged  / (data.len_s[npair] + data.len_t[npair])
        

        return lin_nodemaps, edit_costs, geds




class GATModel(torch.nn.Module):
    def __init__(self, num_input_features, noise=0.0, gnn_num_layers=0, embedding_dim=64, hidden_dim=0):
        super(GATModel, self).__init__()
        self.num_input_features = num_input_features
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = self.num_input_features 
        if hidden_dim > 0:
            self.hidden_dim = hidden_dim
        self.noise = noise
        self.embedding_dim = embedding_dim
        
        self.embed_transform = nn.Sequential(
            nn.Linear(2*self.hidden_dim, self.embedding_dim)
        )
        self.conv = nn.ModuleList([GATConv(in_channels=self.num_input_features, out_channels=self.hidden_dim, dropout=0.1)]+[GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, dropout=0.1) for _ in range(gnn_num_layers-1)])
        
        

    def forward(self, data):
        x_s, edge_index_s, edge_attr_s = data.x_s, data.edge_index_s, data.edge_attr_s
        x_t, edge_index_t, edge_attr_t = data.x_t, data.edge_index_t, data.edge_attr_t

        embedding_s = [x_s]
        embedding_t = [x_t]
        for l in range(self.gnn_num_layers):
            embedding_s.append( self.conv[l](embedding_s[-1], edge_index_s.long(), edge_attr=edge_attr_s) )
            embedding_t.append( self.conv[l](embedding_t[-1], edge_index_t.long(), edge_attr=edge_attr_t) )
        
        # take first and last
        embedding_s = torch.cat([embedding_s[0], embedding_s[-1]], dim=-1)
        embedding_t = torch.cat([embedding_t[0], embedding_t[-1]], dim=-1)

        #pool graph level
        embedding_s = global_mean_pool(embedding_s, data.x_s_batch)
        embedding_t = global_mean_pool(embedding_t, data.x_t_batch)

        # project down
        embedding_s = self.embed_transform(embedding_s)
        embedding_t = self.embed_transform(embedding_t)
        # place them on hypersphere
        embedding_s = F.normalize(embedding_s, dim=-1)
        embedding_t = F.normalize(embedding_t, dim=-1)
        
        geds = F.pairwise_distance(embedding_s, embedding_t, eps=0.0)

        return None, None, geds

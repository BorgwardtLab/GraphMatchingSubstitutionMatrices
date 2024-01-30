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
from torch_geometric.nn import GraphConv, WLConv, GCNConv

import torch.autograd.profiler as profiler

INF = 10000.0





class FixedCostModel(torch.nn.Module):
    def __init__(self, num_input_features, cost_matrix=None):
        super(FixedCostModel, self).__init__()
        self.num_input_features = num_input_features

        with torch.no_grad():
            self.cost_matrix = cost_matrix
            if self.cost_matrix is None:
                self.cost_matrix = 1 - torch.eye(num_input_features+1, num_input_features+1)
            self.cost_matrix = nn.Parameter(self.cost_matrix)

            self.enlarge = torch.eye(num_input_features, num_input_features+1)
            self.enlarge = nn.Parameter(self.enlarge)

            self.virtual = torch.zeros(self.num_input_features+1)
            self.virtual[-1] = 1
            self.virtual = nn.Parameter(self.virtual)
        

        

    
    def lsape(self, edit_costs):
        n, m = edit_costs.shape
        n-=1; m-=1
        # python implementation of lsape
        C = torch.zeros((n+m, n+m)) 
        C[:n, :m] = edit_costs[:n, :m]
        C[n:, :m] = INF+0.0
        C[:n, m:] = INF+0.0
        C[(torch.arange(n,n+m), torch.arange(0,m))] = edit_costs[n, 0:m]
        C[(torch.arange(0,n), torch.arange(m,m+n))] = edit_costs[0:n, m]

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

        embedding_s = x_s @ self.enlarge
        embedding_t = x_t @ self.enlarge
        

        # compute edit cost for pairs of source and targets nodes
        npairs = data.x_s_batch[-1]+1
        edit_costs = [None for _ in range(npairs)]
        lin_nodemaps = [None for _ in range(npairs)]
        geds = torch.zeros(npairs, device=x_s.device)
        

        pos_s_st = 0
        pos_t_st = 0
        for npair in range(npairs):
            edit_costs[npair] = torch.zeros((data.len_s[npair]+1, data.len_t[npair]+1), device=x_s.device)

            emb_s = torch.cat( (embedding_s[pos_s_st:pos_s_st+data.len_s[npair]], self.virtual.unsqueeze(0)) ) 
            emb_t = torch.cat( (embedding_t[pos_t_st:pos_t_st+data.len_t[npair]], self.virtual.unsqueeze(0)) )
            
            edit_costs[npair] = (emb_s @ self.cost_matrix) @ emb_t.T 
            assert(edit_costs[npair].shape[0] == data.len_s[npair]+1)
            assert(edit_costs[npair].shape[1] == data.len_t[npair]+1)
            edit_costs[npair] = (edit_costs[npair] / torch.sum(edit_costs[npair])) * (data.len_s[npair]*data.len_t[npair])

            pos_s_st += data.len_s[npair]
            pos_t_st += data.len_t[npair]



        for npair in range(npairs):
            # compute optimal linear assignment given costs 
            lin_nodemap = self.lsape(edit_costs[npair].cpu()).to(edit_costs[npair].device)
            ged = torch.sum(lin_nodemap * edit_costs[npair])

            lin_nodemaps[npair] = lin_nodemap
            geds[npair] = ged  / (data.len_s[npair] + data.len_t[npair])
        

        return lin_nodemaps, edit_costs, geds





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



class WLKernel(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers+1
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(self.num_layers)])

    def forward_single(self, x, x_batch, edge_index):
        x = self.convs[0](x, torch.zeros(((2, 0)), dtype=torch.int64)) # 0-th layer
        for i in range(1, self.num_layers):
            x = self.convs[i](x, edge_index)

    def forward_hists(self, x, x_batch, edge_index):
        hists = []
        x = self.convs[0](x, torch.zeros(((2, 0)), dtype=torch.int64)) # 0-th layer
        hists.append(self.convs[0].histogram(x, x_batch, norm=False))
        for i in range(1, self.num_layers):
            x = self.convs[i](x, edge_index)
            hists.append(self.convs[i].histogram(x, x_batch, norm=False))
        return hists

    def forward(self, data):
        x_s, edge_index_s, edge_attr_s = data.x_s, data.edge_index_s, data.edge_attr_s
        x_t, edge_index_t, edge_attr_t = data.x_t, data.edge_index_t, data.edge_attr_t

        self.forward_single(x_s, data.x_s_batch, edge_index_s)
        self.forward_single(x_t, data.x_t_batch, edge_index_t)
        # now it's consistent between pairs
        hists_s = self.forward_hists(x_s, data.x_s_batch, edge_index_s)
        hists_t = self.forward_hists(x_t, data.x_t_batch, edge_index_t)

        
        hists_s = torch.cat(hists_s, dim=-1).to(float)
        hists_t = torch.cat(hists_t, dim=-1).to(float)

        
        '''
        hists_s = hists_s.to_sparse(layout=torch.sparse_coo)
        hists_t = hists_t.to_sparse(layout=torch.sparse_coo)

        dot = torch.sparse.mm(hists_s, hists_t.transpose(0,1))
        norm1 = torch.sparse.mm(hists_s, hists_s.transpose(0,1))
        norm2 = torch.sparse.mm(hists_t, hists_t.transpose(0,1))

        return dot[0].to_dense()/torch.sqrt(norm1[0,0]*norm2[0,0]) 
        '''
        return F.cosine_similarity(hists_s, hists_t)

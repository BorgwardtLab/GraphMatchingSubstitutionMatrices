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
from torch_geometric.nn import GraphConv, WLConv, GCNConv, WLConvContinuous
import ot


class WL(torch.nn.Module):
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
        hists.append(self.convs[0].histogram(x, x_batch, norm=True))
        for i in range(1, self.num_layers):
            x = self.convs[i](x, edge_index)
            hists.append(self.convs[i].histogram(x, x_batch, norm=True))
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

        ged = torch.zeros(2)
        ged[0] = 1-F.cosine_similarity(hists_s, hists_t)
        return [None], [None], ged[:1]



class WWL(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList([WLConvContinuous() for _ in range(num_layers)])

    # def forward_single(self, x, edge_index, batch=None):
    #     xs = []
    #     for conv in self.convs:
    #         x = conv(x, edge_index)
    #         xs.append(x)
    #     x = torch.cat(xs, dim=-1)
    #     return x

    def forward(self, data):
        x_s, edge_index_s, edge_attr_s = data.x_s, data.edge_index_s, data.edge_attr_s
        x_t, edge_index_t, edge_attr_t = data.x_t, data.edge_index_t, data.edge_attr_t

        xs_s = [x_s]
        xs_t = [x_t]
        for conv in self.convs:
            x_s = conv(x_s, edge_index_s)
            x_t = conv(x_t, edge_index_t)
            xs_s.append(x_s)
            xs_t.append(x_t)
        x_s = torch.cat(xs_s, dim=-1)
        x_t = torch.cat(xs_t, dim=-1)
        costs = ot.dist(x_s, x_t, metric='euclidean').cpu()
        nodemap = ot.emd2(torch.ones(costs.shape[0])/costs.shape[0], torch.ones(costs.shape[1])/costs.shape[1], costs)
        dist = torch.sum(costs * nodemap)
        ged = torch.zeros(2)
        ged[0] = dist
        return [nodemap], [costs], ged[:1]
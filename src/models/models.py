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



class SimpleConvLayer(torch_geometric.nn.conv.MessagePassing):
    def __init__(self, dropout):
        super().__init__("sum")
        self.dropout = dropout

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = torch.cat([x, out], dim=-1)
        return out 

    def message(self, x_j):
        x_j = F.dropout(x_j, p=self.dropout, training=self.training)
        return x_j


class ConvLayer(torch_geometric.nn.conv.MessagePassing):
    def __init__(self, dim, dropout):
        super().__init__("sum")
        self.out_layer = self.mlp = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            #nn.Dropout(p=dropout),
            nn.BatchNorm1d(2*dim),
            nn.ReLU(True),
            nn.Linear(2*dim, dim)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = torch.cat([x, out], dim=-1)
        return self.out_layer(out)

    def message(self, x_j):
        x_j = F.dropout(x_j, p=self.dropout, training=self.training)
        return x_j




def all_pair_distance(X, Y):
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    return 1-cosine_similarity(X[None,:,:], Y[:,None,:])









class GEDModelHomology(torch.nn.Module):
    def __init__(self, num_input_features, noise=0.0, gnn_num_layers=0, embedding_dim=64, dropout=0.5, soft=False, zero_init=False):
        super(GEDModelHomology, self).__init__()
        self.num_input_features = num_input_features
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = num_input_features 
        self.noise = noise
        self.embedding_dim = embedding_dim
        self.soft = soft

        self.embed_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
        )
        self.virtual_embedding = nn.Parameter(torch.rand(self.hidden_dim))
        self.simpleconv = GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, improved=True)


        if zero_init:
            with torch.no_grad(): # makes constant edit costs as starting point
                self.embed_transform[0].weight = torch.nn.Parameter(torch.eye(self.embedding_dim, self.hidden_dim))
                self.embed_transform[0].bias.fill_(0.0)
                self.embed_transform[3].weight = torch.nn.Parameter(torch.eye(self.embedding_dim, self.embedding_dim))
                self.embed_transform[3].bias.fill_(0.0)

                ve = torch.zeros(self.hidden_dim)
                self.virtual_embedding = nn.Parameter(ve)

        #self.scaler = torch.nn.Parameter(torch.randn(1))
        #self.w1 = torch.nn.Parameter(torch.randn(1))
        #self.w2 = torch.nn.Parameter(torch.randn(1))
        #self.b1 = torch.nn.Parameter(torch.randn(1))

        

    
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
        C[(torch.isfinite(C) == False)] = INF
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


        embedding_s = x_s
        embedding_t = x_t
        for _ in range(self.gnn_num_layers):
            embedding_s = self.simpleconv(embedding_s, edge_index_s.long())
            embedding_t = self.simpleconv(embedding_t, edge_index_t.long())
        
        embedding_s = self.embed_transform(embedding_s)
        embedding_t = self.embed_transform(embedding_t)

        virtual = self.embed_transform(self.virtual_embedding)

        # compute edit cost for pairs of source and targets nodes
        npairs = data.x_s_batch[-1]+1
        edit_costs = [None for _ in range(npairs)]
        lin_nodemaps = [None for _ in range(npairs)]
        geds = torch.zeros(npairs, device=x_s.device)


        

        pos_s_st = 0
        pos_t_st = 0
        for npair in range(npairs):
            #edit_costs[npair] = torch.zeros((data.len_s[npair]+1, data.len_t[npair]+1), device=x_s.device)

            edit_costs[npair] = torch.cdist(
                torch.cat( (embedding_s[pos_s_st:pos_s_st+data.len_s[npair]], virtual.unsqueeze(0))), 
                torch.cat( (embedding_t[pos_t_st:pos_t_st+data.len_t[npair]], virtual.unsqueeze(0)))
            )
            
            #edit_costs[npair] = all_pair_distance( 
            #    torch.cat( (embedding_s[pos_s_st:pos_s_st+data.len_s[npair]], virtual.unsqueeze(0))), 
            #    torch.cat( (embedding_t[pos_t_st:pos_t_st+data.len_t[npair]], virtual.unsqueeze(0)))
            #    )
            edit_costs[npair] = (edit_costs[npair] / torch.sum(edit_costs[npair])) * (data.len_s[npair]*data.len_t[npair])

            pos_s_st += data.len_s[npair]
            pos_t_st += data.len_t[npair]



        for npair in range(npairs):
            # compute optimal linear assignment given costs 
            if self.soft and self.training:
                lin_nodemap = self.soft_lsape(edit_costs[npair])
            else:
                lin_nodemap = self.lsape(edit_costs[npair].cpu()).to(edit_costs[npair].device)
            ged = torch.sum(lin_nodemap * edit_costs[npair])

            lin_nodemaps[npair] = lin_nodemap
            geds[npair] = ged  / (data.len_s[npair] + data.len_t[npair])
            #print(ged / (data.len_s[npair] + data.len_t[npair]), data.hom[npair])
            #pros[npair] = sigmoid(self.w1*( -ged / (data.len_s[npair] + data.len_t[npair]) ) + self.b1)
        

        return lin_nodemaps, edit_costs, geds








class GEDModelHomologyGAT(torch.nn.Module):
    def __init__(self, num_input_features, noise=0.0, gnn_num_layers=0, embedding_dim=64, dropout=0.5, soft=False, zero_init=False):
        super(GEDModelHomologyGAT, self).__init__()
        self.num_input_features = num_input_features
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = num_input_features 
        self.noise = noise
        self.embedding_dim = embedding_dim
        self.soft = soft

        self.embed_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
        )
        self.virtual_embedding = nn.Parameter(torch.rand(self.hidden_dim))
        self.conv = nn.ModuleList([GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, dropout=0.1) for _ in range(gnn_num_layers)])


        if zero_init:
            with torch.no_grad(): # makes constant edit costs as starting point
                self.embed_transform[0].weight = torch.nn.Parameter(torch.eye(self.embedding_dim, self.hidden_dim))
                self.embed_transform[0].bias.fill_(0.0)
                self.embed_transform[3].weight = torch.nn.Parameter(torch.eye(self.embedding_dim, self.embedding_dim))
                self.embed_transform[3].bias.fill_(0.0)

                ve = torch.zeros(self.hidden_dim)
                self.virtual_embedding = nn.Parameter(ve)

        

        

    
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
        C[(torch.isfinite(C) == False)] = INF
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


        embedding_s = x_s
        embedding_t = x_t
        for l in range(self.gnn_num_layers):
            embedding_s = self.conv[l](embedding_s, edge_index_s.long())
            embedding_t = self.conv[l](embedding_t, edge_index_t.long())
        
        embedding_s = self.embed_transform(embedding_s)
        embedding_t = self.embed_transform(embedding_t)

        virtual = self.embed_transform(self.virtual_embedding)

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
            
            edit_costs[npair] = (edit_costs[npair] / torch.sum(edit_costs[npair])) * (data.len_s[npair]*data.len_t[npair])

            pos_s_st += data.len_s[npair]
            pos_t_st += data.len_t[npair]



        for npair in range(npairs):
            # compute optimal linear assignment given costs 
            lin_nodemap = self.lsape(edit_costs[npair].cpu()).to(edit_costs[npair].device)
            ged = torch.sum(lin_nodemap * edit_costs[npair])

            lin_nodemaps[npair] = lin_nodemap
            geds[npair] = ged  / (data.len_s[npair] + data.len_t[npair])
            #print(ged / (data.len_s[npair] + data.len_t[npair]), data.hom[npair])
            #pros[npair] = sigmoid(self.w1*( -ged / (data.len_s[npair] + data.len_t[npair]) ) + self.b1)
        

        return lin_nodemaps, edit_costs, geds





class GEDModelHomologyLinear(torch.nn.Module):
    def __init__(self, num_input_features, noise=0.0, gnn_num_layers=0, embedding_dim=64, dropout=0.5, soft=False, zero_init=False):
        super(GEDModelHomologyLinear, self).__init__()
        self.num_input_features = num_input_features
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = num_input_features 
        self.noise = noise
        self.embedding_dim = embedding_dim
        self.soft = soft

        self.embed_transform = nn.Sequential(
            nn.Linear(2*self.hidden_dim, self.embedding_dim)
        )
        with torch.no_grad():
            tmp = 0.01*torch.rand((self.embedding_dim, 2*self.hidden_dim))
            tmp[:,0:self.hidden_dim] = tmp[:,0:self.hidden_dim] + torch.eye(self.embedding_dim, self.hidden_dim)
            self.embed_transform[0].weight = torch.nn.Parameter( tmp )
            self.embed_transform[0].bias.fill_(0.0)
        self.virtual_embedding = nn.Parameter(0.01*torch.rand(2*self.hidden_dim))
        self.conv = nn.ModuleList([GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, dropout=0.1) for _ in range(gnn_num_layers)])
        

    
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
        C[(torch.isfinite(C) == False)] = INF
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


        embedding_s = [x_s]
        embedding_t = [x_t]
        for l in range(self.gnn_num_layers):
            embedding_s.append( self.conv[l](embedding_s[-1], edge_index_s.long()) )
            embedding_t.append( self.conv[l](embedding_t[-1], edge_index_t.long()) )
        
        # take first and last
        embedding_s = torch.cat([x_s, embedding_s[-1]], dim=-1)
        embedding_t = torch.cat([x_t, embedding_t[-1]], dim=-1)

        embedding_s = self.embed_transform(embedding_s)
        embedding_t = self.embed_transform(embedding_t)

        virtual = self.embed_transform(self.virtual_embedding)

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
            
            edit_costs[npair] = (edit_costs[npair] / torch.sum(edit_costs[npair])) * (data.len_s[npair]*data.len_t[npair])

            pos_s_st += data.len_s[npair]
            pos_t_st += data.len_t[npair]



        for npair in range(npairs):
            # compute optimal linear assignment given costs 
            lin_nodemap = self.lsape(edit_costs[npair].cpu()).to(edit_costs[npair].device)
            ged = torch.sum(lin_nodemap * edit_costs[npair])

            lin_nodemaps[npair] = lin_nodemap
            geds[npair] = ged  / (data.len_s[npair] + data.len_t[npair])
            #print(ged / (data.len_s[npair] + data.len_t[npair]), data.hom[npair])
            #pros[npair] = sigmoid(self.w1*( -ged / (data.len_s[npair] + data.len_t[npair]) ) + self.b1)
        

        return lin_nodemaps, edit_costs, geds





class GEDModelHomologyLinear2(torch.nn.Module):
    def __init__(self, num_input_features, noise=0.0, gnn_num_layers=0, embedding_dim=64, dropout=0.5, soft=False, zero_init=False):
        super(GEDModelHomologyLinear2, self).__init__()
        self.num_input_features = num_input_features+1 #add 1 for dummy
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = self.num_input_features 
        self.noise = noise
        self.embedding_dim = embedding_dim
        self.soft = soft


        self.enlarge = nn.Parameter(torch.eye(num_input_features, num_input_features+1))

        self.embed_transform = nn.Sequential(
            nn.Linear(2*self.hidden_dim, self.embedding_dim)
        )
        self.virtual_embedding = nn.Parameter(0.01*torch.rand(2*self.hidden_dim))
        with torch.no_grad():
            tmp = 0.01*torch.rand((self.embedding_dim, 2*self.hidden_dim))
            tmp[:,0:self.hidden_dim] = tmp[:,0:self.hidden_dim] + torch.eye(self.embedding_dim, self.hidden_dim)
            self.embed_transform[0].weight = torch.nn.Parameter( tmp )
            self.embed_transform[0].bias.fill_(0.0)
            self.virtual_embedding[self.hidden_dim-1] += 1.0
        

        self.conv = nn.ModuleList([GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, dropout=0.1) for _ in range(gnn_num_layers)])
        

    
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
        C[(torch.isfinite(C) == False)] = INF
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
            #print(ged / (data.len_s[npair] + data.len_t[npair]), data.hom[npair])
            #pros[npair] = sigmoid(self.w1*( -ged / (data.len_s[npair] + data.len_t[npair]) ) + self.b1)
        

        return lin_nodemaps, edit_costs, geds


class GEDModelHomologyLinear3(torch.nn.Module):
    def __init__(self, num_input_features, noise=0.0, gnn_num_layers=0, hidden_dim=64, embedding_dim=64, dropout=0.5, soft=False, zero_init=False):
        super(GEDModelHomologyLinear3, self).__init__()
        self.num_input_features = num_input_features
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = hidden_dim 
        self.noise = noise
        self.embedding_dim = embedding_dim
        self.soft = soft


        self.enlarge = nn.Parameter(torch.eye(num_input_features, self.hidden_dim))

        self.embed_transform = nn.Sequential(
            nn.Linear(2*self.hidden_dim, self.embedding_dim)
        )
        self.virtual_embedding = nn.Parameter(0.01*torch.rand(2*self.hidden_dim))
        with torch.no_grad():
            tmp = 0.01*torch.rand((self.embedding_dim, 2*self.hidden_dim))
            tmp[:,0:self.hidden_dim] = tmp[:,0:self.hidden_dim] + torch.eye(self.embedding_dim, self.hidden_dim)
            self.embed_transform[0].weight = torch.nn.Parameter( tmp )
            self.embed_transform[0].bias.fill_(0.0)
            self.virtual_embedding[self.hidden_dim-1] += 1.0
        

        self.conv = nn.ModuleList([GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, dropout=0.1) for _ in range(gnn_num_layers)])
        

    
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
        C[(torch.isfinite(C) == False)] = INF
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
            #print(ged / (data.len_s[npair] + data.len_t[npair]), data.hom[npair])
            #pros[npair] = sigmoid(self.w1*( -ged / (data.len_s[npair] + data.len_t[npair]) ) + self.b1)
        

        return lin_nodemaps, edit_costs, geds





class GATModel(torch.nn.Module):
    def __init__(self, num_input_features, noise=0.0, gnn_num_layers=0, embedding_dim=64, dropout=0.5, soft=False, zero_init=False):
        super(GATModel, self).__init__()
        self.num_input_features = num_input_features
        self.gnn_num_layers = gnn_num_layers
        self.hidden_dim = self.num_input_features 
        self.noise = noise
        self.embedding_dim = embedding_dim
        self.soft = soft
        
        self.embed_transform = nn.Sequential(
            nn.Linear(2*self.hidden_dim, self.embedding_dim)
        )
        self.conv = nn.ModuleList([GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, dropout=0.1) for _ in range(gnn_num_layers)])
        

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

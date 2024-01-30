import os, sys
import numpy as np
from timeit import default_timer as timer
import argparse
import scipy
import random
import math
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

INF = 1000000000
ONEHOT_MAX = 20

class PairData(Data):
    def __init__(self, pair=None):
        super().__init__()
        if pair is not None:
            self.x_s = pair[0].x
            self.edge_index_s = pair[0].edge_index
            self.edge_attr_s = pair[0].edge_attr
            self.x_t = pair[1].x
            self.edge_index_t = pair[1].edge_index
            self.edge_attr_t = pair[1].edge_attr
            self.index = pair[2]
            self.ged = pair[3]
            self.hom = pair[4]
            self.len_s = self.x_s.size(0)
            self.len_t = self.x_t.size(0)

    
    def __inc__(self, key, value, *args, **kwargs): # this modifies edges index do deal with mini-batches
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        if key == 'index':
            return 0
        if key == 'ged':
            return 0
        if key == 'hom':
            return 0
        if key == 'len_s':
            return 0
        if key == 'len_t':
            return 0
        return super().__inc__(key, value, *args, **kwargs)
    
    #def __cat_dim__(self, key, value, *args, **kwargs): # this makes sure indexes are batched along new dimension
    #    if key == "index":
    #        return None
    #    else:
    #        return super().__cat_dim__(key, value, *args, **kwargs)








def read_graph_data(filen):
    lines = []
    with open(filen) as file:
        lines = [line for line in file]
    graphs = []
    i = 0 
    while(i < len(lines)):
        tok = lines[i].split(' ')
        n = int(tok[0])
        m = int(tok[1])
        node_labls = torch.zeros((n, ONEHOT_MAX))
        edge_index = torch.zeros(((2, 2*m)), dtype=torch.int64)
        edge_attr = torch.zeros((2*m, 1))
        i+=1
        for j in range(n):
            lab = int(lines[i])
            node_labls[j][lab] = 1
            i+=1
        for j in range(m):
            tok = lines[i].split(' ')
            u = int(tok[0])
            v = int(tok[1])
            lab = int(tok[2])
            edge_index[0][2*j] = u
            edge_index[1][2*j] = v
            edge_index[0][2*j+1] = v
            edge_index[1][2*j+1] = u
            edge_attr[2*j] = lab
            edge_attr[2*j+1] = lab
            i+=1
        graph = Data(x=node_labls, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)
    return graphs


def read_graph_label_data(filen):
    lines = []
    graphs = []
    labels = []
    with open(filen) as file:
        lines = [line for line in file]
    
    print("read", len(lines), "lines")
    i = 0 
    while(i < len(lines)):
        print("line", i)
        tok = lines[i].rstrip().split(' ')
        n = int(tok[0])
        m = int(tok[1])
        labl = tok[2]
        node_labls = torch.zeros((n, ONEHOT_MAX))
        edge_index = torch.zeros(((2, 2*m)), dtype=torch.int64)
        edge_attr = torch.zeros((2*m, 1))
        i+=1
        for j in range(n):
            attr = int(lines[i])
            node_labls[j][attr] = 1
            i+=1
        for j in range(m):
            tok = lines[i].split(' ')
            u = int(tok[0])
            v = int(tok[1])
            attr = int(float(tok[2]))
            edge_index[0][2*j] = u
            edge_index[1][2*j] = v
            edge_index[0][2*j+1] = v
            edge_index[1][2*j+1] = u
            edge_attr[2*j] = attr
            edge_attr[2*j+1] = attr
            i+=1
        graph = Data(x=node_labls, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)
        labels.append(labl)
    #tmp = list(zip(graphs, labels))
    #random.shuffle(tmp)
    #graphs, labels = zip(*tmp)
    return graphs, labels

def read_rna_data(filen):
    lines = []
    graphs = []
    labels = []
    with open(filen) as file:
        lines = [line for line in file]
    
    print("read", len(lines), "lines")
    i = 0 
    while(i < len(lines)):
        print("line", i)
        tok = lines[i].rstrip().split(' ')
        n = int(tok[0])
        m = int(tok[1])
        labl = tok[2]
        node_labls = torch.zeros((n, ONEHOT_MAX))
        edge_index = torch.zeros(((2, 2*m)), dtype=torch.int64)
        edge_attr = torch.zeros((2*m, 1))
        i+=1
        for j in range(n):
            tok = lines[i].split(' ')
            attr = int(tok[1])
            node_labls[j][attr] = 1
            i+=1
        for j in range(m):
            tok = lines[i].split(' ')
            u = int(tok[0])
            v = int(tok[1])
            assert(u < n)
            assert(v < n)
            attr = int(float(tok[2]))
            edge_index[0][2*j] = u
            edge_index[1][2*j] = v
            edge_index[0][2*j+1] = v
            edge_index[1][2*j+1] = u
            edge_attr[2*j] = attr
            edge_attr[2*j+1] = attr
            i+=1
        graph = Data(x=node_labls, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)
        labels.append(labl)
    
    return graphs, labels










def build_triplets(graphs, labels, start, end, num_neg_cla=10, num_neg=3, frac_pos=0.25): 
    pairs_pos = []
    pairs_neg = []
    
    lab2gr = {}
    for i in range(start, end):
        if labels[i] not in lab2gr:
            lab2gr[labels[i]] = []
        
        lab2gr[labels[i]].append(graphs[i])

    labs = list(lab2gr.keys())
    labs_sub = random.choices(labs, k=num_neg_cla)
    for lab in labs:
        graphs_lab = lab2gr[lab]
        if len(graphs_lab) < 2: continue
        for graph1_id in range(len(graphs_lab)):
            graph1 = graphs_lab[graph1_id]
            graphs_lab_ids = list(range(graph1_id))+list(range(graph1_id+1, len(graphs_lab)))
            graphs_lab_sub = random.choices(graphs_lab_ids, k= int(math.ceil(len(graphs_lab)*frac_pos)))
            for graph2_id in graphs_lab_sub:
                graph2 = graphs_lab[graph2_id]
                pair1 = PairData( (graph1, graph2, 0, 0, 1.0) )

                for lab2 in labs_sub: 
                    if lab2 == lab: continue
                    negs = random.choices(lab2gr[lab2], k=num_neg)
                    for neg in negs:
                        pair2 = PairData( (graph1, neg, 0, 0, 0.0) )
                        pairs_pos.append(pair1)
                        pairs_neg.append(pair2)

    print(len(pairs_pos))
    
    tmp = list(zip(pairs_pos, pairs_neg))
    random.shuffle(tmp)
    pairs_pos, pairs_neg = zip(*tmp)
    
    return pairs_pos, pairs_neg



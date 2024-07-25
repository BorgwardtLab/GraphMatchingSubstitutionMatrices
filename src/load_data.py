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
            self.label = pair[2]
            self.len_s = self.x_s.size(0)
            self.len_t = self.x_t.size(0)

    
    def __inc__(self, key, value, *args, **kwargs): # this modifies edges index do deal with mini-batches
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        if key == 'label':
            return 0
        if key == 'len_s':
            return 0
        if key == 'len_t':
            return 0
        return super().__inc__(key, value, *args, **kwargs)
    
    






def build_triplets(graphs, labels, start, end, num_neg_cla=1, num_neg=10, num_pos=100): 
    pairs_pos = []
    pairs_neg = []
    
    lab2gr = {}
    for i in range(start, end):
        if labels[i] not in lab2gr:
            lab2gr[labels[i]] = []
        
        lab2gr[labels[i]].append(graphs[i])

    labs = list(lab2gr.keys())
    labs_sub = labs
    for lab in labs:
        graphs_lab = lab2gr[lab]
        if len(graphs_lab) < 2: continue
        for graph1_id in range(len(graphs_lab)):
            graph1 = graphs_lab[graph1_id]
            graphs_lab_ids = list(range(graph1_id))+list(range(graph1_id+1, len(graphs_lab)))
            graphs_lab_sub = random.choices(graphs_lab_ids, k= int(min(len(graphs_lab), num_pos)) )
            for graph2_id in graphs_lab_sub:
                graph2 = graphs_lab[graph2_id]
                if graph1.x.shape[0] > graph2.x.shape[0]: pair1 = PairData( (graph1, graph2, 0, 0, 1.0) )
                else: pair1 = PairData( (graph2, graph1, 1.0) )

                for lab2 in labs_sub: 
                    if lab2 == lab: continue
                    negs = random.choices(lab2gr[lab2], k=num_neg)
                    for neg in negs:
                        if graph1.x.shape[0] > neg.x.shape[0]: pair2 = PairData( (graph1, neg, 0, 0, 0.0) )
                        else: pair2 = PairData( (neg, graph1, 0.0) )
                        pairs_pos.append(pair1)
                        pairs_neg.append(pair2)

    print(len(pairs_pos))
    
    tmp = list(zip(pairs_pos, pairs_neg))
    random.shuffle(tmp)
    pairs_pos, pairs_neg = zip(*tmp)
    
    return pairs_pos, pairs_neg




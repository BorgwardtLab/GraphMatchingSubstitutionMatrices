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





def compute_edit_distance(pair, assignment, edit_costs):
    cost = 0.0
    n = pair.x_s.shape[0]
    m = pair.x_t.shape[0]
    matched = torch.zeros(m) 
    # nodes
    ass = torch.argmax(assignment, dim=1)
    for i in range(n):
        j = ass[i]
        if j == pair.x_t.shape[0]:
            cost += edit_costs['node_ins_del']
        else:
            matched[j] = 1
            if not torch.equal(pair.x_s[i], pair.x_t[j]):
                cost += edit_costs['node_edit']
    cost += (m - torch.count_nonzero(matched)) * edit_costs['node_ins_del']
    # edges 
    n = pair.edge_index_s.shape[1]
    m = pair.edge_index_t.shape[1]
    matched = torch.zeros(m)
    # build adj list for target
    adj_list_t = [{} for _ in range(pair.x_t.shape[0])]
    for i in range(m):
        u, v = pair.edge_index_t[0, i], pair.edge_index_t[1, i]
        adj_list_t[u.item()][v.item()] = i
    ass = torch.argmax(assignment, dim=1)
    # compute edit distance for edges
    for i in range(n):
        u, v = pair.edge_index_s[0, i], pair.edge_index_s[1, i]
        w, z = ass[u], ass[v]
        if (w == pair.x_t.shape[0]) or (z == pair.x_t.shape[0]):
            cost += edit_costs['edge_ins_del']/2
        else:
            j = -1
            if not z.item() in adj_list_t[w.item()]: # edge is not present
                cost += edit_costs['edge_ins_del']/2
            else:
                j = adj_list_t[w.item()][z.item()]
                matched[j] = 1
                if not torch.equal(pair.edge_attr_s[i], pair.edge_attr_t[j]):
                    cost += edit_costs['edge_edit']/2
    cost += (m - torch.count_nonzero(matched)) * edit_costs['edge_ins_del']/2
    return cost.item()




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

def read_nmap_data(filen, n, limit = 0):
    lines = []
    with open(filen) as file:
        lines = [line for line in file]
    nmaps = np.ndarray((n, n), dtype=object)
    geds = np.zeros((n, n), float)
    l = 0
    for i in range(n):
        if limit > 0 and i > limit: break;
        for j in range(n):
            line = lines[l].split(' ')
            #assert int(line[0]) == i and int(line[1]) == j
            geds[i, j] = float(line[2])
            l+=1
            nnodes = int(lines[l])
            l+=1
            nmap = torch.zeros(nnodes, dtype=int)
            for k in range(nnodes):
                line = lines[l].split(' ')
                u = int(line[0])
                v = int(line[1])
                if v >= INF: v = -1 
                nmap[u] = v 
                l+=1
            nmaps[i][j] = nmap
    
    return nmaps, geds


def nodemap_to_assignment(nmap, n0, n1):
    assignment = torch.zeros((n0+1, n1+1), dtype=int)
    found = [0 for _ in range(n1)]
    for i in range(n0):
        j = nmap[i]
        if j > -1:
            found[j] = 1
            assignment[i,j] = 1
        else:
            assignment[i,-1] = 1
    for j in range(n1):
        if not found[j]:
            assignment[-1, j] = 1
    return assignment



def eval_hom_model(model, loader, pairs, hom_true, device = torch.device('cpu'), df = None):
    model.eval()
    loss = torch.zeros(1)
    acc = torch.zeros(1)
    for batch in tqdm(loader, disable=disable_tqdm):
        tmp_lin_nodemaps, tmp_edit_costs, pro = model(batch.to(device))
        for i in range(len(pro)):
            hom1 = torch.ones(1)*hom_true[batch.index[i].cpu()]
            prediction = torch.ones(1) if pro[i] > 0.5 else torch.zeros(1)
            loss += F.binary_cross_entropy(prediction, hom1)
            if hom1 == prediction:
                acc = acc + 1
    acc = acc/len(hom_true)
    return acc






def build_pairs_and_hom(graphs, labels, start, end, prob = 0.1):
    pairs = []
    hom = []
    idn = 0
    ones = 0
    for i in range(start, end):
        for j in range(i+1, end):
            if labels[i]!=labels[j] and np.random.random() > prob: continue
            pairs.append( PairData( (graphs[i], graphs[j], idn, 0, 0) ) )
            hom.append( 1.0 if labels[i]==labels[j] else 0.0 )
            pairs[-1].hom = hom[-1]
            ones += hom[-1]
            idn += 1
    positive = ones/len(hom)
    print(len(hom), positive)
    return pairs, hom, positive


def build_pairs_and_hom_balanced(graphs, labels, start, end): # subsamples negative pairs 
    pairs = []
    hom = []
    idn = 0
    ones = 0
    for i in range(start, end):
        for j in range(i+1, end):
            ones += 1.0 if labels[i]==labels[j] else 0.0
    pos_rate = 2*ones/((end-start)*(end-start-1))
    prob = pos_rate/(1 - pos_rate)
    ones = 0
    for i in range(start, end):
        for j in range(i+1, end):
            if labels[i]!=labels[j] and np.random.random() > prob: continue
            pairs.append( PairData( (graphs[i], graphs[j], idn, 0, 0) ) )
            hom.append( 1.0 if labels[i]==labels[j] else 0.0 )
            pairs[-1].hom = hom[-1]
            ones += hom[-1]
            idn += 1
    pos_rate = ones/len(hom)
    print(len(hom), pos_rate, prob)
    return pairs, hom, pos_rate, prob


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



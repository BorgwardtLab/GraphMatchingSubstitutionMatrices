import os, sys
import numpy as np
from timeit import default_timer as timer
import argparse
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
import random
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import torch.autograd.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.transforms as T
import torchmetrics.functional as metrics_F


from load_data import *
from models.models import GEDModelHomology, GEDModelHomologyGAT, GEDModelHomologyLinear, GEDModelHomologyLinear2, GATModel
from models.model_fixed import FixedCostModel
from utils import *


INF = 1000000000
disable_tqdm = False
args = None

def build_id_triplets(ids, labels, start, end, num_neg_cla=10, num_neg=3, frac_pos=0.25): 
    pairs_pos = []
    pairs_neg = []
    
    lab2gr = {}
    for i in range(start, end):
        if labels[i] not in lab2gr:
            lab2gr[labels[i]] = []
        
        lab2gr[labels[i]].append(ids[i])

    labs = list(lab2gr.keys())
    labs_sub = random.choices(labs, k=num_neg_cla)
    for lab in labs:
        graphs_lab = lab2gr[lab]
        #print(lab, len(graphs_lab))
        if len(graphs_lab) < 2: continue
        for graph1_id in range(len(graphs_lab)):
            graph1 = graphs_lab[graph1_id]
            graphs_lab_ids = list(range(graph1_id))+list(range(graph1_id+1, len(graphs_lab)))
            graphs_lab_sub = random.choices(graphs_lab_ids, k= int(math.ceil(len(graphs_lab)*frac_pos)))
            for graph2_id in graphs_lab_sub:
                graph2 = graphs_lab[graph2_id]
                pair1 = (graph1, graph2) 

                for lab2 in labs_sub: 
                    if lab2 == lab: continue
                    negs = random.choices(lab2gr[lab2], k=num_neg)
                    for neg in negs:
                        pair2 = (graph1, neg)
                        pairs_pos.append(pair1)
                        pairs_neg.append(pair2)

    print(len(pairs_pos))
    
    tmp = list(zip(pairs_pos, pairs_neg))
    random.shuffle(tmp)
    pairs_pos, pairs_neg = zip(*tmp)
    
    return pairs_pos, pairs_neg


def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training graphs")
    parser.add_argument("--n-val", type=int, default=100, help="Number of validation graphs")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test graphs")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--ckp", type=str, help="Path to saved model")
    parser.add_argument("--samples", type=str, default="10-1-0.25")
    parser.add_argument("--logs", type=str, help="Path to logs", default=".")

    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print(args.device)

    args.basename = args.ckp.split("/")[-1]
    

    torch.set_printoptions(precision=3, linewidth=300)

    return args




def main():
    args = cline()
    print(args, flush=True)
    
    tmp = torch.load(args.graphs)
    graphs = tmp[2] # ids, not graphs
    labels = tmp[args.label]

    print(len(graphs))
    
    aligns = np.genfromtxt(args.ckp, delimiter=' ', dtype=str)
    
    
    # generate test data 
    tmp = args.samples.split("-")
    num_neg_cla=int(tmp[0]) 
    num_neg=int(tmp[1])
    frac_pos=float(tmp[2])

    #pairs_pos, pairs_neg = build_triplets(graphs, labels, 0, args.n_train, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)

    pairs_pos_te, pairs_neg_te = build_id_triplets(graphs, labels, args.n_train+args.n_val, args.n_train+args.n_val+args.n_test, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    #for p in pairs_pos_te:
    #    print(p[0], p[1], 1)
    #for p in pairs_neg_te:
    #    print(p[0], p[1], 0)
    #exit()
    pairs_te = [val for pair in zip(pairs_pos_te, pairs_neg_te) for val in pair]

    id = 0
    
    dist_pos = []
    for pair in pairs_pos_te:
        assert pair[0] == str(aligns[id][0])
        assert pair[1] == str(aligns[id][1])
        dist_pos.append( float(aligns[id][2])/1000.0 )
        id +=1
    
    dist_neg = []
    for pair in pairs_neg_te:
        assert pair[0] == str(aligns[id][0])
        assert pair[1] == str(aligns[id][1])
        dist_neg.append( float(aligns[id][2])/1000.0 )
        id +=1

    trip_acc = sum([x > y for x,y in zip(dist_pos, dist_neg)]) / len(dist_pos)
    print(trip_acc)

    dist = torch.tensor(dist_pos+dist_neg)
    hom = torch.cat([torch.ones(len(pairs_pos_te)), torch.zeros(len(pairs_neg_te))] ).long()

    pair_auroc = metrics_F.auroc(torch.exp(dist), hom, task='binary').item()
    print(pair_auroc)
    
    exit()
    with open(args.logs+"/test_"+args.basename+".log", "w") as myfile:
        myfile.write(str(args)+"\n")
        myfile.write(str(0) +" "+ str(trip_acc) +" "+ str(pair_auroc))

    
    



        
if __name__ == '__main__':
    main()
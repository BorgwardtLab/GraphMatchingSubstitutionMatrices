import os, sys
import numpy as np
from timeit import default_timer as timer
import argparse
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
import pandas as pd
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
from utils import *


INF = 1000000000
disable_tqdm = False
args = None




def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--label", type=int, default=1, help="1=Pfam, 2=SCOP-SF")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training graphs")
    parser.add_argument("--n-val", type=int, default=100, help="Number of validation graphs")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test graphs")
    parser.add_argument("--hits", type=str, help="Path to hits")
    parser.add_argument("--k", type=int, default=100)
    #parser.add_argument("--logs", type=str, help="Path to logs", default=".")

    args = parser.parse_args()
    

    torch.set_printoptions(precision=3, linewidth=300)

    return args


# pf --n-train 18700 --n-val 2340 --n-test 2337  --samples 10-2-0.01
# scop --n-train 6240 --n-val 780 --n-test 781  --samples 30-2-0.02
# ec --n-train 11240 --n-val 1410 --n-test 1405  --samples 25-2-0.002
# rna --n-train 688 --n-val 86 --n-test 87  --samples 50-10-0.1

# mutagenicity --n-train 3470 --n-val 433 --n-test 434 --samples 1-10-0.1
# nci1 --n-train 3288 --n-val 411 --n-test 411 --samples 1-10-0.1
# aids --n-train 1600 --n-val 200 --n-test 200 --samples 1-100-0.2

def main():
    args = cline()
    print(args, flush=True)
    
    tmp = torch.load(args.graphs)
    graphs = tmp[0]
    labels = tmp[args.label]

    with open(args.hits, "r") as f:
        lines = [s.rstrip() for s in f]
    
    acc_at_k = []
    for i in range(2, len(lines)):
        if i == args.n_test+2: 
            break
        toks = lines[i].split(": ")
        ids = int(toks[0])
        toks = toks[1].split(" ")
        hits = [int(x) for x in toks]
        
        hits = hits[0:min(len(hits), args.k)]

        if ids >= args.n_train:
            ids -= (args.n_train+args.n_val)

        all_positives = np.array([1 if label == labels[ids+(args.n_train+args.n_val)] else 0 for label in labels[0:args.n_train]]).sum()
        if all_positives < 0.005* args.n_train:
            continue

        hom = np.array([1 if labels[hit] == labels[ids+(args.n_train+args.n_val)] else 0 for hit in hits])

        #print(hom)
        acc_at_k.append(hom.sum()/hom.shape[0])
        #print(hom.sum()/hom.shape[0])

        
        #break
    

    acc_at_k = np.array(acc_at_k)

    print(acc_at_k.shape[0])
    print(acc_at_k.mean(), acc_at_k.std())
    
    



        
if __name__ == '__main__':
    main()
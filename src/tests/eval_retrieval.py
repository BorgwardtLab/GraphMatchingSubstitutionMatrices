import os, sys
import numpy as np
import scipy
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.autograd.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.transforms as T
import torchmetrics.functional as metrics_F


from load_data import *
from utils import *


disable_tqdm = False
args = None




def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--hits", type=str, help="Path to hits")
    #parser.add_argument("--k", type=int, default=100)
    #parser.add_argument("--logs", type=str, help="Path to logs", default=".")

    args = parser.parse_args()
    

    torch.set_printoptions(precision=3, linewidth=300)

    return args



def main():
    args = cline()
    print(args, flush=True)
    
    k1=10
    k2=50

    
    tmp = torch.load(args.graphs)
    train_lab, val_lab, test_lab = tmp[1]
    labels = train_lab + val_lab + test_lab
    args.n_train = len(train_lab)
    args.n_val = len(val_lab)
    args.n_test = len(test_lab)

    with open(args.hits, "r") as f:
        lines = [s.rstrip() for s in f]
    
    acc_at_k1 = []
    acc_at_k2 = []
    for i in range(2, len(lines)):
        if i == args.n_test+2: 
            break
        toks = lines[i].split(": ")
        ids = int(toks[0])
        toks = toks[1].split(" ")
        hits = [int(x) for x in toks]
        

        if ids >= args.n_train:
            ids -= (args.n_train+args.n_val)

        all_positives = np.array([1 if label == labels[ids+(args.n_train+args.n_val)] else 0 for label in labels[0:args.n_train]]).sum()
        # if all_positives < 0.005* args.n_train:
        #     continue

        hom1 = np.array([1 if labels[hit] == labels[ids+(args.n_train+args.n_val)] else 0 for hit in hits[0:min(len(hits), k1)] ])
        hom2 = np.array([1 if labels[hit] == labels[ids+(args.n_train+args.n_val)] else 0 for hit in hits[0:min(len(hits), k2)] ])

        #print(hom)
        acc_at_k1.append(hom1.sum()/hom1.shape[0])
        acc_at_k2.append(hom2.sum()/hom2.shape[0])
        #print(hom.sum()/hom.shape[0])

        
        #break
    

    acc_at_k1 = np.array(acc_at_k1)
    acc_at_k2 = np.array(acc_at_k2)

    print(acc_at_k1.shape[0])
    print(acc_at_k1.mean(), acc_at_k1.std())

    print(acc_at_k2.shape[0])
    print(acc_at_k2.mean(), acc_at_k2.std())
    

    
    print(r"%.3f \tsmall{%.3f} & %.3f \tsmall{%.3f}"% (acc_at_k1.mean(), acc_at_k1.std(), acc_at_k2.mean(), acc_at_k2.std()))


        
if __name__ == '__main__':
    main()
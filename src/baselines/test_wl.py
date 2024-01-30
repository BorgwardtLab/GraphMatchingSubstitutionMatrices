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
from models.wl import WLKernel



INF = 1000000000
disable_tqdm = False
args = None




def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--label", type=int, default=1, help="1=Pfam, 2=SCOP-SF")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training graphs")
    parser.add_argument("--n-val", type=int, default=100, help="Number of validation graphs")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test graphs")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cuda", action='store_true')
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

    args.basename = f'{args.graphs.split("/")[-1]}_lab{args.label}_n{args.n_train}_l{args.layers}_wl'

    torch.set_printoptions(precision=3, linewidth=300)

    return args



def val_step(model, loader_val, device):
    model.eval()
    
    all_sim = []
    all_homs = []
    num_pairs = 0
    num_batch = 0
    zero = torch.zeros(1).to(device)

    with torch.no_grad():
        for batch in tqdm(loader_val, disable=disable_tqdm, ncols=64):
            sim = model(batch.to(device))
            
            all_sim.append(sim)
            all_homs.append(batch.hom)
            num_pairs += len(batch)
            num_batch += 1

    
    all_homs = torch.cat(all_homs).long()
    all_sim = torch.cat(all_sim)
    val_auroc = metrics_F.auroc(all_sim, all_homs, task='binary').item()

    pos_sim = all_sim[torch.nonzero(all_homs, as_tuple=True)]
    neg_sim = all_sim[torch.nonzero(1-all_homs, as_tuple=True)]
    return val_auroc, pos_sim, neg_sim


def val_step_triplet(model, loader_pos, loader_neg, device):
    loader_neg_it = iter(loader_neg)

    model.eval()

    trip_acc = 0
    all_sim_pos = []
    all_sim_neg = []
    num_pairs = 0
    num_batch = 0
    zero = torch.zeros(1).to(device)

    with torch.no_grad():
        for batch_pos in tqdm(loader_pos, disable=disable_tqdm, ncols=64):
            batch_neg = next(loader_neg_it)

            #with profiler.profile(with_stack=True, profile_memory=True) as prof:
            sim_pos = model(batch_pos.to(device))
            sim_neg = model(batch_neg.to(device))

            #print(sim_pos)
            #print((torch.count_nonzero(torch.maximum(zero, sim_pos - sim_neg))).detach().item())
            #print(prof.key_averages(group_by_stack_n=10).table(sort_by='self_cpu_time_total', row_limit=10))    
            
            trip_acc += (torch.count_nonzero(torch.maximum(zero, sim_pos - sim_neg))).detach().item()
            all_sim_pos.append(sim_pos)
            all_sim_neg.append(sim_neg)

            
            num_pairs += len(batch_pos)
            num_batch += 1
            
    
    all_sim = torch.cat(all_sim_pos+all_sim_neg)
    all_sim_pos = torch.cat(all_sim_pos)
    all_sim_neg = torch.cat(all_sim_neg)
    #print(all_sim)
    trip_acc = trip_acc/num_pairs

    all_homs = torch.cat([torch.ones_like(all_sim_pos), torch.zeros_like(all_sim_neg)] ).long()
    pair_auroc = metrics_F.auroc(all_sim, all_homs, task='binary').item()
    
    return trip_acc, pair_auroc, all_sim_pos, all_sim_neg



def main():
    args = cline()
    print(args, flush=True)
    
    tmp = torch.load(args.graphs)
    graphs = tmp[0]
    labels = tmp[args.label]

    print(len(graphs))
    
    


    print(graphs[0].x.shape)
    
    # generate training data and validation data 
    tmp = args.samples.split("-")
    num_neg_cla=int(tmp[0]) 
    num_neg=int(tmp[1])
    frac_pos=float(tmp[2])

    pairs_pos_te, pairs_neg_te = build_triplets(graphs, labels, args.n_train+args.n_val, args.n_train+args.n_val+args.n_test, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    pairs_te = [val for pair in zip(pairs_pos_te, pairs_neg_te) for val in pair]

    #pairs_pos, pairs_neg = build_triplets(graphs, labels, 0, args.n_train, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    #pairs_pos = pairs_pos[:len(pairs_pos_te)] 
    #pairs_neg = pairs_neg[:len(pairs_pos_te)]
    #pairs = [val for pair in zip(pairs_pos, pairs_neg) for val in pair]

    #loader_train_pos = DataLoader(pairs_pos, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    #loader_train_neg = DataLoader(pairs_neg, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    #loader_train = DataLoader(pairs, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)

    loader_test_pos = DataLoader(pairs_pos_te, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_test_neg = DataLoader(pairs_neg_te, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    #loader_test = DataLoader(pairs_te, batch_size=1, follow_batch=['x_s', 'x_t'], shuffle = False)
    

    model = WLKernel(args.layers).to(args.device) 
    model.eval()

    


    #val_auroc, pos_geds, neg_geds = val_step(model, loader_test, args.device)
    val_acc, val_auroc, pos_geds, neg_geds = val_step_triplet(model, loader_test_pos, loader_test_neg, args.device)

    print(val_auroc, val_acc)
    with open(args.logs+"/wl_"+args.basename+".log", "w") as myfile:
        myfile.write(str(args)+"\n")
        myfile.write(str(0) +" "+ str(val_acc) +" "+ str(val_auroc))

    '''
    plt.figure()
    bins = np.histogram(np.hstack((pos_geds.cpu().numpy(), neg_geds.cpu().numpy())), bins=20, range=(0.0,1.0))[1] #get the bin edges
    plt.hist(pos_geds.cpu().numpy(), bins, alpha=0.4)
    plt.hist(neg_geds.cpu().numpy(), bins, alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.logs+"/wl_"+args.basename+".png")
    '''

    



        
if __name__ == '__main__':
    main()
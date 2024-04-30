import os, sys
import numpy as np
from timeit import default_timer as timer
import argparse
import scipy
from tqdm import tqdm
#import matplotlib.pyplot as plt
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
from models.models import GEDModel
from utils import *


INF = 1000000000
disable_tqdm = False
args = None



def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training graphs")
    parser.add_argument("--n-val", type=int, default=100, help="Number of validation graphs")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test graphs")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--save-ckp", type=str, help="Path to save model")
    parser.add_argument("--logs", type=str, help="Path to logs", default=".")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples", type=str, default="10-1-0.25")
    parser.add_argument("--patience", type=int, default=2)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print(args.device)

    args.basename = f'{args.graphs.split("/")[-1]}_n{args.n_train}_l{args.layers}_emb{args.embedding_dim}_hid{args.hidden_dim}_noise{args.noise}_margin{args.margin}_seed{args.seed}'
    

    torch.set_printoptions(precision=3, linewidth=300)

    return args




def main():
    args = cline()
    print(args, flush=True)

    tmp = torch.load(args.graphs)
    train_g, val_g, test_g = tmp[0]
    train_l, val_l, test_l = tmp[1]
    
    # generate training data 
    tmp = args.samples.split("-")
    num_neg_cla=int(tmp[0]) 
    num_neg=int(tmp[1])
    frac_pos=float(tmp[2])
    pairs_pos, pairs_neg = build_triplets(train_g, train_l, 0, len(train_g), num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    
    # generate validation data 
    pairs_pos_val, pairs_neg_val = build_triplets(val_g, val_l, 0, len(val_g), num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    pairs_val = [val for pair in zip(pairs_pos_val, pairs_neg_val) for val in pair]

    # loaders
    loader_pos = DataLoader(pairs_pos, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_neg = DataLoader(pairs_neg, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)

    loader_pos_val = DataLoader(pairs_pos_val, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_neg_val = DataLoader(pairs_neg_val, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_val = DataLoader(pairs_val, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    

    # model
    model = GEDModel(train_g[0].x.size(1), noise=args.noise, gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim).to(args.device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, min_lr=1e-6
        )
    
    
    with open(args.logs+"/"+args.basename+".log", "w") as myfile:
        myfile.write(str(args)+"\n")
        myfile.write("Epoch, train trip loss, train tripl acc, val trip loss, val pair auroc, val tripl acc\n")
        
    best_val = 0 # auroc
    best_model = None
    cnt_pat = 0

    margin = args.margin


    for epoch in range(args.epochs):
        # training
        loss, trip_acc = train_step_triplet_retrieval(model, loader_pos, loader_neg, optimizer, margin, args,  val=loader_val, val_p=loader_pos_val, val_n=loader_neg_val)
        
        # validation
        val_loss, val_acc, val_auroc, tmp1, tmp2 = val_step_triplet(model, loader_pos_val, loader_neg_val, margin, args.device)

        # logging
        log_data(args.logs+"/"+args.basename+".log", epoch, loss, trip_acc, val_loss, val_acc, val_auroc)

        if val_auroc > best_val:
            cnt_pat = 0
            best_val = val_auroc
            best_model = model.state_dict()
            print("Epoch", epoch, ": best model", val_auroc)
        
            if args.save_ckp:   
                torch.save(best_model, args.save_ckp+f'model_{args.basename}_epoch{epoch}.pt')
        else:
            cnt_pat += 1
            if cnt_pat == args.patience:
                break
    



        
if __name__ == '__main__':
    main()
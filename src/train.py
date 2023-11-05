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
from models.models import GEDModelHomology, GEDModelHomologyGAT, GEDModelHomologyLinear, GEDModelHomologyLinear2, GEDModelHomologyLinear3
from utils import *


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
    parser.add_argument("--noise", type=float, default=0.00)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--gat", action='store_true')
    parser.add_argument("--lin", action='store_true')
    parser.add_argument("--lin2", action='store_true')
    parser.add_argument("--lin3", action='store_true')
    #parser.add_argument("--uniform-costs", action='store_true')
    parser.add_argument("--save-ckp", type=str, help="Path to save model")
    parser.add_argument("--logs", type=str, help="Path to logs", default=".")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--samples", type=str, default="10-1-0.25")
    parser.add_argument("--acc", type=int, default=1)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print(args.device)

    args.basename = f'{args.graphs.split("/")[-1]}_lab{args.label}_n{args.n_train}_l{args.layers}_emb{args.embedding_dim}_dr{args.dropout}_noise{args.noise}_margin{args.margin}_seed{args.seed}'
    

    torch.set_printoptions(precision=3, linewidth=300)

    return args




def train_step_triplet_retrieval2(model, loader_pos, loader_neg, optimizer, margin, args,
                        val=None, val_p=None, val_n=None):
    loader_neg_it = iter(loader_neg)

    device = args.device
    basename = args.logs+"/"+args.basename
    logging_loss = 0
    all_geds = []
    all_homs = []
    num_pairs = 0
    num_batch = 0
    trip_acc = 0
    zero = torch.zeros(1).to(device)

    
    i = 0
    for batch_pos in tqdm(loader_pos, disable=disable_tqdm, ncols=64):
        model.train()
        batch_neg = next(loader_neg_it)

        lin_nodemaps, edit_costs, geds_pos = model(batch_pos.to(device))
        lin_nodemaps, edit_costs, geds_neg = model(batch_neg.to(device))
        del lin_nodemaps, edit_costs
        
        loss = torch.sum(torch.maximum(zero, -(margin-0.1) + geds_pos)) + torch.sum(torch.maximum(zero, 2*((margin+0.1) - geds_neg) ))


        trip_acc += (torch.count_nonzero(torch.maximum(zero, geds_neg - geds_pos))).detach().item()


        loss.backward()

        
        optimizer.step()
        optimizer.zero_grad()
          

        num_pairs += len(batch_pos)
        num_batch += 1
        logging_loss += loss.detach().item()

        if basename is not None and ((i) % 1000 == 0):
            ls, acc, auroc, tmp1, tmp2 = val_step_triplet(model, val_p, val_n, margin, device)
            log_data(basename+".log", i, logging_loss/num_pairs, trip_acc/num_pairs, ls, acc, auroc)
            torch.save(model.state_dict(), args.save_ckp+f'model_{args.basename}_step{i}.pt')

            trip_acc = 0
            logging_loss = 0
            num_batch = 0
            num_pairs = 0
          

        i+=1
    
    return logging_loss, trip_acc



def main():
    args = cline()
    print(args, flush=True)

    tmp = torch.load(args.graphs)
    graphs = tmp[0]
    labels = tmp[args.label]

    print(len(graphs))
    
    # generate training data 
    tmp = args.samples.split("-")
    num_neg_cla=int(tmp[0]) 
    num_neg=int(tmp[1])
    frac_pos=float(tmp[2])
    pairs_pos, pairs_neg = build_triplets(graphs, labels, 0, args.n_train, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    #pairs = [val for pair in zip(pairs_pos, pairs_neg) for val in pair]
    
    # generate validation data 
    pairs_pos_val, pairs_neg_val = build_triplets(graphs, labels, args.n_train, args.n_train+args.n_val, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    pairs_val = [val for pair in zip(pairs_pos_val, pairs_neg_val) for val in pair]

    # loaders
    loader_pos = DataLoader(pairs_pos, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_neg = DataLoader(pairs_neg, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)

    loader_pos_val = DataLoader(pairs_pos_val, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_neg_val = DataLoader(pairs_neg_val, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_val = DataLoader(pairs_val, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    

    # model
    model = GEDModelHomology(graphs[0].x.size(1), noise=args.noise, gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, dropout=args.dropout).to(args.device) 
    if args.gat: model = GEDModelHomologyGAT(graphs[0].x.size(1), noise=args.noise, gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, dropout=args.dropout).to(args.device) 
    if args.lin: model = GEDModelHomologyLinear(graphs[0].x.size(1), noise=args.noise, gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, dropout=args.dropout).to(args.device) 
    if args.lin2: model = GEDModelHomologyLinear2(graphs[0].x.size(1), noise=args.noise, gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, dropout=args.dropout).to(args.device) 
    if args.lin3: model = GEDModelHomologyLinear3(graphs[0].x.size(1), noise=args.noise, gnn_num_layers=args.layers, hidden_dim=args.hidden_dim, embedding_dim=args.embedding_dim, dropout=args.dropout).to(args.device) 


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, min_lr=1e-6
        )
    
    
    with open(args.logs+"/"+args.basename+".log", "w") as myfile:
        myfile.write(str(args)+"\n")
        myfile.write("Epoch, train trip loss, train tripl acc, val trip loss, val pair auroc, val tripl acc\n")
        
    best_val = 0 # auroc
    best_model = None
    
    margin = args.margin


    for epoch in range(args.epochs):
        # training
        loss, trip_acc = train_step_triplet_retrieval2(model, loader_pos, loader_neg, optimizer, margin, args,  val=loader_val, val_p=loader_pos_val, val_n=loader_neg_val)
        #loss, auroc = train_step(model, loader, optimizer, margin, args.device)

        # validation
        val_loss, val_acc, val_auroc, tmp1, tmp2 = val_step_triplet(model, loader_pos_val, loader_neg_val, margin, args.device)

        # logging
        log_data(args.logs+"/"+args.basename+".log", epoch, loss, trip_acc, val_loss, val_acc, val_auroc)

        if val_auroc > best_val:
            best_val = val_auroc
            best_model = model.state_dict()
            print("Epoch", epoch, ": best model", val_auroc)
        
        if args.save_ckp:   
            torch.save(best_model, args.save_ckp+f'model_{args.basename}_epoch{epoch}.pt')

    



        
if __name__ == '__main__':
    main()
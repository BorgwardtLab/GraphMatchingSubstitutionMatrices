import os, sys
import numpy as np
import argparse
import scipy
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.transforms as T
import torchmetrics.functional as metrics_F


from load_data import *
from models.models import DistanceNet, GEDNet
from utils import *


disable_tqdm = False
args = None



def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-val", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--save-ckp", type=str, help="Path to save model")
    parser.add_argument("--logs", type=str, help="Path to logs", default=".")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples", type=str)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--net", type=str)
    parser.add_argument("--shuffled", action='store_true')
    parser.add_argument("--reg", type=float, default=0.1)

    args = parser.parse_args()

    assert args.net in ['dist', 'ot']

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print(args.device)

    if args.net == 'ot': args.basename = f'{args.graphs.split("/")[-1]}_l{args.layers}_emb{args.embedding_dim}_hid{args.hidden_dim}_net{args.net}{args.reg}_sh{args.shuffled}_seed{args.seed}'
    else: args.basename = f'{args.graphs.split("/")[-1]}_l{args.layers}_emb{args.embedding_dim}_hid{args.hidden_dim}_net{args.net}_sh{args.shuffled}_seed{args.seed}'

    torch.set_printoptions(precision=5, linewidth=300)

    return args




def train_step_triplet(model, loader_pos, loader_neg, optimizer, margin=0.5, device='cpu'):
    loader_neg_it = iter(loader_neg)

    logging_loss = 0
    num_pairs = 0
    num_batch = 0
    zero = torch.zeros(1).to(device)

    
    i = 0
    for batch_pos in tqdm(loader_pos, disable=disable_tqdm, ncols=64):
        model.train()
        batch_neg = next(loader_neg_it)

        lin_nodemaps, edit_costs, geds_pos = model(batch_pos.to(device))
        lin_nodemaps, edit_costs, geds_neg = model(batch_neg.to(device))
        del lin_nodemaps, edit_costs
        
        loss = torch.sum(torch.maximum(zero, -(margin-0.1) + geds_pos)) + torch.sum(torch.maximum(zero, ((margin+0.1) - geds_neg) ))


        loss.backward()

        
        optimizer.step()
        optimizer.zero_grad()
          

        num_pairs += len(batch_pos)
        logging_loss += loss.detach().item()


    
    return logging_loss/num_pairs



def train_step(model, loader, optimizer, margin = 0.5, device='cpu'):
    logging_loss = 0
    num_pairs = 0
    zero = torch.zeros(1).to(device)

    model.train()
    for batch in tqdm(loader, disable=disable_tqdm, ncols=64):

        lin_nodemaps, edit_costs, geds = model(batch.to(device))
        del lin_nodemaps, edit_costs
        

        loss_pos = batch.label * torch.maximum(zero, -(margin-0.1) + geds)
        loss_neg = (1.0 - batch.label) * torch.maximum(zero, (margin-0.1) - geds)

        loss = torch.sum(loss_pos) + torch.sum(loss_neg)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
          

        num_pairs += len(batch)
        logging_loss += loss.detach().item()

    
    return logging_loss/num_pairs






def main():
    args = cline()
    print(args, flush=True)

    tmp = torch.load(args.graphs)
    graphs = tmp[0]
    labels = tmp[1]

    print(len(graphs))
    
    # generate training data 
    tmp = args.samples.split("-")
    num_neg_cla=int(tmp[0]) 
    num_neg=int(tmp[1])
    num_pos=int(tmp[2])
    pairs_pos, pairs_neg = build_triplets(graphs, labels, 0, args.n_train, num_neg_cla=num_neg_cla, num_neg=num_neg, num_pos=num_pos)
    pairs = list(pairs_pos) + list(pairs_neg)
    random.shuffle(pairs)
    
    # generate validation data 
    pairs_pos_val, pairs_neg_val = build_triplets(graphs, labels, args.n_train, args.n_train+args.n_val, num_neg_cla=num_neg_cla, num_neg=num_neg, num_pos=num_pos)
    

    # loaders
    loader_pos_tr = DataLoader(pairs_pos, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_neg_tr = DataLoader(pairs_neg, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    train_loader = DataLoader(pairs, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = True)
    
    loader_pos_val = DataLoader(pairs_pos_val, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_neg_val = DataLoader(pairs_neg_val, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_pos_tr_short = DataLoader(pairs_pos[:len(pairs_pos_val)], batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_neg_tr_short = DataLoader(pairs_neg[:len(pairs_pos_val)], batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    

    # model
    if args.net == 'ot': model = GEDNet(graphs[0].x.size(1), gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, reg=args.reg).to(args.device) 
    elif args.net == 'dist': model = DistanceNet(graphs[0].x.size(1), gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim).to(args.device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    
    with open(args.logs+"/"+args.basename+".log", "w") as myfile:
        myfile.write(str(args)+"\n")
        myfile.write("Epoch, train trip loss, train tripl acc, val trip loss, val pair auroc, val tripl acc\n")
        
    best_val = 0 # auroc
    best_model = None
    cnt_pat = 0

    margin = args.margin

    val_loss, val_acc, val_auroc, tmp1, tmp2 = val_step_triplet(model, loader_pos_val, loader_neg_val, margin=margin, device=args.device)
    print('Baseline:', val_acc, val_auroc)

    for epoch in range(args.epochs):
        # training
        if args.shuffled: loss = train_step(model, train_loader, optimizer, margin=margin, device=args.device)
        else: loss = train_step_triplet(model, loader_pos_tr, loader_neg_tr, optimizer, margin=margin, device=args.device)

        # train metrics
        train_loss, train_acc, train_auroc = 0,0,0
        train_loss, train_acc, train_auroc, tmp1, tmp2 = val_step_triplet(model, loader_pos_tr_short, loader_neg_tr_short, margin=margin, device=args.device)
        # val metrics
        val_loss, val_acc, val_auroc, tmp1, tmp2 = val_step_triplet(model, loader_pos_val, loader_neg_val, margin=margin, device=args.device)

        # logging
        log_data(args.logs+"/"+args.basename+".log", epoch, loss, train_acc, train_auroc, val_acc, val_auroc)

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
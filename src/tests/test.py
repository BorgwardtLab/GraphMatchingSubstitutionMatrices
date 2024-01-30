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
from models.models import GEDModel, GATModel
from models.model_fixed import FixedCostModel
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
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--gnn", action='store_true')
    parser.add_argument("--uniform-costs", action='store_true')
    parser.add_argument("--ckp", type=str, help="Path to saved model")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--samples", type=str)
    parser.add_argument("--logs", type=str, help="Path to logs")

    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print(args.device)

    if args.uniform_costs:
        args.basename = args.graphs.split("/")[-1]+"_uniform_seed"+str(args.seed)
    else:
        args.basename = args.ckp.split("/")[-1]+"_seed"+str(args.seed)
    

    torch.set_printoptions(precision=3, linewidth=300)

    return args




def main():
    args = cline()
    print(args, flush=True)
    
    tmp = torch.load(args.graphs)
    graphs = tmp[0]
    labels = tmp[1]

    print(len(graphs))
    

    # generate test data 
    tmp = args.samples.split("-")
    num_neg_cla=int(tmp[0]) 
    num_neg=int(tmp[1])
    frac_pos=float(tmp[2])
    
    pairs_pos_te, pairs_neg_te = build_triplets(graphs, labels, args.n_train+args.n_val, args.n_train+args.n_val+args.n_test, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    
    loader_test_pos = DataLoader(pairs_pos_te, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    loader_test_neg = DataLoader(pairs_neg_te, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
    
    


    if args.uniform_costs:
        model = FixedCostModel(graphs[0].x.size(1)).to(args.device)
    elif args.gnn:
        model = GATModel(graphs[0].x.size(1), gnn_num_layers=args.layers, embedding_dim=args.embedding_dim)
        model.load_state_dict(torch.load(args.ckp))
        model = model.to(args.device) 
    else:
        model = GEDModel(graphs[0].x.size(1), gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim)
        model.load_state_dict(torch.load(args.ckp))
        model = model.to(args.device) 

    model.eval()


    
    
    
    margin = args.margin
    loss, acc, auroc, pos_geds, neg_geds = val_step_triplet(model, loader_test_pos, loader_test_neg, margin, args.device)

    print(loss, acc, auroc)

    if args.logs:
        with open(args.logs+"/test_"+args.basename+".log", "w") as myfile:
            myfile.write(str(args)+"\n")
            myfile.write(str(loss) +" "+ str(acc) +" "+ str(auroc))

        torch.save([pos_geds, neg_geds], args.logs+"/test_"+args.basename+".pt")

        if args.seed == 0:
            plt.figure()
            bins = np.histogram(np.hstack((pos_geds.cpu().numpy(), neg_geds.cpu().numpy())), bins=40, range=(0.0,2.0))[1] #get the bin edges
            plt.hist(pos_geds.cpu().numpy(), bins, alpha=0.4)
            plt.hist(neg_geds.cpu().numpy(), bins, alpha=0.4)
            plt.tight_layout()
            plt.savefig(args.logs+"/test_"+args.basename+".png")
    



        
if __name__ == '__main__':
    main()
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
from models.models import DistanceNet, GEDNet
from models.model_fixed import FixedCostModel
from models.wl import WWL, WL
from utils import *


disable_tqdm = False
args = None




def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--uniform-costs", action='store_true')
    parser.add_argument("--ckp", type=str, help="Path to saved model")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--samples", type=str)
    parser.add_argument("--logs", type=str, help="Path to logs")
    parser.add_argument("--net", type=str)
    parser.add_argument("--reg", type=float, default=0.0)

    args = parser.parse_args()
    
    
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print(args.device)

    if args.uniform_costs:
        args.basename = args.graphs.split("/")[-1]+"_uniform"
    elif args.net == 'wl':
        args.basename = args.graphs.split("/")[-1]+"_wl"
    else:
        if args.net == 'ot':
            args.basename = args.ckp.split("/")[-1]+"reg_"+str(args.reg)
        else:
            args.basename = args.ckp.split("/")[-1]
    

    torch.set_printoptions(precision=3, linewidth=300)

    return args




def main():
    args = cline()
    print(args, flush=True)
    
    # generate test data 
    tmp = args.samples.split("-")
    num_neg_cla=int(tmp[0]) 
    num_neg=int(tmp[1])
    num_pos=int(tmp[2])


    accs = []
    aurocs = []
    for seed in range(args.runs):
        # set seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        
        tmp = torch.load(args.graphs)
        graphs_tr, graphs_val, graphs_te = tmp[0]
        labels_tr, labels_val, labels_te  = tmp[1]
        print(len(graphs_tr), len(graphs_val), len(graphs_te))
        graphs = graphs_tr
        pairs_pos_te, pairs_neg_te = build_triplets(graphs_te, labels_te, 0, len(graphs_te), num_neg_cla=num_neg_cla, num_neg=num_neg, num_pos=num_pos)
            
        loader_test_pos = DataLoader(pairs_pos_te, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
        loader_test_neg = DataLoader(pairs_neg_te, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle = False)
        


        if args.uniform_costs:
            model = FixedCostModel(graphs[0].x.size(1)).to(args.device)
        elif args.net == 'wl':
            model = WL(args.layers)
            model.to(args.device)
            assert args.batch_size == 1
        elif args.net == 'dist':
            model = DistanceNet(graphs[0].x.size(1), gnn_num_layers=args.layers, embedding_dim=args.embedding_dim)
            model.load_state_dict(torch.load(args.ckp, map_location=torch.device('cpu')))
            model = model.to(args.device) 
        elif args.net == 'ot':
            model = GEDNet(graphs[0].x.size(1), gnn_num_layers=args.layers, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, reg=args.reg)
            model.load_state_dict(torch.load(args.ckp, map_location=torch.device('cpu')))
            model = model.to(args.device) 

        model.eval()


        
        margin = args.margin
        loss, acc, auroc, pos_geds, neg_geds = val_step_triplet(model, loader_test_pos, loader_test_neg, margin, args.device)
        print(loss, acc, auroc)
        accs.append(acc)
        aurocs.append(auroc)

    accs = np.array(accs)
    aurocs = np.array(aurocs)

    print(r"%.3f \tsmall{%.3f} & %.3f \tsmall{%.3f}" % (np.mean(accs), np.std(accs), np.mean(aurocs), np.std(aurocs)))


    if args.logs:
        with open(args.logs+"/test_"+args.basename+".log", "w") as myfile:
            myfile.write(str(args)+"\n")
            for ii in range(5):
                myfile.write(str(0) +" "+ str(accs[ii]) +" "+ str(aurocs[ii])+"\n")
            myfile.write(r"%.3f \tsmall{%.3f} & %.3f \tsmall{%.3f}" % (np.mean(accs), np.std(accs), np.mean(aurocs), np.std(aurocs)))


        #torch.save([pos_geds, neg_geds], args.logs+"/test_"+args.basename+".pt")
        if args.seed == 0:
            plt.figure()
            bins = np.histogram(np.hstack((pos_geds.cpu().numpy(), neg_geds.cpu().numpy())), bins=40, range=(0.0,2.0))[1] #get the bin edges
            plt.hist(pos_geds.cpu().numpy(), bins, alpha=0.4)
            plt.hist(neg_geds.cpu().numpy(), bins, alpha=0.4)
            plt.tight_layout()
            plt.savefig(args.logs+"/test_"+args.basename+".png")
    



        
if __name__ == '__main__':
    main()
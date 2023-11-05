import os, sys
import numpy as np
from timeit import default_timer as timer
import argparse
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
import pandas as pd
import random
import pickle
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import torch.autograd.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.transforms as T
import torchmetrics.functional as metrics_F
from torch_geometric.utils import to_networkx


from load_data import *
from models.models import GEDModelHomology, GEDModelHomologyLinear2, GEDModelHomologyLinear3



INF = 1000000000
disable_tqdm = False
args = None




def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--label", type=int, default=1, help="1=Pfam, 2=EC1, 3=SCOP-SF")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--noise", type=float, default=0.00)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--ckp", type=str, help="Path to saved model")
    parser.add_argument("--logs", type=str, help="Path to logs", default=".")
    parser.add_argument("--lin2", action='store_true')
    parser.add_argument("--uniform-costs", action='store_true')
    parser.add_argument("--id", type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(torch.cuda.current_device()) \
        if (torch.cuda.is_available() and args.cuda) else torch.device('cpu')
    print(args.device)

    #args.basename = f'{args.graphs.split("/")[-1]}_lab{args.label}_n{args.n_train}_l{args.layers}_soft-{args.soft}_seed{args.seed}'
    

    torch.set_printoptions(precision=3, linewidth=300)

    return args


def log_data(filename, epoch, loss, val_loss, auroc, val_auroc):
    print("epoch:", epoch, loss, val_loss, auroc, val_auroc, flush=True)
    with open(filename, "a") as myfile:
        myfile.write(str(epoch)+", "+str(loss)+", "+str(val_loss)+", "+str(auroc)+", "+str(val_auroc)+'\n')
    




def draw_alignment(graph_pair, assignment, edit_costs, pos1=None, pos2=None, return_pos=False, x_shift=4.0, ax=None):
    """
    Arguments
    -----------

    g_1: torch_geometric.Data
        First PyG Graph
    g_2: torch_geometric.Data
        Second PyG Graph
    assignment: numpy array
        N_g1 x N_g2  node assignment matrix
    """

    ax.axis('off')

    g_1_nx = to_networkx(Data(x=graph_pair.x_s, edge_index=graph_pair.edge_index_s))
    g_1_nx = nx.relabel_nodes(g_1_nx, {n: i for i, n in enumerate(sorted(g_1_nx.nodes()))})

    g_2_nx = to_networkx(Data(x=graph_pair.x_t, edge_index=graph_pair.edge_index_t))
    g_2_nx = nx.relabel_nodes(g_2_nx, {n: i+len(g_1_nx.nodes()) for i, n in enumerate(sorted(g_2_nx.nodes()))})

    #print("--")
    #print(graph_pair.edge_index_s)
    #print(graph_pair.edge_index_t)
    #print(g_1_nx.nodes())
    #print(g_2_nx.nodes())


    big_g = nx.Graph()

    big_g.add_nodes_from(g_1_nx.nodes(), label='1')
    big_g.add_edges_from(g_1_nx.edges(), label='1')

    big_g.add_nodes_from(g_2_nx.nodes(), label='2')
    big_g.add_edges_from(g_2_nx.edges(), label='2')


    map_edges = np.nonzero(assignment.cpu().numpy()[:len(g_1_nx.nodes()),:len(g_2_nx.nodes())])
    new_edges = [(u, v + len(g_1_nx.nodes())) for u, v in zip(map_edges[0], map_edges[1])]
    big_g.add_edges_from(new_edges, label='map')

    #if pos1 is None:
    #    pos1 = nx.spring_layout(g_1_nx)
    #if pos2 is None:
    #    pos2 = nx.spring_layout(g_2_nx)
    #pos = pos1 | pos2 #merge dicts

    pos = nx.spring_layout(big_g)

    # spread out the graphs
    for n in g_2_nx.nodes():
        pos[n][0] += x_shift

    nx.draw_networkx_nodes(big_g, pos, nodelist=g_1_nx.nodes(),  node_color='blue', ax=ax)
    nx.draw_networkx_edges(big_g, pos, edgelist=g_1_nx.edges(), ax=ax)

    nx.draw_networkx_nodes(big_g, pos, nodelist=g_2_nx.nodes(), node_color='red', ax=ax)
    nx.draw_networkx_edges(big_g, pos, edgelist=g_2_nx.edges(), ax=ax)

    map_edges = [(u, v) for u, v, d in big_g.edges(data=True) if d['label'] == 'map']
    nx.draw_networkx_edges(big_g, pos, edgelist=map_edges, style='dashed', edge_color='grey', ax=ax)

    nx.draw_networkx_labels(big_g, pos, ax=ax)

    if return_pos:
        return pos








def main():
    args = cline()
    print(args, flush=True)
    
    if args.graphs[-3:] == '.pt':
        tmp = torch.load(args.graphs)
        graphs = tmp[0]
        labels = tmp[1] #not really used
    else:
        graphs, labels = read_graph_label_data(args.graphs)

    print(len(graphs))
    print(graphs[0].x.shape)

    
    # generate training data and validation data 
    pairs_pos, pairs_neg = build_triplets(graphs, labels, 0, 500, num_neg_cla=1, num_neg=1, frac_pos=1e-12)
    
    pairs = []
    maxsize = 50
    minsize = 10
    for pair in pairs_pos:
        if pair.x_s.shape[0] > maxsize or  pair.x_t.shape[0] > maxsize:
            continue
        if pair.x_s.shape[0] < minsize or  pair.x_t.shape[0] < minsize:
            continue
        pairs.append(pair)



    if args.uniform_costs:
        model = FixedCostModel(graphs[0].x.size(1)).to(args.device)
    elif args.lin2:
        model = GEDModelHomologyLinear2(graphs[0].x.size(1), gnn_num_layers=args.layers, embedding_dim=args.embedding_dim)
        model.load_state_dict(torch.load(args.ckp))
        model = model.to(args.device) 
    else:
        model = GEDModelHomology(graphs[0].x.size(1), embedding_dim=args.embedding_dim, gnn_num_layers=args.layers)
        model.load_state_dict(torch.load(args.ckp))
        model = model.to(args.device) 
    
    
    model.eval()

    loader = DataLoader(pairs, batch_size=1, follow_batch=['x_s', 'x_t'], shuffle = False)
    
    for batch in loader:
        with torch.no_grad():
            lin_nodemaps, edit_costs, geds = model(batch.to(args.device))

            #plt.imshow(edit_costs[0].cpu().numpy(), cmap='hot', interpolation='nearest')
            fig, ax = plt.subplots()
            sns.heatmap(edit_costs[0].cpu().numpy())
            plt.tight_layout()
            plt.savefig(args.logs+"/costs.png")

            fig, ax = plt.subplots()
            draw_alignment(batch, lin_nodemaps[0], edit_costs[0], ax=ax)
            plt.tight_layout()
            plt.savefig(args.logs+"/alignment.png")
            break

        
if __name__ == '__main__':
    main()
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

import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm


from load_data import *
from models.wl import WLKernel


INF = 1000000000
disable_tqdm = False
args = None





@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.
    Code stolen from https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def distribute_function(func, X, n_jobs = 1, **kwargs,):
    """Distributes function `func` over iterable `X` using `n_jobs` cores.
    Args:
        func (Callable): function to be distributed
        X (Iterable): iterable over which the function is distributed
        n_jobs (int): number of cores to use
        description (str, optional): Description of the progress. Defaults to "".
    Returns:
        Any: result of the `func` applied to `X`.
    """

    with tqdm_joblib(tqdm(total=len(X), ncols=64)):
        Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    return Xt




def retrieve_hits(query_graph, model=None, database_graphs=None, k=0, device='cpu', batch_size=256):
    query_pairs = build_pairs_query(query_graph, database_graphs, [0], [0]*len(database_graphs))
    query_pairs_loader = DataLoader(query_pairs, batch_size=batch_size, follow_batch=['x_s', 'x_t'], shuffle = False, num_workers = 0)

    all_geds = []
    with torch.no_grad():
        for batch in query_pairs_loader:
            sim = model(batch.to(device))
            all_geds.append(-sim)

    all_geds = torch.cat(all_geds).cpu().numpy()

    indexes = all_geds.argsort()[:k]
    return indexes

def build_pairs_query(query, dataset, label_query, labels):
    pairs = []
    for i in range(len(dataset)):
        hom = 1.0 if label_query == labels[i] else 0.0
        pairs.append( PairData( (dataset[i], query, i, 0, hom) ) )
    return pairs



def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--label", type=int, default=1, help="1=Pfam, 2=SCOP-SF")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training graphs")
    parser.add_argument("--n-val", type=int, default=100, help="Number of validation graphs")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test graphs")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--knn", type=int, default=50)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--cpus", type=int, default=0)
    parser.add_argument("--logs", type=str, help="Path to logs", default="./")

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





def main():
    args = cline()
    print(args, flush=True)
    
    tmp = torch.load(args.graphs)
    graphs = tmp[0]
    labels = tmp[args.label]

    print(len(graphs))

    model = WLKernel(args.layers).to(args.device) 
    model.eval()

    train_graphs = graphs[0:args.n_train]
    test_graphs = graphs[args.n_train+args.n_val:args.n_train+args.n_val+args.n_test]

    with open(args.logs+"retrieve_"+args.basename+".log", "w") as myfile:
        myfile.write(str(args)+"\n")
        myfile.write("Query: hits\n")

    
    results = []

    if args.cpus <= 1:
        for i in tqdm(range(len(test_graphs)), ncols=64):
            indexes = retrieve_hits(test_graphs[i], model=model, database_graphs=train_graphs, k=args.knn, device=args.device, batch_size=args.batch_size)
            results.append(indexes)
    else:
        torch.set_num_threads(1)
        results = distribute_function(retrieve_hits, test_graphs, model=model, n_jobs=args.cpus, database_graphs=train_graphs, k=args.knn, device='cpu', batch_size=args.batch_size)

    with open(args.logs+"retrieve_"+args.basename+".log", "a") as myfile:
        for i in range(len(test_graphs)):
            indexes = results[i]
            myfile.write(str(i)+":")
            for idx in indexes:
                myfile.write(" "+str(idx))
            myfile.write("\n")
    



if __name__ == '__main__':
    main()
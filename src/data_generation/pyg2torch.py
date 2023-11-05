import os, sys
import numpy as np
from timeit import default_timer as timer
import argparse
import scipy
from tqdm import tqdm
import pickle
import argparse
import pandas as pd
import random
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet, TUDataset

from load_data import *

def main():
    data = TUDataset(root = sys.argv[1], name= sys.argv[2])

    graphs = []
    labels = []
    print(len(data))
    for i in range(len(data)):
        y = data[i].y.item()
        graphs.append(data[i])
        labels.append(y)

    for i in range(len(graphs)):
        if 'edge_attr' not in graphs[i]:
            graphs[i] = Data(x = graphs[i].x, edge_index = graphs[i].edge_index, edge_attr = torch.zeros(graphs[i].edge_index.shape[1]))
        #print(graphs[i])

    tmp = list(zip(graphs, labels))
    random.shuffle(tmp)
    graphs, labels = zip(*tmp)



    torch.save([graphs, labels], sys.argv[3]+".pt")


if __name__ == '__main__':
    main()




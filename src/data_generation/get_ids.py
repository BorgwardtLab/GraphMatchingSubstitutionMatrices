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




def main():
    tmp = torch.load(sys.argv[1])
    graphs = tmp[0]
    labels = tmp[1]
    ids = tmp[-1]

    ii = int(sys.argv[2])
    print(graphs[ii], ids[ii])

    exit()

    print(len(graphs))

    with open(sys.argv[1].split("/")[-1]+"-ids.txt", "w") as myfile:
        for i in range(len(graphs)):
            split = 0
            if i >= int(sys.argv[2]) and i < int(sys.argv[2])+int(sys.argv[3]):
                split = 1
            if i >= int(sys.argv[2])+int(sys.argv[3]):
                split = 2

            myfile.write(ids[i]+" "+str(labels[i])+" "+str(split)+"\n")

if __name__ == '__main__':
    main()
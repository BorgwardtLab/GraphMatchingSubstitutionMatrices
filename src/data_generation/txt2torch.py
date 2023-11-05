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

from load_data import *

def main():
    graphs, labels = read_graph_label_data(sys.argv[1])

    tmp = list(zip(graphs, labels))
    random.shuffle(tmp)
    graphs, labels = zip(*tmp)

    torch.save([graphs, labels], sys.argv[1][:-3]+"pt")


if __name__ == '__main__':
    main()




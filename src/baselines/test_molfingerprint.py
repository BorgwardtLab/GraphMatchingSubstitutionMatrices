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
import seaborn as sns 

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import DataStructs

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
from utils import *

from matplotlib import rc

#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

mut_2_atoms = {
    0:  'C',
  1:  'O',
  2:  'Cl',
  3:  'H',
  4:  'N',
  5:  'F',
  6:  'Br',
  7:  'S',
  8:  'P',
  9:  'I',
  10:  'Na',
  11:  'K',
  12:  'Li',
  13:  'Ca',
}

aids_2_atoms = {
    0: 'C',
    1: 'O',
    2: 'N',  
    3: 'Cl', 
    4: 'F',  
    5: 'S',
    6: 'Se',
    7: 'P',
    8: 'Na',
    9: 'I',
    10: 'Co',
    11: 'Br',
    12: 'Li',
    13: 'Si',
    14: 'Mg',
    15: 'Cu',
    16: 'As',
    17: 'B',
    18: 'Pt',
    19: 'Ru',
    20: 'K',
    21: 'Pd',
    22: 'Au',
    23: 'Te',
    24: 'W',
    25: 'Rh',
    26: 'Zn',
    27: 'Bi',
    28: 'Pb',
    29: 'Ge',
    30: 'Sb',
    31: 'Sn',
    32: 'Ga',
    33: 'Hg',
    34: 'Ho',
    35: 'Tl',
    36: 'Ni',
    37: 'Tb',
}

nci1_2_atoms = {
    0: "O",
    1: "N",
    2: "C",
    3: "S",
    4: "Cl",
    5: "P",
    6: "F",
    7: "Na",
    8: "Sn",
    9: "Pt",
    10: "Ni",
    11: "Zn",
    12: "Mn",
    13: "Br",
    14: "Cu",
    15: "Co",
    16: "Se",
    17: "Au",
    18: "Pb",
    19: "Ge",
    20: "I",
    21: "Si",
    22: "Fe",
    23: "Cr",
    24: "Hg",
    25: "As",
    26: "B",
    27: "Ga",
    28: "Ti",
    29: "Bi",
    30: "Y",
    31: "Nd",
    32: "Eu",
    33: "Tl",
    34: "Zr",
    35: "Hf",
    36: "In",
    37: "K",
    38: "La",
}

tox21_2_atoms = {
    0: 'O',
    1: 'C',
    2: 'N',
    3: 'F',
    4: 'Cl',
    5: 'S',
    6: 'Br',
    7: 'Si',
    8: 'Na',
    9: 'I',
    10: 'Hg',
    11: 'B',
    12: 'K',
    13: 'P',
    14: 'Au',
    15: 'Cr',
    16: 'Sn',
    17: 'Ca',
    18: 'Cd',
    19: 'Zn',
    20: 'V',
    21: 'As',
    22: 'Li',
    23: 'Cu',
    24: 'Co',
    25: 'Ag',
    26: 'Se',
    27: 'Pt',
    28: 'Al',
    29: 'Bi',
    30: 'Sb',
    31: 'Ba',
    32: 'Fe',
    33: 'H',
    34: 'Ti',
    35: 'Tl',
    36: 'Sr',
    37: 'In',
    38: 'Dy',
    39: 'Ni',
    40: 'Be',
    41: 'Mg',
    42: 'Nd',
    43: 'Pd',
    44: 'Mn',
    45: 'Zr',
    46: 'Pb',
    47: 'Yb',
    48: 'Mo',
    49: 'Ge',
    50: 'Ru',
    51: 'Eu',
    52: 'Sc'
}

mut_2_bond = {
    0: Chem.rdchem.BondType.SINGLE,
    1: Chem.rdchem.BondType.DOUBLE,
    2: Chem.rdchem.BondType.TRIPLE
}

nci1_2_bond = {
    0: Chem.rdchem.BondType.UNSPECIFIED,
}

tox21_2_bond = {
    0: Chem.rdchem.BondType.SINGLE,
    1: Chem.rdchem.BondType.DOUBLE,
    2: Chem.rdchem.BondType.AROMATIC,
    3: Chem.rdchem.BondType.TRIPLE
}

def molFromGraph(node_list, edge_index, edge_attr, mapa, mapb):
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        atom = mapa[node_list[i]]
        a = Chem.Atom(atom)
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for i in range(edge_index.shape[1]):
        # add relevant bond type (there are many more of these)
        if edge_index[0,i] >= edge_index[1,i]:
            continue 
        bond_type = mapb[edge_attr[i]]
        
        mol.AddBond(int(edge_index[0,i]), int(edge_index[1,i]), bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()            

    return mol


def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, help="Path to edgelist file")
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--n-train", type=int, default=100, help="Number of training graphs")
    parser.add_argument("--n-val", type=int, default=100, help="Number of validation graphs")
    parser.add_argument("--n-test", type=int, default=100, help="Number of test graphs")
    parser.add_argument("--samples", type=str, default="10-1-0.25")
    parser.add_argument("--logs", type=str, help="Path to logs", default=".")
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.basename = f'{args.graphs.split("/")[-1]}'

    torch.set_printoptions(precision=3, linewidth=300)

    return args


def main():
    args = cline()
    print(args, flush=True)
    
    tmp = torch.load(args.graphs)
    graphs = tmp[0]
    labels = tmp[args.label]
    
    print(len(graphs))
    

    # generate test data 
    tmp = args.samples.split("-")
    num_neg_cla=int(tmp[0]) 
    num_neg=int(tmp[1])
    frac_pos=float(tmp[2])

    #pairs_pos, pairs_neg = build_triplets(graphs, labels, 0, args.n_train, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)

    pairs_pos_te, pairs_neg_te = build_triplets(graphs, labels, args.n_train+args.n_val, args.n_train+args.n_val+args.n_test, num_neg_cla=num_neg_cla, num_neg=num_neg, frac_pos=frac_pos)
    pairs_te = [val for pair in zip(pairs_pos_te, pairs_neg_te) for val in pair]

    fpgen = AllChem.GetRDKitFPGenerator()

    if args.graphs.split("/")[-1] == 'AIDS.pt':
        mapa = aids_2_atoms
        mapb = mut_2_bond
    elif args.graphs.split("/")[-1] == 'Mutagenicity.pt':
        mapa = mut_2_atoms
        mapb = mut_2_bond
    else:
        mapa = nci1_2_atoms
        mapb = nci1_2_bond


    sim_pos = []
    sim_neg = []

    mol_pos = []

    for graph in tqdm(pairs_pos_te, ncols=64):
        x_s = graph.x_s
        edge_index_s = graph.edge_index_s
        edge_attr_s = graph.edge_attr_s
        x_s = np.argmax(x_s.numpy(), axis=1)
        edge_index_s = edge_index_s.numpy()
        if edge_attr_s.dim() > 1:
            edge_attr_s = np.argmax(edge_attr_s.numpy(), axis=1)
        else:
            edge_attr_s = edge_attr_s.numpy().astype(int)

        x_t = graph.x_t
        edge_index_t = graph.edge_index_t
        edge_attr_t = graph.edge_attr_t
        x_t = np.argmax(x_t.numpy(), axis=1)
        edge_index_t = edge_index_t.numpy()
        if edge_attr_t.dim() > 1:
            edge_attr_t = np.argmax(edge_attr_t.numpy(), axis=1)
        else:
            edge_attr_t = edge_attr_t.numpy().astype(int)

        m_s = molFromGraph(x_s, edge_index_s, edge_attr_s, mapa, mapb)
        m_t = molFromGraph(x_t, edge_index_t, edge_attr_t, mapa, mapb)

        mol_pos.append((m_s, m_t))

        fp_s = fpgen.GetFingerprint(m_s)
        fp_t = fpgen.GetFingerprint(m_t)

        sim_pos.append(DataStructs.TanimotoSimilarity(fp_s, fp_t))

    for graph in tqdm(pairs_neg_te, ncols=64):
        x_s = graph.x_s
        edge_index_s = graph.edge_index_s
        edge_attr_s = graph.edge_attr_s
        x_s = np.argmax(x_s.numpy(), axis=1)
        edge_index_s = edge_index_s.numpy()
        if edge_attr_s.dim() > 1:
            edge_attr_s = np.argmax(edge_attr_s.numpy(), axis=1)
        else:
            edge_attr_s = edge_attr_s.numpy().astype(int)

        x_t = graph.x_t
        edge_index_t = graph.edge_index_t
        edge_attr_t = graph.edge_attr_t
        x_t = np.argmax(x_t.numpy(), axis=1)
        edge_index_t = edge_index_t.numpy()
        if edge_attr_t.dim() > 1:
            edge_attr_t = np.argmax(edge_attr_t.numpy(), axis=1)
        else:
            edge_attr_t = edge_attr_t.numpy().astype(int)

        m_s = molFromGraph(x_s, edge_index_s, edge_attr_s, mapa, mapb)
        m_t = molFromGraph(x_t, edge_index_t, edge_attr_t, mapa, mapb)

        fp_s = fpgen.GetFingerprint(m_s)
        fp_t = fpgen.GetFingerprint(m_t)

        sim_neg.append(DataStructs.TanimotoSimilarity(fp_s, fp_t))

    mini = np.argmin(np.array(sim_pos))
    maxi = np.argmax(np.array(sim_pos))

    #print(Chem.MolToSmiles(mol_pos[mini][0]))
    #print(Chem.MolToSmiles(mol_pos[mini][1]))
    #print()
    #print(Chem.MolToSmiles(mol_pos[maxi][0]))
    #print(Chem.MolToSmiles(mol_pos[maxi][1]))
    
    trip_acc = sum([x > y for x,y in zip(sim_pos, sim_neg)]) / len(sim_pos)
    print(trip_acc)

    sim = torch.tensor(sim_pos+sim_neg)
    hom = torch.cat([torch.ones(len(pairs_pos_te)), torch.zeros(len(pairs_neg_te))] ).long()

    pair_auroc = metrics_F.auroc(sim, hom, task='binary').item()
    print(pair_auroc)

    with open(args.logs+"molfp_"+args.basename+".log", "a") as myfile:
        myfile.write(str(trip_acc)+" "+ str(pair_auroc)+"\n")
    


if __name__ == '__main__':
    main()
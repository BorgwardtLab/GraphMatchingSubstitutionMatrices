import os, sys
import numpy as np
import scipy
from tqdm import tqdm
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.autograd.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
import torch_geometric.transforms as T
import torchmetrics.functional as metrics_F


disable_tqdm = False



def log_data(filename, epoch, train_loss, train_acc, train_auroc, val_acc, val_auroc):
    print("epoch:", epoch, train_loss, train_acc, train_auroc, " " ,val_acc, val_auroc, flush=True)
    with open(filename, "a") as myfile:
        myfile.write(str(epoch)+", "+str(train_loss)+", "+str(train_acc)+", "+str(train_auroc)+", "+str(val_acc)+", "+str(val_auroc)+'\n')
    





def val_step_triplet(model, loader_pos, loader_neg, margin, device):
    loader_neg_it = iter(loader_neg)

    model.eval()
    logging_loss = 0
    trip_acc = 0
    all_geds_pos = []
    all_geds_neg = []
    num_pairs = 0
    num_batch = 0
    zero = torch.zeros(1).to(device)

    with torch.no_grad():
        for batch_pos in tqdm(loader_pos, disable=disable_tqdm, ncols=64):
            batch_neg = next(loader_neg_it)


            #with profiler.profile(with_stack=True, profile_memory=True) as prof:
            lin_nodemaps, edit_costs, geds_pos = model(batch_pos.to(device))
            lin_nodemaps, edit_costs, geds_neg = model(batch_neg.to(device))
            #print(prof.key_averages(group_by_stack_n=10).table(sort_by='self_cpu_time_total', row_limit=10))    
            del lin_nodemaps, edit_costs
            #loss = (batch.hom @ geds)/batch.hom.sum() + ((1 - batch.hom) @ torch.maximum(zero, margin - geds))/(1 - batch.hom).sum()
            loss = torch.sum( torch.maximum(zero, geds_pos - geds_neg + margin) )
            trip_acc += (torch.count_nonzero(torch.maximum(zero, geds_neg - geds_pos))).detach().item()
            all_geds_pos.append(geds_pos)
            all_geds_neg.append(geds_neg)

            
            num_pairs += len(batch_pos)
            num_batch += 1
            logging_loss += loss.detach().item()
    
    all_geds = torch.cat(all_geds_pos+all_geds_neg)
    all_geds_pos = torch.cat(all_geds_pos)
    all_geds_neg = torch.cat(all_geds_neg)
    all_homs = torch.cat([torch.ones_like(all_geds_pos), torch.zeros_like(all_geds_neg)] ).long()
    logging_loss = logging_loss/num_pairs
    trip_acc = trip_acc/num_pairs
    pair_auroc = metrics_F.auroc(torch.exp(-all_geds), all_homs, task='binary').item()
    
    return logging_loss, trip_acc, pair_auroc, all_geds_pos, all_geds_neg


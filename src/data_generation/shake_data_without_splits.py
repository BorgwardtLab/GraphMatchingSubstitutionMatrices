from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from proteinshake.tasks import Task
from proteinshake.datasets import Dataset 
from proteinshake.datasets import EnzymeCommissionDataset, ProteinFamilyDataset, GeneOntologyDataset, SCOPDataset
from proteinshake.utils import write_avro
import torch
from torch_geometric.data import Data
import random

class MergedDataset(Dataset):
    def __init__(self, root, datasets, use_precomputed=True):
        self.datasets = datasets
        super().__init__(root, use_precomputed=use_precomputed)
        pass

    def download(self):
        pass

    def parse(self):
        protein_counts = Counter()
        all_prots = {}
        for dset in self.datasets:
            for _,prot in dset:
                pid = prot['protein']['ID']
                protein_counts.update([pid])
                if prot['protein']['ID'] in all_prots:
                    # merge the protein dicts 
                    all_prots[pid] = {'protein': {**all_prots[pid]['protein'], **prot['protein']},
                                      'residue': {**all_prots[pid]['residue'], **prot['residue']}
                                     }
                else:
                    all_prots[pid] = prot

        new_prots = [p for pid, p in all_prots.items() if protein_counts[pid] == len(self.datasets)]
        write_avro(new_prots, f'{self.root}/{self.name}.residue.avro')
    pass


if __name__ == "__main__":
    # tasks to merge
    dset1 = EnzymeCommissionDataset(root='../data/merge_ec').to_graph(eps=8).pyg()
    dset2 = ProteinFamilyDataset(root='../data/merge_pf').to_graph(eps=8).pyg()
    dset3 = SCOPDataset(root='../data/merge_scop').to_graph(eps=8).pyg()

    dsets = [dset1]

    folder = "merge"
    if dset1 in dsets:
        folder+= "_ec"
    if dset2 in dsets:
        folder+= "_pf"
    if dset3 in dsets:
        folder+= "_scop"

    dataset = MergedDataset('../data/'+folder, dsets, use_precomputed=True).to_graph(eps=8).pyg()
    print(len(dataset))

    

    maxsize = 512

    graphs = []
    lables_pfam = []
    labels_EC = []
    labels_scop_sf = []
    ids = []

    count_pfam = {}
    count_ec = {}
    count_scop = {}
    for data, protein_dict in dataset:
        if data.x.shape[0] > maxsize: continue

        if dset1 in dsets:
            ec = ((protein_dict['protein']['EC']).split('.'))[0]
            count_ec[ec] = count_ec.get(ec, 0) + 1

        if dset2 in dsets:
            for fam in protein_dict['protein']['Pfam']:
                count_pfam[fam] = count_pfam.get(fam, 0) + 1

        if dset3 in dsets:
            scop_sf = protein_dict['protein']['SCOP-SF']
            count_scop[scop_sf] = count_scop.get(scop_sf, 0) + 1

    cnt = 0
    for data, protein_dict in dataset:
        if data.x.shape[0] > maxsize: continue
        
        if dset1 in dsets:
            ec = ((protein_dict['protein']['EC']).split('.'))[0]
            if count_ec[ec] < 5: continue

        if dset2 in dsets:
            pfam = '' # get the most prevalent one
            for fam in protein_dict['protein']['Pfam']:
                if count_pfam[fam] > count_pfam.get(pfam, 0):
                    pfam = fam
            if count_pfam[pfam] < 5: continue # we want at least 5

        if dset3 in dsets:
            scop_sf = protein_dict['protein']['SCOP-SF']
            if count_scop[scop_sf] < 5: continue

        # finished filtering
        edge_attr = torch.zeros(data.edge_index.shape[1])
        x = torch.nn.functional.one_hot(data.x, num_classes=20).to(torch.float32)
        graph = Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)

        graphs.append(graph)

        if dset1 in dsets:
            labels_EC.append(ec)
        if dset2 in dsets:
            lables_pfam.append(pfam)
        if dset3 in dsets:
            labels_scop_sf.append(scop_sf)
        
        ids.append(protein_dict['protein']['ID'])

        cnt +=1


    out = [graphs]
    if dset1 in dsets:
        out.append(labels_EC)
    if dset2 in dsets:
        out.append(lables_pfam)
    if dset3 in dsets:
        out.append(labels_scop_sf) 
    out.append(ids)

    random.seed(0)
    tmp = list(zip(*out))
    random.shuffle(tmp)
    out = list(zip(*tmp))


    torch.save(
        out,
        folder+".pt"
    )

    print(cnt)

        
    
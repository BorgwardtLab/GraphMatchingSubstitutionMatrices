from pathlib import Path
import glob

from tqdm import tqdm
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from proteinshake.tasks import Task
from proteinshake.datasets import Dataset 
from proteinshake.datasets import EnzymeCommissionDataset 
from proteinshake.datasets import ProteinFamilyDataset
from proteinshake.datasets import SCOPDataset
from proteinshake.utils import write_avro, protein_to_pdb
from shake_release.structure_split import compute_structure_split
from shake_release.sequence_split import compute_sequence_split
from shake_release.random_split import compute_random_split

class MergedDataset(Dataset):
    def __init__(self, root, datasets, use_precomputed=True):
        super().__init__(root, use_precomputed=use_precomputed)
        if not use_precomputed:
            self.datasets = datasets
            self.new_prots = self.merge_prots()
            self.pdbids = {p['protein']['ID'] for p in self.new_prots}
        pass

    def download(self):
        print("download")
        for p in tqdm(self.new_prots, total=len(self.new_prots)):
            protein_to_pdb(p, f'{self.root}/raw/files/{p["protein"]["ID"]}.pdb')
        pass

    def get_id_from_filename(self, filename):
        return Path(filename).stem
    
    def get_raw_files(self):
        return glob.glob(f'{self.root}/raw/files/*.pdb')

    def parse(self):
        residue_proteins = [{'protein':p['protein'], 'residue':p['residue']} for p in self.new_prots]
        atom_proteins = [{'protein':p['protein'], 'atom':p['atom']} for p in self.new_prots]
        write_avro(residue_proteins, f'{self.root}/{self.name}.residue.avro')
        write_avro(atom_proteins, f'{self.root}/{self.name}.atom.avro')

    def merge_prots(self):
        protein_counts = Counter()
        all_prots = {}
        print("merging proteins")
        for dset in self.datasets:
            print(dset)
            i = 0
            for prot_atom, prot_res in zip(dset.proteins(resolution='atom'), dset.proteins(resolution='residue')):
                if not i % 1000:
                    print(i)
                pid = prot_atom['protein']['ID']
                protein_counts.update([pid])
                if prot_res['protein']['ID'] in all_prots:
                    # merge the protein dicts 
                    all_prots[pid] = {'protein': {**all_prots[pid]['protein'], **prot_res['protein']},
                                      'residue': {**all_prots[pid]['residue'], **prot_res['residue']},
                                      'atom': {**all_prots[pid]['atom'], **prot_atom['atom']}
                                     }
                else:
                    all_prots[pid] = {'protein': prot_res['protein'], 
                                      'residue': prot_res['residue'],
                                      'atom': prot_atom['atom']}
                i += 1

        new_prots = [p for pid, p in all_prots.items() if protein_counts[pid] == len(self.datasets)]
        return new_prots
    pass

if __name__ == "__main__":
    # tasks to merge
    dset1 = EnzymeCommissionDataset(root='ec')
    dset2 = ProteinFamilyDataset(root='pf')
    dset3 = SCOPDataset(root='scop')
    m = MergedDataset('scop_ec_pfam', [dset1, dset2, dset3], use_precomputed=True)
    da = m.to_graph(eps=9).pyg()
    print(da[0])

    # compute_random_split(m)
    # compute_sequence_split(m)
    # compute_structure_split(m)

    

import os
import numpy as np
from pathlib import Path
from Bio.PDB import *

AA_THREE_TO_ONE = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


def calc_residue_dist(residue_one, residue_two, atom='CA') :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one[atom].coord - residue_two[atom].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two, atom='CA') :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two, atom=atom)
    return answer

class ProtSelect(Select):
    def accept_residue(self, residue):
        try:
            residue['CA']
            return 1
        except:
            return 0
        else:
            if residue.get_resname() in AA_THREE_TO_ONE:
                return 1


for p in os.listdir("pdbs_query"):
    parser = PDBParser()
    s = parser.get_structure("", str(Path("pdbs_query", p)))
    chain = list(s.get_chains())[0]
    chain = [res for res in chain.get_residues() if 'CA' in res and 'CB' in res and res.get_resname() in AA_THREE_TO_ONE]
    DM_ca = calc_dist_matrix(chain, chain)
    DM_cb = calc_dist_matrix(chain, chain, atom='CB')
    seq = ''.join([AA_THREE_TO_ONE[r.get_resname()] for r in chain])
    np.savez(f"pdbs_query_ca/{p.split('.')[0]}.npz", C_alpha=DM_ca, C_beta=DM_cb, seqres=seq)
    print(f"success on {p}")



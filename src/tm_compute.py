# -*- coding: utf-8 -*-
'''
TMalign needs to be in your $PATH. Follow the instructions at https://zhanggroup.org/TM-align/readme.c++.txt
'''
import glob
import requests
import os
import itertools
import re
import subprocess
from pathlib import Path
import tempfile
import shutil
from tqdm import tqdm
import numpy as np
from biopandas.pdb import PandasPdb
from collections import defaultdict
from joblib import Parallel, delayed
from functools import cached_property

from proteinshake.utils import (extract_tar,
                                download_url,
                                save,
                                load,
                                unzip_file,
                                global_distance_test,
                                local_distance_difference_test,
                                progressbar
                                )

def dl_pdbs(pdbids, root='../data/pdbs/', debug=False):
    paths = []
    lim = 10 if debug else len(pdbids)
    for pdbid in tqdm(pdbids[:lim], total=lim, desc="Downloading"):
        pdb_path = os.path.join(root, pdbid + ".pdb")
        if not os.path.exists(pdb_path):
            ppdb = PandasPdb().fetch_pdb(pdbid)
            ppdb.to_pdb(path=pdb_path, gz=False, records=None)
        paths.append(Path(pdb_path))
    return paths

def align_structures(paths, root='.', n_jobs=96, name='merge_ec.pt-ids', debug=False):
    """ Calls TMalign on all pairs of structures and saves the output"""
    path_dict = {Path(f).stem:f for f in paths}
    pdbids = list(path_dict.keys())
    num_proteins = 10 if debug else len(paths)
    combinations = np.array(list(itertools.combinations(range(num_proteins), 2)))
    TM, RMSD, GDT, LDDT = [np.ones((num_proteins,num_proteins), dtype=np.float16) * np.nan for _ in ['tm','rmsd','gdt','lddt']]
    np.fill_diagonal(TM, 1.0), np.fill_diagonal(RMSD, 0.0), np.fill_diagonal(GDT, 1.0), np.fill_diagonal(LDDT, 1.0)
    d = Parallel(n_jobs=n_jobs)(delayed(tmalign_wrapper)(paths[i], paths[j]) for i,j in progressbar(combinations, desc='Aligning'))
    x,y = tuple(combinations[:,0]), tuple(combinations[:,1])
    TM[x,y] = [x['TM1'] for x in d]
    TM[y,x] = [x['TM2'] for x in d]
    RMSD[x,y] = [x['RMSD'] for x in d]
    RMSD[y,x] = [x['RMSD'] for x in d]
    GDT[x,y] = [x['GDT'] for x in d]
    GDT[y,x] = [x['GDT'] for x in d]
    LDDT[x,y] = [x['LDDT'] for x in d]
    LDDT[y,x] = [x['LDDT'] for x in d]
    # save
    np.save(f'{root}/{name}.tmscore.npy', TM)
    np.save(f'{root}/{name}.rmsd.npy', RMSD)
    np.save(f'{root}/{name}.gdt.npy', GDT)
    np.save(f'{root}/{name}.lddt.npy', LDDT)


def tmalign_wrapper(pdb1, pdb2):
    """Compute TM score with TMalign between two PDB structures.
    Parameters
    ----------
    pdb1: str
        Path to PDB.
    pdb2 : str
        Path to PDB.
    return_superposition: bool
        If True, returns a protein dataframe with superposed structures.
    Returns
    -------
    dict
        Metric values TM1/TM2 (TM-Scores normalized to pdb1 or pdb2), RMSD, GDT
    """
    assert shutil.which('TMalign') is not None,\
           "No TMalign installation found. Go here to install : https://zhanggroup.org/TM-align/TMalign.cpp"
    with tempfile.TemporaryDirectory() as tmpdir:
        lines = subprocess.run(['TMalign','-outfmt','-1', pdb1, pdb2, '-o', f'{tmpdir}/superposition'], stdout=subprocess.PIPE).stdout.decode().split('\n')
        TM1 = lines[7].split()[1]
        TM2 = lines[8].split()[1]
        RMSD = lines[6].split()[4][:-1]
        seq1, ali, seq2 = lines[12], lines[13], lines[14]
        i, j, alignmentA, alignmentB = 0, 0, [], []
        for s1,a,s2 in zip(seq1,ali,seq2):
            if a != ' ': alignmentA.append(i)
            if a != ' ': alignmentB.append(j)
            if s1 != '-': i += 1
            if s2 != '-': j += 1
        os.rename(f'{tmpdir}/superposition_all', f'{tmpdir}/superposition_all.pdb')
        superposition = PandasPdb().read_pdb(f'{tmpdir}/superposition_all.pdb')
        df = superposition.df['ATOM']
        A = df[df['chain_id'] == 'A']
        B = df[df['chain_id'] == 'B']
        coordsA = np.array(list(zip(A['x_coord'], A['y_coord'], A['z_coord'])))[alignmentA]
        coordsB = np.array(list(zip(B['x_coord'], B['y_coord'], B['z_coord'])))[alignmentB]
        GDT = global_distance_test(coordsA, coordsB)
        LDDT = local_distance_difference_test(coordsA, coordsB)
    return {
        'TM1': float(TM1),
        'TM2': float(TM2),
        'RMSD': float(RMSD),
        'GDT': GDT,
        'LDDT': LDDT
    }

if __name__ == "__main__":
    debug = False
    todo = ["../data/merge_pf.pt-ids.txt", "../data/merge_scop.pt-ids.txt", "../data/merge_ec.pt-ids.txt"]
    for pdbid_path in todo:
        print(pdbid_path)

        pdbids = [l.split()[0] for l in open(pdbid_path, 'r').readlines()]

        paths = dl_pdbs(pdbids[:1000], debug=debug) 
        align_structures(paths, root='.', name=f'{Path(pdbid_path).stem}_first1000', debug=debug)

        paths = dl_pdbs(pdbids[-1000:], debug=debug) 
        align_structures(paths, root='.', name=f'{Path(pdbid_path).stem}_last1000', debug=debug)
    pass

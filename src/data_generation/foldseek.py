import os, itertools, tempfile, random, subprocess, shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
from biopandas.pdb import PandasPdb
from proteinshake.utils import progressbar


def pdb_dl(pdbid, dl_path):
    if not os.path.exists(dl_path):
        ppdb = PandasPdb().fetch_pdb(pdbid)
        ppdb.to_pdb(path=dl_path, gz=False, records=None)

def foldseek_create_databases(pdblist, root='../data', debug=False):

    name = Path(pdblist).stem
    pdb_dir = f'{root}/foldseek/{name}/pdbs_target'
    pdb_dir_query = f'{root}/foldseek/{name}/pdbs_query'
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(pdb_dir_query, exist_ok=True)

    pdbids = open(pdblist, 'r').readlines()
    lim = 100 if debug else len(pdbids)
    todo = []
    for entry in tqdm(pdbids[:lim], total=lim, desc="Downloading"):
        pdbid,_,split = entry.split()
        split = int(split)
        if split == 1: continue
        dl_path = pdb_dir if split == 0 else pdb_dir_query
        dl_path = Path(dl_path, pdbid + ".pdb")
        todo.append((pdbid, dl_path))

    d = Parallel(n_jobs=100)(delayed(pdb_dl)(pdbid, dl_path) for pdbid, dl_path in progressbar(todo, desc='Downloading'))
 
    out_path = f'{root}/foldseek/{name}'
    db_path = f'{out_path}/foldseekDB'

    os.makedirs(out_path, exist_ok=True)
    cmd = ['foldseek', 'createdb', pdb_dir, db_path]
    out = subprocess.run(cmd, capture_output=True, text=True)
    cmd = ['foldseek', 'createindex', db_path, out_path]
    out = subprocess.run(cmd, capture_output=True, text=True)

def foldseek_wrapper(name='merge_scop.pt-ids', n_jobs=8, root='../data'):

    assert shutil.which('foldseek') is not None,\
    "FoldSeek installation not found. Go here https://github.com/steineggerlab/foldseek to install"

    db_path = f'{root}/foldseek/{name}/foldseekDB'
    out_path = f'{root}/foldseek/{name}'
    out_file = f'{out_path}/output.m8'
    query_path = f'{root}/foldseek/{name}/pdbs_query'

    pdbid_to_int = {line.split()[0]: i for i, line in enumerate(open(f"../data/{name}.txt", 'r').readlines())}

    try:
        cmd = ['foldseek', 'easy-search', query_path, db_path, out_file, out_path,
            '--threads', str(min(n_jobs, 10)),
            '--max-seqs', '1000000000',
            '--format-output', 'query,target'
        ]
        out = subprocess.run(cmd, capture_output=True, text=True)
        with open(f"{name}_fs.out", "w") as f:
            f.write(str(out))

        hit_dict = defaultdict(set)
        with open(out_file, 'r') as file:
            for line in file:
                query, hit  = line.split()[:2]
                hit_dict[pdbid_to_int[query.split(".")[0]]].add(pdbid_to_int[hit.split(".")[0]])
        with open(name + "_foldseek_hits.txt", "w") as file:
            file.write("Id query: id hits\n")
            for query in sorted(hit_dict.keys()):
                file.write(f"{query}: {' '.join(map(str, hit_dict[query]))}\n")
    except Exception as e:
        print(e)
        return None 

if __name__ == "__main__":
    todo = ["../data/merge_pf.pt-ids.txt", "../data/merge_scop.pt-ids.txt", "../data/merge_ec.pt-ids.txt"]
    for dset in todo:
        foldseek_create_databases(dset, debug=False, root='../data')
        foldseek_wrapper(name=Path(dset).stem)

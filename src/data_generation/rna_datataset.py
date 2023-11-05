import itertools
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem 

from rnaglib.data_loading import RNADataset
from rnaglib.utils import bfs 
from rnaglib.config import LIGAND_TO_SMILES
from rnaglib.utils import NODE_FEATURE_MAP
from rnaglib.config.graph_keys import EDGE_MAP_RGLIB

def build_elist(G, g_label="", node_key='nt_hot', edge_key='etype_hot'):
    lines = [f"{len(G.nodes())} {len(G.edges())} {g_label}"]
    for n in sorted(G.nodes()):
        lines.append(f"{n} {G.nodes[n]['nt_hot']}")
    for u, v, d in sorted(G.edges(data=True)):
        lines.append(f"{u} {v} {d[edge_key]}")
    return lines

dset = RNADataset(redundancy='all')

# grab all the pockets as lists of node IDs
pocket_dict = defaultdict(list) 
ligands = []
for i, rna in enumerate(dset):
    graph = rna['rna']
    for n, d in graph.nodes(data=True):
        try:
            lig = d['binding_small-molecule']
            if not lig is None:
                pocket_dict[(graph.graph['pdbid'][0], lig[0], lig[1])].append(n)
                ligands.append(lig[0].lstrip("H_"))
        except KeyError:
            pass

# cluster the ligands

unique_ligs = sorted(list(set(ligands)))
cluster_ligs = [l for l in unique_ligs if l in LIGAND_TO_SMILES]
cluster_smiles = [LIGAND_TO_SMILES[l] for l in cluster_ligs]


## Build distance matrix between ligands
fpgen = AllChem.GetRDKitFPGenerator()
fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in cluster_smiles]
dists = 1 - np.array([DataStructs.TanimotoSimilarity(fp1,fp2) for fp1, fp2 in itertools.combinations(fps, 2)])
DM = np.zeros((len(fps), len(fps)))
DM[np.triu_indices(len(fps), 1)] = dists
DM += DM.T

clustering = AgglomerativeClustering(affinity='precomputed', 
                                     linkage='complete',
                                     n_clusters=None,
                                     distance_threshold=0.3).fit(DM)

id_to_clust = {lig: clustering.labels_[i] for i, lig in enumerate(cluster_ligs)}
clust_to_id = {clustering.labels_[i]: lig  for i, lig in enumerate(cluster_ligs)}

all_ligs_clust = [id_to_clust[lig] for lig in ligands if lig in cluster_ligs]
cluster_counts = Counter(all_ligs_clust)
keep_clusts = [idx for idx, count in cluster_counts.items() if count > 90 and count < 1100]
keep_ligs = [clust_to_id[clust] for clust in all_ligs_clust if clust in keep_clusts]
print(keep_ligs)


## Keep clusts with > 90 and < 1100 points
# turn pockets to subgraphs
pocket_graphs = []
for pocket, nodes in pocket_dict.items():
    lig = pocket[1].lstrip("H_")
    if lig not in keep_ligs:
        continue
    G = dset.get_pdbid(pocket[0])['rna']
    pocket_G = G.subgraph(nodes)

    # add 1-hop neighbors
    expanded_nodes = bfs(G, list(nodes), depth=1, label='LW')
    pocket_expand = G.subgraph(expanded_nodes).copy()
    pocket_graphs.append((pocket_expand, lig))
    pass

## Get some stats

rows = []
for i, (G, lig) in enumerate(pocket_graphs):
    rows.append({"idx": i, 
                 "connected components": len(list(nx.connected_components(G))),
                 "nodes": len(G.nodes()),
                 "edges": len(G.edges()),
                 "lig": lig,
                 "avg degree": np.mean([G.degree(n) for n in G.nodes()])
                 }
                )

df = pd.DataFrame(rows).groupby("lig").mean()
print(df.to_markdown(df))


lig_map = {lig: i for i, lig in enumerate(sorted(list(set(keep_ligs))))}

### dump 

elists = []
for G, lig in pocket_graphs:
    n = len(G.nodes())
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    remove_nodes = []
    for n,d in G.nodes(data=True):
        try:
            NODE_FEATURE_MAP['nt_code'].mapping[d['nt_code']]
        except KeyError:
            print(d['nt_code'])
            remove_nodes.append(n)
    print(f"killing {len(remove_nodes)}")
    G.remove_nodes_from(remove_nodes)
    e_one_hot = {edge: EDGE_MAP_RGLIB[label] for edge, label in
               (nx.get_edge_attributes(G, 'LW')).items()}
    nx.set_edge_attributes(G, name='etype_hot', values=e_one_hot)
    one_hot_nucs = {node: NODE_FEATURE_MAP['nt_code'].mapping[label] for node, label in
                    (nx.get_node_attributes(G, 'nt_code')).items()}
    nx.set_node_attributes(G, name='nt_hot', values=one_hot_nucs)
    elists.extend(build_elist(G, g_label=lig_map[lig]))

with open("rna_lig_class.txt", "w") as o:
    o.write("\n".join(elists))

# GraphMatchingSubstitutionMatrices

Code and data repository for the GMSM (Graph Matching Substitution Matrices) model from the RECOMB 2024 paper "Structure- and Function-Aware Substitution Matrices via Learnable Graph Matching". The model learns substitution matrices for biochemical structures over structural alphabets based on class labels (functional information).

| ![](imgs/Figure1.png) |
|:--| 
| *Architecture of GMSM. (a) Biochemical structures are transformed into graphs. (b) For each graph, its nodes are represented as a structure-aware embeddings using the same GNN. (c) The model computes the substitution matrix from node embeddings and obtains the graph alignment with respect to the learned substitution matrix.*|


### Citing our work
> Paolo Pellizzoni, C. Oliver and K. Borgwardt. “Structure- and function-aware substitution matrices via learnable graph matching”, in RECOMB, 2024. [[PDF]](https://link.springer.com/chapter/10.1007/978-1-0716-3989-4_18)



### Running the code

Our code is based on PyTorch and PyTorch Geometric. 
Run ```source s``` within the ```src/``` folder before running the code.

    python tests/test.py --graphs ../data/pf_split.pt --samples 10-2-0.01  --layers 3 --embedding-dim 64 --ckp ../out/out_pf/checkpoints/model_merge_pf.pt_l3_emb64.pt --cuda --seed 0
    python tests/test.py --graphs ../data/scop_split.pt --samples 30-2-0.02  --layers 3 --embedding-dim 64 --ckp ../out/out_scop/checkpoints/model_merge_scop.pt_l3_emb64.pt --cuda
    python tests/test.py --graphs ../data/ec_split.pt --samples 25-2-0.002  --layers 3 --embedding-dim 32 --ckp ../out/out_ec/checkpoints/model_merge_ec.pt_l3_emb32.pt --cuda
    python tests/test.py --graphs ../data/rna_lig_class_split.pt --samples 20-4-0.1 --embedding-dim 64 --layers 3 --ckp ../out/out_rna/checkpoints/model_rna_lig_class_l3_emb64.pt --cuda

    python tests/test.py --graphs ../data/Mutagenicity_split.pt --samples 1-10-0.1  --layers 3 --embedding-dim 64 --ckp ../out/out_mut/checkpoints/model_Mutagenicity.pt_l3_emb64.pt --cuda
    python tests/test.py --graphs ../data/NCI1_split.pt --samples 1-10-0.1 --layers 3 --embedding-dim 64 --ckp ../out/out_nci/checkpoints/model_NCI1.pt_l3_emb64.pt --cuda
    python tests/test.py  --graphs ../data/AIDS_split.pt --samples 1-100-0.2 --layers 3 --embedding-dim 64 --ckp ../out/out_aids/checkpoints/model_AIDS.pt_l3_emb64.pt --cuda

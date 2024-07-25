# GraphMatchingSubstitutionMatrices

Code and data repository for the GMSM (Graph Matching Substitution Matrices) model from the paper "Structure- and Function-Aware Substitution Matrices via Learnable Graph Matching" presented at RECOMB 2024 and at ICML 2024 Differentiable Almost Everything Workshop. The model learns substitution matrices for biochemical structures over structural alphabets based on class labels (functional information).

| ![](imgs/Figure1.png) |
|:--| 
| *Architecture of GMSM. (a) Biochemical structures are transformed into graphs. (b) For each graph, its nodes are represented as a structure-aware embeddings using the same GNN. (c) The model computes the substitution matrix from node embeddings and obtains the graph alignment with respect to the learned substitution matrix.*|


### Citing our work
> Paolo Pellizzoni, C. Oliver and K. Borgwardt. “Structure- and function-aware substitution matrices via learnable graph matching”, in RECOMB, 2024. [[PDF]](https://link.springer.com/chapter/10.1007/978-1-0716-3989-4_18)



### Running the code

Our code is based on PyTorch and PyTorch Geometric. 
Run ```source s``` within the ```src/``` folder before running the code.

Tests can be run with:

    python tests/test.py --graphs ../data/Mutagenicity_split.pt --ckp ../checkpoints/mut/model_Mutagenicity.pt_l2_emb64_hid64_netot1.0.pt --samples 1-10-10 --net ot
    python tests/test.py --graphs ../data/NCI1_split.pt --ckp ../checkpoints/nci/model_NCI1.pt_l2_emb64_hid64_netot1.0.pt --samples 1-10-10 --net ot
    python tests/test.py --graphs ../data/AIDS_split.pt --ckp ../checkpoints/aids/model_AIDS.pt_l2_emb64_hid64_netot0.1.pt --samples 1-10-10 --net ot

Retrieval can be run with:

    python tests/retrieve.py --graphs ../data/Mutagenicity_split.pt --ckp ../checkpoints/mut/model_Mutagenicity.pt_l2_emb64_hid64_netot1.0.pt --net ot --logs .
    python tests/retrieve.py --graphs ../data/NCI1_split.pt --ckp ../checkpoints/nci/model_NCI1.pt_l2_emb64_hid64_netot1.0.pt --net ot --logs .
    python tests/retrieve.py --graphs ../data/AIDS_split.pt --ckp ../checkpoints/aids/model_AIDS.pt_l2_emb64_hid64_netot0.1.pt --net ot --logs .

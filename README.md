# GraphMatchingSubstitutionMatrices

Scripts to test models

    python tests/test.py --graphs ../data/merge_ec.pt --n-train 11240 --n-val 1410 --n-test 1405  --samples 25-2-0.002  --ckp ../outc_ec/checkpoints/model_merge_ec.pt_lab1_n11240_l3_emb32.pt --lin2 --embedding-dim 32 --layers 3 --logs ../outc_ec/test/
    python tests/test.py --graphs ../data/merge_pf.pt --n-train 18700 --n-val 2340 --n-test 2337  --samples 10-2-0.01  --ckp ../outc_pfam/checkpoints/model_merge_pf.pt_lab1_n18700_l3_emb64.pt --lin2 --embedding-dim 64 --layers 3 --logs ../outc_pfam/test/ 
    python tests/test.py --graphs ../data/merge_scop.pt --n-train 6240 --n-val 780 --n-test 781  --samples 30-2-0.02  --ckp ../outc_scop/checkpoints/model_merge_scop.pt_lab1_n6240_l2_emb64.pt --lin2 --embedding-dim 64 --layers 2 --logs ../outc_scop/ 
    python tests/test.py --graphs ../data/rna/rna_lig_class.pt --n-train 688 --n-val 86 --n-test 87  --samples 50-10-0.1  --ckp ../outc_rna/checkpoints/model_rna_lig_class.pt_lab1_n688_l3_emb32.pt  -lin2 --embedding-dim 32 --layers 3 --logs ../outc_rna/
    
    python tests/test.py  --graphs ../data/AIDS.pt --n-train 1600 --n-val 200 --n-test 200 --samples 1-100-0.2 --ckp ../outc_mut/checkpoints/model_AIDS.pt_lab1_n1600_l3_emb32.pt --layers 3 --embedding-dim 32  --lin2 --logs ../outc_mut/test/
    python tests/test.py --graphs ../data/NCI1.pt --n-train 3288 --n-val 411 --n-test 411 --samples 1-10-0.1 --ckp ../outc_mut/baselines/model_NCI1.pt_lab1_n3288_l3_emb32.pt --lin2 --embedding-dim 32 --layers 3 --logs ../outc_mut/test/
    python tests/test.py --graphs ../data/Mutagenicity.pt --n-train 3470 --n-val 433 --n-test 434 --samples 1-10-0.1 --ckp ../outc_mut/baselines/model_Mutagenicity.pt_lab1_n3470_l3_emb64.pt --lin2 --embedding-dim 64 --layers 3 --logs ../outc_mut/test/

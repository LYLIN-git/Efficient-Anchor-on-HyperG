python Anchor_Figure_Generation.py - -dataset mushroom - -lam 1e-4 - -T 1000 - -anchor_k 32 2 > &1 | Tee-Object - FilePath mushroom_lam1e4_T1000_k32_run.txt
python Anchor_Figure_Generation.py - -dataset mushroom - -lam 1e-4 - -T 1000 - -anchor_k 64 2 > &1 | Tee-Object - FilePath mushroom_lam1e4_T1000_k64_run.txt
python Anchor_Figure_Generation.py - -dataset mushroom - -lam 1e-4 - -T 3000 - -anchor_k 32 2 > &1 | Tee-Object - FilePath mushroom_lam1e4_T3000_k32_run.txt
python Anchor_Figure_Generation.py - -dataset mushroom - -lam 1e-4 - -T 3000 - -anchor_k 64 2 > &1 | Tee-Object - FilePath mushroom_lam1e4_T3000_k64_run.txt

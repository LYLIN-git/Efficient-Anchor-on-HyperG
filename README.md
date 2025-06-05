--Anchor Method--

**Prerequisites**

pip install pybind11
conda install -c conda-forge eigen

**1. Generate the .pyd file**
#(Note: Whenever you modify the C++ modules, you must regenerate this file. Delete any existing .pyd before rebuilding. If you haven’t changed any C++ code, you can skip this step.)
python diffusion_wrapper_setup.py build_ext --inplace

**2. Run the Anchor Method and compare with PPR**

(1)Each run will create a new set of anchor data in the anchors/ folder. If you rerun or change parameters, delete the previous contents of anchors/ first.
(2)Basic commands to generate figures (you can adjust parameters here or directly inside Anchor_Figure_Generation.py):

python Anchor_Figure_Generation.py --dataset zoo --lam 1e-4 --T 1000 --anchor_k 64 2>&1 | Tee-Object -FilePath zoo_lam1e4_T1000_run.txt
python Anchor_Figure_Generation.py --dataset mushroom --lam 1e-7 --T 3000 --anchor_k 128 2>&1 | Tee-Object -FilePath mushroom_lam1e7_T1000_k128_run.txt

(3)To run with different loss methods (default is least squares):

python Anchor_Figure_Generation.py --dataset zoo --lam 1e-4 --T 1000 --anchor_k 64 --anchor_infer_method least_squares
python Anchor_Figure_Generation.py --dataset mushroom --lam 1e-4 --T 3000 --anchor_k 128 --anchor_infer_method linear_combo

**3. Anchor Method Algorithm (as shown in the presentation)**

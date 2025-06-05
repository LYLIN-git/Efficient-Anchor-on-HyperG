--Anchor Method--

**前置作業**
pip install pybind11
conda install -c conda-forge eigen
**

**1.生pyd (有修改到C+模組 都要重新生一次，舊的要砍掉；如果沒有動就忽略這個步驟)**
python diffusion_wrapper_setup.py build_ext --inplace


**2.跑anchor method 跟ppr比較**

(1)每一次都會生該次anchory 資料在anchors資料夾，重跑、改參數都要砍掉

(2)基礎跑圖指令： 參數可以調整 也可在Anchor_Figure_Generation.py裡面調


python Anchor_Figure_Generation.py --dataset zoo --lam 1e-4 --T 1000 --anchor_k 64 2>&1 | Tee-Object -FilePath zoo_lam1e4_T1000_run.txt
python Anchor_Figure_Generation.py --dataset mushroom --lam 1e-7 --T 1000 --anchor_k 128 2>&1 | Tee-Object -FilePath mushroom_lam1e7_T1000_k128_run.txt


(3)跑不同loss方法指令(預設least squares)：
python Anchor_Figure_Generation.py --dataset zoo --lam 1e-4 --T 3000 --anchor_k 128 --anchor_infer_method least_squares

python Anchor_Figure_Generation.py --dataset mushroom --lam 1e-4 --T 3000 --anchor_k 128 --anchor_infer_method linear_combo


三、Anchor method 演算法如簡報



# quick_test2.py
import os
import scipy.sparse as sp
from diffusion import GraphSolver
import faulthandler
faulthandler.enable()

print("exists?", os.path.exists("data/Paper_datasets/zoo.hmetis"))

# 1) 先用第一构造加载超图
tmp = GraphSolver("data/Paper_datasets/zoo.hmetis", "", "degree", 0)
n, m = tmp.n, tmp.m
degree = tmp.degree      # numpy array
hg = tmp.hypergraph      # list of lists

# 2) 构造 seeds 信息
seeds = [0]  # 测试用单节点
labels = [0 if i in seeds else 1 for i in range(n)]

# 3) 用第二构造明确传入 label_count、labels
solver = GraphSolver(n, m, degree, hg, len(seeds), labels, 1)
print("label_count =", solver.label_count)

# 4) 构造 (label_count×n) 稀疏 s
rows, cols, data = [0], [seeds[0]], [1.0]
s_sp = sp.csr_matrix((data, (rows, cols)), shape=(1, n))

# 5) 调用 diffusion
X = solver.diffusion(s_sp, 1, 1.0, 1e-3, 0)
print("X OK, shape =", X.shape)

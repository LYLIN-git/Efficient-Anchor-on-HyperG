from diffusion_functions import diffusion
import numpy as np
import time

# 加這一段
from reading import read_hypergraph
from anchor_util import _degree_vector   # 假設 degree 計算在這

# 換成你的檔案路徑
graph_path = "data/Paper_datasets/mushroom.hmetis"

# 正確解析出 n, m, hypergraph, weights ...
n, m, _node_w, hypergraph, weights, _center_id, _hnode_w = read_hypergraph(
    graph_path)
D = _degree_vector(n, hypergraph)

x0 = np.zeros((n, 1))
t0 = time.time()
it_times, X, *_ = diffusion(
    x0, n, m, D,
    hypergraph=hypergraph,
    weights=weights if weights else None,
    s=[0],    # 單一 seed
    lamda=0.1,
    eps=1e-3,
    T=30
)
t1 = time.time()
print("Single diffusion time:", t1 - t0)

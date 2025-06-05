# test_cpp_diffusion.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_cpp_diffusion.py — 验证 diffusion C++ 扩展能否正确返回 (n, T+1) 矩阵
"""

import os
import numpy as np
import scipy.sparse as sp

# 直接 import 已编译的 pyd，里面有 GraphSolver
import diffusion


def main():
    graph = "data/Paper_datasets/zoo.hmetis"
    labels = "data/Paper_datasets/zoo.label.npy"

    # 1) 检查文件存在
    assert os.path.exists(graph), f"{graph} not found"
    assert os.path.exists(labels), f"{labels} not found"

    # 2) 读取标签，拿到节点数 n
    y = np.load(labels)    # shape = (n,)
    n = y.shape[0]
    print(f"[test] n = {n}")

    # 3) 构造 seed 向量 s：长度 n，只在节点 0 上设 1.0
    s = np.zeros(n, dtype=np.float32)
    s[0] = 1.0
    s_sp = sp.csr_matrix(s.reshape(-1, 1))

    # 4) 实例化 GraphSolver
    solver = diffusion.GraphSolver(
        str(graph),    # hmetis 文件路径
        "",            # preconditioner (degree / empty)
        "degree",      # 我们用 degree 预处理
        0              # verbose=0 静默
    )
    print("[test] GraphSolver instantiated OK")

    # 5) 调用 diffusion，跑 T=1 步
    T, lam, eps, sched = 1, 1.0, 1e-3, 0
    X = solver.diffusion(s_sp, T, lam, eps, sched)

    # 6) 打印返回的矩阵形状
    print(f"[test] Got X with shape = {X.shape}")
    assert X.shape == (n, T+1), "Output shape mismatch"


if __name__ == "__main__":
    main()

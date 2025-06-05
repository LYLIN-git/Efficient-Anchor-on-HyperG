#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_diffusion_interface.py — 验证 C++ diffusion 接口
"""

from diffusion import GraphSolver
import scipy.sparse as sp


def main():
    # 假设我们测试 zoo（n=101），seed 选 0
    seed_idx = 0
    n = 101

    # 构造 (1 × n) 的 one-hot 稀疏行向量
    rows = [0]
    cols = [seed_idx]
    data = [1.0]
    s = sp.csr_matrix((data, (rows, cols)), shape=(1, n))

    # 注意：把 verbose 作为第 4 个位置参数
    solver = GraphSolver("data/Paper_datasets/zoo.hmetis",
                         "",              # 不需要标签文件
                         "degree",        # preconditioner
                         1)               # verbose=1

    # T=1 立刻返回
    X = solver.diffusion(s, 1, 1.0, 1e-3, 0)
    print("Returned X.shape =", X.shape)


if __name__ == "__main__":
    main()

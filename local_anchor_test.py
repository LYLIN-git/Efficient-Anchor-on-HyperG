#!/usr/bin/env python3
# local_anchor_test.py

from anchor_util import precompute_anchors, anchor_infer
from diffusion_wrapper_setup import diffusion as cpp_diffusion
from pathlib import Path
import numpy as np
print(">>> DEBUG: loaded local_anchor_test.py", flush=True)


def main():
    ds = "zoo"
    root = Path(__file__).parent
    graph = root/"data"/"Paper_datasets"/f"{ds}.hmetis"
    lbl = graph.with_suffix(".label.npy")
    y = np.load(lbl)
    print(f">>> n = {len(y)}", flush=True)

    # 固定参数
    lam, eps, delta, T, seed = 1.0, 1e-3, 1e-6, 10, 1234

    # 随机打乱并取前 20 个 seed
    perm = np.random.RandomState(seed).permutation(len(y))
    seeds = perm[:20]
    print(f">>> seeds = {seeds}", flush=True)

    # 调用 PPR
    s = np.zeros(len(y), dtype=np.float32)
    s[seeds] = lam
    print(">>> calling cpp_diffusion…", flush=True)
    X = cpp_diffusion(str(graph), s, T, lam, eps, 0)
    print(">>> cpp_diffusion returned shape =", X.shape, flush=True)

    # 调用 Anchor‐Infer
    anchors_npz = precompute_anchors(
        graph_path=str(graph), k=64, lam=lam,
        eps=eps, seed=seed,
        out_dir=root/"anchors", T=T, overwrite=False
    )
    print(">>> anchors_npz =", anchors_npz, flush=True)

    x_hat, _ = anchor_infer(
        graph_path=str(graph),
        anchors_npz=str(anchors_npz),
        seeds=seeds, lam=lam, eps=eps, delta=delta
    )
    print(">>> anchor_infer returned len =", len(x_hat), flush=True)


if __name__ == "__main__":
    main()

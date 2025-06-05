# run_anchor_figs.py
from anchor_util import precompute_anchors, anchor_infer
from diffusion_wrapper_setup import diffusion as cpp_diffusion
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import argparse
print(">>> DEBUG: running local run_anchor_figs.py", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["zoo", "mushroom"])
    p.add_argument("--out_dir", type=str, default="Anchor_online_figs")
    return p.parse_args()


def run():
    args = parse_args()
    print(f">>> dataset={args.dataset}", flush=True)
    ROOT = Path(__file__).parent
    graph = ROOT/"data"/"Paper_datasets"/f"{args.dataset}.hmetis"
    lbl = graph.with_suffix(".label.npy")
    y = np.load(lbl)
    print(f">>> n={len(y)}", flush=True)

    # 简化测试：只跑一次 reveal=20
    perm = np.random.RandomState(1234).permutation(len(y))
    seeds = perm[:20]
    T, lam, eps, delta = 10, 1.0, 1e-3, 1e-6

    # PPR via cpp_diffusion
    print(">>> calling cpp_diffusion …", flush=True)
    s = np.zeros(len(y), dtype=np.float32)
    s[seeds] = lam
    X = cpp_diffusion(str(graph), s, T, lam, eps, 0)
    print(">>> cpp_diffusion returned shape", X.shape, flush=True)

    # Anchor via anchor_infer
    anchors_npz = precompute_anchors(str(graph), k=64, lam=lam,
                                     eps=eps, seed=1234, out_dir=ROOT/"anchors",
                                     T=T, overwrite=False)
    print(">>> anchors file:", anchors_npz, flush=True)
    x_hat, _ = anchor_infer(str(graph), str(anchors_npz),
                            seeds, lam, eps, delta)
    print(">>> anchor inference done, len(x_hat) =", len(x_hat), flush=True)


if __name__ == "__main__":
    run()

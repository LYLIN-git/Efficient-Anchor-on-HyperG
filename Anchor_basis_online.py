#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from anchor_util import anchor_infer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph",     type=str, required=True)
    p.add_argument("--anchors",   type=str, required=True)
    p.add_argument("--seed-list", type=str, required=True)
    p.add_argument("--lam",       type=float, default=1.0)
    p.add_argument("--eps",       type=float, default=1e-3)
    p.add_argument("--delta",     type=float, default=1e-6)
    p.add_argument("--T",         type=int,   default=300)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(v) for v in args.seed_list.split(",") if v.strip() != ""]
    x_hat, loss = anchor_infer(
        graph_path=args.graph,
        anchors_npz=args.anchors,
        seeds=seeds,
        lam=args.lam,
        eps=args.eps,
        delta=args.delta
    )
    print(json.dumps({"x_hat": x_hat.tolist(), "loss": loss}))


if __name__ == "__main__":
    main()

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(run_dir, tag):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    s = ea.Scalars(tag)
    return np.array([x.step for x in s]), np.array([x.value for x in s])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--tag", default="charts/return_avg100")
    p.add_argument("--out", default="assets/learning_curves.png")
    p.add_argument("--points", type=int, default=300)
    args = p.parse_args()

    groups = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(args.runs, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        m = re.match(r"(.+)_seed\d+$", name)
        key = m.group(1) if m else name
        loaded = load_scalar(d, args.tag)
        if loaded:
            groups[key].append(loaded)

    plt.figure(figsize=(8, 5))
    for key, seeds in sorted(groups.items()):
        x_max = min(x[-1] for x, _ in seeds)
        grid = np.linspace(0, x_max, args.points)
        curves = np.stack([np.interp(grid, x, y) for x, y in seeds])
        mean = curves.mean(0)
        std = curves.std(0)
        line, = plt.plot(grid, mean, label=f"{key} (n={len(seeds)})")
        plt.fill_between(grid, mean - std, mean + std, alpha=0.2, color=line.get_color())
        final = curves[:, -1]
        print(f"{key}: final {mean[-1]:.2f} +/- {std[-1]:.2f}  (seeds: {', '.join(f'{v:.2f}' for v in final)})")

    plt.xlabel("frames")
    plt.ylabel("100-episode average return")
    plt.title("MinAtar Breakout")
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

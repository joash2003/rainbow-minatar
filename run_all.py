import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

GAME = "breakout"
SEEDS = [1, 2, 3]

# each variant is (name, extra flags); add components here as they land
VARIANTS = [
    ("dqn", []),
    ("ddqn", ["--double"]),
    ("dueling", ["--dueling"]),
]


def run_one(name, flags, seed, frames):
    log = f"{name}_{GAME}_seed{seed}"
    if os.path.exists(os.path.join("runs", log, "model.pt")):
        return f"skip {log} (already done)"
    cmd = [sys.executable, "dqn.py", "--game", GAME, "--seed", str(seed), "--log-name", log] + flags
    if frames:
        cmd += ["--frames", str(frames)]
    with open(f"{log}.log", "w") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)
    return f"done {log}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--frames", type=int, default=None)
    args = p.parse_args()

    tasks = [(name, flags, seed) for name, flags in VARIANTS for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(run_one, *t, args.frames) for t in tasks]
        for f in as_completed(futures):
            print(f.result(), flush=True)
    print("all runs done")


if __name__ == "__main__":
    main()

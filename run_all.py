import os
import subprocess
import sys

GAME = "breakout"
SEEDS = [1, 2, 3]

# each variant is (name, extra flags); add components here as they land
VARIANTS = [
    ("dqn", []),
    ("ddqn", ["--double"]),
]


def main():
    for name, flags in VARIANTS:
        for seed in SEEDS:
            log = f"{name}_{GAME}_seed{seed}"
            if os.path.exists(f"runs/{log}/model.pt"):
                print(f"skip {log} (already done)")
                continue
            cmd = [sys.executable, "dqn.py", "--game", GAME, "--seed", str(seed), "--log-name", log] + flags
            print("running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    print("all runs done")


if __name__ == "__main__":
    main()

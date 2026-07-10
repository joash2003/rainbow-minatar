# rainbow-minatar

Reproducing Rainbow, one component at a time, on MinAtar. A competence-building
project: a seeded, documented deep-RL codebase in PyTorch with honest learning
curves showing what each Rainbow component contributes.

## Setup

```
uv sync
uv run python dqn.py --game breakout
```

Games: `breakout`, `asterix`, `freeway`, `seaquest`, `space_invaders`.

Component flags: `--double`, `--dueling`, `--per`, `--c51`. Run the full ablation
sweep (each variant, 3 seeds, in parallel across cores):

```
uv run python run_all.py --jobs 6
```

Logs go to `runs/<name>`. View with `uv run tensorboard --logdir runs`.

## Notes

- Runs on CPU by default. MinAtar's network is tiny (one 16-filter conv on a
  10x10 input), so CPU beats MPS here — MPS kernel-launch overhead dominates.
  ~1000 fps on CPU means a 5M-frame run is roughly 80 minutes.
- Architecture and hyperparameters follow the MinAtar paper (Young & Tian, 2019)
  so results are comparable to the reference DQN.

## Results

`uv run python aggregate.py` overlays each variant as a mean ± std band across
seeds (`assets/learning_curves.png`).

| Variant       | Breakout (5M frames)    | vs DQN |
|---------------|-------------------------|--------|
| DQN           | 9.45 ± 0.50  (5 seeds)  | —      |
| + Double DQN  | 7.61 ± 0.57  (3 seeds)  | -1.84  |
| + Dueling     | 9.52 ± 0.74  (3 seeds)  | +0.07  |
| + PER         | 9.61 ± 0.99  (5 seeds)  | +0.16  |
| + C51         | 9.56 ± 0.83  (3 seeds)  | +0.11  |

Each row adds one component to the DQN baseline in isolation (independent
ablations, not cumulative). Values are the 100-episode average return at 5M
frames, mean ± std over seeds.

**What each component contributed:**

- **Double DQN** (-1.84) clearly hurt: every DDQN seed finished below every DQN
  seed. On MinAtar Breakout the overestimation it corrects isn't severe, so the
  decoupled target lookup mostly slows value propagation — a reminder that
  Rainbow components aren't universally beneficial.
- **No component shows a gain beyond seed noise.** At 3 seeds PER looked like a
  clear win (+0.89 against a deceptively tight ±0.13 baseline); two more seeds
  of DQN and PER each killed the claim (+0.16 ± ~1σ). The first version of this
  table over-claimed, and the replication is kept here deliberately: small-n
  deep-RL comparisons on single environments are noise until proven otherwise.
- Seeds 4–5 were run with fully seeded environment RNG (same-seed runs are
  bit-identical); seeds 1–3 predate that fix. Both sample the same run
  distribution, so they are pooled.

Returns peak mid-training (~2.5–3.5M frames, ~13–15) then settle lower by 5M,
driven by MinAtar's difficulty ramping plus the fixed ε = 0.1 floor; this affects
the DQN baseline identically, so it's not a regression. Final@5M is reported for
comparability with the MinAtar reference — see `assets/learning_curves.png` for the
full curves.

## Roadmap

- [x] DQN baseline — 9.45 ± 0.50 (5 seeds)
- [x] Double DQN (`--double`) — 7.61 ± 0.57 (-1.84 vs DQN; hurts on this env)
- [x] Dueling network (`--dueling`) — 9.52 ± 0.74 (+0.07; within noise)
- [x] Prioritized experience replay (`--per`) — 9.61 ± 0.99, 5 seeds (+0.16; the
  n=3 "+0.89" did not survive replication)
- [x] C51 (distributional) (`--c51`) — 9.56 ± 0.83 (+0.11; within noise)

Each component is added on its own to the DQN baseline (independent ablations),
evaluated over 3 seeds, and written up above.

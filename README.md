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

| Variant | Breakout (3 seeds, 5M frames) |
|---------|-------------------------------|
| DQN     | 9.23 ± 0.13                   |

## Roadmap

- [x] DQN baseline
- [ ] Double DQN — implemented (`--double`), not yet benchmarked
- [ ] Dueling network — implemented (`--dueling`), not yet benchmarked
- [ ] Prioritized experience replay — implemented (`--per`), not yet benchmarked
- [ ] C51 (distributional)

Each component is added on its own, evaluated over multiple seeds against the
previous baseline, and written up.

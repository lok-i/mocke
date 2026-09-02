# mocke

Minimal `mjlab` implementations of pretrained whole-body tracking recipes for
G1.

`mocke` provides the MDP contracts needed to train and run these controllers:

- flat-ground tracking from standard `motion.npz` clips
- future reference windows and command conditioning
- checkpoint-compatible observation and action layouts
- Isaac Lab ↔ MuJoCo joint and body mappings
- play-only environments for checking frozen controllers

## install

Requires Python 3.10+ and GitHub SSH access for the pinned `rsl_rl` fork.

```bash
bash scripts/setup/install.sh
```

## play

Run either frozen controller on the cached `mjlab` demo clip:

```bash
python scripts/play_textop.py --num_envs 1
python scripts/play_sonic.py --num_envs 1
```

Use `--il_ordered` only when a custom clip follows Isaac Lab joint order:

```bash
python scripts/play_sonic.py --motion path/to/motion.npz --il_ordered
```

## library surface

- `mocke.mdp` — joint maps, motion loading, and future-motion commands
- `mocke.textop.profile` — TextOp observation, robot, and action contract
- `mocke.sonic.profile` — SONIC observation, robot, and action contract
- `mocke.{textop,sonic}.env_cfg` — flat-ground tracking environment factories

Observation order, action order, and future-window shape must match the
checkpoint.

## checkpoints

Port the public SONIC release checkpoint to the native `mjlab` layout:

```bash
python scripts/port_sonic_checkpoint.py
python scripts/port_sonic_checkpoint.py --smpl
```

Ported TextOp and SONIC checkpoints live in `pretrained/` and retain their
upstream licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## export

Export either frozen controller to ONNX and run the two-world parity check:

```bash
pip install -e ".[export]"
python scripts/export_onnx.py sonic
python scripts/export_onnx.py textop
python scripts/export_onnx.py sonic --adapter --rank 16
python scripts/export_onnx.py sonic --viewer native
```

World 0 compares PyTorch and ONNX on the same observations. World 1 runs ONNX
closed-loop. Artifacts are kept only when both checks pass. Use `--output-dir`
to choose their directory. With `--viewer native`, results print at
`--check-steps` and playback continues.

## tasks

Importing `mocke` registers two play-only tasks when the demo clip is available:

- `Mocke-Tracking-Textop-G1`
- `Mocke-Tracking-Sonic-G1`

```bash
python -c "import mocke; from mjlab.tasks.registry import list_tasks; print('\n'.join(list_tasks()))"
```

## supported recipes

- `textop` — TextOpTracker with Isaac Lab-ordered observations and actions
- `sonic` — SONIC with MuJoCo-ordered observations and actions

Datasets and task-specific robot logic are outside this package.

## acknowledgements

`mocke` builds on
[SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl),
[TextOp](https://github.com/TeleHuman/TextOp),
[mjlab](https://github.com/mujocolab/mjlab), and
[RSL-RL](https://github.com/leggedrobotics/rsl_rl). Please cite the relevant
research when using a corresponding recipe or checkpoint.

## third-party code and models

`mocke`'s original code and documentation are BSD-3-Clause licensed. Adapted
implementations and bundled model artifacts retain their upstream terms:

| component | upstream | terms |
|---|---|---|
| SONIC-compatible implementation | [GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl) | Apache-2.0 |
| SONIC checkpoints and derivatives | [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) | NVIDIA Open Model License |
| TextOp-compatible implementation and checkpoints | [TextOp](https://github.com/TeleHuman/TextOp) | MIT |

The complete scope, attribution, and redistributed license texts are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/).

## license

`mocke`'s original material is available under the
[BSD 3-Clause License](LICENSE). Third-party components are excluded from that
grant and remain under the terms above.

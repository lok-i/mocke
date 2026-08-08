# mocke

mock implementations of pre-trained whole-body tracker recipes in `mjlab`.

`mocke` recreates the MDP contracts needed to train and run G1 whole-body
controllers:

- flat-ground tracking from standard `motion.npz` clips
- command conditioning and future reference windows
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
- `mocke.textop.profile` — textop observation, robot, and action contract
- `mocke.sonic.profile` — SONIC observation, robot, and action contract
- `mocke.{textop,sonic}.env_cfg` — flat-ground tracking environment factories

Observation order, action order, and future-window shape are checkpoint
contracts.

## checkpoints

Port the public SONIC release checkpoint to the native `mjlab` layout:

```bash
python scripts/port_sonic_checkpoint.py
python scripts/port_sonic_checkpoint.py --smpl
```

Ported textop and SONIC checkpoints live under `pretrained/`.

## export

Export either frozen controller to ONNX and run the two-world parity check:

```bash
pip install -e ".[export]"
python scripts/export_onnx.py sonic
python scripts/export_onnx.py textop
python scripts/export_onnx.py sonic --adapter --rank 16
python scripts/export_onnx.py sonic --viewer native
```

World 0 runs PyTorch and gates ONNX on the same observations. World 1 runs ONNX
closed-loop. Artifacts are kept only when both checks pass. Use `--output-dir`
to choose the artifact directory. With `--viewer native`, the visible rollout
is the check: results print at `--check-steps`, then playback continues.

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

*"standing on the shoulders of giants"*

1. [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
2. [Textop](https://github.com/TeleHuman/Textop)
3. [mjlab](https://github.com/mujocolab/mjlab)
4. [RSL-RL](https://github.com/leggedrobotics/rsl_rl) framework

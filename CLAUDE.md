# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`mocke` is a **frozen-WBC tracking library** for mjlab G1 projects — shared plumbing, no dataset.
It ships two frozen whole-body-controller (WBC) bases (`textop`, `sonic`) plus the joint/body
remapping, motion-command, and observation machinery consumers need to drive them. It also
registers two **play-only sandbox tasks** against mjlab's cached demo clip.

Consumers (e.g. `repose`, `vibe`) do **not** subclass mocke's envs — they import from
`mocke.{textop,sonic}.profile` + `mocke.mdp` and assemble their own envs. Treat the profiles as a
**public contract**: changing obs term order, action ordering, or the future-window shape breaks the
pretrained checkpoints and every downstream repo.

## Commands

```bash
bash scripts/setup/install.sh              # pip install -e . + rsl_rl fork pin (see Install below)
python scripts/play_textop.py --num_envs 1 # frozen textop base on demo clip
python scripts/play_sonic.py  --num_envs 1 # frozen SONIC base on demo clip
python scripts/play_sonic.py  --motion clip.npz --il_ordered   # IL-ordered dataset clip
python scripts/port_sonic_checkpoint.py    # HF download -> pretrained/sonic/last_ported.pt
ruff check src                             # lint (E4/E7/E9/F/I/B)
pytest                                     # tests (dev extra)

# list registered tasks
python -c "import mocke; from mjlab.tasks.registry import list_tasks; print('\n'.join(list_tasks()))"
```

## Install (the rsl_rl fork dance)

`pip install -e .` gets mjlab. The `lok-i/rsl_rl` fork (provides `ModularNormMLP*`,
`SonicBaseModel`) **cannot** live in `[project.dependencies]`: mjlab pins PyPI `rsl-rl-lib==5.4.0`,
the fork reports `5.4.1` → resolver conflict. The single source of the fork SHA is `[tool.mocke].rsl_rl`
in [pyproject.toml](pyproject.toml). `install.sh` installs it `--no-deps` and **leaves an existing
checkout alone** when its HEAD already contains the pinned SHA (equal or descendant). The fork is
required at play time (models load through `ModularNormMLP`).

## Architecture

Two frozen bases, same skeleton. Each `mocke/{textop,sonic}/` package has:

| file | role |
|---|---|
| `profile.py` | **the WBC contract** — `policy_obs_terms()` + `extra_obs_groups()` + `robot_cfg(base=None)` + `action_cfg()`. Layout/order matches the trained ckpt. |
| `env_cfg.py` | pure tracking-env factory; starts from mjlab's `make_tracking_env_cfg()`, swaps in the profile, wires G1 via `g1_env`. |
| `mdp/` | package-specific obs/action terms; `mdp/__init__.py` does `from mjlab.tasks.tracking.mdp import *` then adds mocke terms. |

Shared pieces:

- [src/mocke/mdp/joint_maps.py](src/mocke/mdp/joint_maps.py) — **single source of truth** for joint/body order.
  The retargeted dataset is **IsaacLab BFS order (IL)**; mjlab/MuJoCo is **XML DFS order (MJ)**.
  `mj = il[:, IL2MJ]`, `il = mj[:, MJ2IL]`. `G1_TRACKED_BODIES` = 14 FK-verified (name, IL-index) pairs.
- [src/mocke/mdp/commands.py](src/mocke/mdp/commands.py) — `FutureMotionCommand` (mjlab `MotionCommand` + future-window
  accessors) and `MjMotionLoader` (IL→MJ remapping loader). The future-window **shape is owned by the
  obs terms**, not the command: textop reads 5 frames @ skip 1, SONIC reads 10 @ skip 5, both via
  `future_frames(steps, skip)`. Named `motion_*_future` props mirror repose's `OmniObjectMotionCommand`
  so tracking obs terms are **duck-typed** across both commands.
- [src/mocke/g1_env.py](src/mocke/g1_env.py) — shared G1 wiring (`wire_g1`) + `play_overrides` (infinite episode,
  no noise/push, deterministic clip start).
- [src/mocke/__init__.py](src/mocke/__init__.py) — registers `Mocke-Tracking-{Textop,Sonic}-G1` at import (mjlab
  entry point `mjlab.tasks`). **Auto-skips when offline** (`default_motion_file()` returns `""`).

### Key differences between the two bases

|  | textop | sonic |
|---|---|---|
| anchor body | `torso_link` | `pelvis` |
| joint order to/from base | **IL** (action + obs remapped) | **MJ** (native, baked into ckpt) |
| future window | 5 frames @ skip 1 | 10 frames @ skip 5 (in `tokenizer` obs group) |
| proprio history | none | 10 (decoder) |
| action | `JointPositionActionIL` (remaps MJ↔IL) | stock `JointPositionActionCfg`, per-joint `0.25*effort/stiffness` scale |
| robot cfg | mjlab G1 as-is | G1 with SONIC hip_pitch actuator regrouping (7520_22) |

### `il_ordered` flag

Threads through `FutureMotionCommandCfg` → env factories → play scripts. `True` = clip npz is
IL-ordered → use `MjMotionLoader` (remaps). `False` = MJ-native npz (e.g. the mjlab demo clip) → keep
mjlab's stock `MotionLoader`. The registered sandboxes and play scripts default to `False` for the
demo clip; pass `--il_ordered` with `--motion` for retargeted-dataset clips.

## Conventions (from global instructions)

- `readme.md` not `README.md`; docstrings describe + link to reference, don't restate the code.
- Prefer tables/snippets over prose. Minimal, compact, consistent code.
- `dependencies/rsl_rl/` is a **vendored checkout of the fork**, not first-party code — don't edit it as
  part of mocke changes.

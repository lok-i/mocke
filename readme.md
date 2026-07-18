# mocke

Frozen-WBC tracking plumbing for mjlab G1 projects — shared library, no dataset.

- `src/mocke/mdp/` — canonical IL↔MJ joint/body maps, IL-remapping `MjMotionLoader`,
  `FutureMotionCommand` (future-window accessor; `il_ordered=False` for MJ-native clips)
- `src/mocke/{textop,sonic}/profile.py` — the WBC contract: `policy_obs_terms()` +
  `extra_obs_groups()` + `robot_cfg(base=None)` + `action_cfg()`; consumers assemble
  their own envs from these (pass a custom EntityCfg as `base` to override the robot)
- `src/mocke/{textop,sonic}/env_cfg.py` — pure tracking sandbox factories
- `pretrained/` — ported base checkpoints (textop `model_75000_ported.pt`,
  sonic `last_ported.pt`); git-tracked, so any consumer repo gets them on sync
- registers `Mocke-Tracking-{Textop,Sonic}-G1` (play-only sandboxes) against
  mjlab's cached demo clip; auto-skips offline

## Install
```bash
bash scripts/setup/install.sh
```
`pip install -e .` + the `lok-i/rsl_rl` fork pin (`[tool.mocke]` in pyproject.toml, SSH).
The fork can't live in `[project.dependencies]` (mjlab pins PyPI `rsl-rl-lib==5.4.0`,
resolver conflict); the script installs it `--no-deps` — and leaves the env's rsl_rl
alone when it's a checkout already containing the pinned SHA (equal or newer).

## Play the sandboxes
```bash
python scripts/play_textop.py --num_envs 1          # frozen textop base, demo clip
python scripts/play_sonic.py  --num_envs 1          # frozen SONIC base, demo clip
python scripts/play_sonic.py  --motion clip.npz --il_ordered   # IL-ordered dataset clip
```

## Port the SONIC release checkpoint
```bash
python scripts/port_sonic_checkpoint.py    # HF download -> pretrained/sonic/last_ported.pt
python scripts/port_sonic_checkpoint.py --smpl  # regenerate pretrained/sonic/smpl_ported.pt (tracked; only needed if the port changes)
```

## List registered tasks
```bash
python -c "import mocke; from mjlab.tasks.registry import list_tasks; print('\n'.join(list_tasks()))"
```

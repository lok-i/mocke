# mock

Frozen-WBC tracking plumbing for mjlab G1 projects — shared library, no dataset.

- `src/mock/mdp/` — canonical IL↔MJ joint/body maps, IL-remapping `MjMotionLoader`,
  `FutureMotionCommand` (future-window accessor; `il_ordered=False` for MJ-native clips)
- `src/mock/{textop,sonic}/profile.py` — the WBC contract: `policy_obs_terms()` +
  `extra_obs_groups()` + `robot_cfg(base=None)` + `action_cfg()`; consumers assemble
  their own envs from these (pass a custom EntityCfg as `base` to override the robot)
- `src/mock/{textop,sonic}/env_cfg.py` — pure tracking sandbox factories
- `pretrained/` — ported base checkpoints (textop `model_75000_ported.pt`,
  sonic `last_ported.pt`); git-tracked, so any consumer repo gets them on sync
- registers `Mock-Tracking-{Textop,Sonic}-G1` (play-only sandboxes) against
  mjlab's cached demo clip; auto-skips offline

## Install
```bash
pip install -e .
```
Play/port scripts additionally need the `lok-i/rsl_rl` fork in the env
(the package itself depends on mjlab only).

## Play the sandboxes
```bash
python scripts/play_textop.py --num_envs 1          # frozen textop base, demo clip
python scripts/play_sonic.py  --num_envs 1          # frozen SONIC base, demo clip
python scripts/play_sonic.py  --motion clip.npz --il_ordered   # IL-ordered dataset clip
```

## Port the SONIC release checkpoint
```bash
python scripts/port_sonic_checkpoint.py    # HF download -> pretrained/sonic/last_ported.pt
```

## List registered tasks
```bash
python -c "import mock; from mjlab.tasks.registry import list_tasks; print('\n'.join(list_tasks()))"
```

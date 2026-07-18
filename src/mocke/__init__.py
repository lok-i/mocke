"""Frozen-WBC tracking library + sandbox tasks (no object, no dataset).

Registers Mocke-Tracking-{Textop,Sonic}-G1 against mjlab's cached demo clip
(MJ-native npz -> il_ordered=False). Consumers compose their own envs from
mocke.{textop,sonic}.profile and mocke.mdp; ported base checkpoints ship in
<repo>/pretrained (see PRETRAINED_DIR).
"""

from pathlib import Path

PRETRAINED_DIR = Path(__file__).resolve().parents[2] / "pretrained"
# NOTE: PRETRAINED_DIR must precede the mjlab imports — mjlab's task discovery
# imports consumer packages that do `from mocke import PRETRAINED_DIR`, so the
# name must exist before this module re-enters via mjlab.

from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.tasks.tracking.config.g1.rl_cfg import (  # noqa: E402
    unitree_g1_tracking_ppo_runner_cfg,
)


def default_motion_file() -> str:
    """mjlab's GCS demo clip (downloaded once, sha-verified cache); "" offline."""
    try:
        from mjlab.scripts.gcs import ensure_default_motion
        return str(ensure_default_motion())
    except Exception:
        return ""


def _register() -> None:
    motion = default_motion_file()
    if not motion:
        return
    from mocke.sonic.env_cfg import sonic_tracking_env_cfg
    from mocke.textop.env_cfg import textop_tracking_env_cfg

    for task_id, factory in (
        ("Mocke-Tracking-Textop-G1", textop_tracking_env_cfg),
        ("Mocke-Tracking-Sonic-G1", sonic_tracking_env_cfg),
    ):
        register_mjlab_task(
            task_id=task_id,
            env_cfg=factory(motion_file=motion, il_ordered=False),
            play_env_cfg=factory(motion_file=motion, play=True, il_ordered=False),
            rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),  # placeholder (play-only sandboxes)
        )


_register()

"""G1 SONIC tracking env — 1:1 minimal port of the SONIC training env to mjlab.

Mirrors the gear_sonic release training env (sonic_release.yaml): pelvis anchor,
50 Hz PD control, obs layouts from the sonic profile. mjlab's G1 gains are
BeyondMimic-lineage == SONIC's, except hip_pitch (regrouped in the profile).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from mocke import g1_env
from mocke.mdp import G1_TRACKED_BODY_NAMES, FutureMotionCommandCfg
from mocke.sonic import profile


def sonic_tracking_env_cfg(
    motion_file: str, play: bool = False, il_ordered: bool = True
) -> ManagerBasedRlEnvCfg:
    """G1 SONIC tracking env config."""
    cfg = make_tracking_env_cfg()

    robot = profile.robot_cfg()
    cfg.scene.entities = {"robot": robot}
    cfg.actions["joint_pos"] = profile.action_cfg(robot)

    base_cmd = cfg.commands.pop("motion")
    cfg.commands["motion"] = FutureMotionCommandCfg(
        entity_name="robot",
        motion_file=motion_file,
        anchor_body_name="pelvis",
        body_names=G1_TRACKED_BODY_NAMES,
        resampling_time_range=base_cmd.resampling_time_range,
        pose_range=base_cmd.pose_range,
        velocity_range=base_cmd.velocity_range,
        joint_position_range=(-0.1, 0.1),
        sampling_mode="start",
        debug_vis=True,
        il_ordered=il_ordered,
    )

    cfg.observations = {
        "policy": ObservationGroupCfg(
            terms=profile.policy_obs_terms(noisy=True),
            concatenate_terms=True,
            enable_corruption=True,
        ),
        **profile.extra_obs_groups("motion"),
    }

    g1_env.wire_g1(cfg)
    if play:
        g1_env.play_overrides(cfg)
    return cfg

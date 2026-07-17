"""G1 textop tracking env — TextOpTracker's G1FlatProjGravObs on the mjlab base.

mjlab's make_tracking_env_cfg IS the BeyondMimic re-implementation TextOpTracker
derives from; this cfg swaps in the textop profile (flat-hand G1, IL action,
IL-ordered obs, 5-frame future refs) on top of it.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from mocke import g1_env
from mocke.mdp import G1_TRACKED_BODY_NAMES, FutureMotionCommandCfg
from mocke.textop import mdp, profile


def _critic_obs_terms(command_name: str) -> dict[str, ObservationTermCfg]:
    """Privileged critic: policy terms un-noised + body pose feedback."""
    _p = {"command_name": command_name}
    return {
        **profile.policy_obs_terms(command_name),
        "body_pos": ObservationTermCfg(func=mdp.robot_body_pos_b, params=_p),
        "body_ori": ObservationTermCfg(func=mdp.robot_body_ori_b, params=_p),
    }


def textop_tracking_env_cfg(
    motion_file: str, play: bool = False, il_ordered: bool = True
) -> ManagerBasedRlEnvCfg:
    """G1 textop tracking env config."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": profile.robot_cfg()}
    cfg.actions["joint_pos"] = profile.action_cfg()

    base_cmd = cfg.commands.pop("motion")
    cfg.commands["motion"] = FutureMotionCommandCfg(
        entity_name="robot",
        motion_file=motion_file,
        anchor_body_name="torso_link",
        body_names=G1_TRACKED_BODY_NAMES,
        future_steps=profile.FUTURE_STEPS,
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
        "critic": ObservationGroupCfg(
            terms=_critic_obs_terms("motion"),
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    g1_env.wire_g1(cfg)
    if play:
        g1_env.play_overrides(cfg)
    return cfg

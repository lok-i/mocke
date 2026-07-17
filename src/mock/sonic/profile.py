"""SONIC WBC profile — everything the frozen base needs from an env.

Consumed by both the pure tracking env (env_cfg.py) and repose's
``_sonic_helpers`` assembler. Layout matches the ported release ckpt
(gear_sonic sonic_release.yaml, g1 mode, MuJoCo joint order baked in).
"""

from __future__ import annotations

import dataclasses

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mock.sonic import mdp

HISTORY = 10
"""Proprio history length the decoder was trained with."""


def policy_obs_terms(noisy: bool = False) -> dict[str, ObservationTermCfg]:
    """The decoder's proprio layout (order = SONIC PolicyCfg attribute order)."""
    _h = {"history_length": HISTORY, "flatten_history_dim": True}

    def _n(term: ObservationTermCfg, n_min: float, n_max: float) -> ObservationTermCfg:
        if noisy:
            term.noise = Unoise(n_min=n_min, n_max=n_max)
        return term

    return {
        "base_ang_vel": _n(ObservationTermCfg(func=mdp.base_ang_vel, **_h), -0.2, 0.2),
        "joint_pos": _n(ObservationTermCfg(func=mdp.joint_pos_rel, **_h), -0.01, 0.01),
        "joint_vel": _n(ObservationTermCfg(func=mdp.joint_vel_rel, **_h), -0.5, 0.5),
        "actions": ObservationTermCfg(func=mdp.last_action, **_h),
        "gravity_dir": _n(ObservationTermCfg(func=mdp.projected_gravity, **_h), -0.05, 0.05),
    }


def extra_obs_groups(command_name: str = "motion") -> dict[str, ObservationGroupCfg]:
    """The g1-encoder input: 10 future ref frames @ 0.1 s, own obs group."""
    return {
        "tokenizer": ObservationGroupCfg(
            terms={
                "g1_tokenizer": ObservationTermCfg(
                    func=mdp.sonic_g1_tokenizer, params={"command_name": command_name}
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }


def robot_cfg(base=None):
    """G1 with SONIC's hip_pitch actuator grouping (7520_22, effort 139).

    Args:
        base: EntityCfg to regroup (e.g. repose's flat-hand G1); default mjlab G1.
    """
    if base is None:
        from mjlab.asset_zoo.robots import get_g1_robot_cfg
        base = get_g1_robot_cfg()
    act_14 = dataclasses.replace(
        G1_ACTUATOR_7520_14, target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint")
    )
    act_22 = dataclasses.replace(
        G1_ACTUATOR_7520_22,
        target_names_expr=(".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"),
    )
    actuators = tuple(
        act_14 if a is G1_ACTUATOR_7520_14 else act_22 if a is G1_ACTUATOR_7520_22 else a
        for a in base.articulation.actuators
    )
    base.articulation = dataclasses.replace(base.articulation, actuators=actuators)
    return base


def action_cfg(robot) -> JointPositionActionCfg:
    """MuJoCo-order PD-target action, 0.25 * effort / stiffness scale per joint."""
    scale = {
        name: 0.25 * a.effort_limit / a.stiffness
        for a in robot.articulation.actuators
        for name in a.target_names_expr
    }
    return JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=scale,
        use_default_offset=True,
    )

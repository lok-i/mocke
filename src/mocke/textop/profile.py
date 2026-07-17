"""Textop WBC profile — everything the frozen base needs from an env.

Consumed by both the pure tracking env (env_cfg.py) and repose's
``_textop_helpers`` assembler. Layout matches the trained ckpt
(TextOpTracker G1FlatProjGravObs variant, IL joint order).
"""

from __future__ import annotations

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mocke.mdp.joint_maps import MJ2IL
from mocke.textop import mdp
from mocke.textop.mdp import JointPositionActionILCfg

FUTURE_STEPS = 5
"""Reference lookahead the WBC was trained with (consecutive frames)."""


def policy_obs_terms(
    command_name: str = "motion",
    wose: bool = False,
    noisy: bool = False,
) -> dict[str, ObservationTermCfg]:
    """The frozen WBC's input layout (order matters — matches the trained ckpt)."""
    _p = {"command_name": command_name}
    _il_asset = SceneEntityCfg("robot", joint_ids=list(MJ2IL))

    def _n(term: ObservationTermCfg, n_min: float, n_max: float) -> ObservationTermCfg:
        if noisy:
            term.noise = Unoise(n_min=n_min, n_max=n_max)
        return term

    return {
        "command": ObservationTermCfg(func=mdp.generated_commands_il, params=_p),
        "motion_anchor_pos_b": _n(ObservationTermCfg(
            func=mdp.motion_anchor_pos_b_future_zero if wose
            else mdp.motion_anchor_pos_b_future, params=_p), -0.25, 0.25),
        "motion_anchor_ori_b": _n(ObservationTermCfg(
            func=mdp.motion_anchor_ori_b_future, params=_p), -0.05, 0.05),
        "projected_gravity": _n(ObservationTermCfg(func=mdp.projected_gravity), -0.07, 0.07),
        "base_lin_vel": _n(ObservationTermCfg(func=mdp.base_lin_vel), -0.5, 0.5),
        "base_ang_vel": _n(ObservationTermCfg(func=mdp.base_ang_vel), -0.2, 0.2),
        "joint_pos": _n(ObservationTermCfg(
            func=mdp.joint_pos_rel, params={"asset_cfg": _il_asset}), -0.01, 0.01),
        "joint_vel": _n(ObservationTermCfg(
            func=mdp.joint_vel_rel, params={"asset_cfg": _il_asset}), -0.5, 0.5),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }


def extra_obs_groups(command_name: str = "motion") -> dict:
    """Additional obs groups the base needs beyond the policy stream (none)."""
    return {}


def robot_cfg(base=None):
    """G1 for the textop WBC; consumers may pass their own EntityCfg (e.g. flat-hand G1).

    Args:
        base: EntityCfg to use as-is; default mjlab G1.
    """
    if base is None:
        from mjlab.asset_zoo.robots import get_g1_robot_cfg
        base = get_g1_robot_cfg()
    return base


def action_cfg() -> JointPositionActionILCfg:
    """IL-ordered joint position action (the WBC emits IL-ordered targets)."""
    return JointPositionActionILCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
    )

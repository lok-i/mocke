"""Shared G1 wiring for pure tracking envs (textop + sonic)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

G1_TRACKING_EE_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def wire_g1(cfg: ManagerBasedRlEnvCfg) -> None:
    """Per-robot wiring shared by G1 tracking envs (mirrors mjlab's g1 tracking cfg)."""
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.terminations["ee_body_pos"].params["body_names"] = G1_TRACKING_EE_BODIES
    cfg.viewer.body_name = "torso_link"


def play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Infinite episode, no noise/push, deterministic clip start."""
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

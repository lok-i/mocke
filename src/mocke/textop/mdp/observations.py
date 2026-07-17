"""Textop WBC tracking obs terms (IL-ordered command, future anchor refs).

Duck-typed on the shared future-window command interface
(``motion_anchor_{pos,quat}_w_future``, ``motion_joint_{pos,vel}_future``,
``robot_anchor_{pos,quat}_w``) — served by both
``mocke.mdp.FutureMotionCommand`` (pure tracking) and consumer-side commands
(e.g. vibe's ``OmniObjectMotionCommand``).
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from mocke.mdp.joint_maps import MJ2IL as _MJ2IL  # noqa: F401 — re-exported

__all__ = [
    "generated_commands_il",
    "motion_anchor_pos_b_future",
    "motion_anchor_pos_b_future_zero",
    "motion_anchor_ori_b_future",
]


def motion_anchor_pos_b_future(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Future N-step anchor position in robot body frame -> (B, N*3)."""
    cmd = env.command_manager.get_term(command_name)
    future_pos = cmd.motion_anchor_pos_w_future   # (B, N, 3)
    future_quat = cmd.motion_anchor_quat_w_future  # (B, N, 4)
    robot_pos = cmd.robot_anchor_pos_w[:, None, :].expand_as(future_pos)
    robot_quat = cmd.robot_anchor_quat_w[:, None, :].expand_as(future_quat)
    pos_b, _ = subtract_frame_transforms(robot_pos, robot_quat, future_pos, future_quat)
    return pos_b.reshape(env.num_envs, -1)


def motion_anchor_ori_b_future(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Future N-step anchor orientation in robot body frame (mat6d) -> (B, N*6)."""
    cmd = env.command_manager.get_term(command_name)
    future_pos = cmd.motion_anchor_pos_w_future
    future_quat = cmd.motion_anchor_quat_w_future
    robot_pos = cmd.robot_anchor_pos_w[:, None, :].expand_as(future_pos)
    robot_quat = cmd.robot_anchor_quat_w[:, None, :].expand_as(future_quat)
    _, ori_b = subtract_frame_transforms(robot_pos, robot_quat, future_pos, future_quat)
    mat = matrix_from_quat(ori_b)  # (B, N, 3, 3)
    return mat[..., :2].reshape(env.num_envs, -1)


def generated_commands_il(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """N-step [joint_pos, joint_vel] reordered from MJ to IL order."""
    cmd = env.command_manager.get_term(command_name)
    jpos = cmd.motion_joint_pos_future  # (B, N, 29) MJ order
    jvel = cmd.motion_joint_vel_future  # (B, N, 29) MJ order
    B = env.num_envs
    return torch.cat([
        jpos[:, :, _MJ2IL].reshape(B, -1),
        jvel[:, :, _MJ2IL].reshape(B, -1),
    ], dim=1)


def motion_anchor_pos_b_future_zero(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """WoSe: exact zeros replacing motion_anchor_pos_b_future (no odometry). -> (B, N*3)."""
    cmd = env.command_manager.get_term(command_name)
    N = cmd.cfg.future_steps
    return torch.zeros(env.num_envs, N * 3, device=env.device)

"""Shared tracking observations — reference-anchor error in the robot frame.

These express the ROBOT's pose against a motion reference, which is the tracking
layer itself rather than any one pre-trained tracker: `textop` and `sonic` mean
the same thing by them, and a downstream task that tracks a reference needs them
without importing a specific WBC's package.

`textop.mdp.observations` re-exports them — its `__all__` is the term set its
profile is built from — and keeps the WoSe zero variant, which IS textop-specific.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

__all__ = ["motion_anchor_pos_b_future", "motion_anchor_ori_b_future"]


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

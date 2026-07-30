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

from mocke.mdp.joint_maps import MJ2IL as _MJ2IL  # noqa: F401 — re-exported
from mocke.mdp.observations import (  # noqa: F401 — re-export; canonical home
    motion_anchor_ori_b_future,
    motion_anchor_pos_b_future,
)

__all__ = [
    "generated_commands_il",
    "motion_anchor_pos_b_future",
    "motion_anchor_pos_b_future_zero",
    "motion_anchor_ori_b_future",
]






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

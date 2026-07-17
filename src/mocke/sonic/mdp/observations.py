"""SONIC observation terms.

The tokenizer term replicates the SONIC g1-encoder input op-chain *verbatim*
(gear_sonic commands.py::command_multi_future + root_rot_dif_l_multi_future,
"nonflat" reshapes included) in mjlab/MuJoCo joint order. The ported checkpoint
(scripts/sonic/port_sonic_checkpoint.py) bakes the matching joint permutation
into the encoder weights — the two layouts are coupled by construction.

Duck-typed on the shared command interface (``future_frames(steps, skip)`` +
``motion`` arrays) — works on both ``FutureMotionCommand`` (pure tracking) and
``OmniObjectMotionCommand`` (repose).
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul

FUTURE_STEPS = 10
FRAME_SKIP = 5  # 0.1 s @ 50 fps motion data


def sonic_g1_tokenizer(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    future_steps: int = FUTURE_STEPS,
    frame_skip: int = FRAME_SKIP,
    anchor_body: str = "pelvis",
) -> torch.Tensor:
    """SONIC g1 tokenizer obs: (num_envs, future_steps * (2*num_dof + 6)).

    Per SONIC's flat layout: cat([jp_future.flat | jv_future.flat]) chopped to
    (F, 2*num_dof) rows, concatenated with per-frame 6D pelvis-relative
    reference orientation, then flattened. The live pelvis pose is the robot
    root (independent of the command's anchor cfg).
    """
    command = env.command_manager.get_term(command_name)
    num_envs = env.num_envs
    idx = command.future_frames(future_steps, frame_skip)  # (B, F)

    jp = command.motion.joint_pos[idx].reshape(num_envs, -1)
    jv = command.motion.joint_vel[idx].reshape(num_envs, -1)
    # command_multi_future (flat) -> "nonflat" chop, exactly as in gear_sonic
    chop = torch.cat([jp, jv], dim=-1).reshape(num_envs, future_steps, -1)

    # root_rot_dif_l_multi_future: quat_inv(robot_pelvis) * ref_pelvis(f), 6D (first
    # two rotation-matrix columns, row-major flatten)
    anchor_index = command.cfg.body_names.index(anchor_body)
    ref_quat = command.motion.body_quat_w[idx, anchor_index]  # (B, F, 4)
    robot_quat = env.scene["robot"].data.root_link_quat_w  # pelvis = G1 root
    rot_dif = quat_mul(
        quat_inv(robot_quat[:, None, :].expand(num_envs, future_steps, 4).reshape(-1, 4)),
        ref_quat.reshape(-1, 4),
    )
    ori = matrix_from_quat(rot_dif)[..., :2].reshape(num_envs, future_steps, 6)

    return torch.cat([chop, ori], dim=-1).reshape(num_envs, -1)

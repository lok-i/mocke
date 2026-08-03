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
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

from mocke.mdp.joint_maps import IL2MJ

FUTURE_STEPS = 10
FRAME_SKIP = 5  # 0.1 s @ 50 fps motion data

SMPL_FUTURE_STEPS = 10
SMPL_FRAME_SKIP = 1  # 0.02 s @ 50 fps (sonic_release smpl_dt_future_ref_frames)
# The release smpl encoder's wrist slots are IsaacLab joint indices 23..28
# (L/R interleaved roll, pitch, yaw) — mapped here to MJ-order joint_pos slots.
SMPL_WRIST_MJ_IDX = tuple(IL2MJ.index(i) for i in range(23, 29))


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


def sonic_smpl_tokenizer(
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    future_steps: int = SMPL_FUTURE_STEPS,
    frame_skip: int = SMPL_FRAME_SKIP,
) -> torch.Tensor:
    """SONIC smpl-encoder input: (num_envs, future_steps * 84).

    Per-frame layout mirrors sonic_release.yaml's smpl encoder inputs, in order:
      [smpl_joints_multi_future_local_nonflat (72) |
       smpl_root_ori_b_multi_future (6) |
       joint_pos_multi_future_wrist_for_smpl (6)]

    Op-chain verbatim from gear_sonic (commands.py smpl properties +
    observations.py smpl terms): joints rotated into each future frame's OWN
    smpl-root frame; root ori = 6D of quat_inv(robot_root) * smpl_root; wrist
    refs = the retargeted G1 wrist joint targets (zeros when unavailable).

    Requires ``command.motion`` to carry ``smpl_joints (T, 24, 3)`` RAW as in
    SONIC's smpl pkl (**z-up**, root-centered, no transl — the motion lib
    never converts joints, only the root quat) and ``smpl_root_quat (T, 4)``
    z-up, wxyz, SMPL base rot removed.

    **Both z-up, and that is load-bearing.** The op-chain below is
    ``quat_apply_inverse(root_q, joints)``, so a frame mismatch does not error —
    it silently redefines the leading 72 dims as a body frame that rotates with
    heading. gear_sonic's ``smpl_y_up`` flag converts the ROOT ONLY
    (``motion_lib_base.py`` loads ``smpl_joints`` untouched), so a pkl with a
    y-up ``pose_aa`` still ships z-up joints. Zero-shot tracking reward on 63
    GRAIL curb clips: z-up **4.735** vs y-up **0.348** (g1 encoder 5.677).
    """
    command = env.command_manager.get_term(command_name)
    num_envs = env.num_envs
    idx = command.future_frames(future_steps, frame_skip)  # (B, F)

    joints = command.motion.smpl_joints[idx]      # (B, F, 24, 3)
    root_q = command.motion.smpl_root_quat[idx]   # (B, F, 4)

    q = root_q[:, :, None, :].expand(*joints.shape[:-1], 4)
    joints_local = quat_apply_inverse(
        q.reshape(-1, 4), joints.reshape(-1, 3)
    ).reshape(num_envs, future_steps, -1)          # (B, F, 72)

    robot_quat = env.scene["robot"].data.root_link_quat_w
    rot_dif = quat_mul(
        quat_inv(robot_quat[:, None, :].expand(num_envs, future_steps, 4).reshape(-1, 4)),
        root_q.reshape(-1, 4),
    )
    ori = matrix_from_quat(rot_dif)[..., :2].reshape(num_envs, future_steps, 6)

    wrist = command.motion.joint_pos[idx][:, :, SMPL_WRIST_MJ_IDX]  # (B, F, 6)

    return torch.cat([joints_local, ori, wrist], dim=-1).reshape(num_envs, -1)

"""Shared tracking command: mjlab MotionCommand + IL->MJ loader + future window.

One command serves every frozen WBC — the future-window *shape* is owned by the
obs terms via :meth:`FutureMotionCommand.future_frames` (textop reads 5 frames
@ skip 1, SONIC reads 10 @ skip 5). The named ``motion_*_future`` properties
(window = ``cfg.future_steps`` @ ``cfg.future_frame_skip``) mirror repose's
``OmniObjectMotionCommand`` interface, so tracking obs terms are duck-typed
across both commands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

from mock.mdp.joint_maps import G1_TRACKED_BODIES, IL2MJ


class MjMotionLoader:
    """mjlab MotionLoader-compatible loader with IL->MJ remapping.

    joint_pos/joint_vel columns are reordered IL->MJ; body arrays are selected
    via the FK-verified IL body-index table, ordered to match ``body_names``.
    """

    def __init__(self, motion_file: str, body_names: tuple[str, ...], device: str) -> None:
        il_index = dict(G1_TRACKED_BODIES)
        missing = [n for n in body_names if n not in il_index]
        assert not missing, f"bodies missing from IL index table: {missing}"
        body_ids = [il_index[n] for n in body_names]

        data = np.load(motion_file)
        _t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.joint_pos = _t(data["joint_pos"])[:, IL2MJ]
        self.joint_vel = _t(data["joint_vel"])[:, IL2MJ]
        self.body_pos_w = _t(data["body_pos_w"])[:, body_ids]
        self.body_quat_w = _t(data["body_quat_w"])[:, body_ids]
        self.body_lin_vel_w = _t(data["body_lin_vel_w"])[:, body_ids]
        self.body_ang_vel_w = _t(data["body_ang_vel_w"])[:, body_ids]
        self.time_step_total = self.joint_pos.shape[0]


class FutureMotionCommand(MotionCommand):
    """MotionCommand over a single IL-ordered clip, with future-frame access."""

    cfg: FutureMotionCommandCfg

    def __init__(self, cfg: FutureMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if cfg.il_ordered:
            # Stock MotionLoader misreads the IL-ordered npz; swap in the remapping loader.
            self.motion = MjMotionLoader(cfg.motion_file, cfg.body_names, device=self.device)

    # ── future window ──

    def future_frames(self, steps: int, skip: int = 1) -> torch.Tensor:
        """(num_envs, steps) motion frame indices at ``skip`` spacing, clamped to clip end."""
        offsets = torch.arange(steps, device=self.device, dtype=torch.long) * skip
        return torch.clamp(
            self.time_steps[:, None] + offsets[None, :],
            max=self.motion.time_step_total - 1,
        )

    def _future_time_indices(self) -> torch.Tensor:
        return self.future_frames(self.cfg.future_steps, self.cfg.future_frame_skip)

    # ── named props (OmniObjectMotionCommand-compatible) ──

    @property
    def command(self) -> torch.Tensor:
        """Future N-step [joint_pos | joint_vel], flattened. (B, 2*N*J)."""
        return torch.cat(
            [
                self.motion_joint_pos_future.reshape(self.num_envs, -1),
                self.motion_joint_vel_future.reshape(self.num_envs, -1),
            ],
            dim=1,
        )

    @property
    def motion_anchor_pos_w_future(self) -> torch.Tensor:
        """Future N-step anchor position, world frame. (B, N, 3)."""
        idx = self._future_time_indices()
        ai = self.motion_anchor_body_index
        return self.motion.body_pos_w[idx, ai] + self._env.scene.env_origins[:, None, :]

    @property
    def motion_anchor_quat_w_future(self) -> torch.Tensor:
        """Future N-step anchor quaternion. (B, N, 4)."""
        return self.motion.body_quat_w[self._future_time_indices(), self.motion_anchor_body_index]

    @property
    def motion_joint_pos_future(self) -> torch.Tensor:
        """Future N-step reference joint positions. (B, N, J)."""
        return self.motion.joint_pos[self._future_time_indices()]

    @property
    def motion_joint_vel_future(self) -> torch.Tensor:
        """Future N-step reference joint velocities. (B, N, J)."""
        return self.motion.joint_vel[self._future_time_indices()]


@dataclass(kw_only=True)
class FutureMotionCommandCfg(MotionCommandCfg):
    future_steps: int = 5
    """Window length of the named ``motion_*_future`` properties."""
    future_frame_skip: int = 1
    """Motion frames between window entries (obs terms may request other shapes
    via :meth:`FutureMotionCommand.future_frames`)."""
    il_ordered: bool = True
    """Clip npz is IL-ordered (IL->MJ remap loader); False = MJ-native npz,
    keep mjlab's stock MotionLoader (e.g. the mjlab demo clip)."""

    def build(self, env: ManagerBasedRlEnv) -> FutureMotionCommand:
        return FutureMotionCommand(self, env)

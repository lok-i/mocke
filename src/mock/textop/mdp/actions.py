"""IL-order joint position action for aug-WBC.

The pretrained WBC operates in IsaacLab joint order (PhysX BFS traversal),
while mjlab/MuJoCo uses XML DFS order. This thin wrapper accepts IL-ordered
actions from the policy, reorders to MJ for the sim, and exposes IL-ordered
raw_action so ``last_action`` obs feeds the correct order back to the WBC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from mock.mdp.joint_maps import IL2MJ as _IL2MJ

__all__ = ["JointPositionActionILCfg", "JointPositionActionIL"]


@dataclass(kw_only=True)
class JointPositionActionILCfg(JointPositionActionCfg):
    """JointPositionAction that expects IL-ordered actions from the policy."""

    il2mj: list[int] = field(default_factory=lambda: list(_IL2MJ))

    def build(self, env):
        return JointPositionActionIL(self, env)


class JointPositionActionIL(JointPositionAction):
    """Reorders IL→MJ on process, exposes IL-ordered raw_action."""

    def __init__(self, cfg: JointPositionActionILCfg, env):
        super().__init__(cfg, env)
        self._il2mj = torch.tensor(cfg.il2mj, device=self.device, dtype=torch.long)
        self._il_raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

    def process_actions(self, actions: torch.Tensor):
        self._il_raw_actions[:] = actions
        super().process_actions(actions[:, self._il2mj])

    @property
    def raw_action(self) -> torch.Tensor:
        """Raw actions in IL order (what the policy actually output)."""
        return self._il_raw_actions

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._il_raw_actions[env_ids] = 0.0

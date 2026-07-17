"""Play the frozen textop WBC tracking a motion clip in mjlab (base only).

Builds the Mock-Tracking-Textop-G1 play env and runs the ported checkpoint
deterministically. Needs the lok-i/rsl_rl fork in the env (ModularNormMLP).

Usage:
    python scripts/play_textop.py --num_envs 1
    python scripts/play_textop.py --motion path/to/motion.npz --il_ordered --viewer native
"""

from __future__ import annotations

import argparse
import os

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from rsl_rl.models import ModularNormMLPWithAdapterModel

from mock import PRETRAINED_DIR, default_motion_file
from mock.textop.env_cfg import textop_tracking_env_cfg

_DEFAULT_CKPT = PRETRAINED_DIR / "textop/model_75000_ported.pt"
# Frozen base arch (must match the ckpt).
_WBC_HIDDEN = (2048, 1024, 512)
_DIST = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


def main():
    parser = argparse.ArgumentParser(description="Play the frozen textop WBC in mjlab.")
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CKPT))
    parser.add_argument("--motion", type=str, default=None,
                        help="motion.npz clip to track (default: mjlab demo clip).")
    parser.add_argument("--il_ordered", action="store_true",
                        help="--motion clip is IL-ordered (retargeted-dataset format).")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", choices=["auto", "native", "viser"], default="auto")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    motion = args.motion or default_motion_file()
    assert motion, "no motion clip found — pass --motion"
    il_ordered = args.il_ordered if args.motion else False

    env_cfg = textop_tracking_env_cfg(motion_file=motion, play=True, il_ordered=il_ordered)
    env_cfg.scene.num_envs = args.num_envs

    print(f"[play_textop] motion={motion}")
    print(f"[play_textop] checkpoint={args.checkpoint}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    obs = env.get_observations()
    # Adapter model with all layers skipped == pure frozen base (loads the ckpt
    # through the same ModularNormMLP path).
    model = ModularNormMLPWithAdapterModel(
        obs=obs, obs_groups={"actor": ["policy"]}, obs_set="actor", output_dim=29,
        hidden_dims=list(_WBC_HIDDEN), activation="elu", obs_normalization=True,
        distribution_cfg=dict(_DIST), base_checkpoint=args.checkpoint,
        adapter_obs_group="policy", rank=[None] * len(_WBC_HIDDEN) + [None],
    ).to(device)
    model.eval()

    @torch.no_grad()
    def policy(obs_td):
        return model(obs_td)

    if args.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved = "native" if has_display else "viser"
    else:
        resolved = args.viewer

    if resolved == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()

"""Play the frozen SONIC base WBC tracking a motion clip in mjlab (base only).

Builds the Mock-Tracking-Sonic-G1 play env and runs the ported checkpoint
(scripts/port_sonic_checkpoint.py output) deterministically. Needs the
lok-i/rsl_rl fork in the env (SonicBaseModel).

Usage:
    python scripts/play_sonic.py --num_envs 1
    python scripts/play_sonic.py --motion path/to/motion.npz --il_ordered --viewer native
"""

from __future__ import annotations

import argparse
import os

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from rsl_rl.models import SonicBaseModel

from mock import PRETRAINED_DIR, default_motion_file
from mock.sonic.env_cfg import sonic_tracking_env_cfg

_DEFAULT_CKPT = PRETRAINED_DIR / "sonic/last_ported.pt"


def main():
    parser = argparse.ArgumentParser(description="Play the frozen SONIC base in mjlab.")
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

    env_cfg = sonic_tracking_env_cfg(motion_file=motion, play=True, il_ordered=il_ordered)
    env_cfg.scene.num_envs = args.num_envs

    print(f"[play_sonic] motion={motion}")
    print(f"[play_sonic] checkpoint={args.checkpoint}")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    obs = env.get_observations()
    model = SonicBaseModel(
        obs=obs, obs_groups={"actor": ["policy"]}, obs_set="actor",
        output_dim=29, base_checkpoint=args.checkpoint,
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

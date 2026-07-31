"""Export the frozen SONIC base (or SONIC + zero-init LoRA adapter) to ONNX + manifest.

The tracking twin of vibe's `export-onnx`: same artifact contract (vibe.onnx.v1 —
self-contained graph, manifest in a sibling .json AND in the onnx metadata), but built
directly like play_sonic.py — no runner, no wandb, no repose dataset plumbing. The
manifest builder and the OnnxAgent parity harness are imported from vibe.deploy (the
schema owner) until the exporter migrates here.

The parity gate mirrors vibe's open-loop check: the torch checkpoint drives the env;
every step the exported graph is asked for an action on the same observation and the
delta is gated at --tolerance (cpu fp32 both sides). With --adapter the adapters are
zero-initialized, so the gate ALSO proves adapter == base bit-near-exactly.

Usage:
    python scripts/export_sonic_onnx.py                       # base SONIC, demo clip
    python scripts/export_sonic_onnx.py --adapter --rank 16   # + zero-init LoRA
    python scripts/export_sonic_onnx.py --motion clip.npz --il_ordered --output-dir /tmp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from rsl_rl.models import SonicBaseModel, SonicWithAdapterModel

from mocke import PRETRAINED_DIR, default_motion_file
from mocke.sonic.env_cfg import sonic_tracking_env_cfg

_DEFAULT_CKPT = PRETRAINED_DIR / "sonic/last_ported.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CKPT))
    parser.add_argument("--motion", type=str, default=None,
                        help="motion.npz to build the env with (default: mjlab demo clip).")
    parser.add_argument("--il_ordered", action="store_true",
                        help="--motion clip is IL-ordered (retargeted-dataset format).")
    parser.add_argument("--adapter", action="store_true",
                        help="export SonicWithAdapterModel with ZERO-INIT LoRA adapters.")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="artifact directory (default: the checkpoint's folder).")
    parser.add_argument("--stem", type=str, default=None,
                        help="artifact basename (default: g1_sonic_base / g1_sonic_adapter0).")
    parser.add_argument("--check-steps", type=int, default=64)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu keeps the parity gate fp32-exact on both sides.")
    args = parser.parse_args()

    # Schema owner (see module docstring) — hard requirement, soft import.
    from vibe.deploy.onnx_agent import OnnxAgent
    from vibe.deploy.onnx_manifest import build_manifest

    motion = args.motion or default_motion_file()
    assert motion, "no motion clip found — pass --motion"
    il_ordered = args.il_ordered if args.motion else False

    env_cfg = sonic_tracking_env_cfg(
        motion_file=motion, play=True, il_ordered=il_ordered, adapter=args.adapter
    )
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    obs = env.get_observations()

    common = dict(obs=obs, obs_groups={"actor": ["policy"]}, obs_set="actor",
                  output_dim=29, base_checkpoint=args.checkpoint)
    if args.adapter:
        model = SonicWithAdapterModel(
            adapter_obs_group="augmentation", rank=args.rank, alpha=args.alpha,
            freeze_base=True, adapt_encoder=False, adapt_decoder=True, **common)
    else:
        model = SonicBaseModel(**common)
    model = model.to(args.device).eval()

    stem = args.stem or ("g1_sonic_adapter0" if args.adapter else "g1_sonic_base")
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{stem}.onnx"

    onnx_model = model.as_onnx(verbose=False)
    onnx_model.to("cpu").eval()
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
        dynamic_axes={},
        dynamo=False,
    )
    print(f"[export] inputs : {onnx_model.input_names}")
    print(f"[export] outputs: {onnx_model.output_names}")

    manifest = build_manifest(
        env.unwrapped, onnx_model,
        task_id="Mocke-Tracking-Sonic-G1",
        run_path=args.checkpoint,
        checkpoint=args.checkpoint,
        model_class=type(model).__name__,
    )
    manifest_path = out_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    attach_metadata_to_onnx(str(onnx_path), {"manifest": json.dumps(manifest)})

    # --- open-loop parity gate: torch drives, ONNX shadows ---
    agent = OnnxAgent(str(onnx_path), provider="cpu")
    max_delta = 0.0
    with torch.no_grad():
        for step in range(args.check_steps):
            torch_actions = model(obs)
            onnx_actions = agent.act(obs, world=0)
            delta = float(np.abs(
                torch_actions[0].cpu().numpy() - onnx_actions[0]).max())
            max_delta = max(max_delta, delta)
            if delta > args.tolerance:
                onnx_path.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"open-loop parity failed at step {step}: "
                    f"max|delta a| = {delta:.3e} > {args.tolerance:.1e}")
            obs, _, _, _ = env.step(torch_actions)
    print(f"[export] parity ok over {args.check_steps} steps: "
          f"max|delta a| = {max_delta:.3e}")
    print(f"[export] wrote {onnx_path}")
    print(f"[export] wrote {manifest_path}")
    env.close()


if __name__ == "__main__":
    main()

"""Export frozen Textop or SONIC policies and verify them in two worlds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from rsl_rl.models import (
    ModularNormMLPModel,
    SonicBaseModel,
    SonicWithAdapterModel,
)

from mocke import PRETRAINED_DIR, default_motion_file
from mocke.export.manifest import build_manifest
from mocke.export.onnx_agent import DualPolicy, OnnxAgent, WorldStats
from mocke.sonic.env_cfg import sonic_tracking_env_cfg
from mocke.textop.env_cfg import textop_tracking_env_cfg

Recipe = Literal["sonic", "textop"]

_DEFAULT_CHECKPOINT = {
    "sonic": PRETRAINED_DIR / "sonic/last_ported.pt",
    "textop": PRETRAINED_DIR / "textop/model_75000_ported.pt",
}
_TASK_ID = {
    "sonic": "Mocke-Tracking-Sonic-G1",
    "textop": "Mocke-Tracking-Textop-G1",
}


def _make_env_cfg(recipe: Recipe, motion: str, il_ordered: bool, adapter: bool):
    if recipe == "sonic":
        return sonic_tracking_env_cfg(
            motion_file=motion,
            play=True,
            il_ordered=il_ordered,
            adapter=adapter,
        )
    return textop_tracking_env_cfg(
        motion_file=motion, play=True, il_ordered=il_ordered
    )


def _make_model(recipe: Recipe, obs, checkpoint: str, adapter: bool, rank: int, alpha: float):
    common = {
        "obs": obs,
        "obs_groups": {"actor": ["policy"]},
        "obs_set": "actor",
        "output_dim": 29,
    }
    if recipe == "sonic":
        sonic_common = {**common, "base_checkpoint": checkpoint}
        if adapter:
            return SonicWithAdapterModel(
                adapter_obs_group="augmentation",
                rank=rank,
                alpha=alpha,
                freeze_base=True,
                adapt_encoder=False,
                adapt_decoder=True,
                **sonic_common,
            )
        return SonicBaseModel(**sonic_common)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    build_cfg = payload["build_cfg"]
    model = ModularNormMLPModel(
        hidden_dims=build_cfg["hidden_dims"],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        **common,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _textop_layout(obs) -> list[dict]:
    return [
        {
            "name": "obs",
            "shape": [obs["policy"].shape[-1]],
            "groups": ["policy"],
            "term": None,
        }
    ]


def _parser(default_recipe: Recipe | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    if default_recipe is None:
        parser.add_argument("recipe", choices=("sonic", "textop"))
    else:
        parser.set_defaults(recipe=default_recipe)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--motion", default=None)
    parser.add_argument("--il_ordered", action="store_true")
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--check-steps", type=int, default=64)
    parser.add_argument("--closed-loop-min-steps", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viewer", choices=("none", "native"), default="none")
    return parser


class _CheckedViewerEnv:
    """Observe viewer-driven steps without changing the viewer loop."""

    def __init__(self, env, check) -> None:
        self._env = env
        self._check = check

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        result = self._env.step(actions)
        self._check.update(result)
        return result


class _LiveCheck:
    """Gate the first visible rollout window, then let playback continue."""

    def __init__(
        self,
        env,
        policy: DualPolicy,
        *,
        steps: int,
        min_steps: int,
        tolerance: float,
        on_pass,
        on_fail,
    ) -> None:
        self.env = env
        self.policy = policy
        self.total_steps = steps
        self.min_steps = min_steps
        self.tolerance = tolerance
        self.on_pass = on_pass
        self.on_fail = on_fail
        self.stats = WorldStats((0, 1))
        self.step = 0
        self.complete = False
        self.error: Exception | None = None

    def update(self, result) -> None:
        if self.complete or self.error is not None:
            return
        _, rewards, dones, _ = result
        try:
            if self.policy.open_loop_max > self.tolerance:
                raise RuntimeError(
                    f"open-loop parity failed at step {self.step}: "
                    f"max|delta a| = {self.policy.open_loop_max:.3e} "
                    f"> {self.tolerance:.1e}"
                )
            self.stats.update(self.step, rewards, dones, self.env)
            self.step += 1
            if self.step < self.total_steps:
                return

            survived = self.stats.first_reset[1] or self.total_steps
            if survived < self.min_steps:
                raise RuntimeError(
                    f"closed-loop ONNX world reset after {survived} steps "
                    f"(< min_steps {self.min_steps})"
                )
            self.complete = True
            self.policy.finish_check()
            self.on_pass(self.stats)
        except Exception as exc:
            self.error = exc
            self.on_fail()
            raise


def _launch_viewer(env, policy: DualPolicy, check: _LiveCheck) -> None:
    from mjlab.viewer import NativeMujocoViewer

    print(
        "[export] viewer: world 0 = PyTorch, world 1 = ONNX; "
        f"checking the first {check.total_steps} visible steps"
    )
    NativeMujocoViewer(_CheckedViewerEnv(env, check), policy).run()


def main(default_recipe: Recipe | None = None) -> None:
    args = _parser(default_recipe).parse_args()
    recipe: Recipe = args.recipe
    if recipe == "textop" and args.adapter:
        raise ValueError("--adapter is supported only for SONIC")
    if args.check_steps < 1:
        raise ValueError("--check-steps must be at least 1")

    checkpoint = args.checkpoint or str(_DEFAULT_CHECKPOINT[recipe])
    motion = args.motion or default_motion_file()
    if not motion:
        raise RuntimeError("no motion clip found -- pass --motion")
    il_ordered = args.il_ordered if args.motion else False

    env_cfg = _make_env_cfg(recipe, motion, il_ordered, args.adapter)
    env_cfg.scene.num_envs = 2
    ManagerBasedRlEnv.seed(args.seed)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    try:
        obs = env.get_observations()
        model = _make_model(
            recipe, obs, checkpoint, args.adapter, args.rank, args.alpha
        ).to(args.device).eval()

        stem = args.stem or (
            "g1_sonic_adapter0"
            if args.adapter
            else f"g1_{recipe}_base"
        )
        out_dir = Path(args.output_dir) if args.output_dir else Path(checkpoint).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = out_dir / f"{stem}.onnx"
        manifest_path = out_dir / f"{stem}.manifest.json"

        onnx_model = model.as_onnx(verbose=False).to("cpu").eval()
        if recipe == "textop":
            onnx_model.layout = _textop_layout(obs)
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
            env.unwrapped,
            onnx_model,
            task_id=_TASK_ID[recipe],
            checkpoint=checkpoint,
            model_class=type(model).__name__,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2))
        attach_metadata_to_onnx(str(onnx_path), {"manifest": json.dumps(manifest)})

        agent = OnnxAgent(str(onnx_path))
        policy = DualPolicy(model, agent)
        policy.warmup(env.get_observations())
        min_steps = (
            args.check_steps
            if args.closed_loop_min_steps is None
            else args.closed_loop_min_steps
        )

        def fail() -> None:
            onnx_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            print("[export] check FAILED -- no files written")

        def report(stats: WorldStats) -> None:
            print(
                f"[export] open-loop gate : max|delta a| = "
                f"{policy.open_loop_max:.3e} over {args.check_steps} steps"
            )
            print(f"[export] torch world    : {stats.line(0, args.check_steps)}")
            print(f"[export] ONNX world     : {stats.line(1, args.check_steps)}")
            print(f"[export] wrote {onnx_path}")
            print(f"[export] wrote {manifest_path}")

        if args.viewer == "native":
            check = _LiveCheck(
                env,
                policy,
                steps=args.check_steps,
                min_steps=min_steps,
                tolerance=args.tolerance,
                on_pass=report,
                on_fail=fail,
            )
            _launch_viewer(env, policy, check)
            if check.error is not None:
                raise check.error
            if not check.complete:
                fail()
                raise RuntimeError(
                    f"viewer closed before the {args.check_steps}-step check completed"
                )
        else:
            stats = WorldStats((0, 1))
            obs = env.get_observations()
            try:
                with torch.no_grad():
                    for step in range(args.check_steps):
                        actions = policy(obs)
                        if policy.open_loop_max > args.tolerance:
                            raise RuntimeError(
                                f"open-loop parity failed at step {step}: "
                                f"max|delta a| = {policy.open_loop_max:.3e} "
                                f"> {args.tolerance:.1e}"
                            )
                        obs, rewards, dones, _ = env.step(actions)
                        stats.update(step, rewards, dones, env)

                survived = stats.first_reset[1] or args.check_steps
                if survived < min_steps:
                    raise RuntimeError(
                        f"closed-loop ONNX world reset after {survived} steps "
                        f"(< min_steps {min_steps})"
                    )
            except Exception:
                fail()
                raise
            report(stats)
    finally:
        env.close()

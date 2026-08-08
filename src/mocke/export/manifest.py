"""Ordered observation and action contract embedded in exported ONNX files."""

from __future__ import annotations

import subprocess
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction

from mocke.mdp import MJ2IL
from mocke.textop.mdp import JointPositionActionIL

_SCHEMA = "vibe.onnx.v1"


def _action_metadata(env: ManagerBasedRlEnv) -> dict:
    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, JointPositionAction)

    joint_to_ctrl = {act.target.split("/")[-1]: act.id for act in robot.spec.actuators}
    ctrl_ids = [joint_to_ctrl[name] for name in robot.joint_names]
    order = list(MJ2IL) if isinstance(action, JointPositionActionIL) else list(
        range(len(robot.joint_names))
    )

    scale = action._scale
    if hasattr(scale, "cpu"):
        scale = scale[0].cpu().tolist()
    scale = [scale[index] for index in order]
    default_joint_pos = robot.data.default_joint_pos[0].cpu().tolist()
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0].tolist()
    damping = (-env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]).tolist()

    return {
        "type": "joint_position",
        "joint_names": [robot.joint_names[index] for index in order],
        "scale": scale,
        "default_joint_pos": [default_joint_pos[index] for index in order],
        "stiffness": [stiffness[index] for index in order],
        "damping": [damping[index] for index in order],
    }


def _term_table(env: ManagerBasedRlEnv, group: str, only: str | None) -> list[dict]:
    manager = env.observation_manager
    entries, offset = [], 0
    for index, term in enumerate(manager.active_terms[group]):
        shape = tuple(manager.group_obs_term_dim[group][index])
        width = 1
        for axis in shape:
            width *= axis
        if only is None or term == only:
            cfg = manager.get_term_cfg(group, term)
            entries.append(
                {
                    "name": term,
                    "shape": list(shape),
                    "dim": width,
                    "offset": offset,
                    "history_length": getattr(cfg, "history_length", 0),
                    "flatten_history_dim": getattr(
                        cfg, "flatten_history_dim", True
                    ),
                }
            )
        offset += width
    return entries


def _git_sha(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_manifest(
    env: ManagerBasedRlEnv,
    onnx_model,
    *,
    task_id: str,
    checkpoint: str,
    model_class: str,
) -> dict:
    """Build the deploy contract from the graph's own port layout."""
    inputs = []
    for port in onnx_model.layout:
        group = port["groups"][0]
        inputs.append(
            {
                "name": port["name"],
                "shape": list(port["shape"]),
                "groups": list(port["groups"]),
                "terms": _term_table(env, group, only=port["term"]),
            }
        )

    import rsl_rl

    return {
        "schema": _SCHEMA,
        "task_id": task_id,
        "run_path": checkpoint,
        "checkpoint": checkpoint,
        "model_class": model_class,
        "control": {
            "sim_timestep": env.cfg.sim.mujoco.timestep,
            "decimation": env.cfg.decimation,
            "step_dt": env.cfg.sim.mujoco.timestep * env.cfg.decimation,
        },
        "action": _action_metadata(env),
        "inputs": inputs,
        "outputs": list(onnx_model.output_names),
        "versions": {
            "mocke": _git_sha(Path(__file__).resolve().parents[3]),
            "rsl_rl": _git_sha(Path(rsl_rl.__file__).resolve().parent),
        },
    }

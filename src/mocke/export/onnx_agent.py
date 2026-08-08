"""Manifest-driven ONNX policy and two-world rollout helpers."""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import onnxruntime as ort
import torch


class OnnxAgent:
    """Run an exported policy from the manifest embedded in its ONNX file."""

    def __init__(self, onnx_path: str) -> None:
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.manifest = json.loads(
            self.session.get_modelmeta().custom_metadata_map["manifest"]
        )
        self.ports = self.manifest["inputs"]
        self.action_index = self.manifest["outputs"].index("actions")
        self._inputs = {
            port["name"]: np.empty((1, *port["shape"]), dtype=np.float32)
            for port in self.ports
        }

        graph_inputs = {item.name: tuple(item.shape) for item in self.session.get_inputs()}
        for port in self.ports:
            expected = (1, *port["shape"])
            if graph_inputs.get(port["name"]) != expected:
                raise RuntimeError(
                    f"manifest/graph mismatch on '{port['name']}': "
                    f"manifest says {expected}, graph says {graph_inputs.get(port['name'])}"
                )
            width = int(np.prod(port["shape"]))
            declared = sum(term["dim"] for term in port["terms"])
            if declared != width:
                raise RuntimeError(
                    f"manifest terms for '{port['name']}' sum to {declared}, "
                    f"input is {width}"
                )

    def assemble(self, obs, world: int) -> dict[str, np.ndarray]:
        """Assemble one world's named ONNX inputs term by term."""
        for port in self.ports:
            group = obs[port["groups"][0]]
            terms = port["terms"]
            target = self._inputs[port["name"]].reshape(-1)
            if hasattr(group, "keys"):
                source = group[terms[0]["name"]][world].reshape(-1)
                target[:] = source.detach().cpu().numpy().astype(
                    np.float32, copy=False
                )
            else:
                row = group[world]
                cursor = 0
                for term in terms:
                    width = term["dim"]
                    source = row[term["offset"] : term["offset"] + width]
                    target[cursor : cursor + width] = (
                        source.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    cursor += width
        return self._inputs

    def act(self, obs, world: int) -> np.ndarray:
        """Return actions for one world with shape ``(1, num_actions)``."""
        return self.session.run(None, self.assemble(obs, world))[self.action_index]


class DualPolicy:
    """Run PyTorch in world 0, ONNX in world 1, and gate world 0 open-loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        agent: OnnxAgent,
    ) -> None:
        self.model = model
        self.agent = agent
        self.open_loop_max = 0.0
        self.checking = True

    @torch.inference_mode()
    def __call__(self, obs) -> torch.Tensor:
        torch_action = self.model(obs[0:1])
        if self.checking:
            reference = torch_action.detach().cpu().numpy()
            exported = self.agent.act(obs, world=0)
            self.open_loop_max = max(
                self.open_loop_max, float(np.abs(reference - exported).max())
            )
        onnx_action = torch.as_tensor(
            self.agent.act(obs, world=1),
            device=torch_action.device,
            dtype=torch_action.dtype,
        )
        return torch.cat((torch_action, onnx_action), dim=0)

    @torch.inference_mode()
    def warmup(self, obs) -> None:
        """Initialize PyTorch and both batch-1 ONNX paths without stepping the env."""
        self.model(obs[0:1])
        self.agent.act(obs, world=0)
        self.agent.act(obs, world=1)

    def finish_check(self) -> None:
        """Stop the world-0 ONNX shadow pass after the parity window closes."""
        self.checking = False


class WorldStats:
    """Survival and return summary for selected worlds."""

    def __init__(self, worlds: Sequence[int]) -> None:
        self.worlds = list(worlds)
        self.steps = dict.fromkeys(self.worlds, 0)
        self.first_reset = dict.fromkeys(self.worlds, None)
        self.resets = dict.fromkeys(self.worlds, 0)
        self.returns = dict.fromkeys(self.worlds, 0.0)
        self.fired: dict[int, set[str]] = {world: set() for world in self.worlds}

    def update(self, step: int, rewards: torch.Tensor, dones: torch.Tensor, env) -> None:
        manager = env.unwrapped.termination_manager
        for world in self.worlds:
            self.returns[world] += float(rewards[world])
            self.steps[world] += 1
            if bool(dones[world]):
                self.resets[world] += 1
                if self.first_reset[world] is None:
                    self.first_reset[world] = step + 1
                for name in manager.active_terms:
                    if bool(manager.get_term(name)[world]):
                        self.fired[world].add(name)

    def line(self, world: int, total_steps: int) -> str:
        survived = self.first_reset[world] or total_steps
        terms = ",".join(sorted(self.fired[world])) or "none"
        mean_reward = self.returns[world] / max(self.steps[world], 1)
        return (
            f"{survived}/{total_steps} steps to first reset, "
            f"{self.resets[world]} reset(s) [{terms}], r̄ = {mean_reward:.3f}"
        )

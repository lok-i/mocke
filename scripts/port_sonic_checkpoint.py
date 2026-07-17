"""Port the GEAR-SONIC release checkpoint to a mjlab-native SonicBaseModel state dict.

Extracts the g1-mode pipeline (g1 encoder -> FSQ -> g1_dyn decoder) from
``sonic_release/last.pt`` and bakes the IsaacLab->MuJoCo joint permutation into
the first/last linear layers, so the runtime model consumes observations and
emits actions in mjlab (MuJoCo XML) joint order with no runtime converters.

Layout contract (must match src/mock/sonic/mdp/observations.py):

  tokenizer (640) = cat([jp_mf.flat(290) | jv_mf.flat(290)]).reshape(10, 58)
                    cat ori_mf(10, 6) -> flatten          # SONIC op-chain, verbatim
  proprio (930)   = [ang_vel hist(30) | joint_pos hist(290) | joint_vel hist(290)
                     | actions hist(290) | gravity hist(30)], history oldest-first

The source ckpt (447 MB, gitignored) is auto-downloaded from
huggingface.co/nvidia/GEAR-SONIC when missing.

Usage:
    python scripts/port_sonic_checkpoint.py \
        [--ckpt pretrained/sonic/last.pt] \
        [--out pretrained/sonic/last_ported.pt]
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import shutil
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]

# G1 29-dof body joints in IsaacLab (BFS) order — copied from
# gear_sonic/envs/env_utils/joint_utils.py::G1_ISAACLab_ORDER.
G1_ISAACLAB_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

NUM_FUTURE_FRAMES = 10
HIST = 10
NUM_DOF = 29


def fetch_release_checkpoint(dest: Path) -> None:
    """Download nvidia/GEAR-SONIC sonic_release/last.pt from Hugging Face."""
    from huggingface_hub import hf_hub_download

    print(f"[port] fetching sonic_release/last.pt from HF -> {dest}")
    cached = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="sonic_release/last.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)


def load_release_checkpoint(path: str) -> dict:
    """torch.load with a fallback unpickler: the ckpt pickles internal `trl.*`
    config classes that don't exist here — stub anything unresolvable."""

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            for candidate in (module, module.replace("trl", "gear_sonic.trl", 1)):
                try:
                    return getattr(importlib.import_module(candidate), name)
                except Exception:
                    continue
            return type(name, (_Dummy,), {"__module__": module})

    class _PickleModule:
        Unpickler = _Unpickler
        load = staticmethod(pickle.load)

    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PickleModule)


def mjlab_g1_joint_order() -> list[str]:
    import mujoco
    from mjlab.asset_zoo.robots import get_g1_robot_cfg

    model = get_g1_robot_cfg().spec_fn().compile()
    hinge = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    assert len(hinge) == NUM_DOF, f"expected {NUM_DOF} joints, got {len(hinge)}"
    return hinge


def tokenizer_layout(joint_slot_ids: torch.Tensor) -> torch.Tensor:
    """Replay the SONIC tokenizer op-chain on symbolic ids.

    joint_slot_ids: (29,) canonical id of the joint occupying each slot.
    Returns (640,) canonical quantity-ids in flat layout order.
    """
    f = torch.arange(NUM_FUTURE_FRAMES).view(-1, 1)
    jp = 0 * 1000 + f * NUM_DOF + joint_slot_ids.view(1, -1)  # (10, 29)
    jv = 1 * 1000 + f * NUM_DOF + joint_slot_ids.view(1, -1)
    cmd = torch.cat([jp.reshape(-1), jv.reshape(-1)])  # (580,) = command_multi_future
    chop = cmd.reshape(NUM_FUTURE_FRAMES, 2 * NUM_DOF)  # nonflat reshape (10, 58)
    ori = 2 * 1000 + torch.arange(NUM_FUTURE_FRAMES * 6).reshape(NUM_FUTURE_FRAMES, 6)
    return torch.cat([chop, ori], dim=-1).reshape(-1)  # (640,)


def proprio_layout(joint_slot_ids: torch.Tensor) -> torch.Tensor:
    """[ang_vel hist | joint_pos hist | joint_vel hist | actions hist | gravity hist],
    each term flattened history-major (oldest first). Returns (930,) ids."""
    h = torch.arange(HIST).view(-1, 1)
    parts = [3 * 1000 + torch.arange(HIST * 3)]  # base_ang_vel
    for q in (4, 5, 6):  # joint_pos, joint_vel, actions
        parts.append((q * 1000 + h * NUM_DOF + joint_slot_ids.view(1, -1)).reshape(-1))
    parts.append(7 * 1000 + torch.arange(HIST * 3))  # gravity_dir
    return torch.cat(parts)


def perm_from_layouts(theirs: torch.Tensor, ours: torch.Tensor) -> torch.Tensor:
    """src[i] = our flat index holding the quantity at their flat index i."""
    # ids are not contiguous (blocks of 1000); match via sorting
    order_ours = torch.argsort(ours)
    order_theirs = torch.argsort(theirs)
    src = torch.empty_like(theirs)
    src[order_theirs] = order_ours  # matching sorted ids pairs positions
    assert torch.equal(ours[src], theirs)
    return src


def permute_in_columns(weight: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(weight)
    out[:, src] = weight
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", default=str(_REPO_ROOT / "pretrained/sonic/last.pt")
    )
    parser.add_argument(
        "--out", default=str(_REPO_ROOT / "pretrained/sonic/last_ported.pt")
    )
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        fetch_release_checkpoint(Path(args.ckpt))
    ckpt = load_release_checkpoint(args.ckpt)
    policy = ckpt["policy_state_dict"]

    mj_order = mjlab_g1_joint_order()
    # slot ids: for their tensors, slot k holds isaaclab joint k -> canonical (mjlab) id
    il_slot_ids = torch.tensor([mj_order.index(n) for n in G1_ISAACLAB_ORDER])
    mj_slot_ids = torch.arange(NUM_DOF)

    # --- permutations ---
    tok_src = perm_from_layouts(tokenizer_layout(il_slot_ids), tokenizer_layout(mj_slot_ids))
    prop_src = perm_from_layouts(proprio_layout(il_slot_ids), proprio_layout(mj_slot_ids))
    token_total_dim = 64
    dec_src = torch.cat([torch.arange(token_total_dim), token_total_dim + prop_src])
    act_dst = il_slot_ids  # their action row k belongs to mjlab joint il_slot_ids[k]

    # --- remap keys + bake permutations ---
    state = {}
    for key, val in policy.items():
        if key.startswith("actor_module.encoders.g1.module."):
            new_key = key.replace("actor_module.encoders.g1.module.", "encoder.")
            if new_key == "encoder.0.weight":
                val = permute_in_columns(val, tok_src)
            state[new_key] = val.clone()
        elif key.startswith("actor_module.decoders.g1_dyn.module."):
            new_key = key.replace("actor_module.decoders.g1_dyn.module.", "decoder.")
            if new_key == "decoder.0.weight":
                val = permute_in_columns(val, dec_src)
            elif new_key in ("decoder.12.weight", "decoder.12.bias"):
                out = torch.empty_like(val)
                out[act_dst] = val
                val = out
            state[new_key] = val.clone()

    std = torch.empty_like(policy["std"])
    std[act_dst] = policy["std"]

    # --- verify: original-layout model vs ported model on random input ---
    from rsl_rl.models import SonicBaseModel
    from rsl_rl.models.sonic_base_model import _mlp, fsq_quantize
    from tensordict import TensorDict

    enc_il = _mlp(640, (2048, 1024, 512, 512), 64, "SiLU")
    dec_il = _mlp(994, (2048, 2048, 1024, 1024, 512, 512), 29, "SiLU")
    enc_il.load_state_dict(
        {k.replace("actor_module.encoders.g1.module.", ""): v for k, v in policy.items()
         if k.startswith("actor_module.encoders.g1.module.")}
    )
    dec_il.load_state_dict(
        {k.replace("actor_module.decoders.g1_dyn.module.", ""): v for k, v in policy.items()
         if k.startswith("actor_module.decoders.g1_dyn.module.")}
    )

    obs = TensorDict(
        {"policy": torch.randn(4, 930), "tokenizer": torch.randn(4, 640)}, batch_size=[4]
    )
    model = SonicBaseModel(
        obs=obs, obs_groups={"actor": ["policy"]}, obs_set="actor", output_dim=29
    )
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        act_mj = model(obs)
        # original pipeline on il-layout inputs
        tok_il = obs["tokenizer"][:, tok_src]
        prop_il = obs["policy"][:, prop_src]
        levels = torch.full((32,), 32, dtype=torch.long)
        z = enc_il(tok_il).view(4, 2, 32)
        tokens = fsq_quantize(z, levels).reshape(4, 64)
        act_il = dec_il(torch.cat([tokens, prop_il], dim=-1))
        act_roundtrip = torch.empty_like(act_il)
        act_roundtrip[:, act_dst] = act_il

    err = (act_mj - act_roundtrip).abs().max().item()
    assert err < 1e-5, f"port verification failed: max err {err}"
    print(f"[port] permutation round-trip max err: {err:.2e}")

    # optional: FSQ cross-check against vector_quantize_pytorch
    try:
        from vector_quantize_pytorch import FSQ

        vq = FSQ(levels=[32] * 32)
        z = torch.randn(8, 2, 32)
        ours = fsq_quantize(z, torch.full((32,), 32, dtype=torch.long))
        theirs, _ = vq(z)
        fsq_err = (ours - theirs).abs().max().item()
        print(f"[port] FSQ vs vector_quantize_pytorch max err: {fsq_err:.2e}")
        assert fsq_err < 1e-5
    except ImportError:
        print("[port] vector_quantize_pytorch not installed — FSQ cross-check skipped")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "meta": {
                "joint_order": mj_order,
                "action_std": std,
                "num_tokens": 2,
                "token_dim": 32,
                "fsq_levels": 32,
                "num_future_frames": NUM_FUTURE_FRAMES,
                "future_dt": 0.1,
                "history_length": HIST,
                "proprio_dim": 930,
                "tokenizer_dim": 640,
                "action_dim": 29,
                "source": str(args.ckpt),
            },
        },
        out_path,
    )
    print(f"[port] saved: {out_path}")


if __name__ == "__main__":
    main()

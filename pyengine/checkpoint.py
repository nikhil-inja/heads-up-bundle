"""Checkpoint directories shared by the smoke trainer and the evaluator.

A directory holds manifest.json plus one flat float32 weight file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from game import Action, Observation, default_action
from pyengine.encoder_ref import ENCODER_VERSION, encode_observation, legal_from_observation, masked_argmax
from pyengine.mlp import flatten, forward, masked_probs, unflatten


def _net_arch(params: dict[str, np.ndarray], order: list[str]) -> dict:
    hidden = int(params["W0"].shape[1])
    return {
        "in": int(params["W0"].shape[0]),
        "hidden": hidden,
        "layers": sum(1 for key in order if key.startswith("W") and key[1:].isdigit()),
        "out": int(params["Wpi"].shape[1]),
        "value": True,
    }


def save_checkpoint(directory: Path | str, params: dict[str, np.ndarray], iteration: int, notes: str = "",
                    algo: str = "ppo", extras: dict[str, dict[str, np.ndarray]] | None = None,
                    manifest_extra: dict | None = None) -> Path:
    """Write the primary net as weights.f32. `extras` are additional named nets.

    A name in `extras` is stored as `{name}.f32` with its own param order. The
    primary file stays first so `load_params` and `CheckpointPolicy` still read
    the deployed net (PPO actor, ESCHER strategy net, SD-CFR button net).
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    flat, order, shapes = flatten(params)
    weight_name = "weights.f32"
    flat.tofile(path / weight_name)
    files = [weight_name]
    extra_meta: dict[str, dict] = {}
    for name, extra_params in (extras or {}).items():
        fname = name if str(name).endswith(".f32") else f"{name}.f32"
        extra_flat, extra_order, extra_shapes = flatten(extra_params)
        extra_flat.tofile(path / fname)
        files.append(fname)
        extra_meta[fname] = {"param_order": extra_order, "shapes": extra_shapes}
    manifest = {
        "algo": algo,
        "iteration": int(iteration),
        "wall_time": time.time(),
        "encoder_version": ENCODER_VERSION,
        "net_arch": _net_arch(params, order),
        "files": files,
        "notes": notes,
        "param_order": order,
        "shapes": shapes,
    }
    if extra_meta:
        manifest["extras"] = extra_meta
    if manifest_extra:
        manifest.update(manifest_extra)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(directory: Path | str) -> dict:
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("encoder_version") != ENCODER_VERSION:
        raise ValueError(f"checkpoint encoder {manifest.get('encoder_version')!r} != {ENCODER_VERSION}")
    return manifest


def load_params(directory: Path | str) -> dict[str, np.ndarray]:
    path = Path(directory)
    manifest = load_manifest(path)
    weight_name = manifest["files"][0]
    flat = np.fromfile(path / weight_name, dtype=np.float32)
    return unflatten(flat, manifest["param_order"], manifest["shapes"])


def load_named(directory: Path | str, filename: str) -> dict[str, np.ndarray]:
    """Load one weight file from a checkpoint directory. The primary file uses the top-level order."""
    path = Path(directory)
    manifest = load_manifest(path)
    if filename == manifest["files"][0]:
        flat = np.fromfile(path / filename, dtype=np.float32)
        return unflatten(flat, manifest["param_order"], manifest["shapes"])
    meta = manifest["extras"][filename]
    flat = np.fromfile(path / filename, dtype=np.float32)
    return unflatten(flat, meta["param_order"], meta["shapes"])


class CheckpointPolicy:
    """Greedy masked action from a saved MLP. Always returns a legal Action."""

    def __init__(self, directory: Path | str | None = None, params: dict[str, np.ndarray] | None = None) -> None:
        if params is not None:
            self.params = params
        elif directory is not None:
            self.params = load_params(directory)
        else:
            raise ValueError("pass a checkpoint directory or a param dict")

    def __call__(self, obs: Observation) -> Action:
        legal = legal_from_observation(obs)
        if float(legal.sum()) <= 0.0:
            return default_action(obs.to_call)
        logits, _value, _cache = forward(self.params, encode_observation(obs))
        action = Action(masked_argmax(logits[0], legal))
        if action not in obs.legal_actions:
            return default_action(obs.to_call)
        return action

    def sample(self, feat: np.ndarray, legal: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch. Returns actions, log-probs, and values."""
        logits, value, _cache = forward(self.params, feat)
        probs = masked_probs(logits, legal)
        batch = probs.shape[0]
        actions = np.empty(batch, dtype=np.int64)
        logp = np.empty(batch, dtype=np.float32)
        for i in range(batch):
            choices = np.flatnonzero(legal[i] > 0.0)
            if choices.size == 0:
                actions[i] = 1
                logp[i] = 0.0
                continue
            weights = probs[i, choices]
            weights = weights / weights.sum()
            pick = int(rng.choice(choices.size, p=weights))
            actions[i] = choices[pick]
            logp[i] = np.log(max(float(probs[i, actions[i]]), 1e-12))
        return actions, logp, value.astype(np.float32)

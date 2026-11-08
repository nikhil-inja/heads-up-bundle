"""Decision wrapper for the submitted bot.

Loads the PPO actor at import and samples a legal action. Any failure becomes
the judge's own default (check if nothing is owed, else fold).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from game import ACTION_TIMEOUT_SECONDS, Action, AgentConfig, HandResult, Observation, default_action
from pyengine.checkpoint import CheckpointPolicy
from pyengine.encoder_ref import encode_observation, legal_from_observation

_MODEL = Path(__file__).resolve().parents[1] / "model"
_POLICY: CheckpointPolicy | None = None
_RNG = np.random.default_rng()


def _load() -> CheckpointPolicy | None:
    try:
        if not (_MODEL / "manifest.json").exists():
            return None
        return CheckpointPolicy(_MODEL)
    except Exception:
        return None


_POLICY = _load()


def inner_action(obs: Observation, memory: dict) -> Action:
    """Sample the shipped PPO actor. Check or fold if it is not loaded."""
    if _POLICY is None:
        return default_action(obs.to_call)
    legal = legal_from_observation(obs)
    if float(legal.sum()) <= 0.0:
        return default_action(obs.to_call)
    feat = encode_observation(obs).reshape(1, -1)
    actions, _logp, _value = _POLICY.sample(feat, legal.reshape(1, -1), _RNG)
    return Action(int(actions[0]))


def safe_choose_action(obs: Observation, memory: dict, time_limit: float = ACTION_TIMEOUT_SECONDS) -> Action:
    """Return a legal action. Failures use `default_action`."""
    fallback = default_action(obs.to_call)
    try:
        started = time.perf_counter()
        action = inner_action(obs, memory)
        if time.perf_counter() - started > time_limit:
            return fallback
        if not isinstance(action, Action) or action not in obs.legal_actions:
            return fallback
        return action
    except Exception:
        return fallback


def prepare_half(config: AgentConfig, memory: dict) -> None:
    """Reset the per-half counters. `config` is accepted so callers can record it later."""
    memory["hands_played"] = 0


def on_hand_end(result: HandResult, memory: dict) -> None:
    memory["hands_played"] = memory.get("hands_played", 0) + 1

"""Small MLP used by the PPO smoke trainer.

Architecture: `layers` hidden layers of width `hidden`, ReLU, then a 4-way
policy head and a scalar value head. Weights are a dict of float arrays so a
checkpoint is just those arrays concatenated.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

PARAM_SUFFIXES = ("Wpi", "bpi", "Wv", "bv")


def hidden_layers(params: dict[str, np.ndarray]) -> int:
    count = 0
    while f"W{count}" in params:
        count += 1
    return count


def param_order(params: dict[str, np.ndarray]) -> list[str]:
    order = []
    for i in range(hidden_layers(params)):
        order.extend((f"W{i}", f"b{i}"))
    order.extend(PARAM_SUFFIXES)
    return order


def init_params(rng: np.random.Generator, in_dim: int, hidden: int = 256, layers: int = 3, out: int = 4,
                dtype=np.float32) -> dict[str, np.ndarray]:
    """He-initialize a 3x256 network by default."""
    params: dict[str, np.ndarray] = {}
    fan = in_dim
    for i in range(layers):
        scale = math.sqrt(2.0 / fan)
        params[f"W{i}"] = rng.normal(0.0, scale, size=(fan, hidden)).astype(dtype)
        params[f"b{i}"] = np.zeros(hidden, dtype=dtype)
        fan = hidden
    scale = math.sqrt(2.0 / hidden)
    params["Wpi"] = rng.normal(0.0, scale, size=(hidden, out)).astype(dtype)
    params["bpi"] = np.zeros(out, dtype=dtype)
    params["Wv"] = rng.normal(0.0, scale, size=(hidden, 1)).astype(dtype)
    params["bv"] = np.zeros(1, dtype=dtype)
    return params


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return logits (batch, 4), values (batch,), and (input, preactivation) per hidden layer."""
    h = np.asarray(x, dtype=params["W0"].dtype)
    if h.ndim == 1:
        h = h.reshape(1, -1)
    cache: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(hidden_layers(params)):
        pre = h @ params[f"W{i}"] + params[f"b{i}"]
        cache.append((h, pre))
        h = np.maximum(pre, 0)
    logits = h @ params["Wpi"] + params["bpi"]
    value = (h @ params["Wv"] + params["bv"]).reshape(-1)
    return logits, value, cache


def masked_probs(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Softmax over legal actions. `legal` is 0/1 with shape (batch, 4)."""
    masked = np.where(legal > 0.0, logits, -1e9)
    shifted = masked - masked.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    exp = np.where(legal > 0.0, exp, 0.0)
    denom = np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    return exp / denom


def ppo_loss_and_grads(params: dict[str, np.ndarray], x: np.ndarray, actions: np.ndarray, legal: np.ndarray,
                       old_logp: np.ndarray, adv: np.ndarray, ret: np.ndarray, clip_eps: float = 0.2,
                       vf_coef: float = 0.5, ent_coef: float = 0.01, kl_coef: float = 0.0,
                       magnet_probs: np.ndarray | None = None) -> tuple[float, dict[str, np.ndarray]]:
    """Mean clipped-PPO loss and gradients. `old_logp`, `adv`, and `ret` are constants."""
    logits, value, cache = forward(params, x)
    probs = masked_probs(logits, legal)
    batch = probs.shape[0]
    idx = np.arange(batch)
    new_logp = np.log(np.maximum(probs[idx, actions], 1e-12))
    log_ratio = np.clip(new_logp - old_logp, -20.0, 20.0)
    ratio = np.exp(log_ratio)
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    unclipped_obj = ratio * adv
    clipped_obj = clipped * adv
    take_unclipped = unclipped_obj <= clipped_obj
    policy_loss = -np.minimum(unclipped_obj, clipped_obj)

    safe_log = np.log(np.maximum(probs, 1e-30))
    safe_log = np.where(probs > 0.0, safe_log, 0.0)
    entropy = -(probs * safe_log).sum(axis=1)
    value_sq = (value - ret) ** 2
    loss = float(policy_loss.mean() + vf_coef * 0.5 * value_sq.mean() - ent_coef * entropy.mean())

    dlogp = np.zeros(batch, dtype=probs.dtype)
    dlogp[take_unclipped] = -adv[take_unclipped] * ratio[take_unclipped] / batch
    dlogits = probs * (-dlogp[:, None])
    dlogits[idx, actions] += dlogp

    # dH/dz_j = p_j * (-log p_j - H). Illegal actions are constant under the mask.
    d_entropy = probs * (-safe_log - entropy[:, None])
    dlogits = dlogits - (ent_coef / batch) * d_entropy
    dlogits = np.where(legal > 0.0, dlogits, 0.0)

    if kl_coef and magnet_probs is not None:
        log_rho = np.log(np.maximum(magnet_probs, 1e-30))
        log_rho = np.where(magnet_probs > 0.0, log_rho, 0.0)
        kl_vec = (probs * (safe_log - log_rho)).sum(axis=1)
        loss += kl_coef * float(kl_vec.mean())
        d_kl = probs * (safe_log - log_rho - kl_vec[:, None])
        dlogits = dlogits + (kl_coef / batch) * np.where(legal > 0.0, d_kl, 0.0)

    h_last = np.maximum(cache[-1][1], 0)
    grads: dict[str, np.ndarray] = {}
    grads["Wpi"] = h_last.T @ dlogits
    grads["bpi"] = dlogits.sum(axis=0)
    dh = dlogits @ params["Wpi"].T

    dv = vf_coef * (value - ret) / batch
    grads["Wv"] = h_last.T @ dv[:, None]
    grads["bv"] = dv.sum(keepdims=True)
    dh = dh + dv[:, None] * params["Wv"].reshape(1, -1)

    for i in reversed(range(hidden_layers(params))):
        h_in, pre = cache[i]
        dpre = dh * (pre > 0)
        grads[f"W{i}"] = h_in.T @ dpre
        grads[f"b{i}"] = dpre.sum(axis=0)
        dh = dpre @ params[f"W{i}"].T
    for key, grad in grads.items():
        grads[key] = np.asarray(grad, dtype=params[key].dtype)
    return loss, grads


class Adam:
    """Adam over a param dict. Updates arrays in place."""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 3e-4) -> None:
        self.lr = lr
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray], max_norm: float = 1.0) -> None:
        self.t += 1
        total = math.sqrt(sum(float(np.sum(grads[k] * grads[k])) for k in grads))
        scale = 1.0 if total <= max_norm or total == 0.0 else max_norm / total
        b1, b2, eps = 0.9, 0.999, 1e-8
        for key, param in params.items():
            if key not in grads:
                continue
            g = grads[key] * scale
            self.m[key] = b1 * self.m[key] + (1.0 - b1) * g
            self.v[key] = b2 * self.v[key] + (1.0 - b2) * (g * g)
            mhat = self.m[key] / (1.0 - b1 ** self.t)
            vhat = self.v[key] / (1.0 - b2 ** self.t)
            param -= self.lr * mhat / (np.sqrt(vhat) + eps)


def copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in params.items()}


def ema_update(target: dict[str, np.ndarray], source: dict[str, np.ndarray], decay: float) -> None:
    """In-place EMA: target = decay * target + (1 - decay) * source."""
    for key, value in source.items():
        target[key] *= decay
        target[key] += (1.0 - decay) * value


def _backward(params: dict[str, np.ndarray], cache: list[tuple[np.ndarray, np.ndarray]],
              dlogits: np.ndarray | None, dvalue: np.ndarray | None) -> dict[str, np.ndarray]:
    """Backprop from the policy head, the value head, or both. Missing heads get zero grads."""
    h_last = np.maximum(cache[-1][1], 0)
    grads: dict[str, np.ndarray] = {}
    dh = np.zeros_like(h_last)
    if dlogits is not None:
        grads["Wpi"] = h_last.T @ dlogits
        grads["bpi"] = dlogits.sum(axis=0)
        dh = dh + dlogits @ params["Wpi"].T
    else:
        grads["Wpi"] = np.zeros_like(params["Wpi"])
        grads["bpi"] = np.zeros_like(params["bpi"])
    if dvalue is not None:
        grads["Wv"] = h_last.T @ dvalue[:, None]
        grads["bv"] = np.asarray(dvalue.sum(keepdims=True))
        dh = dh + dvalue[:, None] * params["Wv"].reshape(1, -1)
    else:
        grads["Wv"] = np.zeros_like(params["Wv"])
        grads["bv"] = np.zeros_like(params["bv"])
    for i in reversed(range(hidden_layers(params))):
        h_in, pre = cache[i]
        dpre = dh * (pre > 0)
        grads[f"W{i}"] = h_in.T @ dpre
        grads[f"b{i}"] = dpre.sum(axis=0)
        dh = dpre @ params[f"W{i}"].T
    for key, grad in grads.items():
        grads[key] = np.asarray(grad, dtype=params[key].dtype)
    return grads


def regret_loss_and_grads(params: dict[str, np.ndarray], x: np.ndarray, target: np.ndarray, legal: np.ndarray,
                          weights: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    """Weighted MSE of the policy head against regret targets. Illegal actions are masked out."""
    logits, _value, cache = forward(params, x)
    w = np.asarray(weights, dtype=logits.dtype).reshape(-1)
    err = (logits - target) * legal
    loss = float((w * (err * err).sum(axis=1)).mean())
    dlogits = (2.0 / logits.shape[0]) * w[:, None] * err
    return loss, _backward(params, cache, dlogits, None)


def strategy_loss_and_grads(params: dict[str, np.ndarray], x: np.ndarray, target: np.ndarray, legal: np.ndarray,
                            weights: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    """Weighted cross-entropy of a masked softmax against a target distribution."""
    logits, _value, cache = forward(params, x)
    probs = masked_probs(logits, legal)
    w = np.asarray(weights, dtype=probs.dtype).reshape(-1)
    safe_log = np.log(np.maximum(probs, 1e-30))
    safe_log = np.where(probs > 0.0, safe_log, 0.0)
    per = -(target * safe_log).sum(axis=1)
    loss = float((w * per).mean())
    dlogits = (w[:, None] / probs.shape[0]) * (probs - target)
    dlogits = np.where(legal > 0.0, dlogits, 0.0)
    return loss, _backward(params, cache, dlogits, None)


def value_loss_and_grads(params: dict[str, np.ndarray], x: np.ndarray, target: np.ndarray,
                         weights: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    """Weighted MSE of the scalar value head."""
    _logits, value, cache = forward(params, x)
    w = np.asarray(weights, dtype=value.dtype).reshape(-1)
    diff = value - np.asarray(target, dtype=value.dtype).reshape(-1)
    loss = float((w * diff * diff).mean())
    dvalue = (2.0 / value.shape[0]) * w * diff
    return loss, _backward(params, cache, None, dvalue)


def flatten(params: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    order = param_order(params)
    shapes = {k: list(params[k].shape) for k in order}
    flat = np.concatenate([params[k].astype(np.float32).ravel() for k in order])
    return flat, order, shapes


def unflatten(flat: np.ndarray, order: Iterable[str], shapes: dict[str, list[int]]) -> dict[str, np.ndarray]:
    params = {}
    offset = 0
    data = np.asarray(flat, dtype=np.float32)
    for key in order:
        shape = tuple(shapes[key])
        size = int(np.prod(shape))
        params[key] = data[offset:offset + size].reshape(shape).copy()
        offset += size
    if offset != data.size:
        raise ValueError(f"weight file has {data.size} floats, used {offset}")
    return params

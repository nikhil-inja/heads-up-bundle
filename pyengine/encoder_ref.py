"""Canonical state encoder (v1).

The submitted bot and every trainer call this module. Feature order is fixed
for ENCODER_VERSION; checkpoints store that string and refuse a mismatch.

Layout (182 floats):
    street one-hot (4),
    hole-card multi-hot (36), hole rank counts (9), hole suit counts (4),
    board multi-hot (36),
    scalars / starting stack: pot, to_call, my_bet, opp_bet, my_stack,
        opp_stack, min_raise_to, then raises_this_street/4 and is_button,
    legal-action mask (4),
    last 16 actions, oldest first, each (is_me, action one-hot 4).

Opponent hole cards are not an input. This is the runtime encoder.

`encode_critic` appends the opponent's hole cards for the training-only
value networks. The submitted bot never calls it.
"""

from __future__ import annotations

import numpy as np

from game import NUM_CARDS, STARTING_STACK, Observation

ENCODER_VERSION = "v1"

STREET_DIM = 4
HOLE_DIM = NUM_CARDS
RANK_DIM = 9
SUIT_DIM = 4
BOARD_DIM = NUM_CARDS
SCALAR_DIM = 9
LEGAL_DIM = 4
HISTORY_LEN = 16
HISTORY_SLOT = 5  # is_me + action one-hot
HISTORY_DIM = HISTORY_LEN * HISTORY_SLOT

OFF_STREET = 0
OFF_HOLE = OFF_STREET + STREET_DIM
OFF_RANK = OFF_HOLE + HOLE_DIM
OFF_SUIT = OFF_RANK + RANK_DIM
OFF_BOARD = OFF_SUIT + SUIT_DIM
OFF_SCALAR = OFF_BOARD + BOARD_DIM
OFF_LEGAL = OFF_SCALAR + SCALAR_DIM
OFF_HISTORY = OFF_LEGAL + LEGAL_DIM
FEATURE_DIM = OFF_HISTORY + HISTORY_DIM
CRITIC_EXTRA = NUM_CARDS
CRITIC_DIM = FEATURE_DIM + CRITIC_EXTRA

SCALAR_NAMES = (
    "pot",
    "to_call",
    "my_bet",
    "opp_bet",
    "my_stack",
    "opp_stack",
    "min_raise_to",
    "raises_frac",
    "is_button",
)

_STACK = float(STARTING_STACK)


def encode_observation(obs: Observation) -> np.ndarray:
    """Encode one Observation into a float32 vector of FEATURE_DIM."""
    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    feat[OFF_STREET + int(obs.street)] = 1.0

    ranks = feat[OFF_RANK:OFF_SUIT]
    suits = feat[OFF_SUIT:OFF_BOARD]
    for card in obs.hole_cards:
        feat[OFF_HOLE + int(card)] = 1.0
        ranks[int(card) >> 2] += 1.0
        suits[int(card) & 3] += 1.0
    for card in obs.board:
        feat[OFF_BOARD + int(card)] = 1.0

    scalars = feat[OFF_SCALAR:OFF_LEGAL]
    scalars[0] = obs.pot / _STACK
    scalars[1] = obs.to_call / _STACK
    scalars[2] = obs.my_bet / _STACK
    scalars[3] = obs.opp_bet / _STACK
    scalars[4] = obs.my_stack / _STACK
    scalars[5] = obs.opp_stack / _STACK
    scalars[6] = obs.min_raise_to / _STACK
    scalars[7] = obs.raises_this_street / 4.0
    scalars[8] = 1.0 if obs.is_button else 0.0

    legal = feat[OFF_LEGAL:OFF_HISTORY]
    for action in obs.legal_actions:
        legal[int(action)] = 1.0

    history = obs.action_history[-HISTORY_LEN:]
    base = OFF_HISTORY + (HISTORY_LEN - len(history)) * HISTORY_SLOT
    for i, record in enumerate(history):
        slot = base + i * HISTORY_SLOT
        feat[slot] = 1.0 if record.is_me else 0.0
        feat[slot + 1 + int(record.action)] = 1.0
    return feat


def encode_critic(obs: Observation, opp_hole) -> np.ndarray:
    """Runtime features plus the opponent's hole cards. Training only."""
    feat = np.zeros(CRITIC_DIM, dtype=np.float32)
    feat[:FEATURE_DIM] = encode_observation(obs)
    for card in opp_hole:
        feat[FEATURE_DIM + int(card)] = 1.0
    return feat


def legal_from_observation(obs: Observation) -> np.ndarray:
    """Float mask of length 4, 1 where the action is legal."""
    mask = np.zeros(LEGAL_DIM, dtype=np.float32)
    for action in obs.legal_actions:
        mask[int(action)] = 1.0
    return mask


def masked_argmax(logits: np.ndarray, legal: np.ndarray) -> int:
    """Index of the largest logit among legal actions. `legal` is 0/1."""
    masked = np.where(legal > 0.0, logits, -1e9)
    return int(np.argmax(masked))


__all__ = [
    "ENCODER_VERSION",
    "CRITIC_DIM",
    "FEATURE_DIM",
    "LEGAL_DIM",
    "OFF_LEGAL",
    "SCALAR_NAMES",
    "encode_critic",
    "encode_observation",
    "legal_from_observation",
    "masked_argmax",
]

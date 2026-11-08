"""The decision wrapper always returns a legal action."""

from __future__ import annotations

import runtime.policy as policy
from game import Action, Game, default_action, make_deal


def _flop_observation():
    game = Game(make_deal("flop", 0), button_seat=0)
    game.apply(Action.CHECK_CALL)
    game.apply(Action.CHECK_CALL)
    assert game.street.name == "FLOP"
    obs = game.observation(game.to_act)
    assert Action.FOLD not in obs.legal_actions
    return obs


def test_illegal_action_falls_back():
    obs = _flop_observation()
    original = policy.inner_action
    policy.inner_action = lambda obs, memory: Action.FOLD
    try:
        action = policy.safe_choose_action(obs, {})
    finally:
        policy.inner_action = original
    assert action == default_action(obs.to_call)
    assert action in obs.legal_actions


def test_slow_call_falls_back():
    obs = Game(make_deal("time", 0), 0).observation(0)
    original = policy.inner_action
    policy.inner_action = lambda obs, memory: Action.RAISE_POT
    try:
        action = policy.safe_choose_action(obs, {}, time_limit=-1.0)
    finally:
        policy.inner_action = original
    assert action == default_action(obs.to_call)


def test_shipped_policy_is_legal_preflop():
    obs = Game(make_deal("pre", 1), 1).observation(1)
    action = policy.safe_choose_action(obs, {})
    assert action in obs.legal_actions

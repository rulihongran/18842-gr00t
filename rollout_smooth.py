#!/usr/bin/env python
"""`lerobot-rollout` with smoother GR00T actions: zero initial noise + a Kalman filter on the joint targets.

Usage: the same options as `lerobot-rollout`, plus (all optional):

    --zero_noise=true        start GR00T's flow-matching sampler from zeros instead of random noise
                             (same image/state -> same action chunk; less jitter, smaller chunk jumps)
    --filter=kalman          kalman | ema | none   filter applied to the arm joint targets
    --kalman_q=0.01          Kalman process noise (bigger = follows faster, smooths less)
    --kalman_r=9             Kalman measurement noise (bigger = smooths more, more lag)
    --ema_alpha=0.3          only for --filter=ema (smaller = smoother, more lag)
    --gripper_filter=none    none | same   filter the gripper too (default: no, keep grasps fast)

Nothing in the installed `lerobot` package is modified; the changes are applied in memory:
  * `ThreadSafeRobot.send_action` filters `<joint>.pos` targets before they reach the robot
    (the robot's own `max_relative_target` clamp still applies afterwards);
  * GR00T N1.7's action head draws its initial noise as zeros during inference;
  * the return-to-start motion at the end bypasses the filter.
"""

import argparse
import logging
import os
import sys
import threading
import types

logger = logging.getLogger("rollout_smooth")


def _bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if v.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def parse_own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--zero_noise", type=_bool, default=True)
    p.add_argument("--filter", choices=["kalman", "ema", "none"], default="kalman")
    p.add_argument("--kalman_q", type=float, default=0.01)
    p.add_argument("--kalman_r", type=float, default=9.0)
    p.add_argument("--ema_alpha", type=float, default=0.3)
    p.add_argument("--gripper_filter", choices=["none", "same"], default="none")
    own, rest = p.parse_known_args(sys.argv[1:])
    if own.kalman_q <= 0 or own.kalman_r <= 0:
        p.error("--kalman_q and --kalman_r must be > 0")
    if not 0 < own.ema_alpha <= 1:
        p.error("--ema_alpha must be in (0, 1]")
    sys.argv = [sys.argv[0]] + rest
    return own


# --------------------------------------------------------------------------- filters


class KalmanCV:
    """1-D constant-velocity Kalman filter, one step per control tick (white-acceleration model)."""

    def __init__(self, q: float, r: float):
        self.q, self.r = q, r
        self.x = None  # [position, velocity per tick]

    def reset(self):
        self.x = None

    def __call__(self, z: float) -> float:
        if self.x is None:
            self.x = [z, 0.0]
            self.p = [[self.r, 0.0], [0.0, self.r]]
            return z
        q, r = self.q, self.r
        (x0, x1), ((p00, p01), (p10, p11)) = self.x, self.p
        # predict: x = F x, P = F P F^T + Q, F = [[1,1],[0,1]], Q = q*[[1/4,1/2],[1/2,1]]
        x0 = x0 + x1
        p00, p01, p10, p11 = (
            p00 + p01 + p10 + p11 + q / 4,
            p01 + p11 + q / 2,
            p10 + p11 + q / 2,
            p11 + q,
        )
        # update with measurement z of the position
        s = p00 + r
        k0, k1 = p00 / s, p10 / s
        y = z - x0
        x0, x1 = x0 + k0 * y, x1 + k1 * y
        p00, p01, p10, p11 = (1 - k0) * p00, (1 - k0) * p01, p10 - k1 * p00, p11 - k1 * p01
        self.x, self.p = [x0, x1], [[p00, p01], [p10, p11]]
        return x0


class EMA:
    def __init__(self, alpha: float):
        self.alpha, self.y = alpha, None

    def reset(self):
        self.y = None

    def __call__(self, z: float) -> float:
        self.y = z if self.y is None else self.alpha * z + (1 - self.alpha) * self.y
        return self.y


class ActionFilter:
    """Filters the `<joint>.pos` entries of an action dict, one filter per joint."""

    def __init__(self, own):
        self.own = own
        self.filters = {}
        self.lock = threading.Lock()
        self.bypass = False
        self.n = 0
        self.sum_abs_change = {}

    def _make(self):
        if self.own.filter == "kalman":
            return KalmanCV(self.own.kalman_q, self.own.kalman_r)
        return EMA(self.own.ema_alpha)

    def _wanted(self, key: str) -> bool:
        if not key.endswith(".pos"):
            return False
        if key.startswith("gripper") and self.own.gripper_filter == "none":
            return False
        return True

    def reset(self):
        with self.lock:
            for f in self.filters.values():
                f.reset()

    def __call__(self, action):
        if self.own.filter == "none" or self.bypass or not isinstance(action, dict):
            return action
        out = dict(action)
        with self.lock:
            for k, v in action.items():
                if not self._wanted(k):
                    continue
                try:
                    z = float(v)
                except (TypeError, ValueError):
                    continue
                f = self.filters.get(k)
                if f is None:
                    f = self.filters[k] = self._make()
                y = f(z)
                out[k] = y
                self.sum_abs_change[k] = self.sum_abs_change.get(k, 0.0) + abs(y - z)
            self.n += 1
        return out

    def summary(self) -> str:
        if not self.n:
            return "no actions filtered"
        parts = [f"{k.removesuffix('.pos')} {v / self.n:.2f}" for k, v in self.sum_abs_change.items()]
        return f"{self.n} actions filtered; mean |filtered - raw| per joint: " + ", ".join(parts)


# --------------------------------------------------------------------------- patches


def patch_send_action(action_filter: ActionFilter):
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    original = ThreadSafeRobot.send_action

    def send_action(self, action):
        return original(self, action_filter(action))

    ThreadSafeRobot.send_action = send_action


def patch_return_to_initial(action_filter: ActionFilter):
    """The end-of-run return-to-start already interpolates smoothly (at 50 Hz): don't filter it."""
    from lerobot.rollout.strategies.core import RolloutStrategy

    original = RolloutStrategy._return_to_initial_position  # staticmethod -> plain function here

    def _return_to_initial_position(*args, **kwargs):
        action_filter.bypass = True
        try:
            return original(*args, **kwargs)
        finally:
            action_filter.bypass = False
            action_filter.reset()

    RolloutStrategy._return_to_initial_position = staticmethod(_return_to_initial_position)


def patch_zero_noise():
    """During GR00T N1.7 action sampling, `torch.randn` (initial flow-matching noise) returns zeros."""
    import torch

    from lerobot.policies.groot import groot_n1_7

    def zeros_like_randn(*size, **kwargs):
        kwargs.pop("generator", None)
        if "size" in kwargs:
            size = (kwargs.pop("size"),)
        return torch.zeros(*size, **kwargs)

    zero_torch = types.ModuleType("torch_zero_noise")
    zero_torch.__dict__.update({k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
    zero_torch.randn = zeros_like_randn

    module_globals = vars(groot_n1_7)
    patched = 0
    for obj in list(module_globals.values()):
        if isinstance(obj, type) and "get_action_with_features" in vars(obj):
            original = obj.get_action_with_features

            def get_action_with_features(self, *args, _orig=original, **kwargs):
                # Only code in groot_n1_7 sees the zero-noise torch, and only during this call
                # (it runs in the single RTC inference thread).
                g = module_globals
                saved = g["torch"]
                g["torch"] = zero_torch
                try:
                    return _orig(self, *args, **kwargs)
                finally:
                    g["torch"] = saved

            obj.get_action_with_features = get_action_with_features
            patched += 1
    if patched == 0:
        raise RuntimeError("zero_noise: GR00T N1.7 action head not found (LeRobot version changed?)")
    return patched


# --------------------------------------------------------------------------- main


def main():
    own = parse_own_args()
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("HF_TOKEN"):
        logger.warning(
            "HF_TOKEN is set. If loading GR00T fails with 403 / gated repo Cosmos-Reason2-2B, "
            "run `unset HF_TOKEN` first."
        )

    action_filter = ActionFilter(own)
    patch_send_action(action_filter)
    patch_return_to_initial(action_filter)
    if own.zero_noise:
        patch_zero_noise()

    if own.filter == "kalman":
        desc = f"kalman q={own.kalman_q} r={own.kalman_r}"
    elif own.filter == "ema":
        desc = f"ema alpha={own.ema_alpha}"
    else:
        desc = "none"
    logger.info(
        "rollout_smooth: zero_noise=%s, filter=%s, gripper_filter=%s", own.zero_noise, desc, own.gripper_filter
    )

    from lerobot.scripts.lerobot_rollout import main as rollout_main

    try:
        rollout_main()
    finally:
        logger.info("rollout_smooth: %s", action_filter.summary())


if __name__ == "__main__":
    main()

"""Vectorized Flappy world for genetic-algorithm evaluation.

All N birds fly through ONE shared pipe sequence (same seed → identical
gapYs, spawn times, positions). Bird physics are exact copies of
autoflappy.env (fixed-point ints, 120 Hz), vectorized with numpy so a whole
population evaluates in a few big array ops per tick.
"""

import numpy as np

from ..env import (
    SCALE, HZ, GRAVITY_T, FLAP_V, FALL_CAP, GROUND_BH, BIRD_R, SPEED_T,
    SPAWN_X, KILL_X, SPACING, GAP_MIN, GAP_MAX, HALF_GAP, OPENING_GAP,
    PIPE_HALF_W, COLL_R2, FIRST_PIPE_STEP, N_FEATURES, _NEG10, _POS10,
)


def _trunc_div(x: np.ndarray, d: int) -> np.ndarray:
    """Vectorized JS Math.trunc(x / d) for integer arrays (trunc toward 0)."""
    return np.where(x < 0, -((-x) // d), x // d)


class VectorWorld:
    """n birds + one pipe world. step() advances `frame_skip` sub-ticks."""

    def __init__(self, n: int, seed: int, frame_skip: int = 4):
        self.n = n
        self.frame_skip = frame_skip
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.tick = 0
        self.y = np.zeros(self.n, dtype=np.int64)          # fxp
        self.vy = np.full(self.n, FLAP_V, dtype=np.int64)  # post-first-flap
        self.alive = np.ones(self.n, dtype=bool)
        self.score = np.zeros(self.n, dtype=np.int64)
        self.death_tick = np.full(self.n, -1, dtype=np.int64)
        self.death_cause = np.array([None] * self.n, dtype=object)
        self.pipes: list[list] = []   # [x, gap_y, half_gap, passed]
        self.spawn_count = 0
        self._zeros = np.zeros(self.n, dtype=bool)
        return self

    # ------------------------------------------------------------------ #
    def _kill(self, mask: np.ndarray, cause: str):
        mask &= self.alive
        if mask.any():
            self.alive[mask] = False
            self.death_tick[mask] = self.tick
            self.death_cause[mask] = cause

    def _next_pipe(self):
        for p in self.pipes:
            if p[0] + PIPE_HALF_W + BIRD_R >= 0:
                return p
        return None

    def _tick(self, flap: np.ndarray):
        a = self.alive

        # --- bird physics (alive only) ---
        self.vy[a] = np.where(flap[a], FLAP_V, self.vy[a])
        vy = self.vy[a]
        vy = np.where(vy < -FALL_CAP, -FALL_CAP, vy) - GRAVITY_T
        self.vy[a] = vy
        self.y[a] += _trunc_div(vy, HZ)

        self._kill(self.y - BIRD_R < GROUND_BH, "ground")
        self.y[~self.alive & (self.y - BIRD_R < GROUND_BH)] = GROUND_BH + BIRD_R

        # --- pipe collision ---
        for p in self.pipes:
            l = p[0] - PIPE_HALF_W
            u = p[0] + PIPE_HALF_W
            f = -max(l, min(0, u))
            if abs(f) >= BIRD_R:
                continue
            h = p[1] - p[2]
            w = p[1] + p[2]
            dyb = np.maximum(0, self.y - h)
            self._kill(f * f + dyb * dyb < COLL_R2, "pipeBottom")
            dyt = np.minimum(0, self.y - w)
            self._kill(f * f + dyt * dyt < COLL_R2, "pipeTop")

        # --- advance pipes (shared world) ---
        last = self.pipes[-1] if self.pipes else None
        if (last is not None and SPAWN_X - last[0] >= SPACING) or (
            last is None and self.tick >= FIRST_PIPE_STEP
        ):
            gap = int(GAP_MIN + self.rng.random() * (GAP_MAX - GAP_MIN))
            hg = OPENING_GAP[self.spawn_count] if self.spawn_count < len(OPENING_GAP) else HALF_GAP
            self.pipes.append([SPAWN_X, gap, hg, False])
            self.spawn_count += 1

        passes = 0
        for p in self.pipes:
            p[0] -= SPEED_T
            if not p[3] and p[0] <= 0:
                p[3] = True
                passes += 1
        if passes:
            self.score[self.alive] += passes

        if self.pipes and self.pipes[0][0] < KILL_X:
            self.pipes.pop(0)

        self.tick += 1

    # ------------------------------------------------------------------ #
    def step(self, actions: np.ndarray):
        """actions: (n,) int/bool — flap applies on first sub-tick only."""
        for i in range(self.frame_skip):
            if not self.alive.any():
                break
            self._tick(actions if i == 0 else self._zeros)

    def features(self) -> np.ndarray:
        """(n, 8) — identical layout to FlappyEnv.features()."""
        s = float(SCALE)
        y = self.y / s
        out = np.empty((self.n, N_FEATURES), dtype=np.float32)
        out[:, 0] = y
        out[:, 1] = self.vy / s / 2.2

        p1 = self._next_pipe()
        idx = self.pipes.index(p1) if p1 is not None else -1
        p2 = self.pipes[idx + 1] if 0 <= idx + 1 < len(self.pipes) else None

        if p1 is None:
            out[:, 2] = 1.5; out[:, 3] = 0.0; out[:, 4] = 0.0
        else:
            out[:, 2] = p1[0] / s / 1.2
            out[:, 3] = p1[1] / s - y
            out[:, 4] = p1[2] / s / 0.31
        if p2 is None:
            out[:, 5] = 1.5; out[:, 6] = 0.0; out[:, 7] = 0.0
        else:
            out[:, 5] = p2[0] / s / 1.2
            out[:, 6] = p2[1] / s - y
            out[:, 7] = p2[2] / s / 0.31
        return out

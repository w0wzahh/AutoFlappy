"""
Bit-exact replica of the flappybird.io simulation (simVersion 4).

All state is stored as fixed-point integers with scale 2**20, matching the
game bundle's `Ne()`/`$t()` conversion functions and its 120 Hz step order:

    per tick (1/120 s), while state == "play":
        vy = max(vy, -FALL_CAP);  vy -= GRAVITY_T;  y += trunc(vy / 120)
        if y - BIRD_R < GROUND_Y:  die ("ground")
        for each pipe: circle-vs-pipe collision check -> die ("pipeTop"/"pipeBottom")
        advance pipes (spawn / move / score / despawn)
"""

import math
import numpy as np

SCALE = 1 << 20
HZ = 120


def _ne(x: float) -> int:
    """JS Math.round(x * SCALE) — halves round toward +inf."""
    return math.floor(x * SCALE + 0.5)


def _trunc120(x: int) -> int:
    """JS Math.trunc(x / 120) for integer x (trunc toward zero)."""
    return -((-x) // HZ) if x < 0 else x // HZ


# ---- game constants (extracted from the site's bundle) ----
GRAVITY_T   = _ne(5.0) // HZ                # 43690  (Xv)
FLAP_V      = _ne(1.4)                      # flap velocity, sim v4
FALL_CAP    = _ne(2.2)                      # max fall speed, sim v4
GROUND_BH   = _ne(-0.975)                   # lowest height
BIRD_R      = _ne(0.068)                    # bird circle radius
SPEED_T     = _ne(0.6) // HZ                # 5242  (Si) pipe speed per tick
SPAWN_X     = _ne(1.2)
KILL_X      = _ne(-1.2)
SPACING     = _ne(1.0)
GAP_MIN     = _ne(-0.2)
GAP_MAX     = _ne(0.8)
HALF_GAP    = _ne(0.47) // 2                # wf  (246415)
OPENING_GAP = [_ne(g) // 2 for g in (0.62, 0.59, 0.56, 0.53, 0.50)]
PIPE_HALF_W = (26 * _ne(0.01)) // 2         # kf  (136318)
COLL_R2     = BIRD_R * BIRD_R               # Af
FIRST_PIPE_STEP = 180                       # $v
N_FEATURES = 8

_NEG10 = _ne(-10.0)
_POS10 = _ne(10.0)

# reward shaping
R_ALIVE = 0.01
R_PASS = 1.0
R_DEAD = -1.0
SHAPE_W = 0.4          # potential-shaping weight toward next gap center
IDLE_TARGET = _ne(0.25)  # where to hover before the first pipe spawns


class Pipe:
    __slots__ = ("x", "gap_y", "half_gap", "passed")

    def __init__(self, gap_y: int, half_gap: int):
        self.x = SPAWN_X
        self.gap_y = gap_y
        self.half_gap = half_gap
        self.passed = False


class FlappyEnv:
    """Single-player env. Action: 0 = idle, 1 = flap.

    frame_skip sub-ticks per step(): an action of 1 flaps on the first
    sub-tick only, then idles — matching a ~30 Hz decision rate.
    """

    def __init__(self, seed: int | None = None, frame_skip: int = 4):
        self.rng = np.random.default_rng(seed)
        self.frame_skip = frame_skip
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> np.ndarray:
        """State right after the first flap (beginRun + birdVy = FLAP_V)."""
        self.y = 0
        self.vy = FLAP_V
        self.pipes: list[Pipe] = []
        self.score = 0
        self.play_step = 0
        self.spawn_count = 0
        self.done = False
        self.death = None
        return self.features()

    def _spawn_gap_y(self) -> int:
        # uniform in [GAP_MIN, GAP_MAX)
        return int(GAP_MIN + (self.rng.random() * (GAP_MAX - GAP_MIN)))

    def _half_gap_for(self, idx: int) -> int:
        return OPENING_GAP[idx] if idx < len(OPENING_GAP) else HALF_GAP

    def _next_pipe(self):
        for p in self.pipes:
            if p.x + PIPE_HALF_W + BIRD_R >= 0:
                return p
        return None

    def _potential(self) -> float:
        """-|y - target| where target is the next gap center (or idle hover)."""
        p = self._next_pipe()
        target = p.gap_y if p is not None else IDLE_TARGET
        return -abs(self.y - target) / SCALE

    # ------------------------------------------------------------------ #
    def features(self) -> np.ndarray:
        s = SCALE
        y, vy = self.y, self.vy
        p1 = self._next_pipe()
        p2 = self.pipes[self.pipes.index(p1) + 1] if p1 is not None and self.pipes.index(p1) + 1 < len(self.pipes) else None
        f = np.empty(N_FEATURES, dtype=np.float32)
        f[0] = y / s
        f[1] = (vy / s) / 2.2
        if p1 is None:
            f[2] = 1.5; f[3] = 0.0; f[4] = 0.0
        else:
            f[2] = (p1.x / s) / 1.2
            f[3] = (p1.gap_y - y) / s
            f[4] = (p1.half_gap / s) / 0.31
        if p2 is None:
            f[5] = 1.5; f[6] = 0.0; f[7] = 0.0
        else:
            f[5] = (p2.x / s) / 1.2
            f[6] = (p2.gap_y - y) / s
            f[7] = (p2.half_gap / s) / 0.31
        return f

    # ------------------------------------------------------------------ #
    def _tick(self, flap: bool):
        """One physics sub-tick. Returns base reward, or None when dead."""
        if flap:
            self.vy = FLAP_V
        if self.vy < -FALL_CAP:
            self.vy = -FALL_CAP
        self.vy -= GRAVITY_T
        self.y += _trunc120(self.vy)

        if self.y - BIRD_R < GROUND_BH:
            self.y = GROUND_BH + BIRD_R
            self.vy = 0
            self.done = True
            self.death = "ground"
            return None

        for p in self.pipes:
            l = p.x - PIPE_HALF_W
            u = p.x + PIPE_HALF_W
            f = -max(l, min(0, u))
            if abs(f) < BIRD_R:
                h = p.gap_y - p.half_gap
                dy = self.y - max(_NEG10, min(self.y, h))
                if f * f + dy * dy < COLL_R2:
                    self.done = True
                    self.death = "pipeBottom"
                    return None
                w = p.gap_y + p.half_gap
                dy = self.y - max(w, min(self.y, _POS10))
                if f * f + dy * dy < COLL_R2:
                    self.done = True
                    self.death = "pipeTop"
                    return None

        # --- advance pipes ---
        last = self.pipes[-1] if self.pipes else None
        if (last is not None and SPAWN_X - last.x >= SPACING) or (
            last is None and self.play_step >= FIRST_PIPE_STEP
        ):
            self.pipes.append(Pipe(self._spawn_gap_y(), self._half_gap_for(self.spawn_count)))
            self.spawn_count += 1

        r = R_ALIVE
        for p in self.pipes:
            p.x -= SPEED_T
            if not p.passed and p.x <= 0:
                p.passed = True
                self.score += 1
                r += R_PASS

        if self.pipes and self.pipes[0].x < KILL_X:
            self.pipes.pop(0)

        self.play_step += 1
        return r

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self.features(), 0.0, True, {"score": self.score}

        phi_prev = self._potential()
        reward = 0.0
        for i in range(self.frame_skip):
            r = self._tick(action == 1 and i == 0)
            if r is None:
                reward += R_DEAD
                break
            reward += r

        reward += SHAPE_W * (self._potential() - phi_prev)
        return self.features(), reward, self.done, {
            "score": self.score, "death": self.death}

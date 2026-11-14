"""Pygame flock visualization for GA evolution — NEAT-video style.

Every bird of the current generation flies the same pipe layout at once.
Alive birds are colored dots (rainbow by index), the current leader gets a
white ring, dead birds leave a fading grey ghost.

Controls:  SPACE pause   UP/DOWN speed   S skip generation   ESC/Q quit
"""

import colorsys
import math

import numpy as np
import pygame

from ..env import SCALE, PIPE_HALF_W, GROUND_BH, BIRD_R

W, H = 960, 660
SKY = (78, 192, 202)
GROUND_C = (222, 216, 149)
PIPE_C = (115, 191, 46)
PIPE_EDGE = (74, 140, 26)
TXT = (255, 255, 255)

PX_PER_UNIT = 470.0          # horizontal scale
PY_PER_UNIT = 250.0          # vertical scale
BIRD_SX = 210                # bird screen x (world x = 0)
GROUND_SY = 596              # screen y of the ground line


def sx(xw: float) -> float:
    return BIRD_SX + xw * PX_PER_UNIT


def sy(yw: float) -> float:
    return GROUND_SY - (yw + 0.975) * PY_PER_UNIT


class View:
    def __init__(self, pop_size: int, title="AutoFlappy — GA evolution"):
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)
        self.speed = 8                    # sim steps per drawn frame
        self.speeds = [1, 2, 4, 8, 16, 32, 64, 128]
        self.paused = False
        self.skip_gen = False
        self.quit = False
        self.frame = 0
        n = pop_size
        hues = np.linspace(0.0, 0.83, n, endpoint=False)
        self.colors = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.85, 1.0))
                       for h in hues]

    # ------------------------------------------------------------------ #
    def _events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit = True
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.quit = True
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_UP:
                    self.speed = self.speeds[min(self.speeds.index(self.speed) + 1,
                                                 len(self.speeds) - 1)]
                elif e.key == pygame.K_DOWN:
                    self.speed = self.speeds[max(self.speeds.index(self.speed) - 1, 0)]
                elif e.key == pygame.K_s:
                    self.skip_gen = True

    def on_step(self, world, gen: int, seed_idx: int, info: dict) -> bool:
        """Called every sim step (4 ticks). Returns False to abort the gen."""
        self._events()
        if self.quit:
            return False
        if self.skip_gen:
            self.skip_gen = False
            return False
        if self.paused:
            # keep pumping events while paused; don't advance sim
            while self.paused and not self.quit:
                self._events()
                self.clock.tick(30)
            if self.quit:
                return False
        self.frame += 1
        if self.frame % max(1, int(self.speed)) != 0:
            return True
        self.draw(world, gen, seed_idx, info)
        self.clock.tick(60)
        return True

    # ------------------------------------------------------------------ #
    def draw(self, world, gen: int, seed_idx: int, info: dict):
        s = self.screen
        s.fill(SKY)

        # pipes (world x in fxp -> float)
        for p in world.pipes:
            x, gap, half = p[0] / SCALE, p[1] / SCALE, p[2] / SCALE
            l = sx(x) - PIPE_HALF_W / SCALE * PX_PER_UNIT
            wpx = 2 * PIPE_HALF_W / SCALE * PX_PER_UNIT
            top_gap = sy(gap + half)
            bot_gap = sy(gap - half)
            pygame.draw.rect(s, PIPE_C, (l, -4, wpx, top_gap + 4))
            pygame.draw.rect(s, PIPE_EDGE, (l, top_gap - 5, wpx, 5))
            pygame.draw.rect(s, PIPE_C, (l, bot_gap, wpx, H - bot_gap))
            pygame.draw.rect(s, PIPE_EDGE, (l, bot_gap, wpx, 5))

        # ground
        pygame.draw.rect(s, GROUND_C, (0, GROUND_SY, W, H - GROUND_SY))

        # birds
        ys = world.y / SCALE
        leader = int(np.argmax(world.score + world.alive * 1e-6))
        for i in range(world.n):
            if not world.alive[i]:
                continue
            col = self.colors[i % len(self.colors)]
            cy = int(sy(ys[i]))
            pygame.draw.circle(s, col, (BIRD_SX, cy), 6)
        if world.alive[leader]:
            pygame.draw.circle(s, (255, 255, 255),
                               (BIRD_SX, int(sy(ys[leader]))), 9, 2)

        # HUD
        alive = int(world.alive.sum())
        lines = [
            f"GEN {gen}   seed {seed_idx+1}/{info['seeds']}",
            f"alive {alive}/{world.n}",
            f"score now {int(world.score.max())}",
            f"gen best run {info.get('gen_best_run', 0)}   "
            f"best run {info.get('best_run', 0)}   best fit {info['best_ever']:.0f}",
            f"speed x{int(self.speed)}   [space]=pause [up/dn]=speed [s]=skip [esc]=quit",
        ]
        for j, t in enumerate(lines):
            img = (self.big if j == 0 else self.font).render(t, True, TXT)
            s.blit(img, (10, 8 + j * (26 if j == 0 else 20)))
        pygame.display.flip()

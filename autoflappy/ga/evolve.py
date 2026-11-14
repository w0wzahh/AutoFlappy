"""Genetic-algorithm trainer for the flappybird.io replica.

Each generation evaluates the whole population in a shared VectorWorld
(identical pipes for every bird). Fitness = mean score over eval seeds +
small survival tiebreak. Selection: elitism + tournament + uniform crossover
+ gaussian mutation (implemented in genome.Population.reproduce).

`run_generation` is driven step-wise so view.py can render every frame.
"""

import json
from pathlib import Path

import numpy as np

from .genome import Population
from .world import VectorWorld

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "models" / "ga_policy.json"
DEFAULT_LOG = ROOT / "models" / "evolution_log.csv"


class Evolver:
    def __init__(self, pop_size=200, dims=(8, 32, 32, 2), seeds_per_gen=2,
                 elite=6, tourn_k=4, mut_p=0.06, mut_sigma=0.3,
                 frame_skip=4, seed=None,
                 out_path=DEFAULT_OUT, log_path=DEFAULT_LOG):
        self.rng = np.random.default_rng(seed)
        self.pop = Population(pop_size, dims, self.rng)
        self.seeds_per_gen = seeds_per_gen
        self.elite = elite
        self.tourn_k = tourn_k
        self.mut_p = mut_p
        self.mut_sigma = mut_sigma
        self.frame_skip = frame_skip
        self.max_ticks = 150_000
        self.out_path = Path(out_path)
        self.log_path = Path(log_path)
        self.gen = 0
        self.best_ever = -1.0        # best mean-fitness (selection metric)
        self.best_run_ever = 0       # best single-run score seen anywhere
        self.gen_best_run = 0        # best single-run score this generation
        self.history = []            # (gen, best, mean, median)
        self.checkpoint_path = self.out_path.parent / "ga_checkpoint.npz"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text(
                "gen,best,mean,median,best_ever,best_run\n")

    # ------------------------------------------------------------------ #
    def save_checkpoint(self):
        np.savez_compressed(self.checkpoint_path,
                            genes=self.pop.genes, gen=self.gen,
                            best_ever=self.best_ever)

    def load_checkpoint(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        d = np.load(self.checkpoint_path)
        self.pop.genes[:] = d["genes"]
        self.gen = int(d["gen"])
        self.best_ever = float(d["best_ever"])
        return True

    def seed_from_policy(self, path, clones=8, jitter=0.1):
        """Restart the population from an exported policy champion:
        `clones` exact copies + the rest jittered mutants."""
        d = json.loads(Path(path).read_text())
        flat = np.concatenate([
            np.asarray(d["W"][0], dtype=np.float32).ravel(),
            np.asarray(d["b"][0], dtype=np.float32).ravel(),
            np.asarray(d["W"][1], dtype=np.float32).ravel(),
            np.asarray(d["b"][1], dtype=np.float32).ravel(),
            np.asarray(d["W"][2], dtype=np.float32).ravel(),
            np.asarray(d["b"][2], dtype=np.float32).ravel(),
        ])
        assert flat.size == self.pop.n_genes, \
            f"policy has {flat.size} genes, pop expects {self.pop.n_genes}"
        self.pop.genes[:] = flat + self.rng.standard_normal(
            self.pop.genes.shape) * jitter
        self.pop.genes[:clones] = flat
        print(f"seeded population from {path}", flush=True)

    # ------------------------------------------------------------------ #
    def _eval_seed(self, seed: int, on_step=None, max_ticks=150_000) -> VectorWorld:
        world = VectorWorld(self.pop.size, seed, self.frame_skip)
        while world.alive.any() and world.tick < max_ticks:
            if on_step is not None and on_step(world) is False:
                self._aborted = True
                break
            rows = np.nonzero(world.alive)[0]
            feats = world.features()[rows]
            acts = self.pop.batch_forward(rows, feats).argmax(axis=1)
            a = np.zeros(self.pop.size, dtype=np.int64)
            a[rows] = acts
            world.step(a)
        return world

    def evaluate(self, on_step=None) -> np.ndarray:
        """Fitness per genome: mean score across seeds + survival tiebreak.
        `on_step(world)` is called once per 4-tick step; returning False
        aborts the whole generation early."""
        self._aborted = False
        fit = np.zeros(self.pop.size)
        done = 0
        for si in range(self.seeds_per_gen):
            seed = self.gen * 104729 + si * 37     # fresh layout every gen
            w = self._eval_seed(
                seed, (lambda w, si=si: on_step(w, si)) if on_step else None,
                self.max_ticks)
            ticks = np.where(w.death_tick >= 0, w.death_tick, w.tick)
            fit += w.score + ticks / 2e4
            done += 1
            self.gen_best_run = max(self.gen_best_run, int(w.score.max()))
            if self._aborted:
                break
        return fit / max(done, 1)

    # ------------------------------------------------------------------ #
    def next_generation(self, fitness: np.ndarray):
        order = np.argsort(-fitness)
        best_i = int(order[0])
        stats = (float(fitness[best_i]), float(fitness.mean()),
                 float(np.median(fitness)))
        self.history.append((self.gen, *stats))
        self.best_ever = max(self.best_ever, stats[0])
        self.last_gen_best_run = self.gen_best_run
        self.best_run_ever = max(self.best_run_ever, self.gen_best_run)
        with open(self.log_path, "a") as f:
            f.write(f"{self.gen},{stats[0]:.2f},{stats[1]:.2f},"
                    f"{stats[2]:.2f},{self.best_ever:.2f},{self.gen_best_run}\n")
        self.gen_best_run = 0

        self.pop.reproduce(fitness, self.elite, self.tourn_k,
                           self.mut_p, self.mut_sigma)
        self.gen += 1
        self.save_checkpoint()
        return best_i, stats

    def export_best(self, idx: int):
        """Write genome idx as a policy.json-compatible dict."""
        W1, b1, W2, b2, W3, b3 = self.pop.decode(self.pop.genes[idx])
        data = {
            "dims": list(self.pop.dims),
            "W": [W1.tolist(), W2.tolist(), W3.tolist()],
            "b": [b1.tolist(), b2.tolist(), b3.tolist()],
            "feature_version": 1,
            "origin": f"ga gen {self.gen}",
        }
        self.out_path.write_text(json.dumps(data))


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Evolve a flappy bot with a GA")
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=2,
                    help="eval worlds (pipe layouts) per generation")
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--mut-p", type=float, default=0.06)
    ap.add_argument("--mut-sigma", type=float, default=0.3)
    ap.add_argument("--speed", type=int, default=8,
                    help="initial sim-steps per drawn frame")
    ap.add_argument("--no-view", action="store_true",
                    help="run headless (no pygame window)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore the checkpoint and start from scratch")
    ap.add_argument("--seed-from", type=Path, default=None,
                    help="init population from an exported policy champion")
    ap.add_argument("--max-ticks", type=int, default=150_000,
                    help="cap per eval world (~750 pipes) so immortal "
                         "genomes can't stall a generation")
    args = ap.parse_args()

    ev = Evolver(pop_size=args.pop, dims=(8, args.hidden, args.hidden, 2),
                 seeds_per_gen=args.seeds, elite=args.elite,
                 mut_p=args.mut_p, mut_sigma=args.mut_sigma,
                 out_path=args.out)
    ev.max_ticks = args.max_ticks

    if not args.fresh and ev.load_checkpoint():
        print(f"resumed checkpoint: gen {ev.gen} best_ever {ev.best_ever:.1f}",
              flush=True)
    elif args.seed_from is not None:
        ev.seed_from_policy(args.seed_from)

    if args.no_view:
        while True:
            fitness = ev.evaluate()
            best_i, stats = ev.next_generation(fitness)
            print(f"gen {ev.gen-1:>4}: best {stats[0]:8.1f}  mean {stats[1]:8.1f}"
                  f"  median {stats[2]:8.1f}  best_ever {ev.best_ever:.1f}"
                  f"  best_run {ev.last_gen_best_run}", flush=True)
            if stats[0] >= ev.best_ever:
                ev.export_best(best_i)

    from .view import View
    view = View(ev.pop.size)
    view.speed = args.speed
    info = {"seeds": ev.seeds_per_gen, "best_ever": ev.best_ever,
            "best_run": ev.best_run_ever, "gen_best_run": 0}

    def _step(w, si):
        info["gen_best_run"] = max(info["gen_best_run"], int(w.score.max()))
        return view.on_step(w, ev.gen, si, info)

    try:
        while not view.quit:
            info["gen_best_run"] = 0
            fitness = ev.evaluate(on_step=_step)
            if view.quit:
                break
            best_i, stats = ev.next_generation(fitness)
            info["best_ever"] = ev.best_ever
            info["best_run"] = ev.best_run_ever
            if stats[0] >= ev.best_ever:
                ev.export_best(best_i)
            print(f"gen {ev.gen-1:>4}: best {stats[0]:8.1f}  "
                  f"mean {stats[1]:8.1f}  best_ever {ev.best_ever:.1f}"
                  f"  best_run {ev.last_gen_best_run}", flush=True)
    finally:
        import pygame
        pygame.quit()

"""Genome = weights of a fixed-topology MLP (default 8-32-32-2).

Population weights live in stacked arrays (pop, fan_in, fan_out) so a batched
forward pass for every bird is a few einsums. Flat "gene" vectors are used
for crossover/mutation.
"""

import numpy as np


class Population:
    def __init__(self, size: int, dims=(8, 32, 32, 2), rng=None):
        self.dims = dims
        self.size = size
        self.rng = rng or np.random.default_rng()
        # gene layout: W1 | b1 | W2 | b2 | W3 | b3  (each flattened)
        self.shapes = [(dims[0], dims[1]), (dims[1],),
                       (dims[1], dims[2]), (dims[2],),
                       (dims[2], dims[3]), (dims[3],)]
        self.bounds = np.cumsum([int(np.prod(s)) for s in self.shapes])
        self.n_genes = int(self.bounds[-1])
        self.genes = np.zeros((size, self.n_genes), dtype=np.float32)
        self._init_random()

    # ------------------------------------------------------------------ #
    def _init_random(self):
        lo = 0
        for s, hi in zip(self.shapes, self.bounds):
            n = hi - lo
            if len(s) == 2:                       # weights: He init
                self.genes[:, lo:hi] = self.rng.standard_normal(
                    (self.size, n)) * np.sqrt(2.0 / s[0])
            # biases stay 0
            lo = hi

    def decode(self, genes: np.ndarray) -> list:
        """flat (n_genes,) -> [W1,b1,W2,b2,W3,b3]"""
        parts, lo = [], 0
        for s, hi in zip(self.shapes, self.bounds):
            parts.append(genes[lo:hi].reshape(s))
            lo = hi
        return parts

    # ------------------------------------------------------------------ #
    def tensors(self, rows: np.ndarray):
        """Per-bird weight tensors for the given genome rows."""
        g = self.genes[rows]                     # (r, n_genes)
        lo = 0
        out = []
        for s, hi in zip(self.shapes, self.bounds):
            out.append(g[:, lo:hi].reshape(len(rows), *s))
            lo = hi
        return out

    def batch_forward(self, rows: np.ndarray, X: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2, W3, b3 = self.tensors(rows)
        h1 = np.einsum("ni,nih->nh", X, W1) + b1
        np.maximum(h1, 0, out=h1)
        h2 = np.einsum("ni,nih->nh", h1, W2) + b2
        np.maximum(h2, 0, out=h2)
        return np.einsum("ni,nih->nh", h2, W3) + b3

    # ------------------------------------------------------------------ #
    def reproduce(self, fitness: np.ndarray, elite: int, tourn_k: int,
                  mut_p: float, mut_sigma: float):
        """Produce the next generation's gene matrix in-place (contents of
        self.genes are overwritten)."""
        order = np.argsort(-fitness)
        parents = np.empty((self.size, self.n_genes), dtype=np.float32)

        # elitism: exact copies
        elites = order[:elite]
        parents[:elite] = self.genes[elites]

        # rest: tournament select 2, uniform crossover, gaussian mutation
        for j in range(elite, self.size):
            a = self._tournament(fitness, tourn_k)
            b = self._tournament(fitness, tourn_k)
            mask = self.rng.random(self.n_genes) < 0.5
            child = np.where(mask, self.genes[a], self.genes[b])
            mut = self.rng.random(self.n_genes) < mut_p
            child[mut] += self.rng.standard_normal(mut.sum()) * mut_sigma
            # rare full-gene reset keeps diversity
            rst = self.rng.random(self.n_genes) < 0.005
            child[rst] = self.rng.standard_normal(rst.sum()) * 0.5
            parents[j] = child

        self.genes[:] = parents
        return elites

    def _tournament(self, fitness: np.ndarray, k: int) -> int:
        idx = self.rng.integers(0, self.size, size=k)
        return int(idx[np.argmax(fitness[idx])])

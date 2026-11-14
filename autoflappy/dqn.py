"""NumPy MLP + Double DQN agent. No deep-learning framework required."""

import json
import math
import numpy as np


class QNet:
    """MLP with ReLU hidden layers:  n_in -> h -> h -> n_out (linear)."""

    def __init__(self, n_in: int, n_hid: int, n_out: int, rng: np.random.Generator):
        self.dims = (n_in, n_hid, n_hid, n_out)
        self.W = []
        self.b = []
        dims = (n_in, n_hid, n_hid, n_out)
        for i in range(3):
            fan_in = dims[i]
            self.W.append(rng.standard_normal((fan_in, dims[i + 1])) * math.sqrt(2.0 / fan_in))
            self.b.append(np.zeros(dims[i + 1]))
        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(x) for x in self.b]
        self.vb = [np.zeros_like(x) for x in self.b]
        self.t = 0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        a1 = x @ self.W[0] + self.b[0]
        h1 = np.maximum(a1, 0)
        a2 = h1 @ self.W[1] + self.b[1]
        h2 = np.maximum(a2, 0)
        q = h2 @ self.W[2] + self.b[2]
        return q, (x, a1, h1, a2, h2)

    def q_values(self, x: np.ndarray) -> np.ndarray:
        q, _ = self.forward(x)
        return q

    def backward(self, cache: tuple, dq: np.ndarray, lr: float = 3e-4):
        x, a1, h1, a2, h2 = cache
        B = x.shape[0]

        gW3 = h2.T @ dq
        gb3 = dq.sum(axis=0)
        dh2 = dq @ self.W[2].T
        da2 = dh2 * (a2 > 0)

        gW2 = h1.T @ da2
        gb2 = da2.sum(axis=0)
        dh1 = da2 @ self.W[1].T
        da1 = dh1 * (a1 > 0)

        gW1 = x.T @ da1
        gb1 = da1.sum(axis=0)

        self._adam([gW1, gW2, gW3], [gb1, gb2, gb3], lr)

    def _adam(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        for i in range(3):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * gW[i] ** 2
            self.W[i] -= lr * (self.mW[i] / bc1) / (np.sqrt(self.vW[i] / bc2) + eps)

            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb[i] ** 2
            self.b[i] -= lr * (self.mb[i] / bc1) / (np.sqrt(self.vb[i] / bc2) + eps)

    def copy_from(self, other: "QNet"):
        for i in range(3):
            self.W[i][...] = other.W[i]
            self.b[i][...] = other.b[i]

    def export(self) -> dict:
        return {
            "dims": list(self.dims),
            "W": [w.tolist() for w in self.W],
            "b": [x.tolist() for x in self.b],
        }

    @staticmethod
    def load(data: dict) -> "QNet":
        net = QNet.__new__(QNet)
        net.dims = tuple(data["dims"])
        net.W = [np.array(w, dtype=np.float64) for w in data["W"]]
        net.b = [np.array(x, dtype=np.float64) for x in data["b"]]
        net.mW = [np.zeros_like(w) for w in net.W]
        net.vW = [np.zeros_like(w) for w in net.W]
        net.mb = [np.zeros_like(x) for x in net.b]
        net.vb = [np.zeros_like(x) for x in net.b]
        net.t = 0
        return net


class ReplayBuffer:
    def __init__(self, capacity: int, n_feat: int):
        self.s = np.zeros((capacity, n_feat), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, n_feat), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.i = 0
        self.n = 0
        self.cap = capacity

    def push(self, s, a, r, s2, d):
        i = self.i
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DQNAgent:
    def __init__(self, n_feat: int, n_act: int = 2, buffer: int = 100_000,
                 gamma: float = 0.99, lr: float = 3e-4, hid: int = 128,
                 seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.online = QNet(n_feat, hid, n_act, self.rng)
        self.target = QNet(n_feat, hid, n_act, self.rng)
        self.target.copy_from(self.online)
        self.buf = ReplayBuffer(buffer, n_feat)
        self.gamma = gamma
        self.lr = lr
        self.n_act = n_act
        self.learn_steps = 0

    def act(self, s: np.ndarray, eps: float, flap_prob: float = 0.35) -> int:
        if self.rng.random() < eps:
            # flap-biased exploration: uniform 50/50 keeps the bird pinned too
            # high (above every possible gap), so random flaps are rarer
            return int(self.rng.random() < flap_prob)
        return int(np.argmax(self.online.q_values(s[None])[0]))

    def learn(self, batch: int = 256) -> float:
        s, a, r, s2, d = self.buf.sample(batch, self.rng)
        q, cache = self.online.forward(s)
        # Double DQN target
        next_a = np.argmax(self.online.q_values(s2), axis=1)
        q_next = self.target.q_values(s2)
        y = r + self.gamma * q_next[np.arange(batch), next_a] * (1 - d)
        # gradient of huber-free MSE on chosen action only
        dq = np.zeros_like(q)
        err = q[np.arange(batch), a] - y
        err = np.clip(err, -10, 10)  # mild robustification
        dq[np.arange(batch), a] = 2 * err / batch
        self.online.backward(cache, dq, self.lr)
        self.learn_steps += 1
        return float(np.mean(err ** 2))

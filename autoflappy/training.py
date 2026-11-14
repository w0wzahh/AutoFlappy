"""Train a Double DQN agent on the bit-exact flappybird.io replica.

Outputs models/policy.json — weights consumed by web/agent.js in the browser.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .env import FlappyEnv, N_FEATURES, SCALE
from .dqn import DQNAgent, QNet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "models" / "policy.json"

TOTAL_STEPS  = 250_000         # steps are 4-tick windows (~30 Hz decisions)
WARMUP       = 5_000
EPS_START    = 0.5
EPS_END      = 0.02
EPS_DECAY    = 100_000         # linear decay horizon (env steps)
EXPERT_EPS   = 80              # scripted-demonstration episodes used to
EXPERT_CAP   = 6_000           # max windows per expert episode
BATCH        = 256
TARGET_SYNC  = 2_000           # learn steps between target-net syncs
EVAL_EVERY   = 20_000          # env steps between evaluations
EVAL_EPS     = 5
EVAL_CAP     = 9_000           # windows per eval episode (~5 game-minutes)
MASTERY      = 150             # mean score that counts as "solved"
FLAP_PROB    = 0.2             # flap probability of exploratory actions


def epsilon_at(t: int, start: float = EPS_START) -> float:
    if t >= EPS_DECAY:
        return EPS_END
    return start + (EPS_END - start) * (t / EPS_DECAY)


def evaluate(agent: DQNAgent, n_eps: int, cap: int, seed_off: int):
    env = FlappyEnv(777_000 + seed_off)
    scores, ticks = [], []
    for _ in range(n_eps):
        s = env.reset()
        t = 0
        while t < cap:
            a = int(np.argmax(agent.online.q_values(s[None])[0]))
            s, _, d, info = env.step(a)
            t += 1
            if d:
                break
        scores.append(info["score"])
        ticks.append(t)
    return float(np.mean(scores)), int(max(scores)), float(np.mean(ticks))


def save_policy(agent: DQNAgent, path: str):
    data = agent.online.export()
    data["feature_version"] = 1
    with open(path, "w") as f:
        json.dump(data, f)


def fill_expert(agent: DQNAgent, n_eps: int):
    """Pre-fill the replay buffer with a scripted near-expert policy.

    Rule: flap when the bird is below the (jittered) gap center. Jitter gives
    coverage of off-center approach/recovery states, not just the ideal line.
    """
    rng = np.random.default_rng(99)
    env = FlappyEnv(seed=555)
    total = 0
    for ep in range(n_eps):
        s = env.reset()
        jit = float(rng.uniform(-0.15, 0.12))      # per-episode aim offset
        for _ in range(EXPERT_CAP):
            p = env._next_pipe()
            if p is None:
                a = 1 if env.y < int(0.22 * SCALE) else 0
            else:
                a = 1 if env.y < p.gap_y + jit * SCALE + float(rng.normal(0, 0.015)) * SCALE else 0
            s2, r, d, info = env.step(a)
            agent.buf.push(s, a, r, s2, float(d))
            s = s2
            total += 1
            if d:
                break
        if (ep + 1) % 10 == 0:
            print(f"  expert ep {ep+1}: score {info['score']}  buf {agent.buf.n}",
                  flush=True)
    print(f"expert buffer filled: {total} transitions", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Train the flappybird.io DQN agent")
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--out", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="warm-start both nets from an existing policy.json")
    ap.add_argument("--eps-start", type=float, default=EPS_START)
    ap.add_argument("--mastery", type=float, default=MASTERY,
                    help="eval-mean early-stop threshold (use a huge value to "
                         "train uninterrupted)")
    ap.add_argument("--eval-cap", type=int, default=EVAL_CAP,
                    help="max windows per eval episode")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    env = FlappyEnv(seed=1234)
    agent = DQNAgent(N_FEATURES, n_act=2, seed=42)

    if args.init_from is not None:
        net = QNet.load(json.loads(Path(args.init_from).read_text()))
        agent.online.copy_from(net)
        agent.target.copy_from(net)
        print(f"warm-started from {args.init_from}", flush=True)

    print("filling replay buffer with expert demonstrations...")
    fill_expert(agent, EXPERT_EPS)

    s = env.reset()
    best = -1.0
    ep_scores = []
    ep_t = 0
    loss_acc, loss_n = 0.0, 0
    next_eval = EVAL_EVERY

    for t in range(1, args.steps + 1):
        eps = epsilon_at(t, args.eps_start)
        a = agent.act(s, eps, FLAP_PROB)
        s2, r, d, info = env.step(a)
        agent.buf.push(s, a, r, s2, float(d))
        s = s2
        ep_t += 1

        if t >= WARMUP:
            loss_acc += agent.learn(BATCH)
            loss_n += 1
            if agent.learn_steps % TARGET_SYNC == 0:
                agent.target.copy_from(agent.online)

        if d:
            ep_scores.append(info["score"])
            s = env.reset()
            ep_t = 0

        if t >= next_eval:
            next_eval += EVAL_EVERY
            mean_s, max_s, mean_t = evaluate(agent, EVAL_EPS, args.eval_cap, seed_off=t)
            recent = np.mean(ep_scores[-50:]) if ep_scores else 0.0
            loss = loss_acc / max(loss_n, 1)
            loss_acc, loss_n = 0.0, 0
            el = time.time() - t_start
            print(f"[{el:6.0f}s] step {t:>7}  eps {eps:.3f}  "
                  f"train_recent {recent:8.1f}  eval mean {mean_s:8.1f}  "
                  f"eval max {max_s:7}  ep_len {mean_t:8.0f}  loss {loss:.4f}",
                  flush=True)
            if mean_s > best:
                best = mean_s
                save_policy(agent, args.out)
                print(f"          -> new best, saved to {args.out}", flush=True)
            if mean_s >= args.mastery:
                print("MASTERED — stopping early.", flush=True)
                break

    # a cold run keeps its final net even without mastery; a warm-started run
    # never overwrites the proven file with a possibly-regressed last net
    if best < args.mastery and args.init_from is None:
        save_policy(agent, args.out)
    print(f"done in {time.time()-t_start:.0f}s, best eval mean {best:.1f}")


if __name__ == "__main__":
    main()

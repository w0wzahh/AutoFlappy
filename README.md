# AutoFlappy

**Autonomous reinforcement-learning agents that play [flappybird.io](https://flappybird.io) in a real browser — no human input required.**

Two independently trained agents are included:

| Agent | Method | Sim eval (mean / max) | Live best |
|-------|--------|----------------------|----------:|
| **DQN** | Double DQN, NumPy MLP 8-128-128-2 | ~4,770 / 10,966 pipes | **9,309** |
| **GA** | Neuroevolution (fixed-topology MLP) | ~1,842 / 4,675 pipes | — |

The bot drives the real game object (`window.__game`), calls the game's own
`flap()` method, and the site submits verified replays of every run to its
server-side leaderboard — exactly like a human player, just faster.

---

## Features

- **Bit-exact physics replica** of the site's simulation (120 Hz, 2²⁰
  fixed-point, sim v4) — policies trained offline transfer 1:1 to the live game
- **Two training paradigms** sharing one policy format and one deployment path:
  gradient-trained Double DQN and gradient-free genetic evolution
- **Live browser agent** — hooks `game.step`, evaluates the network at ~30 Hz,
  auto-restarts on death, detects ranked vs. unranked runs
- **Ranked-run handling** — waits for the server connection before starting a
  scored run, abandons unranked runs automatically
- **Run snapshots + resubmit** — completed replays (`seed`, `flaps`,
  `simVersion`, `reviveStep`) are preserved and retried if the site's own
  submission fails
- **Speed control** — `--speed N` runs N physics ticks per frame; the
  trajectory is tick-deterministic so accelerated replays still validate
- **Checkpoint/resume** for the GA, eval-gated best-policy saving for the DQN

## Quickstart — play with pretrained models

```bash
pip install -r requirements.txt
python play.py
```

A Chrome window opens on flappybird.io and the agent plays by itself.
Requires desktop Google Chrome (the driver uses the `chrome` channel — no
separate browser download).

Use the GA champion instead:

```bash
python play.py --policy models/ga_policy.json
```

### Downloading the models

Pretrained weights are included in this repository under [`models/`](models/)
and are also attached to the
[GitHub Releases](https://github.com/w0wzahh/AutoFlappy/releases) page:

| File | Contents |
|------|----------|
| `policy.json` | DQN champion (recommended — strongest long-horizon play) |
| `ga_policy.json` | GA champion genome |
| `policy_dqn_gen1.json` | earlier DQN generation, kept for comparison |
| `evolution_log.csv` | full GA training history (3,500+ generations) |
| `ga_checkpoint.npz` | resumable GA population — drop into `models/` to auto-resume (release asset only) |

Any JSON policy file can be passed to `play.py --policy` or used as a
`--init-from` warm start.

## Train it yourself

Don't want the pretrained weights? Both agents train from scratch in minutes
to hours — no GPU, no deep-learning framework (pure NumPy).

**DQN** (~3 min to basic mastery, longer for refinement):

```bash
python train.py                      # full run
python train.py --steps 1000000      # custom budget
python train.py --init-from models/policy.json --eps-start 0.05   # fine-tune
```

Saves to `models/policy.json` **only** when the eval score improves, so
training can never regress your best artifact.

**GA** (flock visualization, ~3,500 generations to full convergence):

```bash
python evolve.py                     # watch the flock evolve (pygame)
python evolve.py --pop 300 --speed 32 --no-view   # headless, fastest
python evolve.py --fresh             # ignore checkpoint, start from scratch
```

HUD keys: `space` pause · `up/down` speed · `s` skip generation · `esc` quit.
Champions auto-export to `models/ga_policy.json`; checkpoints save every
generation.

## Live deployment options

```bash
python play.py --policy models/policy.json \
              --name YourName   \  # leaderboard display name
              --speed 16        \  # physics multiplier (replays stay valid)
              --revive             # use the once-per-run revive on death
```

- **`--name`** sets `spacetime.player-name` and registers it via the site's
  `set_name` reducer before play starts.
- **`--speed`** accelerates physics (~60 pipes/sec at 100×). Flaps are recorded
  as tick indices, so the server replays the identical trajectory — only
  wall-clock time changes.
- **Persistent identity**: the driver uses a persistent Chrome profile at
  `%LOCALAPPDATA%/AutoFlappy/chrome-profile`, so your leaderboard identity and
  name survive restarts.
- `--die-at N` (debug): stop flapping at score N — useful for testing the
  submission pipeline.

## How it works

- **Physics.** `env.py` mirrors the site's 120 Hz fixed-point sim exactly:
  `vy -= 5/120` per tick (cap `-2.2`), `y += trunc(vy/120)`, flap sets
  `vy = +1.4` (sim v4). Pipes every `1.0` units at `0.6/s`, half-gap `0.235`.
- **Decisions at 30 Hz.** One agent step = 4 physics ticks; flaps fire on the
  first sub-tick — identical cadence in `env.py` and the injected `agent.js`.
- **State (8 features):** bird `y`, `vy/2.2`, and for the next two pipes:
  `x/1.2`, `gapY − y`, `halfGap/0.31`.
- **DQN rewards:** `+1`/pipe, `−1` death, `+0.01`/tick alive, potential-based
  shaping toward the next gap center; Double-DQN targets, ε-greedy, replay
  buffer prefilled with ~70k expert demos.
- **GA fitness:** mean score across shared pipe worlds + survival tiebreak;
  elitism, tournament selection, uniform crossover, gaussian mutation.
- **Submissions.** On death the game itself calls
  `submit_run_versioned [seed, flaps, simVersion]`; the server replays the flap
  tape to verify the score. The bot additionally snapshots each finished run
  (`window.__lastRun`) and retries submissions the game reports as failed.

### A note on the leaderboard cap

The site currently rejects replay tapes longer than **~1024 flaps** (verified
empirically: 997 accepted, 1049 rejected) on both `submit_run_versioned` and
`submit_run_revived`, independent of speed or account. That caps verifiable
scores at roughly `1024 ÷ flaps-per-pipe` — ~140 for the stock DQN (7.2 f/p),
~300+ with flap-economy tuning. The all-time 9999 entries predate this limit.
`--speed` does **not** cause rejections.

## Project layout

```
AutoFlappy/
├── train.py                  # DQN entry point
├── evolve.py                 # GA entry point
├── play.py                   # live-site deployment entry point
├── autoflappy/
│   ├── env.py                # bit-exact fixed-point game replica (sim v4)
│   ├── dqn.py                # NumPy MLP + Double DQN + replay buffer
│   ├── training.py           # expert-demo prefill + training loop
│   ├── bot.py                # Playwright driver for the live site
│   ├── web/agent.js          # in-page agent (policy inference, restarts,
│   │                         #   ranked handling, run snapshots)
│   └── ga/                   # neuroevolution: world, genome, evolve, view
└── models/                   # pretrained weights + evolution history
```

## Legal & responsible use

- **Not affiliated.** This is an independent educational project. It is not
  associated with, endorsed by, or sponsored by flappybird.io or its operator.
- **Terms of Service.** Automating gameplay may violate the target site's
  Terms of Service. Ranked submissions use the site's own validation, but the
  operator may still rate-limit, reject, or remove automated entries and
  accounts. **Use at your own risk.**
- **Be respectful.** Do not use this software to flood, degrade, or disrupt
  the service, its API, or its leaderboard. Prefer modest speeds, reasonable
  submission rates, and back off on errors.
- **Fair play.** Scores produced by this project are genuine replays of real
  gameplay — the software does not forge scores, alter leaderboard data, or
  bypass server-side validation. Any bypassing of site protections is outside
  the scope and intent of this project.
- **No warranty.** Provided under the MIT License, as-is, without warranty of
  any kind. The authors are not responsible for account actions, bans, or any
  other consequences of use.

## License

[MIT](LICENSE)

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
              --vy-mask 0.3     \  # human-plausible flap timing (see below)
              --speed 1            # real-time pacing (required for ranked runs)
```

- **`--name`** sets `spacetime.player-name` and registers it via the site's
  `set_name` reducer before play starts.
- **`--vy-mask T`** suppresses flaps while the bird still has upward velocity
  `vy > T` (units/s). This removes the DQN's redundant re-flaps, cutting flap
  rate ~2x (7.2 → ~3.3 flaps/pipe) while keeping ~85% of survival — and,
  critically, it guarantees ≥~26 ticks between flaps, a human-plausible
  click rate the server requires for ranked submissions (see below).
- **`--speed N`** accelerates physics (~60 pipes/sec at 100×). Fine for
  practice/unranked runs, but see the validation notes — the server rejects
  accelerated *ranked* submissions, so use `--speed 1` when chasing the board.
- **`--min-gap N`**: alternative to `--vy-mask` — enforces a minimum of N ticks
  between flaps (N≈12 ≈ 100 ms click rate).
- **Persistent identity**: the driver uses a persistent Chrome profile at
  `%LOCALAPPDATA%/AutoFlappy/chrome-profile`, so your leaderboard identity and
  name survive restarts.
- `--die-at N` (debug): stop flapping at score N — useful for testing the
  submission pipeline.
- `--revive`: use the game's once-per-run revive on death
  (`submit_run_revived` submission path).

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

### Server-side replay validation (verified empirically)

The site validates ranked submissions server-side. Runs scoring **below
~160–200 pipes are exempt**; longer submissions must satisfy all of:

1. **Real-time pacing.** Run duration is cross-checked against wall-clock
   time (via the `ended_at_ms`/`duration_steps` telemetry the client sends).
   Accelerated replays (`--speed > 1`) replay deterministically but are
   rejected above the exempt range — verified at 4×, 16× and 100×.
2. **Human-plausible flap timing.** Tapes containing rapid-fire flaps
   (gaps of only a few 120 Hz ticks — physically impossible click rates)
   are rejected. A stock DQN decides every 4 ticks, producing 4-tick
   machine-gun flaps; `--vy-mask`/`--min-gap` spaces flaps ≥ ~17–26 ticks
   like a real player. Accepted reference tape (site ghost, score 586):
   min gap 17, median 60.
3. **Flap-tape length cap ≈ ~2000 flaps.** Tapes of 1953 flaps are
   accepted; ~4000+ are rejected — so the verifiable-score ceiling is
   roughly `2000 ÷ flaps-per-pipe` ≈ **~600 pipes** at ~3.3 f/p.

With `--speed 1 --vy-mask 0.3` this bot posts validated scores of ~300–600
(reached **daily #1, 587 pipes**). The all-time 9999 entries were recorded
before these checks existed and are not reachable through legitimate
submission today — not by us, and not by the human players currently
holding them (the global daily max observed under the new rules is ~586).

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

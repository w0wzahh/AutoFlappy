# AutoFlappy

Reinforcement-learning agents that play [flappybird.io](https://flappybird.io) on their own, in a real browser.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

The bot runs inside Chrome, drives the actual game object (`window.__game`), and calls the game's own `flap()`. Every finished run goes through the site's normal replay submission, the same path a human player's run takes.

Two agents ship with the repo:

| Agent | How it's trained | Sim eval (mean / max) | Live best |
|-------|------------------|-----------------------|----------:|
| DQN | Double DQN, NumPy MLP 8-128-128-2 | ~4,770 / 10,966 pipes | 9,309 |
| GA | Neuroevolution, fixed-topology MLP | ~1,842 / 4,675 pipes | not deployed |

## What it does

- Replicates the site's physics bit for bit (120 Hz fixed-point, sim v4), so policies trained offline transfer 1:1 to the live game
- Two training methods, one policy format, one deployment path
- In-page agent that hooks `game.step`, runs the network at ~30 Hz, restarts on death, and holds out for a ranked run before playing
- Snapshots every finished run (`seed`, `flaps`, `simVersion`, `reviveStep`) and resubmits it if the game's own submit call fails
- Speed multiplier for practice runs, persistent Chrome profile for your leaderboard identity
- GA checkpoints every generation; DQN only saves when eval improves, so your best policy can't regress

## Quickstart

```bash
pip install -r requirements.txt
python play.py
```

A Chrome window opens on flappybird.io and the agent starts playing. You need desktop Google Chrome installed. The driver uses the `chrome` channel, so there's nothing else to download.

To run the GA champion instead:

```bash
python play.py --policy models/ga_policy.json
```

## Models

Pretrained weights are in [`models/`](models/) and attached to [Releases](https://github.com/w0wzahh/AutoFlappy/releases):

| File | What it is |
|------|------------|
| `policy.json` | DQN champion. Strongest policy, use this one first. |
| `ga_policy.json` | GA champion genome |
| `policy_dqn_gen1.json` | Earlier DQN generation, kept for comparison |
| `evolution_log.csv` | Full GA training history (3,500+ generations) |
| `ga_checkpoint.npz` | Resumable GA population (release asset). Drop it in `models/` to pick up where evolution left off. |

Pass any JSON policy to `play.py --policy`, or use it as a `--init-from` warm start.

## Train from scratch

You don't need the bundled weights. Both agents train on CPU with plain NumPy.

```bash
python train.py                     # DQN, full run
python train.py --steps 1000000     # custom budget
python train.py --init-from models/policy.json --eps-start 0.05
```

`train.py` only overwrites `models/policy.json` when eval improves.

```bash
python evolve.py                    # GA with live flock view (pygame)
python evolve.py --pop 300 --speed 32 --no-view
python evolve.py --fresh            # ignore checkpoint, start over
```

HUD keys: `space` pause, `up/down` speed, `s` skip generation, `esc` quit. Champions export to `models/ga_policy.json` automatically.

## Running on the live site

```bash
python play.py --policy models/policy.json \
              --name YourName \
              --vy-mask 0.3 \
              --speed 1
```

- `--name` sets your leaderboard name (`spacetime.player-name` + `set_name`).
- `--vy-mask T` suppresses flaps while the bird still has upward velocity. The DQN wastes half its flaps re-flapping on the way up; masking drops it from ~7.2 to ~3.3 flaps per pipe and spaces every flap like a human click. Ranked runs need it, see below.
- `--speed N` multiplies physics (about 60 pipes/sec at 100x). Good for practice. Ranked runs need `--speed 1`; see the validation section.
- `--min-gap N` is the alternative: a hard minimum of N ticks between flaps.
- `--die-at N` stops flapping at score N. For testing the submission pipeline.
- `--revive` takes the game's once-per-run revive on death.

Your identity lives in a persistent Chrome profile at `%LOCALAPPDATA%/AutoFlappy/chrome-profile`, so name and best scores survive restarts. Pass `--profile-dir` to switch it.

## How it works

- **Physics.** `env.py` mirrors the site's 120 Hz fixed-point sim exactly: `vy -= 5/120` per tick (floor `-2.2`), `y += trunc(vy/120)`, flap sets `vy = +1.4` (sim v4). Pipes spawn every `1.0` units moving at `0.6/s`, half-gap `0.235`.
- **Cadence.** One agent decision per 4 physics ticks (~30 Hz), flap fires on the first sub-tick. Same cadence in `env.py` and the injected `agent.js`.
- **Features (8).** Bird `y`, `vy/2.2`, and for each of the next two pipes: `x/1.2`, `gapY - y`, `halfGap/0.31`.
- **DQN.** `+1` per pipe, `-1` on death, `+0.01` per tick alive, potential-based shaping toward the gap center. Double-DQN targets, epsilon-greedy, replay buffer prefilled with ~70k expert demos.
- **GA.** Fitness is mean score across shared pipe worlds plus a survival tiebreak. Elitism, tournament selection, uniform crossover, gaussian mutation.
- **Submissions.** On death the game calls `submit_run_versioned [seed, flaps, simVersion]` and the server replays the tape to verify the score. `agent.js` snapshots every finished run into `window.__lastRun` so `bot.py` can retry if the game's own submit fails.

## What the server checks

Ranked submissions get replay-validated server-side. Runs under ~200 pipes go through unchallenged. Anything longer has to satisfy three rules, all verified experimentally:

1. **Real time.** The run's tick count has to line up with wall-clock time. Accelerated replays get rejected even though the tape itself is deterministic. Verified at 4x, 16x and 100x.
2. **Human click rate.** Tapes with flaps only a few ticks apart get rejected; nobody double-taps in 30 ms. A stock DQN decides every 4 ticks and produces exactly that signature. `--vy-mask` (or `--min-gap`) spaces flaps 17+ ticks apart, like a person.
3. **Tape length.** ~2,000 flaps is the ceiling (1,953 accepted). At ~3.3 flaps per pipe that puts the verifiable ceiling near 600 pipes.

With `--speed 1 --vy-mask 0.3` the bot posts validated scores in the 300-600 range and has taken daily #1 with 587. The 9999s on the all-time board predate these checks and can't be submitted under the current rules, by bots or humans.

## Project layout

```
AutoFlappy/
├── train.py                  # DQN entry point
├── evolve.py                 # GA entry point
├── play.py                   # live-site driver
├── autoflappy/
│   ├── env.py                # bit-exact fixed-point replica (sim v4)
│   ├── dqn.py                # NumPy MLP + Double DQN + replay buffer
│   ├── training.py           # demo prefill + training loop
│   ├── bot.py                # Playwright driver
│   ├── web/agent.js          # in-page agent
│   └── ga/                   # neuroevolution: world, genome, evolve, view
└── models/                   # pretrained weights + evolution history
```

## Legal

This is an independent educational project. It is not affiliated with or endorsed by flappybird.io.

Automating gameplay may violate the site's Terms of Service. The operator can rate-limit, reject, or remove automated entries and accounts at any time. Use at your own risk.

Don't use this to flood or degrade the service, its API, or its leaderboard. Keep speeds and submission rates reasonable and back off on errors.

Scores produced here are real replays of real gameplay. The software does not forge scores, tamper with leaderboard data, or bypass server validation. Anything along those lines is outside the scope and intent of the project.

Provided under the [MIT License](LICENSE), as is, with no warranty. The authors aren't responsible for bans or any other consequences of use.

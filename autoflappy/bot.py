"""Launch flappybird.io in real Chrome and let the trained DQN agent play.

Usage:  python play.py   (or:  python -m autoflappy.bot)
Requires:  pip install playwright   (uses the installed Google Chrome — no
playwright browser download needed)
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "models" / "policy.json"
AGENT_JS = Path(__file__).resolve().parent / "web" / "agent.js"
URL = "https://flappybird.io"
SUBMIT_URL = ("https://maincloud.spacetimedb.com/v1/database/"
              "fb-db-prod-2-8w3a5/call/submit_run_versioned")

ARGS = [
    "--mute-audio",
    "--disable-blink-features=AutomationControlled",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
]


def main():
    ap = argparse.ArgumentParser(description="Run the trained agent on flappybird.io")
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--name", default=None,
                    help="leaderboard player name (sets spacetime.player-name)")
    ap.add_argument("--die-at", type=int, default=None,
                    help="debug: stop flapping at this score (test submission)")
    ap.add_argument("--profile-dir", type=Path,
                    default=Path.home() / "AppData" / "Local" / "AutoFlappy" / "chrome-profile",
                    help="persistent Chrome profile dir (keeps leaderboard identity)")
    ap.add_argument("--speed", type=int, default=1,
                    help="physics ticks per scheduled tick (>1 = faster bird)")
    ap.add_argument("--revive", action="store_true",
                    help="use the once-per-run revive on death (submit_run_revived path)")
    ap.add_argument("--min-gap", type=int, default=0,
                    help="min ticks between flaps (human-plausible click rate)")
    ap.add_argument("--vy-mask", type=float, default=None,
                    help="suppress flaps while vy > this (units/s); cuts flap "
                         "count ~2x for the ~2000-flap submission cap")
    args = ap.parse_args()

    with open(args.policy) as f:
        weights = json.load(f)
    with open(AGENT_JS, encoding="utf-8") as f:
        agent_src = f.read()

    with sync_playwright() as pw:
        if args.profile_dir:
            ctx = pw.chromium.launch_persistent_context(
                str(args.profile_dir),
                channel="chrome",
                headless=False,
                args=ARGS,
                ignore_default_args=["--enable-automation"],
                viewport={"width": 520, "height": 800},
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
        else:
            browser = pw.chromium.launch(
                channel="chrome",
                headless=False,
                args=ARGS,
                ignore_default_args=["--enable-automation"],
            )
            page = browser.new_page(viewport={"width": 520, "height": 800})
        # log leaderboard/server calls so we can see submissions land (or fail)
        def _on_response(resp):
            if "/v1/database/" in resp.url:
                try:
                    body = resp.text()[:200]
                except Exception:
                    body = "<unreadable>"
                print(f"    [net] {resp.request.method} {resp.url.rsplit('/',2)[-2:]} "
                      f"-> {resp.status} {body}", flush=True)
        page.on("response", _on_response)
        if args.name:
            page.add_init_script(
                f'localStorage.setItem("spacetime.player-name", {json.dumps(args.name)});')
            print(f"leaderboard name: {args.name}", flush=True)
        page.goto(args.url, wait_until="domcontentloaded")
        print("page loading, waiting for game object...", flush=True)
        page.wait_for_function("() => !!window.__game", timeout=60_000)

        page.evaluate(f"window.__agentPolicy = {json.dumps(weights)};")
        page.evaluate(agent_src)
        if args.die_at:
            page.evaluate(f"window.__agentDieAt = {args.die_at};")
            print(f"debug: will stop flapping at score {args.die_at}", flush=True)
        if args.speed != 1:
            page.evaluate(f"window.__agentSpeed = {args.speed};")
            print(f"speed: {args.speed}x physics", flush=True)
        if args.revive:
            page.evaluate("window.__agentRevive = true;")
            print("revive mode: on", flush=True)
        if args.min_gap:
            page.evaluate(f"window.__agentMinGap = {args.min_gap};")
            print(f"min flap gap: {args.min_gap} ticks", flush=True)
        if args.vy_mask is not None:
            page.evaluate(f"window.__agentVyMask = {args.vy_mask};")
            print(f"vy flap mask: suppress while vy > {args.vy_mask}", flush=True)
        print("agent injected — it should be playing now.\n", flush=True)

        last = None
        last_mtime = args.policy.stat().st_mtime
        last_prof = 0.0
        resubmits = {}
        submitted = set()
        t0 = time.time()
        try:
            while True:
                time.sleep(1.0)
                # hot-swap the network if the policy file was re-exported
                try:
                    mt = args.policy.stat().st_mtime
                except OSError:
                    mt = last_mtime
                if mt != last_mtime:
                    last_mtime = mt
                    try:
                        weights = json.load(open(args.policy))
                    except (OSError, json.JSONDecodeError):
                        weights = None          # torn write — try again next tick
                        last_mtime = 0
                    if weights:
                        page.evaluate(
                            f"window.__agentSetPolicy && window.__agentSetPolicy({json.dumps(weights)});")
                        print(f"[{int(time.time()-t0):>4}s] policy hot-swapped: "
                              f"{args.policy.name} ({weights.get('origin','?')})",
                              flush=True)
                stats = page.evaluate("() => window.__agentStats || null")
                if not stats:
                    continue
                # every 20 s, ask the server what it recorded for our identity
                if time.time() - last_prof > 20:
                    last_prof = time.time()
                    for tbl in ("my_profile", "my_rank"):
                        try:
                            prof = page.evaluate(
                                "(q) => fetch('https://maincloud.spacetimedb.com/v1/database/"
                                "fb-db-prod-2-8w3a5/sql', {method:'POST', headers:{"
                                "Authorization:'Bearer '+localStorage.getItem('spacetime.identity-token'),"
                                "'Content-Type':'text/plain'}, body:q})"
                                ".then(r=>r.json()).then(j=>j[0]&&j[0].rows||[]).catch(e=>String(e))",
                                f"SELECT * FROM {tbl}")
                            print(f"    [{tbl}] {json.dumps(prof)[:280]}", flush=True)
                        except Exception as e:
                            print(f"    [{tbl}] query failed: {e}", flush=True)
                line = (stats["runs"], stats["score"], stats["best"], stats["state"])
                if line != last:
                    last = line
                    el = int(time.time() - t0)
                    print(f"[{el:>4}s] state={stats['state']:<9} "
                          f"score={stats['score']:<6} best={stats['best']:<6} "
                          f"runs={stats['runs']:<5} net={stats.get('net','?'):<8} "
                          f"ranked={stats.get('ranked','?')}", flush=True)
                # safety net: resubmit the finished run ourselves if the
                # game's own submission failed (530s etc.) — retried over
                # ~90 s so no run is lost to a transient server error
                try:
                    lr = page.evaluate("() => window.__lastRun || null")
                except Exception:
                    lr = None
                if lr and lr.get("ranked") and lr.get("score", 0) > 0 \
                        and lr.get("key") not in resubmits \
                        and lr.get("key") not in submitted:
                    resubmits[lr["key"]] = {"run": lr, "tries": 0, "next": 0.0}
                    try:
                        dump = Path("logs") / f"lastrun_{lr['score']}.json"
                        dump.parent.mkdir(exist_ok=True)
                        dump.write_text(json.dumps(lr))
                    except Exception:
                        pass
                now = time.time()
                for key in list(resubmits):
                    item = resubmits[key]
                    if now < item["next"]:
                        continue
                    if item["tries"] >= 8:
                        # exhausted the fast window — retry every 5 min
                        item["tries"] = 0
                        item["next"] = now + 300.0
                        continue
                    item["tries"] += 1
                    item["next"] = now + 15.0
                    token = page.evaluate(
                        "() => localStorage.getItem('spacetime.identity-token')")
                    if not token:
                        continue
                    run = item["run"]
                    # revived runs submit through a different reducer with
                    # an extra reviveStep arg
                    if run.get("reviveStep") is not None:
                        url = SUBMIT_URL.replace(
                            "submit_run_versioned", "submit_run_revived")
                        payload = [run["seed"], run["flaps"], run["simVersion"],
                                   run["reviveStep"]]
                    else:
                        url = SUBMIT_URL
                        payload = [run["seed"], run["flaps"], run["simVersion"]]
                    try:
                        req = urllib.request.Request(
                            url,
                            data=json.dumps(payload).encode(),
                            headers={"Authorization": f"Bearer {token}",
                                     "Content-Type": "application/json"},
                            method="POST")
                        resp = urllib.request.urlopen(req, timeout=60)
                        print(f"    [resubmit] score={run['score']} "
                              f"flaps={len(run['flaps'])} -> "
                              f"{resp.status} (try {item['tries']})", flush=True)
                        submitted.add(key)
                        del resubmits[key]
                    except Exception as e:
                        print(f"    [resubmit] score={run['score']} "
                              f"flaps={len(run['flaps'])} "
                              f"try {item['tries']} failed: {e}", flush=True)
        except KeyboardInterrupt:
            print("\nstopped by user")
        except Exception as e:
            print(f"browser closed or error: {e}")


if __name__ == "__main__":
    main()

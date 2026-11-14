/* AutoFlappy in-page agent.
 *
 * Injected after window.__game exists. Wraps game.step so that after every
 * physics tick (120 Hz) the policy re-evaluates the state and calls
 * game.flap() when the Q-network prefers the flap action. Also auto-restarts
 * runs: gameover -> restart() -> getready -> flap().
 *
 * Feature layout MUST match env.py features().
 */
(() => {
  if (window.__agentInstalled) return;
  window.__agentInstalled = true;

  const SCALE = 1 << 20;
  const PIPE_HW = 136318;          // kf
  const BIRD_R = 71303;            // Ci
  const DEAD_WAIT_TICKS = 45;      // ~0.37 s on the gameover screen
  const FRAME_SKIP = 4;            // decide every 4 ticks (~30 Hz) — must
                                   // match FlappyEnv(frame_skip) in training

  // ---- tiny MLP forward (row-major weights, ReLU hidden) ----
  let net = [];
  const bufA = new Float64Array(256), bufB = new Float64Array(256);

  // hot-swappable: window.__agentSetPolicy(json) replaces the network
  window.__agentSetPolicy = (P) => {
    const layers = [];
    for (let li = 0; li < P.W.length; li++) {
      const W = P.W[li], nin = W.length, nout = W[0].length;
      const w = new Float64Array(nin * nout);
      for (let r = 0; r < nin; r++) {
        const row = W[r];
        for (let c = 0; c < nout; c++) w[r * nout + c] = row[c];
      }
      layers.push({ w, b: Float64Array.from(P.b[li]), nin, nout });
    }
    net = layers;
  };
  if (window.__agentPolicy) window.__agentSetPolicy(window.__agentPolicy);
  else { console.error("agent: no policy"); return; }

  function decide(feat) {
    let inp = feat, out = bufA, tmp;
    for (let li = 0; li < net.length; li++) {
      const { w, b, nin, nout } = net[li];
      out.fill(0, 0, nout);
      for (let r = 0; r < nin; r++) {
        const x = inp[r];
        if (x === 0) continue;
        const base = r * nout;
        for (let c = 0; c < nout; c++) out[c] += x * w[base + c];
      }
      for (let c = 0; c < nout; c++) {
        let v = out[c] + b[c];
        if (li < net.length - 1 && v < 0) v = 0;
        out[c] = v;
      }
      inp = out; out = (out === bufA ? bufB : bufA);
    }
    return inp[1] > inp[0] ? 1 : 0;   // 1 = flap
  }

  const feat = new Float64Array(8);
  function features(g) {
    const y = g.birdY / SCALE;
    feat[0] = y;
    feat[1] = g.birdVy / SCALE / 2.2;
    let p1 = null, p2 = null;
    const pipes = g.pipes;
    for (let i = 0; i < pipes.length; i++) {
      if (pipes[i].x + PIPE_HW + BIRD_R >= 0) { p1 = pipes[i]; p2 = pipes[i + 1] || null; break; }
    }
    if (p1) {
      feat[2] = p1.x / SCALE / 1.2;
      feat[3] = p1.gapY / SCALE - y;
      feat[4] = p1.halfGap / SCALE / 0.31;
    } else { feat[2] = 1.5; feat[3] = 0; feat[4] = 0; }
    if (p2) {
      feat[5] = p2.x / SCALE / 1.2;
      feat[6] = p2.gapY / SCALE - y;
      feat[7] = p2.halfGap / SCALE / 0.31;
    } else { feat[5] = 1.5; feat[6] = 0; feat[7] = 0; }
    return feat;
  }

  // ---- stats for the python driver ----
  const stats = window.__agentStats = { runs: 0, best: 0, score: 0, state: "", deaths: {}, net: "?", ranked: "?" };
  let deadTicks = 0, wasGameover = false, suicides = 0, abandoned = false;
  let lastFlapStep = -1e9;

  function tick(g) {
    const st = g.state;
    stats.state = st;
    stats.score = g.score;
    stats.net = g.netMode || "?";
    // getready -> flap() launches the run; the game records it as flap 0.
    // Anchor lastFlapStep so the FIRST real flap also respects min-gap —
    // otherwise every tape starts with a ~4-tick inhuman double-tap.
    if (st === "getready") { g.flap(); lastFlapStep = 0; return; }
    if (st === "gameover") {
      if (!wasGameover) {
        wasGameover = true;
        stats.runs++;
        if (g.score > stats.best) stats.best = g.score;
        const c = g.deathCause || "unknown";
        stats.deaths[c] = (stats.deaths[c] || 0) + 1;
        // snapshot the finished run so the driver can resubmit it if the
        // game's own submit_run_versioned fails (530s on big payloads)
        try {
          window.__lastRun = {
            seed: Number(g.currentSeed ?? g.runSeed ?? g.seed),
            simVersion: g.simVersion,
            flaps: Array.from(g.flapTrace || []),
            score: g.score,
            ranked: !!(g.runRanked ?? g.isRanked),
            reviveStep: (g.wasRevived && g.reviveStep != null) ? Number(g.reviveStep) : null,
          };
          window.__lastRun.key = `${window.__lastRun.seed}:${window.__lastRun.flaps.length}:${g.score}`;
        } catch (e) { console.error("agent snapshot", e); }
      }
      if (++deadTicks > DEAD_WAIT_TICKS) {
        deadTicks = 0;
        // window.__agentRevive: use the once-per-run revive instead of
        // restarting — run then submits via submit_run_revived
        if (window.__agentRevive && !g.wasRevived) g.revive();
        else g.restart();
      }
      return;
    }
    wasGameover = false;
    deadTicks = 0;
    // unranked runs can't submit — while online, let the bird fall so the
    // next run starts on a server seed (ranked is fixed at run start)
    const ranked = g.runRanked ?? g.isRanked;
    stats.ranked = ranked === true ? "yes" : ranked === false ? "no" : "?";
    if (ranked === false && g.netMode === "online" && suicides < 8) {
      if (!abandoned) { abandoned = true; suicides++; }
      return;
    }
    abandoned = false;
    // debug: window.__agentDieAt = N makes the agent stop flapping at score N
    if (typeof window.__agentDieAt === "number" && g.score >= window.__agentDieAt) return;
    // act only on 4-tick window boundaries; playStep is the index of the
    // tick that is about to run, so playStep % FRAME_SKIP === 0 marks it
    if (g.playStep % FRAME_SKIP !== 0) return;
    // window.__agentMinGap: min ticks between flaps. Humans can't click
    // faster than ~8-10 ticks (66-83 ms); the server rejects replays whose
    // flap tape contains physically-impossible double-taps, so enforce a
    // human-plausible minimum interval between recorded flaps.
    if (decide(features(g)) === 1) {
      // window.__agentVyMask: suppress flaps while the bird still has
      // strong upward velocity — a re-flap just resets vy to 1.4 (nearly
      // free to skip) and cutting them ~halves the flap count, which both
      // keeps the tape human-plausible and fits under the ~2000-flap cap
      const vm = window.__agentVyMask;
      if (typeof vm === "number" && g.birdVy / SCALE > vm) return;
      const mg = Math.max(0, Math.floor(window.__agentMinGap || 0));
      // playStep resets each run; a stale lastFlapStep would suppress
      // every flap after the first death
      if (g.playStep < lastFlapStep) lastFlapStep = -1e9;
      if (mg > 0 && g.playStep - lastFlapStep < mg) return;
      lastFlapStep = g.playStep;
      g.flap();
    }
  }

  // ---- hook (re-hook if the game object is ever replaced) ----
  function hook(g) {
    if (!g || g.__agentHooked) return;
    const orig = g.step;
    // window.__agentSpeed = N runs N physics ticks per scheduled tick.
    // The trajectory is tick-deterministic, so the submitted flap schedule
    // replays identically server-side — only wall-clock speed changes.
    g.step = function (dt) {
      let r = orig.call(this, dt);
      try { tick(this); } catch (e) { console.error("agent tick", e); }
      const n = Math.max(1, Math.floor(window.__agentSpeed || 1));
      for (let k = 1; k < n && this.state === "play"; k++) {
        r = orig.call(this, dt);
        try { tick(this); } catch (e) { console.error("agent tick", e); }
      }
      return r;
    };
    g.__agentHooked = true;
    console.log("agent: hooked game.step");
  }

  hook(window.__game);
  setInterval(() => hook(window.__game), 500);
})();

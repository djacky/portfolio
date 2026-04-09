"use client";

/* ------------------------------------------------------------------
   MagnetILCDemo — data-driven iterative learning control.

   Reproduces the spirit of Nicoletti et al., "Data-driven approach to
   iterative learning control via convex optimisation" (IET Control
   Theory & Appl., 2020, CERN preprint). A simulated magnet-current
   plant is driven through repetitive reference trials; between each
   trial, a convex QP (solved in-browser with FISTA) synthesises the
   next feedforward input from the previous trial's tracking error.
   No plant model is assumed — the lifted Toeplitz G used by the QP
   is built from a single identification impulse response.

   All physics / optimisation runs live in the browser; no data is
   precomputed. The user watches the error envelope collapse onto the
   reference over ~8-15 trials, and can tune the regularisation λ,
   input saturation u_max, measurement noise σ, and reference shape.
------------------------------------------------------------------ */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { motion } from "framer-motion";
import {
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  ExternalLink,
  Activity,
} from "lucide-react";

/* ================================================================
   plant & problem constants
================================================================ */

const N = 200; // horizon samples per trial
const TS_MS = 5; // sample time (ms)
const TRIAL_MS = N * TS_MS; // 1000ms per trial

// 2nd-order lightly-damped discrete plant with a NON-MINIMUM PHASE
// zero and a long transport delay.  All three are classic enemies
// of naive feedforward and classic wins for ILC:
//
//   • NMP zero at z ≈ 3.4 (outside unit circle) → step response
//     has INVERSE response (dips before rising); any causal model
//     inverse is unstable.  ILC sidesteps this by operating
//     non-causally across the full trial horizon.
//   • Poles at 0.92 ± 0.30j  (|r|=0.968, ~10Hz resonance, very
//     lightly damped — rings for ~150ms after every transient).
//   • 10-sample (50ms) pure transport delay — naive "u = y_ref"
//     lags visibly and has no way to compensate without preview,
//     which ILC gets for free from the previous trial.
//
//   y[k] = -a1·y[k-1] - a2·y[k-2] + b0·u[k-d] + b1·u[k-d-1]
//
// DC gain = (b0+b1)/(1+a1+a2) = 0.0964/0.0964 = 1.0.
const PLANT = {
  a1: -1.84,
  a2: 0.9364,
  b0: -0.04,      // negative lead → non-minimum phase zero at z ≈ 3.4
  b1: 0.1364,
  d: 10,          // 10 samples = 50ms transport delay
};

type Trial = {
  u: Float64Array;
  y: Float64Array;
  e: Float64Array;
  rms: number;
};

type RefKey = "trap" | "pnp" | "sine";

/* ================================================================
   reference trajectories
================================================================ */

// min-jerk smoothstep: 3t² − 2t³
const smoothstep = (t: number) => 3 * t * t - 2 * t * t * t;

function smoothedTrapezoid(): Float64Array {
  // Aggressive ramp — rise/fall in 20 samples (100ms) instead of 40.
  // Peak slope ≈ 7.5 A/ms, which slams hard against the plant's
  // ~10Hz resonance and NMP zero: exactly where naive feedforward
  // falls apart and ILC earns its keep.
  const r = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    if (k < 35) r[k] = 0;
    else if (k < 55) r[k] = 100 * smoothstep((k - 35) / 20);
    else if (k < 140) r[k] = 100;
    else if (k < 160) r[k] = 100 * (1 - smoothstep((k - 140) / 20));
    else r[k] = 0;
  }
  return r;
}

function pickAndPlace(): Float64Array {
  // Multi-segment "pick-and-place" profile, the bread and butter of
  // factory motion control: approach a setpoint, dwell, step up to
  // a second setpoint, dwell, return — each leg with a short
  // smoothstep ramp.  Every transition kicks the NMP zero and the
  // 10Hz resonance, so naive feedforward rings between every hold.
  //
  //   seg 1: 0   → 70A    (samples 20–35,  75ms ramp)
  //   dwell: 70A            (samples 35–65)
  //   seg 2: 70  → 110A   (samples 65–80,  75ms ramp)
  //   dwell: 110A           (samples 80–115)
  //   seg 3: 110 → 40A    (samples 115–135, 100ms ramp)
  //   dwell: 40A            (samples 135–165)
  //   seg 4: 40  → 0      (samples 165–180, 75ms ramp)
  const segs: [number, number, number, number][] = [
    [20, 35, 0, 70],
    [35, 65, 70, 70],
    [65, 80, 70, 110],
    [80, 115, 110, 110],
    [115, 135, 110, 40],
    [135, 165, 40, 40],
    [165, 180, 40, 0],
  ];
  const r = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    let v = 0;
    for (const [k0, k1, a, b] of segs) {
      if (k >= k0 && k < k1) {
        if (a === b) v = a;
        else {
          const t = (k - k0) / (k1 - k0);
          v = a + (b - a) * smoothstep(t);
        }
        break;
      }
    }
    if (k >= 180) v = 0;
    r[k] = v;
  }
  return r;
}

function sineSweep(): Float64Array {
  // Chirp centered at 50A, peak ±40A, 1→4 Hz sweep
  const r = new Float64Array(N);
  const T = TRIAL_MS / 1000;
  for (let k = 0; k < N; k++) {
    const t = (k / (N - 1)) * T;
    const f = 1 + 3 * (t / T);
    r[k] = 50 + 40 * Math.sin(2 * Math.PI * f * t);
  }
  return r;
}

/* ================================================================
   plant simulation + lifted matrix construction
================================================================ */

function simulatePlant(u: Float64Array, noiseStd: number): Float64Array {
  // Keep the noise-free state separate from the measured output.
  // Writing noise back into y[k] would let the AR(2) dynamics treat
  // it as PROCESS noise — the resonance then colours and amplifies
  // it massively (σ_y ≈ 3·σ for this plant), which is not what a
  // measurement-noise model means and not what the paper assumes.
  const yTrue = new Float64Array(N);
  const yMeas = new Float64Array(N);
  const { a1, a2, b0, b1, d } = PLANT;
  for (let k = 0; k < N; k++) {
    const uk = k - d >= 0 ? u[k - d] : 0;
    const uk1 = k - d - 1 >= 0 ? u[k - d - 1] : 0;
    const yk1 = k >= 1 ? yTrue[k - 1] : 0;
    const yk2 = k >= 2 ? yTrue[k - 2] : 0;
    yTrue[k] = -a1 * yk1 - a2 * yk2 + b0 * uk + b1 * uk1;
    if (noiseStd > 0) {
      // Gaussian via Box-Muller, added only to the measurement.
      const r =
        Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) *
        Math.cos(2 * Math.PI * Math.random());
      yMeas[k] = yTrue[k] + noiseStd * r;
    } else {
      yMeas[k] = yTrue[k];
    }
  }
  return yMeas;
}

// Build the lifted impulse-response Toeplitz matrix G (lower tri.)
// so that y = G·u holds for the noise-free plant.  This is the
// "data-driven" piece: in a real setup it would come from a single
// identification trial, not from the model.
function buildG(): Float64Array {
  const imp = new Float64Array(N);
  imp[0] = 1;
  const h = simulatePlant(imp, 0);
  const G = new Float64Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j <= i; j++) {
      G[i * N + j] = h[i - j];
    }
  }
  return G;
}

function gMul(G: Float64Array, u: Float64Array): Float64Array {
  const y = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    const rowBase = i * N;
    for (let j = 0; j <= i; j++) s += G[rowBase + j] * u[j];
    y[i] = s;
  }
  return y;
}

function gTMul(G: Float64Array, e: Float64Array): Float64Array {
  const v = new Float64Array(N);
  for (let j = 0; j < N; j++) {
    let s = 0;
    for (let i = j; i < N; i++) s += G[i * N + j] * e[i];
    v[j] = s;
  }
  return v;
}

/* ================================================================
   ILC update — convex QP solved with FISTA

     minimise   ‖G·Δu − e_prev‖² + λ ‖Δu‖²
     subject to |u_prev[k] + Δu[k]| ≤ u_max
     then       u_next = u_prev + Δu

   e_prev = y_ref − y_prev is the MEASURED tracking error of the
   previous trial — this is what makes the scheme data-driven and
   what lets it reject repeatable plant/model mismatch.  Using the
   lifted predictor ŷ_j = y_{j-1} + G·Δu, demanding ŷ_j = y_ref
   gives G·Δu = e_{j-1}, which is the least-squares term above.
   Box-constrained LS is Lipschitz-smooth with L ≤ ‖G‖_F² + λ, so
   a fixed step η = 1/L suffices for FISTA convergence.
================================================================ */

function ilcSolve(
  G: Float64Array,
  ePrev: Float64Array,
  uPrev: Float64Array,
  lambda: number,
  uMax: number,
  gFrob2: number,
  iters = 140,
): Float64Array {
  const L = gFrob2 + lambda;
  const eta = 1 / L;

  // Q-filter the measured error: only try to cancel the part that
  // sits inside the plant's learnable bandwidth.  Asking the QP to
  // invert HF measurement noise is how ILC blows up.
  const eFilt = qFilter(ePrev);

  // Decision variable is Δu, initialised at 0.
  let xPrev = new Float64Array(N);
  let yk = new Float64Array(N);
  let t = 1;

  for (let it = 0; it < iters; it++) {
    // residual = G·Δu − e_prev
    const resid = gMul(G, yk);
    for (let k = 0; k < N; k++) resid[k] -= eFilt[k];

    // grad = Gᵀ·resid + λ·Δu
    const grad = gTMul(G, resid);
    for (let k = 0; k < N; k++) grad[k] += lambda * yk[k];

    // gradient step + box projection on u_prev + Δu
    const xNew = new Float64Array(N);
    for (let k = 0; k < N; k++) {
      let v = yk[k] - eta * grad[k];
      const uCand = uPrev[k] + v;
      if (uCand > uMax) v = uMax - uPrev[k];
      else if (uCand < -uMax) v = -uMax - uPrev[k];
      xNew[k] = v;
    }

    // Nesterov momentum
    const tNew = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
    const beta = (t - 1) / tNew;
    for (let k = 0; k < N; k++) {
      yk[k] = xNew[k] + beta * (xNew[k] - xPrev[k]);
    }
    xPrev = xNew;
    t = tNew;
  }
  // Second Q-filter pass on Δu itself — kills any remaining HF
  // component the penalty didn't fully suppress — then commit.
  const dU = qFilter(xPrev);
  const uNext = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    let v = uPrev[k] + dU[k];
    if (v > uMax) v = uMax;
    else if (v < -uMax) v = -uMax;
    uNext[k] = v;
  }
  return uNext;
}

/* Zero-phase Q-filter: symmetric Hann FIR.  The cutoff must sit
   below where |G(e^jω)|⁻¹ explodes.  For this plant |G⁻¹| reaches
   ~116 at Nyquist, so a Q with stopband starting around 6–8Hz is
   required.  M=31 Hann at fs=200Hz rolls off past ~4Hz and is deep
   in the stopband by 10Hz — comfortably below the 9Hz resonance
   where inversion starts to blow up, yet still above the slowest
   reference harmonics (trapezoid rise energy below 3Hz, 1–4Hz
   chirp).  This is the Q(z) block the paper relies on to make the
   learning contraction robust to measurement noise. */
const Q_TAPS = (() => {
  const M = 17;
  const t = new Float64Array(M);
  let s = 0;
  for (let i = 0; i < M; i++) {
    t[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (M - 1)));
    s += t[i];
  }
  for (let i = 0; i < M; i++) t[i] /= s;
  return t;
})();

function qFilter(x: Float64Array): Float64Array {
  const M = Q_TAPS.length;
  const half = (M - 1) >> 1;
  const y = new Float64Array(x.length);
  for (let k = 0; k < x.length; k++) {
    let s = 0;
    for (let m = 0; m < M; m++) {
      const idx = k + m - half;
      const xi = idx < 0 ? x[0] : idx >= x.length ? x[x.length - 1] : x[idx];
      s += Q_TAPS[m] * xi;
    }
    y[k] = s;
  }
  return y;
}

function rms(e: Float64Array): number {
  let s = 0;
  for (let k = 0; k < e.length; k++) s += e[k] * e[k];
  return Math.sqrt(s / e.length);
}

/* ================================================================
   main component
================================================================ */

const REF_OPTIONS: { key: RefKey; label: string }[] = [
  { key: "trap", label: "Smoothed trapezoid" },
  { key: "pnp", label: "Pick-and-place" },
  { key: "sine", label: "Sine sweep (chirp)" },
];

const MAX_TRIALS_CAP = 200;

export default function MagnetILCDemo() {
  // ---- user knobs ----
  const [refKey, setRefKey] = useState<RefKey>("trap");
  const [lambda, setLambda] = useState(0.3);
  const [uMax, setUMax] = useState(400);
  const [noiseStd, setNoiseStd] = useState(0.6);
  const [maxTrials, setMaxTrials] = useState(15);

  // ---- run state ----
  const [trials, setTrials] = useState<Trial[]>([]);
  const [autoRun, setAutoRun] = useState(false);
  const [sweepFrac, setSweepFrac] = useState(1); // 0..1 animating newest trial
  const [playing, setPlaying] = useState(false);

  // ---- precomputed plant operators ----
  const G = useMemo(() => buildG(), []);
  const gFrob2 = useMemo(() => {
    let s = 0;
    for (let k = 0; k < G.length; k++) s += G[k] * G[k];
    return s;
  }, [G]);

  // ---- reference ----
  const yRef = useMemo(() => {
    switch (refKey) {
      case "trap":
        return smoothedTrapezoid();
      case "pnp":
        return pickAndPlace();
      case "sine":
        return sineSweep();
    }
  }, [refKey]);

  // ---- reset trials whenever anything that invalidates them changes ----
  useEffect(() => {
    setTrials([]);
    setAutoRun(false);
    setPlaying(false);
    setSweepFrac(1);
  }, [refKey]);

  // ---- run a trial ----
  const runTrial = useCallback(() => {
    setTrials((prev) => {
      if (prev.length >= maxTrials) return prev;
      let uNext: Float64Array;
      if (prev.length === 0) {
        // Trial 0: naive open-loop feedforward (u = y_ref clipped).
        // Deliberately the dumbest possible controller so ILC has
        // something dramatic to improve on.
        uNext = new Float64Array(yRef);
        for (let k = 0; k < N; k++) {
          if (uNext[k] > uMax) uNext[k] = uMax;
          else if (uNext[k] < -uMax) uNext[k] = -uMax;
        }
      } else {
        const last = prev[prev.length - 1];
        uNext = ilcSolve(G, last.e, last.u, lambda, uMax, gFrob2);
      }
      const yNext = simulatePlant(uNext, noiseStd);
      const eNext = new Float64Array(N);
      for (let k = 0; k < N; k++) eNext[k] = yRef[k] - yNext[k];
      return [
        ...prev,
        { u: uNext, y: yNext, e: eNext, rms: rms(eNext) },
      ];
    });
    setSweepFrac(0);
    setPlaying(true);
  }, [G, yRef, lambda, uMax, noiseStd, gFrob2, maxTrials]);

  // ---- sweep animation ----
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const start = performance.now();
    const duration = 620;
    const tick = () => {
      const f = Math.min(1, (performance.now() - start) / duration);
      setSweepFrac(f);
      if (f < 1) raf = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  // ---- auto-run loop ----
  useEffect(() => {
    if (!autoRun || playing) return;
    if (trials.length >= maxTrials) {
      setAutoRun(false);
      return;
    }
    const id = window.setTimeout(() => runTrial(), 180);
    return () => window.clearTimeout(id);
  }, [autoRun, playing, trials.length, runTrial]);

  const reset = () => {
    setTrials([]);
    setAutoRun(false);
    setPlaying(false);
    setSweepFrac(1);
  };

  // ---- derived metrics ----
  const latest = trials.length > 0 ? trials[trials.length - 1] : null;
  const initialRms = trials.length > 0 ? trials[0].rms : 0;
  const improvement =
    latest && initialRms > 1e-9
      ? Math.max(0, 1 - latest.rms / initialRms) * 100
      : 0;
  const converged = trials.length >= 6 && latest !== null && latest.rms < 0.8;

  /* ================================================================
     rendering
  ================================================================ */

  return (
    <div
      id="demo-ilc"
      className="glass rounded-3xl p-6 md:p-8 scroll-mt-24"
    >
      {/* ---- header ---- */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-5">
        <div>
          <div className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.22em] text-[#fbbf24]">
            <Activity className="w-3.5 h-3.5" />
            CERN · 2020 · IET Control Theory &amp; Applications
          </div>
          <h3 className="mt-2 text-2xl md:text-3xl font-semibold text-gradient">
            Magnet Current Ramp — Iterative Learning Control
          </h3>
          <p className="mt-2 text-sm text-gray-400 max-w-2xl leading-relaxed">
            A simulated accelerator magnet executes the same current
            profile every trial. Between trials a convex QP (solved
            in-browser via FISTA) synthesises the next feedforward input
            from the previous error. No plant model is assumed — the QP
            uses only a single identification impulse response.
          </p>
          <a
            href="https://cds.cern.ch/record/2799372"
            target="_blank"
            rel="noreferrer"
            className="mt-2 inline-flex items-center gap-1 text-[11px] font-mono uppercase tracking-wider text-[#fbbf24] hover:text-amber-300 transition-colors"
          >
            Read the paper <ExternalLink className="w-3 h-3" />
          </a>
        </div>
        <TrialBadge
          trials={trials.length}
          latestRms={latest?.rms ?? null}
          improvement={improvement}
          converged={converged}
        />
      </div>

      {/* ---- magnet schematic strip ---- */}
      <MagnetSchematic
        sweepFrac={sweepFrac}
        playing={playing}
        current={latest?.y ?? null}
        converged={converged}
      />

      {/* ---- main grid ---- */}
      <div className="mt-5 grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-5">
        {/* LEFT: charts stack */}
        <div className="flex flex-col gap-4">
          <ChartCard
            title="Tracking — measured output vs reference"
            subtitle="Each trial's y(k) overlaid; fresh trial animates left→right"
            height={260}
          >
            {({ w, h }) => (
              <TrackingChart
                w={w}
                h={h}
                yRef={yRef}
                trials={trials}
                sweepFrac={sweepFrac}
                converged={converged}
              />
            )}
          </ChartCard>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ChartCard
              title="Tracking error e(k) = y_ref − y"
              subtitle="Envelope collapses toward zero"
              height={180}
            >
              {({ w, h }) => (
                <ErrorChart
                  w={w}
                  h={h}
                  trials={trials}
                  converged={converged}
                />
              )}
            </ChartCard>

            <ChartCard
              title="Learned feedforward u(k)"
              subtitle="What the QP synthesises"
              height={180}
            >
              {({ w, h }) => (
                <FeedforwardChart
                  w={w}
                  h={h}
                  trials={trials}
                  yRef={yRef}
                  uMax={uMax}
                  converged={converged}
                />
              )}
            </ChartCard>
          </div>

          {/* Plant transfer function card */}
          <PlantTransferFunctionCard />
        </div>

        {/* RIGHT: convergence + controls */}
        <div className="flex flex-col gap-4">
          <ChartCard
            title="Convergence"
            subtitle="‖e‖₂ per trial (log)"
            height={180}
          >
            {({ w, h }) => (
              <ConvergenceChart
                w={w}
                h={h}
                trials={trials}
                converged={converged}
              />
            )}
          </ChartCard>

          <div className="rounded-2xl border border-white/10 bg-black/20 p-4 flex flex-col gap-4">
            {/* run controls */}
            <div className="flex gap-2">
              <button
                onClick={runTrial}
                disabled={playing || trials.length >= maxTrials}
                className="flex-1 inline-flex items-center justify-center gap-2 rounded-lg bg-[#22d3ee]/15 border border-[#22d3ee]/40 text-[#22d3ee] text-xs font-medium py-2 hover:bg-[#22d3ee]/25 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <SkipForward className="w-3.5 h-3.5" />
                Next trial
              </button>
              <button
                onClick={() => setAutoRun((v) => !v)}
                disabled={trials.length >= maxTrials}
                className={`flex-1 inline-flex items-center justify-center gap-2 rounded-lg border text-xs font-medium py-2 transition-colors disabled:opacity-40 ${
                  autoRun
                    ? "bg-[#fbbf24]/20 border-[#fbbf24]/50 text-[#fbbf24] hover:bg-[#fbbf24]/30"
                    : "bg-white/5 border-white/15 text-gray-300 hover:bg-white/10"
                }`}
              >
                {autoRun ? (
                  <Pause className="w-3.5 h-3.5" />
                ) : (
                  <Play className="w-3.5 h-3.5" />
                )}
                {autoRun ? "Pause" : "Auto-run"}
              </button>
              <button
                onClick={reset}
                className="inline-flex items-center justify-center rounded-lg border border-white/15 bg-white/5 text-gray-300 px-3 py-2 hover:bg-white/10 transition-colors"
                aria-label="Reset"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* trial count */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500">
                  Max trials
                </div>
                <div className="text-[10px] font-mono text-[#22d3ee]">
                  {maxTrials}
                </div>
              </div>
              <input
                type="number"
                min={1}
                max={MAX_TRIALS_CAP}
                step={1}
                value={maxTrials}
                onChange={(e) => {
                  const n = Math.max(
                    1,
                    Math.min(MAX_TRIALS_CAP, Math.floor(Number(e.target.value) || 0)),
                  );
                  setMaxTrials(n);
                }}
                className="w-full rounded-md bg-white/5 border border-white/10 px-2 py-1.5 text-[11px] font-mono text-gray-200 focus:outline-none focus:border-[#22d3ee]/50"
              />
              <div className="text-[9px] text-gray-500 mt-1">
                Cap on how many ILC iterations will run (1–{MAX_TRIALS_CAP})
              </div>
            </div>

            {/* reference selector */}
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-1.5">
                Reference profile
              </div>
              <div className="flex flex-col gap-1">
                {REF_OPTIONS.map((opt) => (
                  <button
                    key={opt.key}
                    onClick={() => setRefKey(opt.key)}
                    className={`text-left text-[11px] font-medium rounded-md px-2.5 py-1.5 border transition-colors ${
                      refKey === opt.key
                        ? "bg-[#22d3ee]/15 border-[#22d3ee]/40 text-[#22d3ee]"
                        : "bg-white/5 border-white/10 text-gray-400 hover:bg-white/10"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* sliders */}
            <Slider
              label="Regularisation λ"
              value={lambda}
              min={0.005}
              max={2}
              step={0.005}
              log
              onChange={setLambda}
              format={(v) => v.toFixed(3)}
              help="Penalises change from previous u"
            />
            <Slider
              label="Input constraint |u|ₘₐₓ"
              value={uMax}
              min={110}
              max={800}
              step={10}
              onChange={setUMax}
              format={(v) => `${v.toFixed(0)} A`}
              help="Hard box constraint in the QP"
            />
            <Slider
              label="Measurement noise σ"
              value={noiseStd}
              min={0}
              max={3}
              step={0.05}
              onChange={setNoiseStd}
              format={(v) => `${v.toFixed(2)} A`}
              help="Additive Gaussian on y"
            />
          </div>

          {/* footnote */}
          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 leading-relaxed">
            Plant: 2nd-order underdamped + 30 ms delay · {N} samples ·{" "}
            {TS_MS} ms sample time · FISTA QP · 140 inner iterations
          </div>
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   trial badge
================================================================ */

function TrialBadge({
  trials,
  latestRms,
  improvement,
  converged,
}: {
  trials: number;
  latestRms: number | null;
  improvement: number;
  converged: boolean;
}) {
  return (
    <div
      className="flex items-center gap-4 rounded-2xl border px-4 py-3 shrink-0"
      style={{
        background: converged
          ? "linear-gradient(135deg, rgba(251,191,36,0.15), rgba(251,191,36,0.02))"
          : "linear-gradient(135deg, rgba(34,211,238,0.10), rgba(34,211,238,0.02))",
        borderColor: converged
          ? "rgba(251,191,36,0.45)"
          : "rgba(34,211,238,0.35)",
        boxShadow: converged
          ? "0 0 26px rgba(251,191,36,0.22), inset 0 0 18px rgba(251,191,36,0.08)"
          : "0 0 20px rgba(34,211,238,0.15)",
      }}
    >
      <div>
        <div className="text-[9px] font-mono uppercase tracking-[0.2em] text-gray-400">
          Trial
        </div>
        <div className="text-2xl font-semibold tabular-nums text-white leading-none mt-0.5">
          {trials}
        </div>
      </div>
      <div className="h-9 w-px bg-white/10" />
      <div>
        <div className="text-[9px] font-mono uppercase tracking-[0.2em] text-gray-400">
          RMS error
        </div>
        <div
          className="text-lg font-semibold tabular-nums leading-none mt-0.5"
          style={{ color: converged ? "#fbbf24" : "#22d3ee" }}
        >
          {latestRms !== null ? latestRms.toFixed(3) : "—"}
        </div>
      </div>
      <div className="h-9 w-px bg-white/10" />
      <div>
        <div className="text-[9px] font-mono uppercase tracking-[0.2em] text-gray-400">
          Improvement
        </div>
        <div className="text-lg font-semibold tabular-nums text-white leading-none mt-0.5">
          {improvement > 0 ? `${improvement.toFixed(1)}%` : "—"}
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   magnet schematic (SVG)
================================================================ */

function MagnetSchematic({
  sweepFrac,
  playing,
  current,
  converged,
}: {
  sweepFrac: number;
  playing: boolean;
  current: Float64Array | null;
  converged: boolean;
}) {
  const pulseX = 30 + sweepFrac * 540;
  const accent = converged ? "#fbbf24" : "#22d3ee";
  // Ambient idle beam — slow drift left→right independent of trials
  const [ambient, setAmbient] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const tick = () => {
      setAmbient(((performance.now() - start) / 3500) % 1);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  const ambX = 30 + ambient * 540;

  return (
    <div className="relative rounded-2xl border border-white/10 bg-black/30 overflow-hidden">
      <svg viewBox="0 0 600 80" className="w-full h-[88px] block">
        <defs>
          <linearGradient id="beamGlow" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={accent} stopOpacity="0" />
            <stop offset="50%" stopColor={accent} stopOpacity="0.85" />
            <stop offset="100%" stopColor={accent} stopOpacity="0" />
          </linearGradient>
          <radialGradient id="pulseGlow">
            <stop offset="0%" stopColor={accent} stopOpacity="1" />
            <stop offset="60%" stopColor={accent} stopOpacity="0.3" />
            <stop offset="100%" stopColor={accent} stopOpacity="0" />
          </radialGradient>
          <linearGradient id="magnetBody" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1f2937" />
            <stop offset="100%" stopColor="#0b0f1a" />
          </linearGradient>
        </defs>

        {/* mounting rails */}
        <line
          x1="20"
          y1="12"
          x2="580"
          y2="12"
          stroke="#1f2937"
          strokeWidth="1"
        />
        <line
          x1="20"
          y1="68"
          x2="580"
          y2="68"
          stroke="#1f2937"
          strokeWidth="1"
        />

        {/* beam tube base */}
        <line
          x1="20"
          y1="40"
          x2="580"
          y2="40"
          stroke="#334155"
          strokeWidth="4"
        />
        <line
          x1="20"
          y1="40"
          x2="580"
          y2="40"
          stroke="url(#beamGlow)"
          strokeWidth="2"
          opacity="0.8"
        />

        {/* dipole magnets — three along the tube */}
        {[130, 300, 470].map((cx) => (
          <g key={cx}>
            {/* upper pole */}
            <rect
              x={cx - 40}
              y="20"
              width="80"
              height="14"
              rx="2"
              fill="url(#magnetBody)"
              stroke="#475569"
              strokeWidth="0.8"
            />
            <text
              x={cx}
              y="31"
              textAnchor="middle"
              fontSize="9"
              fontFamily="monospace"
              fill="#64748b"
            >
              N
            </text>
            {/* lower pole */}
            <rect
              x={cx - 40}
              y="46"
              width="80"
              height="14"
              rx="2"
              fill="url(#magnetBody)"
              stroke="#475569"
              strokeWidth="0.8"
            />
            <text
              x={cx}
              y="57"
              textAnchor="middle"
              fontSize="9"
              fontFamily="monospace"
              fill="#64748b"
            >
              S
            </text>
            {/* coil winding hint lines on the sides */}
            {[-36, -24, -12, 12, 24, 36].map((dx) => (
              <g key={dx}>
                <line
                  x1={cx + dx}
                  y1="20"
                  x2={cx + dx}
                  y2="34"
                  stroke="#64748b"
                  strokeWidth="0.4"
                  opacity="0.5"
                />
                <line
                  x1={cx + dx}
                  y1="46"
                  x2={cx + dx}
                  y2="60"
                  stroke="#64748b"
                  strokeWidth="0.4"
                  opacity="0.5"
                />
              </g>
            ))}
          </g>
        ))}

        {/* ambient idle pulse — always drifting */}
        <circle
          cx={ambX}
          cy="40"
          r="14"
          fill="url(#pulseGlow)"
          opacity="0.35"
        />
        <circle cx={ambX} cy="40" r="2" fill={accent} opacity="0.7" />

        {/* trial playback pulse — only while sweeping */}
        {playing && (
          <>
            <circle cx={pulseX} cy="40" r="22" fill="url(#pulseGlow)" />
            <circle cx={pulseX} cy="40" r="4" fill={accent} />
            <line
              x1={pulseX}
              y1="10"
              x2={pulseX}
              y2="70"
              stroke={accent}
              strokeWidth="0.5"
              opacity="0.35"
              strokeDasharray="2 3"
            />
          </>
        )}
      </svg>

      {/* corner label */}
      <div className="absolute top-2 left-3 text-[9px] font-mono uppercase tracking-[0.18em] text-gray-500">
        Dipole string · beam axis
      </div>
      <div
        className="absolute top-2 right-3 text-[9px] font-mono uppercase tracking-[0.18em]"
        style={{ color: accent }}
      >
        {playing ? "▶ cycle running" : "○ idle"}
      </div>
    </div>
  );
}

/* ================================================================
   chart card wrapper — handles sizing + hi-dpi
================================================================ */

/* =================================================================
   Plant transfer function card — LaTeX-style fraction rendering
================================================================= */
function PlantTransferFunctionCard() {
  const { a1, a2, b0, b1, d } = PLANT;
  const fmt = (v: number) => {
    const s = v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
    return s;
  };
  // Denominator: 1 + a1 z^-1 + a2 z^-2  (display signs explicitly)
  const a1Abs = fmt(Math.abs(a1));
  const a2Abs = fmt(Math.abs(a2));
  const a1Sign = a1 >= 0 ? "+" : "−";
  const a2Sign = a2 >= 0 ? "+" : "−";
  // Numerator: z^-d (b0 + b1 z^-1)
  const b0Str = (b0 < 0 ? "−" : "") + fmt(Math.abs(b0));
  const b1Abs = fmt(Math.abs(b1));
  const b1Sign = b1 >= 0 ? "+" : "−";

  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-[10px] font-mono uppercase tracking-wider text-gray-400">
          Linearized Model
        </div>
        <div className="text-[9px] font-mono text-gray-500">
          2nd-order · NMP zero · 50 ms delay
        </div>
      </div>

      <div className="flex items-center justify-center gap-3 py-2 font-serif text-gray-100 select-none">
        {/* H(z) = */}
        <div className="text-xl italic">
          H<span className="not-italic">(</span>z<span className="not-italic">)</span>
          <span className="not-italic mx-2">=</span>
        </div>

        {/* z^{-d} factor */}
        <div className="text-xl italic">
          z<sup className="text-[0.65em] not-italic">−{d}</sup>
        </div>

        <div className="text-2xl not-italic text-gray-400">·</div>

        {/* Fraction */}
        <div className="inline-flex flex-col items-center leading-tight">
          {/* numerator */}
          <div className="text-[17px] whitespace-nowrap px-3 pb-1">
            <span className="not-italic">{b0Str}</span>
            <span className="not-italic mx-1.5">{b1Sign}</span>
            <span className="not-italic">{b1Abs}</span>
            <span className="italic ml-1">
              z<sup className="text-[0.65em] not-italic">−1</sup>
            </span>
          </div>
          {/* fraction bar */}
          <div className="h-px w-full bg-gray-300/80" />
          {/* denominator */}
          <div className="text-[17px] whitespace-nowrap px-3 pt-1">
            <span className="not-italic">1</span>
            <span className="not-italic mx-1.5">{a1Sign}</span>
            <span className="not-italic">{a1Abs}</span>
            <span className="italic ml-1">
              z<sup className="text-[0.65em] not-italic">−1</sup>
            </span>
            <span className="not-italic mx-1.5">{a2Sign}</span>
            <span className="not-italic">{a2Abs}</span>
            <span className="italic ml-1">
              z<sup className="text-[0.65em] not-italic">−2</sup>
            </span>
          </div>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2 text-[10px] font-mono text-gray-400">
        <div>
          <span className="text-gray-500">poles</span>{" "}
          <span className="text-[#22d3ee]">0.92 ± 0.30j</span>
        </div>
        <div>
          <span className="text-gray-500">zero</span>{" "}
          <span className="text-[#fbbf24]">z ≈ 3.4 (NMP)</span>
        </div>
        <div>
          <span className="text-gray-500">DC gain</span>{" "}
          <span className="text-gray-200">1.00</span>
        </div>
      </div>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  height,
  children,
}: {
  title: string;
  subtitle?: string;
  height: number;
  children: (size: { w: number; h: number }) => React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);

  useLayoutEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  return (
    <div className="rounded-2xl border border-white/10 bg-black/25 p-3 overflow-hidden">
      <div className="flex items-baseline justify-between mb-2 px-1">
        <div className="text-[11px] font-mono uppercase tracking-wider text-gray-300">
          {title}
        </div>
        {subtitle && (
          <div className="text-[9px] font-mono uppercase tracking-wider text-gray-500 hidden md:block">
            {subtitle}
          </div>
        )}
      </div>
      <div ref={ref} style={{ width: "100%", height }}>
        {w > 0 && children({ w, h: height })}
      </div>
    </div>
  );
}

/* ================================================================
   canvas helper
================================================================ */

function useCanvas(
  w: number,
  h: number,
  draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void,
  deps: React.DependencyList,
) {
  const ref = useRef<HTMLCanvasElement>(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.floor(w * dpr);
    c.height = Math.floor(h * dpr);
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    draw(ctx, w, h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w, h, ...deps]);
  return ref;
}

// y range helper
function boundsOf(
  series: Float64Array[],
  pad = 0.12,
): { lo: number; hi: number } {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    for (let k = 0; k < s.length; k++) {
      if (s[k] < lo) lo = s[k];
      if (s[k] > hi) hi = s[k];
    }
  }
  if (!isFinite(lo) || !isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  if (hi - lo < 1e-6) {
    hi = lo + 1;
  }
  const span = hi - lo;
  return { lo: lo - span * pad, hi: hi + span * pad };
}

function drawAxes(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  padL: number,
  padR: number,
  padT: number,
  padB: number,
  yLo: number,
  yHi: number,
  yLabel: string,
  xLabel: string,
  yTicks = 4,
) {
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(148,163,184,0.7)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "middle";

  // y gridlines + ticks
  for (let t = 0; t <= yTicks; t++) {
    const frac = t / yTicks;
    const yVal = yHi - frac * (yHi - yLo);
    const y = padT + frac * (h - padT - padB);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(yVal.toFixed(0), padL - 4, y);
  }

  // frame
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.strokeRect(padL, padT, w - padL - padR, h - padT - padB);

  // labels
  ctx.fillStyle = "rgba(148,163,184,0.6)";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(yLabel, padL + 4, padT + 4);
  ctx.textAlign = "right";
  ctx.fillText(xLabel, w - padR - 4, h - padB - 14);
}

/* ================================================================
   tracking chart — the hero view
================================================================ */

function trialColor(
  i: number,
  total: number,
  converged: boolean,
): string {
  // Gradient from violet (old) → cyan (new), flips to gold on converge
  if (converged && i === total - 1) return "#fbbf24";
  const t = total > 1 ? i / (total - 1) : 1;
  // violet #7c5cff  →  cyan #22d3ee
  const r = Math.round(0x7c + (0x22 - 0x7c) * t);
  const g = Math.round(0x5c + (0xd3 - 0x5c) * t);
  const b = Math.round(0xff + (0xee - 0xff) * t);
  return `rgb(${r},${g},${b})`;
}

function TrackingChart({
  w,
  h,
  yRef,
  trials,
  sweepFrac,
  converged,
}: {
  w: number;
  h: number;
  yRef: Float64Array;
  trials: Trial[];
  sweepFrac: number;
  converged: boolean;
}) {
  const canvasRef = useCanvas(
    w,
    h,
    (ctx) => {
      const padL = 38,
        padR = 12,
        padT = 8,
        padB = 22;
      const ys: Float64Array[] = [yRef, ...trials.map((t) => t.y)];
      const { lo, hi } = boundsOf(ys, 0.15);
      drawAxes(
        ctx,
        w,
        h,
        padL,
        padR,
        padT,
        padB,
        lo,
        hi,
        "current (A)",
        `time (ms)  →  ${TRIAL_MS}`,
      );

      const plotW = w - padL - padR;
      const plotH = h - padT - padB;
      const xAt = (k: number) => padL + (k / (N - 1)) * plotW;
      const yAt = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

      // reference (bold dashed white)
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.lineWidth = 2.2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(xAt(0), yAt(yRef[0]));
      for (let k = 1; k < N; k++) ctx.lineTo(xAt(k), yAt(yRef[k]));
      ctx.stroke();
      ctx.setLineDash([]);

      // prior trials — stacked with opacity ramp
      const total = trials.length;
      for (let i = 0; i < total - 1; i++) {
        const y = trials[i].y;
        const ageT = total > 1 ? i / (total - 1) : 1;
        const opacity = 0.08 + 0.22 * ageT;
        ctx.strokeStyle = trialColor(i, total, false);
        ctx.globalAlpha = opacity;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(y[0]));
        for (let k = 1; k < N; k++) ctx.lineTo(xAt(k), yAt(y[k]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // current trial — bright, animated left→right
      if (total > 0) {
        const cur = trials[total - 1].y;
        const kEnd = Math.max(
          1,
          Math.min(N, Math.floor(sweepFrac * N) + 1),
        );
        // glow pass
        ctx.strokeStyle = trialColor(total - 1, total, converged);
        ctx.shadowColor = trialColor(total - 1, total, converged);
        ctx.shadowBlur = 14;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(cur[0]));
        for (let k = 1; k < kEnd; k++) ctx.lineTo(xAt(k), yAt(cur[k]));
        ctx.stroke();
        ctx.shadowBlur = 0;

        // playhead during sweep
        if (sweepFrac < 1) {
          const px = xAt(kEnd - 1);
          const py = yAt(cur[kEnd - 1]);
          ctx.strokeStyle = "rgba(255,255,255,0.35)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(px, padT);
          ctx.lineTo(px, h - padB);
          ctx.stroke();
          ctx.fillStyle = trialColor(total - 1, total, converged);
          ctx.beginPath();
          ctx.arc(px, py, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // legend
      ctx.font = "10px ui-monospace, monospace";
      ctx.textBaseline = "alphabetic";
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.fillText("— — reference", w - padR - 150, padT + 12);
      ctx.fillStyle = converged ? "#fbbf24" : "#22d3ee";
      ctx.fillText("■ current trial", w - padR - 150, padT + 26);
    },
    [yRef, trials, sweepFrac, converged],
  );
  return <canvas ref={canvasRef} />;
}

/* ================================================================
   error envelope chart
================================================================ */

function ErrorChart({
  w,
  h,
  trials,
  converged,
}: {
  w: number;
  h: number;
  trials: Trial[];
  converged: boolean;
}) {
  const canvasRef = useCanvas(
    w,
    h,
    (ctx) => {
      const padL = 34,
        padR = 10,
        padT = 8,
        padB = 20;

      let maxAbs = 1;
      for (const t of trials)
        for (let k = 0; k < N; k++)
          if (Math.abs(t.e[k]) > maxAbs) maxAbs = Math.abs(t.e[k]);

      const lo = -maxAbs;
      const hi = maxAbs;
      drawAxes(
        ctx,
        w,
        h,
        padL,
        padR,
        padT,
        padB,
        lo,
        hi,
        "error (A)",
        "time",
        4,
      );

      const plotW = w - padL - padR;
      const plotH = h - padT - padB;
      const xAt = (k: number) => padL + (k / (N - 1)) * plotW;
      const yAt = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

      // zero line
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, yAt(0));
      ctx.lineTo(w - padR, yAt(0));
      ctx.stroke();

      const total = trials.length;
      for (let i = 0; i < total; i++) {
        const e = trials[i].e;
        const isLast = i === total - 1;
        const ageT = total > 1 ? i / (total - 1) : 1;
        ctx.strokeStyle = isLast
          ? converged
            ? "#fbbf24"
            : "#f87171"
          : `rgba(248,113,113,${0.08 + 0.2 * ageT})`;
        ctx.lineWidth = isLast ? 1.8 : 1;
        if (isLast) {
          ctx.shadowColor = converged ? "#fbbf24" : "#f87171";
          ctx.shadowBlur = 10;
        }
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(e[0]));
        for (let k = 1; k < N; k++) ctx.lineTo(xAt(k), yAt(e[k]));
        ctx.stroke();
        if (isLast) ctx.shadowBlur = 0;
      }
    },
    [trials, converged, w, h],
  );
  return <canvas ref={canvasRef} />;
}

/* ================================================================
   learned feedforward chart
================================================================ */

function FeedforwardChart({
  w,
  h,
  trials,
  yRef,
  uMax,
  converged,
}: {
  w: number;
  h: number;
  trials: Trial[];
  yRef: Float64Array;
  uMax: number;
  converged: boolean;
}) {
  const canvasRef = useCanvas(
    w,
    h,
    (ctx) => {
      const padL = 34,
        padR = 10,
        padT = 8,
        padB = 20;

      const series: Float64Array[] = [yRef, ...trials.map((t) => t.u)];
      const { lo, hi } = boundsOf(series, 0.18);

      drawAxes(
        ctx,
        w,
        h,
        padL,
        padR,
        padT,
        padB,
        lo,
        hi,
        "u (A)",
        "time",
        4,
      );

      const plotW = w - padL - padR;
      const plotH = h - padT - padB;
      const xAt = (k: number) => padL + (k / (N - 1)) * plotW;
      const yAt = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

      // saturation band
      if (uMax < hi) {
        ctx.fillStyle = "rgba(248,113,113,0.06)";
        ctx.fillRect(padL, padT, plotW, yAt(uMax) - padT);
        ctx.fillRect(padL, yAt(-uMax), plotW, h - padB - yAt(-uMax));
        ctx.strokeStyle = "rgba(248,113,113,0.35)";
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, yAt(uMax));
        ctx.lineTo(w - padR, yAt(uMax));
        ctx.moveTo(padL, yAt(-uMax));
        ctx.lineTo(w - padR, yAt(-uMax));
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // reference — faint dashed white for comparison
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(xAt(0), yAt(yRef[0]));
      for (let k = 1; k < N; k++) ctx.lineTo(xAt(k), yAt(yRef[k]));
      ctx.stroke();
      ctx.setLineDash([]);

      // current learned u
      if (trials.length > 0) {
        const u = trials[trials.length - 1].u;
        const color = converged ? "#fbbf24" : "#c4b5fd";
        ctx.strokeStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(xAt(0), yAt(u[0]));
        for (let k = 1; k < N; k++) ctx.lineTo(xAt(k), yAt(u[k]));
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    },
    [trials, yRef, uMax, converged, w, h],
  );
  return <canvas ref={canvasRef} />;
}

/* ================================================================
   convergence curve — ‖e‖₂ per trial (log y)
================================================================ */

function ConvergenceChart({
  w,
  h,
  trials,
  converged,
}: {
  w: number;
  h: number;
  trials: Trial[];
  converged: boolean;
}) {
  const canvasRef = useCanvas(
    w,
    h,
    (ctx) => {
      const padL = 34,
        padR = 10,
        padT = 10,
        padB = 22;
      ctx.fillStyle = "rgba(148,163,184,0.7)";
      ctx.font = "10px ui-monospace, monospace";

      if (trials.length === 0) {
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "rgba(148,163,184,0.5)";
        ctx.fillText("awaiting first trial", w / 2, h / 2);
        return;
      }

      // log-y axis spanning from min→max with floor for empty case
      const rmsArr = trials.map((t) => t.rms);
      const logs = rmsArr.map((v) => Math.log10(Math.max(v, 1e-3)));
      const lo = Math.min(...logs) - 0.15;
      const hi = Math.max(...logs) + 0.15;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;
      const xAt = (i: number) =>
        padL + (trials.length > 1 ? (i / (trials.length - 1)) * plotW : plotW / 2);
      const yAt = (logv: number) =>
        padT + (1 - (logv - lo) / (hi - lo)) * plotH;

      // gridlines at integer powers
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      const loInt = Math.floor(lo);
      const hiInt = Math.ceil(hi);
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let p = loInt; p <= hiInt; p++) {
        const y = yAt(p);
        if (y < padT || y > h - padB) continue;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        ctx.fillStyle = "rgba(148,163,184,0.6)";
        ctx.fillText(`10${supDigit(p)}`, padL - 4, y);
      }

      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.strokeRect(padL, padT, plotW, plotH);
      ctx.fillStyle = "rgba(148,163,184,0.6)";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("‖e‖₂", padL + 4, padT + 2);
      ctx.textAlign = "right";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("trial", w - padR - 4, h - padB + 12);

      // decay line
      ctx.strokeStyle = converged
        ? "rgba(251,191,36,0.9)"
        : "rgba(34,211,238,0.9)";
      ctx.shadowColor = converged ? "#fbbf24" : "#22d3ee";
      ctx.shadowBlur = 8;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < trials.length; i++) {
        const x = xAt(i);
        const y = yAt(logs[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // dots
      for (let i = 0; i < trials.length; i++) {
        const x = xAt(i);
        const y = yAt(logs[i]);
        ctx.fillStyle =
          i === trials.length - 1
            ? converged
              ? "#fbbf24"
              : "#22d3ee"
            : "rgba(124,92,255,0.8)";
        ctx.beginPath();
        ctx.arc(x, y, i === trials.length - 1 ? 3.5 : 2, 0, Math.PI * 2);
        ctx.fill();
      }
    },
    [trials, converged, w, h],
  );
  return <canvas ref={canvasRef} />;
}

function supDigit(n: number): string {
  const map: Record<string, string> = {
    "-": "⁻",
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
  };
  return String(n)
    .split("")
    .map((c) => map[c] ?? c)
    .join("");
}

/* ================================================================
   slider
================================================================ */

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  help,
  log = false,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
  help?: string;
  log?: boolean;
}) {
  // For log sliders, map slider position 0..1 to value via exp.
  const toPos = (v: number) =>
    log
      ? (Math.log(v) - Math.log(min)) / (Math.log(max) - Math.log(min))
      : (v - min) / (max - min);
  const fromPos = (p: number) =>
    log ? Math.exp(Math.log(min) + p * (Math.log(max) - Math.log(min))) : min + p * (max - min);

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <label className="text-[10px] font-mono uppercase tracking-wider text-gray-400">
          {label}
        </label>
        <span className="text-[11px] tabular-nums text-[#22d3ee] font-mono">
          {format(value)}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={1000}
        step={1}
        value={Math.round(toPos(value) * 1000)}
        onChange={(e) => {
          const p = Number(e.target.value) / 1000;
          let v = fromPos(p);
          // Snap to step
          if (step > 0 && !log) v = Math.round(v / step) * step;
          onChange(v);
        }}
        className="w-full accent-[#22d3ee]"
      />
      {help && (
        <div className="text-[9px] font-mono text-gray-600 mt-0.5">
          {help}
        </div>
      )}
    </div>
  );
}

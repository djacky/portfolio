"use client";

/* ------------------------------------------------------------------
   MPCDemo — container for the 3-phase grid-tied VSC MPC scene.

   Layout:
     • header + solve-time badge
     • main grid: R3F waveform world (left, two stacked panels:
       phase currents + V_dc) + control column (right)
     • LCL + 6-IGBT bridge schematic beneath the main grid

   Owns the MPCEngine singleton, pushes UI state into it, and drives a
   15 Hz HUD re-render pulse so widgets track the live engine.  R3F and
   schematic are dynamically imported (ssr:false).
------------------------------------------------------------------ */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import dynamic from "next/dynamic";
import {
  CircuitBoard,
  Play,
  Pause,
  RotateCcw,
  Zap,
  Waves,
  Radio,
  Activity,
  Gauge,
  Sigma,
} from "lucide-react";
import katex from "katex";

import {
  MPCEngine,
  PARAMS,
  V_GD,
  type ActiveConstraint,
} from "@/lib/mpc-sim";

const MPCWaveformWorld = dynamic(() => import("./MPCWaveformWorld"), {
  ssr: false,
  loading: () => <div className="w-full h-full" />,
});

const MPCCircuitPanel = dynamic(() => import("./MPCCircuitPanel"), {
  ssr: false,
  loading: () => <div className="w-full h-[170px]" />,
});

const ACCENT = "#f472b6";
const ACCENT_GOLD = "#fbbf24";
const ACCENT_CYAN = "#22d3ee";
const ACCENT_VIOLET = "#a78bfa";
const ACCENT_ROSE = "#fb7185";
const DANGER = "#ef4444";

// Slow-mo presets — sim-seconds per render-second.
//   0.2  ≈ 10 grid cycles / s    (brisk)
//   0.05 ≈ 2.5 grid cycles / s   (legible)
//   0.01 ≈ 0.5 grid cycle / s    (inspect)
const SLOW_MO_PRESETS: { label: string; value: number; sub: string }[] = [
  { label: "1×", value: 0.2, sub: "≈ 10 cycles / s" },
  { label: "0.25×", value: 0.05, sub: "≈ 2.5 cycles / s" },
  { label: "0.05×", value: 0.01, sub: "≈ 0.5 cycle / s" },
];

// ==========================================================
//  Main component
// ==========================================================
export default function MPCDemo() {
  const [horizon, setHorizon] = useState(10);
  const [playing, setPlaying] = useState(true);
  const [slowMo, setSlowMo] = useState(SLOW_MO_PRESETS[2].value);
  const [, setTick] = useState(0);

  const engineRef = useRef<MPCEngine | null>(null);
  if (engineRef.current === null) engineRef.current = new MPCEngine(10);

  useEffect(() => {
    engineRef.current?.setHorizon(horizon);
  }, [horizon]);

  useEffect(() => {
    const id = setInterval(() => setTick((n) => (n + 1) % 1_000_000), 66);
    return () => clearInterval(id);
  }, []);

  const engine = engineRef.current!;

  const onReset = useCallback(() => {
    engineRef.current?.reset();
  }, []);
  const onCarCurrent = useCallback((a: number) => {
    engineRef.current?.setLoadCurrent(a);
    setTick((n) => n + 1);
  }, []);
  const onGridHarmonic = useCallback(() => {
    engineRef.current?.toggleGridHarmonic();
    setTick((n) => n + 1);
  }, []);
  const onHarmonicReject = useCallback(() => {
    engineRef.current?.toggleHarmonicReject();
    setTick((n) => n + 1);
  }, []);
  const onNoise = useCallback(() => {
    engineRef.current?.toggleNoise();
    setTick((n) => n + 1);
  }, []);

  const igdDisplay = engine.i_gd_meas.toFixed(2);
  const igqDisplay = engine.i_gq_meas.toFixed(2);
  const irefDisplay = engine.iref_d.toFixed(2);
  const vdcDisplay = engine.V_dc.toFixed(0);
  const pRefDisplay = (engine.P_ref / 1000).toFixed(2);
  const phase = ((engine.theta * 180 / Math.PI) % 360).toFixed(0);
  const costTrack = engine.jTrack;
  const costEffort = engine.jEffort;
  const costTotal = costTrack + costEffort + 1e-9;
  const solveUsShown = engine.solveUsEma > 0 ? engine.solveUsEma : engine.solveUs;

  return (
    <div id="demo-mpc" className="glass rounded-3xl p-6 md:p-8 scroll-mt-24">
      {/* ---------------- Header ---------------- */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-5">
        <div>
          <div
            className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.22em]"
            style={{ color: ACCENT }}
          >
            <CircuitBoard className="w-3.5 h-3.5" />
            Eaton · 3-phase grid-tied VSC · Implicit MPC
          </div>
          <h3 className="mt-2 text-2xl md:text-3xl font-semibold text-gradient">
            The Controller Sees the Future
          </h3>
          <p className="mt-2 text-sm text-gray-400 max-w-2xl leading-relaxed">
            A grid connected converter with an LCL filter, steered by a
            predictive controller that solves a fresh QP every{" "}
            {Math.round(PARAMS.T_s * 1e6)} µs right in your browser. The
            bright arcs riding ahead of each phase trace are the
            controller&apos;s predicted future currents.
          </p>
        </div>
        <SolveBadge
          solveUs={solveUsShown}
          iters={engine.iters}
          itersMax={engine.itersMax}
          horizon={horizon}
        />
      </div>

      {/* ---------------- Main stage ---------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-5 items-start">
        {/* LEFT: waveform + schematic + legend (stacked).  Using an explicit
            height on the waveform instead of aspect-ratio prevents the grid
            row from stretching the canvas container and overflowing into
            the right column. */}
        <div className="flex flex-col gap-4 min-w-0">
          <div
            className="relative rounded-2xl overflow-hidden border border-white/10 bg-[#060912]"
            style={{ height: 480 }}
          >
            <MPCWaveformWorld engineRef={engineRef} playing={playing} slowMo={slowMo} />

          {/* Constraint labels — pixel-fixed positions anchored to the canvas
              so they stay aligned with the R3F scene on every resolution
              (canvas is pinned at 480px, zoom 88 → y=1.7 rail ≈ 90 px,
               y=0 rail ≈ 240 px, V_dc panel top ≈ 306 px, V_dc trace ≈ 359 px). */}
          <div className="pointer-events-none absolute top-[68px] left-[12px] text-[9.5px] font-mono tracking-wide"
               style={{ color: `${DANGER}cc` }}>
            +i<sub>g,max</sub> · {PARAMS.I_G_MAX.toFixed(1)} A
          </div>
          <div className="pointer-events-none absolute top-[244px] left-[12px] text-[9.5px] font-mono tracking-wide"
               style={{ color: `${DANGER}cc` }}>
            −i<sub>g,max</sub> · {PARAMS.I_G_MAX.toFixed(1)} A
          </div>

          {/* i_gd live readout (bottom panel, right axis) — sits in the gap
              above the V_dc panel, well clear of the green i_gd trace. */}
          <div className="pointer-events-none absolute top-[272px] left-[12px] text-[9.5px] font-mono tracking-wide leading-tight">
            <div style={{ color: "#4ade80" }}>
              i<sub>gd</sub> = <span className="tabular-nums">{igdDisplay}</span> A
            </div>
            <div style={{ color: "#4ade8088" }}>
              i<sub>gd</sub>* = <span className="tabular-nums">{irefDisplay}</span> A · ±80 A span
            </div>
          </div>

          {/* V_dc live readout — below the V_dc trace, anchored to canvas bottom. */}
          <div className="pointer-events-none absolute bottom-[12px] left-[12px] text-[9.5px] font-mono tracking-wide leading-tight">
            <div style={{ color: ACCENT_GOLD }}>
              V<sub>dc</sub> = <span className="tabular-nums">{vdcDisplay}</span> V
            </div>
            <div style={{ color: `${ACCENT_GOLD}88` }}>
              V<sub>dc</sub>* = {PARAMS.V_DC_REF.toFixed(0)} V · ±80 V span
            </div>
          </div>

          {/* Optimizer HUD (bottom-right) */}
          <div className="absolute bottom-[12px] right-[12px] max-w-[290px]">
            <OptimizerHUD
              iters={engine.iters}
              itersMax={engine.itersMax}
              solveUs={solveUsShown}
              active={engine.active}
              costTrack={costTrack}
              costEffort={costEffort}
              costTotal={costTotal}
            />
          </div>

          {/* Phase legend (top-right) */}
          <div className="absolute top-[12px] right-[12px] rounded-lg border border-white/10 bg-black/50 px-2.5 py-1.5 text-[10px] font-mono text-gray-400 min-w-[110px]">
            <div className="text-[8.5px] uppercase tracking-wider text-gray-500 mb-1">
              ωt = {phase}°
            </div>
            <div className="flex flex-col gap-0.5">
              <PhaseDot color={ACCENT_CYAN} label="i_a" />
              <PhaseDot color={ACCENT_VIOLET} label="i_b" />
              <PhaseDot color={ACCENT_ROSE} label="i_c" />
            </div>
          </div>
          </div>

          {/* Compact legend above the schematic — horizontal strip so it
              doesn't extend the right column's height. */}
          <div className="rounded-2xl border border-white/10 bg-black/25 px-3 py-2 flex flex-wrap gap-x-4 gap-y-1.5 text-[10px] font-mono text-gray-500 tracking-wide">
            <LegendDot color={ACCENT_CYAN} label="i_a" />
            <LegendDot color={ACCENT_VIOLET} label="i_b" />
            <LegendDot color={ACCENT_ROSE} label="i_c" />
            <LegendDot color="#e8ecf3" label="i_a ref" />
            <LegendDot color="#7dd3fc" label="pred i_a" />
            <LegendDot color="#c4b5fd" label="pred i_b" />
            <LegendDot color="#fda4af" label="pred i_c" />
            <LegendDot color={DANGER} label="±i_g,max rail" />
            <LegendDot color={ACCENT_GOLD} label="V_dc" />
            <LegendDot color="#4ade80" label="i_gd  (dq d-axis, right scale)" />
            <LegendDot color="#86efac" label="pred i_gd" />
          </div>

          {/* Schematic lives directly below the legend, in the LEFT column,
              so it fills the column height beneath the plot rather than
              leaving a gap next to the Live State panel. */}
          <MPCCircuitPanel engineRef={engineRef} slowMo={slowMo} />
        </div>

        {/* RIGHT: controls + live state */}
        <div className="flex flex-col gap-3 min-w-0">
          <RunBar playing={playing} setPlaying={setPlaying} onReset={onReset} />

          <SlowMoSelector slowMo={slowMo} setSlowMo={setSlowMo} />

          <HorizonSlider
            horizon={horizon}
            setHorizon={setHorizon}
            solveUs={solveUsShown}
          />

          <DisturbanceConsole
            carCurrent={engine.i_load}
            harmonicActive={engine.gridHarmonicActive}
            rejectOn={engine.harmonicRejectOn}
            noiseOn={engine.noiseOn}
            onCarCurrent={onCarCurrent}
            onHarmonic={onGridHarmonic}
            onHarmonicReject={onHarmonicReject}
            onNoise={onNoise}
          />

          <LivePanel
            igd={igdDisplay}
            igq={igqDisplay}
            iref={irefDisplay}
            vdc={vdcDisplay}
            pRef={pRefDisplay}
          />
        </div>
      </div>

      <p className="mt-4 text-[10.5px] text-gray-500 leading-relaxed max-w-3xl">
        The MPC commands averaged dq voltages u<sub>d</sub>, u<sub>q</sub>;
        in hardware these realise as 6 IGBT switching patterns at{" "}
        {(PARAMS.f_sw / 1000).toFixed(0)} kHz via SVPWM. The plant simulator
        runs the full 6-state LCL in dq with forward Euler at 20 kHz,
        integrating the non-linear V<sub>dc</sub> power balance alongside.
        Particle flow direction in the schematic follows true phase current
        sign and is gated by slow-mo so it&apos;s always watchable.
      </p>

      <MPCFormulation horizon={horizon} />
    </div>
  );
}

// ==========================================================
//  KaTeX helpers
// ==========================================================

function Tex({
  expr,
  block = false,
  className = "",
}: {
  expr: string;
  block?: boolean;
  className?: string;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(expr, {
        throwOnError: false,
        displayMode: block,
        output: "html",
        strict: "ignore",
      }),
    [expr, block],
  );
  const Tag = block ? "div" : "span";
  return (
    <Tag
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function MPCFormulation({ horizon }: { horizon: number }) {
  const stateExpr = String.raw`x_k = \begin{bmatrix} i_{fd} & i_{fq} & v_{cfd} & v_{cfq} & i_{gd} & i_{gq} \end{bmatrix}_k^{\!\top} \in \mathbb{R}^{6}`;
  const inputExpr = String.raw`u_k = \begin{bmatrix} u_{d} & u_{q} \end{bmatrix}_k^{\!\top},\quad \Delta u_k = u_k - u_{k-1}`;
  const plantExpr = String.raw`x_{k+1} = A_d\,x_k \;+\; B_d\,u_k \;+\; E_d\,v_{g,k}`;
  const outputExpr = String.raw`y_k = C\,x_k = \begin{bmatrix} i_{gd} \\ i_{gq} \end{bmatrix}_k`;
  const costExpr = String.raw`\min_{\,U\,}\; J \;=\; \sum_{k=0}^{N-1} \underbrace{\bigl\lVert\, y_k - y_k^{\star}\,\bigr\rVert_{Q}^{2}}_{\text{tracking}} \;+\; \underbrace{\bigl\lVert\, \Delta u_k \,\bigr\rVert_{R}^{2}}_{\text{effort}}`;
  const decisionExpr = String.raw`U = \bigl[\,u_0^{\!\top},\, u_1^{\!\top},\, \ldots,\, u_{N-1}^{\!\top}\,\bigr]^{\!\top} \in \mathbb{R}^{2N}`;
  const svpwmExpr = String.raw`\sqrt{\,u_{d,k}^{2} + u_{q,k}^{2}\,} \;\le\; \tfrac{V_{dc}}{\sqrt{3}}\cdot 0.95 \qquad \text{(SVPWM hex inscribed circle)}`;
  const railExpr = String.raw`\bigl|\,i_{gd,k}\,\bigr| \le I_{g,\max},\quad \bigl|\,i_{gq,k}\,\bigr| \le I_{g,\max} \qquad \text{(grid-inductor rails)}`;
  const busExpr = String.raw`V_{dc,\min} \;\le\; V_{dc,k} \;\le\; V_{dc,\max} \qquad \text{(DC-bus admissible set)}`;
  const qpExpr = String.raw`\tfrac{1}{2}\,U^{\!\top} H\,U \;+\; f(x_0,\,y^{\star},\,u_{-1})^{\!\top} U \quad \text{s.t.}\quad G\,U \le w(x_0,\,V_{dc})`;

  return (
    <div className="mt-6 rounded-2xl border border-white/10 bg-black/30 p-5 md:p-6">
      <div className="flex items-center gap-2 mb-1">
        <Sigma className="w-4 h-4" style={{ color: ACCENT }} />
        <div
          className="text-[11px] font-mono uppercase tracking-[0.22em]"
          style={{ color: ACCENT }}
        >
          Optimization problem · condensed QP
        </div>
      </div>
      <div className="text-[11px] text-gray-500 leading-relaxed max-w-3xl mb-4">
        At every sample the controller substitutes the plant recursion into
        the cost, eliminating the states and leaving a dense convex QP in the
        stacked input sequence <Tex expr={String.raw`U`} />. Hildreth&apos;s
        dual method solves it in microseconds.
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {/* ---- Plant ---- */}
        <FormSection title="Plant · discrete dq LCL">
          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-1">State</div>
          <Tex expr={stateExpr} block className="text-gray-200 text-[13px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Input</div>
          <Tex expr={inputExpr} block className="text-gray-200 text-[13px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">
            Model · sampled at T<sub>s</sub> = {Math.round(PARAMS.T_s * 1e6)} µs
          </div>
          <Tex expr={plantExpr} block className="text-gray-200 text-[13px] overflow-x-auto" />
          <Tex expr={outputExpr} block className="text-gray-200 text-[13px] overflow-x-auto" />
        </FormSection>

        {/* ---- Cost ---- */}
        <FormSection title="Cost · horizon N = {horizon}" dynamicTitle={`Cost · horizon N = ${horizon}`}>
          <Tex expr={costExpr} block className="text-gray-200 text-[13px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Decision variable</div>
          <Tex expr={decisionExpr} block className="text-gray-200 text-[13px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Condensed QP</div>
          <Tex expr={qpExpr} block className="text-gray-200 text-[13px] overflow-x-auto" />

          <div className="text-[10px] font-mono text-gray-500 mt-3 leading-snug">
            <span style={{ color: ACCENT_CYAN }}>Q</span> penalises dq
            tracking error; <span style={{ color: ACCENT }}>R</span>{" "}
            penalises input rate to tame chatter.
          </div>
        </FormSection>

        {/* ---- Constraints ---- */}
        <FormSection title="Hard constraints · ∀ k = 0…N−1">
          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-1">Bridge voltage</div>
          <Tex expr={svpwmExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Grid current</div>
          <Tex expr={railExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">DC bus</div>
          <Tex expr={busExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />

          <div className="text-[10px] font-mono text-gray-500 mt-3 leading-snug">
            Nonlinear SVPWM circle is linearised into {`{`}8, 12{`}`}{" "}
            tangent half-planes per step, keeping the problem a pure QP.
          </div>
        </FormSection>
      </div>

      <div className="mt-4 text-[10px] font-mono text-gray-500 leading-relaxed">
        Apply receding horizon: send <Tex expr={String.raw`u_0^{\star}`} />,
        discard the tail, re-solve next sample with fresh{" "}
        <Tex expr={String.raw`x_0`} />.
      </div>
    </div>
  );
}

function FormSection({
  title,
  dynamicTitle,
  children,
}: {
  title: string;
  dynamicTitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-white/5 bg-black/20 p-4">
      <div
        className="text-[10.5px] font-mono uppercase tracking-[0.18em] mb-3"
        style={{ color: ACCENT_GOLD }}
      >
        {dynamicTitle ?? title}
      </div>
      <div className="flex flex-col">{children}</div>
    </div>
  );
}

// ==========================================================
//  Sub-components
// ==========================================================

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="inline-block w-2 h-2 rounded-full"
        style={{ background: color, boxShadow: `0 0 6px ${color}` }}
      />
      <span>{label}</span>
    </div>
  );
}

function PhaseDot({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="inline-block w-1.5 h-1.5 rounded-full"
        style={{ background: color, boxShadow: `0 0 4px ${color}` }}
      />
      <span>{label}</span>
    </div>
  );
}

function SolveBadge({
  solveUs,
  iters,
  itersMax,
  horizon,
}: {
  solveUs: number;
  iters: number;
  itersMax: number;
  horizon: number;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-black/30 px-3 py-2 min-w-[230px]">
      <div className="text-[9px] font-mono uppercase tracking-wider text-gray-500">
        QP · Hildreth dual · 6-state LCL
      </div>
      <div className="flex items-baseline gap-2 mt-0.5">
        <span className="text-xl font-semibold tabular-nums" style={{ color: ACCENT }}>
          {solveUs >= 10 ? solveUs.toFixed(0) : solveUs.toFixed(1)}
        </span>
        <span className="text-[10px] font-mono text-gray-400">
          {`μs · ${iters}/${itersMax} iter · N=${horizon}`}
        </span>
      </div>
    </div>
  );
}

function OptimizerHUD({
  iters,
  itersMax,
  solveUs,
  active,
  costTrack,
  costEffort,
  costTotal,
}: {
  iters: number;
  itersMax: number;
  solveUs: number;
  active: ActiveConstraint[];
  costTrack: number;
  costEffort: number;
  costTotal: number;
}) {
  const trackFrac = costTrack / costTotal;
  const effortFrac = costEffort / costTotal;
  return (
    <div className="rounded-lg border border-white/10 bg-black/50 backdrop-blur-sm px-3 py-2.5 text-[10px] font-mono min-w-[260px]">
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-[8.5px] uppercase tracking-wider text-gray-500">
          Cost · J = ‖i_g − i_g*‖²_Q + ‖u_dq‖²_R
        </div>
        <div className="text-[8.5px] text-gray-500 tabular-nums">
          iter {iters} (peak {itersMax}) · {solveUs >= 10 ? solveUs.toFixed(0) : solveUs.toFixed(1)} μs
        </div>
      </div>
      <div className="h-1.5 w-full rounded-full overflow-hidden bg-black/70 border border-white/5 flex">
        <div
          className="h-full transition-all"
          style={{
            width: `${(trackFrac * 100).toFixed(1)}%`,
            background: ACCENT_CYAN,
            boxShadow: `0 0 8px ${ACCENT_CYAN}88`,
          }}
        />
        <div
          className="h-full transition-all"
          style={{
            width: `${(effortFrac * 100).toFixed(1)}%`,
            background: ACCENT,
            boxShadow: `0 0 8px ${ACCENT}88`,
          }}
        />
      </div>
      <div className="flex justify-between mt-1 text-[8.5px] text-gray-500">
        <span style={{ color: ACCENT_CYAN }}>
          ‖i_g − i_g*‖²  {(trackFrac * 100).toFixed(0)}%
        </span>
        <span style={{ color: ACCENT }}>
          ‖u_dq‖²  {(effortFrac * 100).toFixed(0)}%
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {active.length === 0 ? (
          <span className="text-[9px] text-gray-500 italic">interior (no active constraint)</span>
        ) : (
          active.map((a) => (
            <span
              key={a}
              className="text-[9px] px-1.5 py-0.5 rounded-full border"
              style={{
                color: ACCENT_GOLD,
                borderColor: `${ACCENT_GOLD}66`,
                background: `${ACCENT_GOLD}18`,
              }}
            >
              {a}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

function DisturbanceConsole({
  carCurrent,
  harmonicActive,
  rejectOn,
  noiseOn,
  onCarCurrent,
  onHarmonic,
  onHarmonicReject,
  onNoise,
}: {
  carCurrent: number;
  harmonicActive: boolean;
  rejectOn: boolean;
  noiseOn: boolean;
  onCarCurrent: (a: number) => void;
  onHarmonic: () => void;
  onHarmonicReject: () => void;
  onNoise: () => void;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/25 p-3.5 flex flex-col gap-2">
      <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500">
        Disturbance theatre
      </div>
      <EVDemandSlider value={carCurrent} onChange={onCarCurrent} />
      <div className="flex items-stretch gap-2">
        <div className="flex-1 min-w-0">
          <DisturbanceBtn
            label={harmonicActive ? "5th harmonic · holding" : "Grid 5th harmonic"}
            sub="±8% v_g · 300 Hz ripple in dq · IEEE 519 envelope"
            icon={<Waves className="w-3.5 h-3.5" />}
            active={harmonicActive}
            onClick={onHarmonic}
            tint={ACCENT_GOLD}
          />
        </div>
        <RejectToggle rejectOn={rejectOn} onClick={onHarmonicReject} />
      </div>
      <DisturbanceBtn
        label="Inject noise"
        sub="±2.5 A current · ±6 V V_dc"
        icon={<Radio className="w-3.5 h-3.5" />}
        active={noiseOn}
        onClick={onNoise}
        tint={ACCENT}
      />
    </div>
  );
}

function RejectToggle({
  rejectOn,
  onClick,
}: {
  rejectOn: boolean;
  onClick: () => void;
}) {
  const tint = ACCENT_CYAN;
  return (
    <button
      onClick={onClick}
      title="Toggle IMP harmonic-rejection companion (2-state resonator at 6ω, K_r·ξ added to u)"
      className="rounded-lg border transition-colors flex flex-col items-center justify-center w-[78px] shrink-0 px-1 py-2 text-center"
      style={{
        borderColor: rejectOn ? `${tint}66` : "rgba(255,255,255,0.08)",
        background: rejectOn ? `${tint}12` : "rgba(255,255,255,0.02)",
      }}
    >
      <span
        className="text-[9px] font-mono uppercase tracking-[0.12em] leading-tight"
        style={{ color: rejectOn ? tint : "#94a3b8" }}
      >
        IMP reject
      </span>
      <span
        className="text-[10px] font-mono mt-0.5"
        style={{ color: rejectOn ? tint : "#6b7280" }}
      >
        {rejectOn ? "ON" : "OFF"}
      </span>
      <span className="text-[8.5px] font-mono text-gray-500 mt-0.5 leading-tight">
        K<sub>r</sub>·ξ @ 6ω
      </span>
    </button>
  );
}

const EV_DEMAND_MAX = 55;

function EVDemandSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  const pct = Math.max(0, Math.min(1, value / EV_DEMAND_MAX));
  // At V_gd = 326 V, i_gd* ≈ (2/3)·(V_dc·i_car)/V_gd ≈ 2.3·i_car, so the
  // 65 A rail starts biting around i_car ≈ 28 A.  Highlight when we're in
  // the regime where MPC constraint enforcement is visibly active.
  const railBinding = value > 40;
  return (
    <div
      className="rounded-lg border px-3 py-2 transition-colors"
      style={{
        borderColor: railBinding ? `${ACCENT_CYAN}55` : "rgba(255,255,255,0.08)",
        background: railBinding ? `${ACCENT_CYAN}10` : "rgba(255,255,255,0.02)",
      }}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-1.5 text-[11px] font-medium text-gray-200">
          <Zap className="w-3.5 h-3.5" style={{ color: ACCENT_CYAN }} />
          EV charge demand
        </div>
        <div className="text-[11px] font-mono tabular-nums" style={{ color: ACCENT_CYAN }}>
          {value.toFixed(1)} A · {(value * 750 / 1000).toFixed(1)} kW
        </div>
      </div>
      <input
        type="range"
        min={0}
        max={EV_DEMAND_MAX}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[#22d3ee]"
      />
      <div className="flex justify-between text-[9px] font-mono text-gray-500 mt-0.5">
        <span>idle · 0 A</span>
        <span style={{ opacity: 0.8 }}>{(pct * 100).toFixed(0)}% of rated</span>
        <span>peak · {EV_DEMAND_MAX} A</span>
      </div>
    </div>
  );
}

function DisturbanceBtn({
  label,
  sub,
  icon,
  active,
  onClick,
  tint,
}: {
  label: string;
  sub: string;
  icon: React.ReactNode;
  active: boolean;
  onClick: () => void;
  tint: string;
}) {
  return (
    <button
      onClick={onClick}
      className="rounded-lg border text-left px-3 py-2 transition-colors flex items-start gap-2.5"
      style={{
        borderColor: active ? `${tint}66` : "rgba(255,255,255,0.08)",
        background: active ? `${tint}12` : "rgba(255,255,255,0.02)",
      }}
    >
      <span className="mt-0.5" style={{ color: active ? tint : "#94a3b8" }}>
        {icon}
      </span>
      <span className="flex flex-col gap-0.5">
        <span className="text-[11px] font-medium leading-tight"
              style={{ color: active ? tint : "#e5e7eb" }}>
          {label}
        </span>
        <span className="text-[9px] text-gray-500 font-mono leading-tight">
          {active ? "active — click to clear" : sub}
        </span>
      </span>
    </button>
  );
}

function RunBar({
  playing,
  setPlaying,
  onReset,
}: {
  playing: boolean;
  setPlaying: (b: boolean) => void;
  onReset: () => void;
}) {
  return (
    <div className="flex gap-2">
      <button
        onClick={() => setPlaying(!playing)}
        className={`flex-1 inline-flex items-center justify-center gap-2 rounded-lg border text-xs font-medium py-2 transition-colors ${
          playing
            ? "bg-[#fbbf24]/20 border-[#fbbf24]/50 text-[#fbbf24] hover:bg-[#fbbf24]/30"
            : "bg-[#22d3ee]/15 border-[#22d3ee]/40 text-[#22d3ee] hover:bg-[#22d3ee]/25"
        }`}
      >
        {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
        {playing ? "Pause" : "Run"}
      </button>
      <button
        onClick={onReset}
        className="inline-flex items-center justify-center rounded-lg border border-white/15 bg-white/5 text-gray-300 px-3 py-2 hover:bg-white/10 transition-colors"
        aria-label="Reset"
      >
        <RotateCcw className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

function SlowMoSelector({
  slowMo,
  setSlowMo,
}: {
  slowMo: number;
  setSlowMo: (v: number) => void;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/25 p-3.5">
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 flex items-center gap-1.5">
          <Gauge className="w-3 h-3" /> Slow-mo
        </div>
        <div className="text-[10px] font-mono text-[#22d3ee] tabular-nums">
          {(slowMo * 1000).toFixed(0)} ms / s
        </div>
      </div>
      <div className="grid grid-cols-3 gap-1">
        {SLOW_MO_PRESETS.map((p) => (
          <button
            key={p.label}
            onClick={() => setSlowMo(p.value)}
            className={`rounded-md border px-2 py-1.5 text-[10px] font-mono transition-colors ${
              Math.abs(slowMo - p.value) < 1e-6
                ? "bg-[#22d3ee]/20 border-[#22d3ee]/55 text-[#22d3ee]"
                : "bg-white/5 border-white/10 text-gray-400 hover:bg-white/10"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="text-[9px] font-mono text-gray-500 mt-1">
        {SLOW_MO_PRESETS.find((p) => Math.abs(p.value - slowMo) < 1e-6)?.sub}
      </div>
    </div>
  );
}

function HorizonSlider({
  horizon,
  setHorizon,
  solveUs,
}: {
  horizon: number;
  setHorizon: (n: number) => void;
  solveUs: number;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/25 p-3.5">
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500">
          Prediction horizon
        </div>
        <div className="text-[10px] font-mono text-[#f472b6] tabular-nums">
          N = {horizon}
        </div>
      </div>
      <input
        type="range"
        min={4}
        max={16}
        step={1}
        value={horizon}
        onChange={(e) => setHorizon(Number(e.target.value))}
        className="w-full accent-[#f472b6]"
      />
      <div className="flex items-center justify-between mt-1 text-[9px] font-mono text-gray-500">
        <span>look-ahead samples · {(PARAMS.T_s * 1e6).toFixed(0)} μs each</span>
        <span className="tabular-nums">
          {`${solveUs.toFixed(0)} μs/solve`}
        </span>
      </div>
    </div>
  );
}

function LivePanel({
  igd,
  igq,
  iref,
  vdc,
  pRef,
}: {
  igd: string;
  igq: string;
  iref: string;
  vdc: string;
  pRef: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/25 p-3.5 flex flex-col gap-2">
      <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-wider text-gray-500">
        <Activity className="w-3 h-3" style={{ color: ACCENT }} />
        Live state · dq frame
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="i_gd" value={`${igd} A`} tint={ACCENT_CYAN} />
        <Stat label="i_gd*" value={`${iref} A`} tint="#e2e8f0" />
        <Stat label="i_gq" value={`${igq} A`} tint={ACCENT_VIOLET} />
        <Stat label="V_dc" value={`${vdc} V`} tint={ACCENT_GOLD} />
        <Stat label="P_ref" value={`${pRef} kW`} tint={ACCENT} />
        <Stat label="mode" value="MPC" tint="#e2e8f0" />
      </div>
      <div className="text-[9px] font-mono text-gray-500 mt-1 leading-snug">
        v_gd = {V_GD.toFixed(0)} V · v_gq = 0 · PLL locked · unity PF
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  tint,
}: {
  label: string;
  value: string;
  tint: string;
}) {
  return (
    <div className="rounded-lg bg-black/30 border border-white/5 px-2.5 py-1.5">
      <div className="text-[9px] font-mono uppercase tracking-wider text-gray-500">
        {label}
      </div>
      <div className="text-[13px] font-mono tabular-nums" style={{ color: tint }}>
        {value}
      </div>
    </div>
  );
}

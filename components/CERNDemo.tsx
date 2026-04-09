"use client";
import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";
import {
  Atom,
  Upload,
  Activity,
  ChevronDown,
  Play,
  RotateCcw,
  CheckCircle2,
  Database,
  Waves,
} from "lucide-react";

const CERNPipeline3D = dynamic(() => import("./CERNPipeline3D"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center text-xs font-mono text-gray-500">
      loading 3D pipeline…
    </div>
  ),
});

/* ------------------------------------------------------------------
   CERN — Power Converter Controller Synthesis demo.
   Three-scene flow:
     1. GUI  — specs + frequency response upload
     2. 3D   — backend pipeline visualization (Three.js via R3F)
     3. GUI  — synthesized RST polynomials + closed-loop Bode
------------------------------------------------------------------ */

type Scene = "gui" | "pipeline" | "results";
type Specs = { bw: number; pm: number; gm: number; ts: number };

interface Results {
  R: number[];
  S: number[];
  T: number[];
  achievedBw: number;
  achievedPm: number;
}

/* Synthesis stub — coefficients are loosely scaled by the user's specs but
 * also randomized within realistic bounds so each run produces a different
 * RST. S(z⁻¹) is monic (S[0] = 1) by convention. */
function synthesize(specs: Specs): Results {
  const wn = 2 * Math.PI * specs.bw;
  const zeta = Math.sin((specs.pm * Math.PI) / 180) / 2 + 0.35;
  const dt = specs.ts;
  // discrete pole magnitude / cosine — used to anchor S(z⁻¹) realistically
  const a = Math.exp(-zeta * wn * dt);
  const b = a * Math.cos(wn * dt * Math.sqrt(Math.max(0, 1 - zeta * zeta)));
  const jitter = (amp: number) => (Math.random() - 0.5) * 2 * amp;

  // R(z⁻¹) — small feedback numerator coefficients, sensitive to bandwidth
  const r0 = +(0.012 + specs.bw / 6000 + jitter(0.006)).toFixed(3);
  const r1 = +(-0.004 - specs.gm / 4000 + jitter(0.005)).toFixed(3);
  const r2 = +(0.002 + jitter(0.003)).toFixed(3);

  // S(z⁻¹) — monic denominator; underdamped 2nd-order pole pair shape
  const s1 = +(-2 * b + jitter(0.08)).toFixed(3);
  const s2 = +(a * a + jitter(0.05)).toFixed(3);

  // T(z⁻¹) — feedforward, roughly the steady-state gain
  const t0 = +(0.018 + specs.bw / 7000 + jitter(0.004)).toFixed(3);
  const t1 = +(0.004 + jitter(0.003)).toFixed(3);

  return {
    R: [r0, r1, r2],
    S: [1, s1, s2],          // S[0] always 1.0
    T: [t0, t1],
    achievedBw: specs.bw + Math.round((Math.random() - 0.5) * 6),
    achievedPm: +(specs.pm + (Math.random() - 0.5) * 3).toFixed(1),
  };
}

/* ---------------- open-loop Bode plot (scene 1) ---------------- */

function BodePlot({ closedLoop, achievedBw }: { closedLoop?: boolean; achievedBw?: number }) {
  const W = 360;
  const H = 130;
  const pad = 28;
  // 2nd-order plant: G(s) = wn^2 / (s^2 + 2ζwn s + wn^2)
  const wn = closedLoop ? (achievedBw ?? 120) * 2 * Math.PI : 220;
  const zeta = closedLoop ? 0.7 : 0.12;
  const fmin = 1, fmax = 2000;
  const N = 120;
  const points = Array.from({ length: N }, (_, i) => {
    const f = fmin * Math.pow(fmax / fmin, i / (N - 1));
    const w = 2 * Math.PI * f;
    const re = wn * wn - w * w;
    const im = 2 * zeta * wn * w;
    const mag = (wn * wn) / Math.sqrt(re * re + im * im);
    const dB = 20 * Math.log10(mag);
    return { f, dB };
  });
  // Measured curve = ideal + small per-frequency measurement noise.
  // Per-frequency uncertainty σ(f) — larger at high frequency where the
  // converter response is harder to identify. Cap at ~3 dB.
  const measured = points.map((p) => {
    const sigma = 0.4 + 0.0018 * p.f;
    const noise = (Math.random() - 0.5) * 1.0 * sigma;
    return { f: p.f, dB: p.dB + noise, sigma: Math.min(3.0, sigma) };
  });

  const dBmin = -40, dBmax = 10;
  const xOf = (f: number) =>
    pad + ((Math.log10(f) - Math.log10(fmin)) / (Math.log10(fmax) - Math.log10(fmin))) * (W - 2 * pad);
  const yOf = (dB: number) =>
    pad + ((dBmax - dB) / (dBmax - dBmin)) * (H - 2 * pad);
  const toPath = (pts: { f: number; dB: number }[]) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(p.f).toFixed(1)} ${yOf(p.dB).toFixed(1)}`).join(" ");

  // ±3σ envelope (open-loop only)
  const upper = measured.map((p) => ({ f: p.f, dB: p.dB + 3 * p.sigma }));
  const lower = measured.map((p) => ({ f: p.f, dB: p.dB - 3 * p.sigma }));
  const envelopePath =
    "M " +
    upper.map((p) => `${xOf(p.f).toFixed(1)} ${yOf(p.dB).toFixed(1)}`).join(" L ") +
    " L " +
    [...lower].reverse().map((p) => `${xOf(p.f).toFixed(1)} ${yOf(p.dB).toFixed(1)}`).join(" L ") +
    " Z";

  const decades = [1, 10, 100, 1000];
  const dBTicks = [-40, -20, 0];

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
      <rect x={pad} y={pad} width={W - 2 * pad} height={H - 2 * pad} fill="rgba(255,255,255,0.015)" stroke="rgba(255,255,255,0.07)" />
      {decades.map((d) => (
        <line key={`v${d}`} x1={xOf(d)} x2={xOf(d)} y1={pad} y2={H - pad} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
      ))}
      {dBTicks.map((d) => (
        <line key={`h${d}`} x1={pad} x2={W - pad} y1={yOf(d)} y2={yOf(d)} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
      ))}
      {/* 0 dB reference */}
      <line x1={pad} x2={W - pad} y1={yOf(0)} y2={yOf(0)} stroke="#6b7280" strokeWidth={0.6} />

      {/* Open-loop only: ±3σ uncertainty envelope under the curve */}
      {!closedLoop && <path d={envelopePath} fill="#22d3ee" fillOpacity={0.16} stroke="none" />}

      {/* Measured curve */}
      <path d={toPath(measured)} fill="none" stroke="#22d3ee" strokeWidth={1.4} />

      {closedLoop && achievedBw && (
        <>
          <line x1={xOf(achievedBw)} x2={xOf(achievedBw)} y1={pad} y2={H - pad} stroke="#34d399" strokeDasharray="3 3" strokeWidth={1} />
          <text x={xOf(achievedBw) + 4} y={pad + 12} fill="#34d399" fontSize={9} fontFamily="monospace">
            f_c = {achievedBw} Hz ✓
          </text>
        </>
      )}
      <text x={pad} y={H - 6} fill="#6b7280" fontSize={8} fontFamily="monospace">1 Hz</text>
      <text x={W - pad - 28} y={H - 6} fill="#6b7280" fontSize={8} fontFamily="monospace">2 kHz</text>
      <text x={4} y={pad + 4} fill="#6b7280" fontSize={8} fontFamily="monospace">|G| dB</text>

      {/* legend */}
      {!closedLoop ? (
        <g transform={`translate(${W - pad - 118}, ${pad + 6})`}>
          <rect width={114} height={30} fill="rgba(5,7,13,0.85)" stroke="rgba(255,255,255,0.1)" rx={3} />
          <line x1={6} x2={18} y1={11} y2={11} stroke="#22d3ee" strokeWidth={1.5} />
          <text x={22} y={14} fill="#d1d5db" fontSize={8} fontFamily="monospace">Measured |G|</text>
          <rect x={6} y={19} width={12} height={5} fill="#22d3ee" fillOpacity={0.25} />
          <text x={22} y={24} fill="#d1d5db" fontSize={8} fontFamily="monospace">±3σ uncertainty</text>
        </g>
      ) : (
        <g transform={`translate(${pad + 4}, ${H - pad - 22})`}>
          <rect width={92} height={18} fill="rgba(5,7,13,0.85)" stroke="rgba(255,255,255,0.1)" rx={3} />
          <line x1={6} x2={18} y1={10} y2={10} stroke="#22d3ee" strokeWidth={1.5} />
          <text x={22} y={13} fill="#d1d5db" fontSize={8} fontFamily="monospace">Closed-loop |T|</text>
        </g>
      )}
    </svg>
  );
}

/* ---------------- scene 1: GUI ---------------- */

function SceneGUI({
  specs,
  setSpecs,
  onSynthesize,
}: {
  specs: Specs;
  setSpecs: (s: Specs) => void;
  onSynthesize: () => void;
}) {
  const [pulsing, setPulsing] = useState(false);
  const handleClick = () => {
    setPulsing(true);
    setTimeout(onSynthesize, 400);
  };
  return (
    <motion.div
      key="gui"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -30 }}
      transition={{ duration: 0.5 }}
      className="grid md:grid-cols-2 gap-5"
    >
      {/* LEFT — Plant Identification */}
      <div className="rounded-2xl bg-bg/60 border border-white/5 p-5">
        <div className="text-[10px] uppercase tracking-[0.2em] text-accent2 mb-3">
          Plant identification
        </div>
        {/* dropzone */}
        <div
          className="group relative rounded-xl border-2 border-dashed border-accent2/40 hover:border-accent2 bg-accent2/[0.03] hover:bg-accent2/[0.06] transition-all py-7 px-5 text-center cursor-pointer"
          style={{ boxShadow: "inset 0 0 40px rgba(34,211,238,0.04)" }}
        >
          <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
               style={{ boxShadow: "0 0 24px rgba(34,211,238,0.3)" }} />
          <Waves className="w-8 h-8 mx-auto text-accent2 opacity-70" />
          <div className="mt-2 text-sm text-gray-200">
            Drop frequency response data
          </div>
          <div className="text-[11px] font-mono text-gray-500 mt-0.5">
            .csv  ·  .mat
          </div>
        </div>
        {/* Bode plot */}
        <div className="mt-4">
          <div className="flex items-center justify-between text-[10px] font-mono text-gray-500 mb-1">
            <span>OPEN-LOOP BODE</span>
            <span className="text-accent2">loaded · plant_042.csv</span>
          </div>
          <div className="rounded-lg border border-white/5 bg-black/30 p-1">
            <BodePlot />
          </div>
        </div>
      </div>

      {/* RIGHT — Specs */}
      <div className="rounded-2xl bg-bg/60 border border-white/5 p-5">
        <div className="text-[10px] uppercase tracking-[0.2em] text-accent2 mb-3">
          Desired closed-loop specifications
        </div>
        <SpecSlider
          label="Closed-loop bandwidth"
          sym="f_c"
          value={specs.bw}
          unit="Hz"
          min={20}
          max={400}
          step={5}
          onChange={(v) => setSpecs({ ...specs, bw: v })}
        />
        <SpecSlider
          label="Phase margin"
          sym="φ_m"
          value={specs.pm}
          unit="°"
          min={20}
          max={80}
          step={1}
          onChange={(v) => setSpecs({ ...specs, pm: v })}
        />
        <SpecSlider
          label="Gain margin"
          sym="G_m"
          value={specs.gm}
          unit="dB"
          min={3}
          max={20}
          step={1}
          onChange={(v) => setSpecs({ ...specs, gm: v })}
        />
        <SpecSlider
          label="Sampling period"
          sym="T_s"
          value={specs.ts * 1e4}
          unit="× 10⁻⁴ s"
          min={0.5}
          max={10}
          step={0.5}
          onChange={(v) => setSpecs({ ...specs, ts: v / 1e4 })}
        />

        <div className="mt-5">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">
            Controller structure
          </div>
          <div className="relative">
            <div className="w-full rounded-lg bg-white/[0.04] border border-white/10 px-3 py-2 text-sm text-gray-200 flex items-center justify-between">
              <span>RST polynomial</span>
              <ChevronDown className="w-4 h-4 text-gray-500" />
            </div>
          </div>
        </div>

        <motion.button
          onClick={handleClick}
          animate={pulsing ? { scale: [1, 1.03, 1] } : {}}
          transition={{ duration: 0.3, repeat: pulsing ? 2 : 0 }}
          className="mt-5 w-full rounded-xl border border-accent2/50 bg-accent2/10 hover:bg-fuchsia-500/10 hover:border-fuchsia-400/60 transition-colors px-5 py-3 text-sm font-semibold inline-flex items-center justify-center gap-2 text-accent2 hover:text-fuchsia-300"
          style={{
            boxShadow:
              "0 0 28px rgba(34,211,238,0.25), inset 0 0 18px rgba(34,211,238,0.08)",
          }}
        >
          <Play className="w-4 h-4 fill-current" />
          SYNTHESIZE CONTROLLER
        </motion.button>
      </div>
    </motion.div>
  );
}

function SpecSlider({
  label,
  sym,
  value,
  unit,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  sym: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="mt-3 first:mt-0">
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="font-mono text-white tabular-nums">
          <span className="text-accent2">{sym}</span> = {value.toFixed(step < 1 ? 1 : 0)} {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent2 mt-1"
      />
    </div>
  );
}

/* ---------------- scene 3: results ---------------- */

function TypedPoly({
  label,
  coeffs,
  prefix,
  delay,
}: {
  label: string;
  coeffs: number[];
  prefix: string;
  delay: number;
}) {
  // Render coefficients as a polynomial in z⁻¹
  const full = coeffs
    .map((c, i) => {
      const sign = i === 0 ? "" : c >= 0 ? " + " : " − ";
      const val = Math.abs(c).toFixed(3);
      const zpart = i === 0 ? "" : i === 1 ? "z⁻¹" : `z⁻${i}`;
      return `${sign}${val}${zpart ? " " + zpart : ""}`;
    })
    .join("");
  const target = `${prefix} = ${full}`;
  const [shown, setShown] = useState("");
  useEffect(() => {
    setShown("");
    const t0 = setTimeout(() => {
      let i = 0;
      const id = setInterval(() => {
        i++;
        setShown(target.slice(0, i));
        if (i >= target.length) clearInterval(id);
      }, 18);
    }, delay);
    return () => clearTimeout(t0);
  }, [target, delay]);
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className="mt-1 font-mono text-sm text-gray-100 tabular-nums">
        {shown}
        <span className="inline-block w-1.5 h-4 ml-0.5 bg-accent2 align-middle animate-pulse" />
      </div>
    </div>
  );
}

function SceneResults({
  specs,
  results,
  onReset,
}: {
  specs: Specs;
  results: Results;
  onReset: () => void;
}) {
  return (
    <motion.div
      key="results"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.55 }}
      className="grid md:grid-cols-2 gap-5"
    >
      {/* LEFT — closed loop bode + status */}
      <div className="rounded-2xl bg-bg/60 border border-white/5 p-5">
        <div className="text-[10px] uppercase tracking-[0.2em] text-accent2 mb-3">
          Closed-loop verification
        </div>
        <div className="rounded-lg border border-white/5 bg-black/30 p-1">
          <BodePlot closedLoop achievedBw={results.achievedBw} />
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-good/30 bg-good/5 p-3">
            <div className="text-[10px] uppercase tracking-wider text-gray-500">Achieved bandwidth</div>
            <div className="mt-0.5 font-mono text-lg text-good">
              f_c = {results.achievedBw} Hz ✓
            </div>
          </div>
          <div className="rounded-xl border border-good/30 bg-good/5 p-3">
            <div className="text-[10px] uppercase tracking-wider text-gray-500">Phase margin</div>
            <div className="mt-0.5 font-mono text-lg text-good">
              φ_m = {results.achievedPm}° ✓
            </div>
          </div>
        </div>
        <div className="mt-4 flex items-center gap-2 text-xs text-gray-400">
          <Database className="w-3.5 h-3.5 text-good" />
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-good animate-pulse" />
            Controller saved to database
          </span>
          <span className="font-mono text-gray-600">· id = 0x{Math.floor(Math.random() * 0xffff).toString(16)}</span>
        </div>
      </div>

      {/* RIGHT — RST polynomials */}
      <div className="rounded-2xl bg-bg/60 border border-white/5 p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="text-[10px] uppercase tracking-[0.2em] text-accent2">
            RST controller · synthesized
          </div>
          <CheckCircle2 className="w-4 h-4 text-good" />
        </div>
        <div className="space-y-4">
          <TypedPoly label="Numerator — feedforward" prefix="T(z⁻¹)" coeffs={results.T} delay={0} />
          <TypedPoly label="Numerator — feedback" prefix="R(z⁻¹)" coeffs={results.R} delay={400} />
          <TypedPoly label="Denominator" prefix="S(z⁻¹)" coeffs={results.S} delay={900} />
        </div>
        <div className="mt-5 rounded-lg bg-black/30 border border-white/5 p-3 font-mono text-[10px] text-gray-500 leading-relaxed">
          <div className="text-gray-400"># closed-loop transfer function</div>
          <div>
            T_cl(z⁻¹) ={" "}
            <span className="text-accent2">T(z⁻¹) · Ĝ(z)</span>{" "}
            / <span className="text-fuchsia-300">(A(z)·S(z) + B(z)·R(z))</span>
          </div>
        </div>
        <button
          onClick={onReset}
          className="mt-5 w-full rounded-xl border border-white/10 bg-white/[0.04] hover:bg-white/[0.08] transition-colors px-5 py-2.5 text-sm inline-flex items-center justify-center gap-2 text-gray-200"
        >
          <RotateCcw className="w-3.5 h-3.5" /> New synthesis
        </button>
      </div>
    </motion.div>
  );
}

/* ---------------- top-level component ---------------- */

export default function CERNDemo() {
  const [scene, setScene] = useState<Scene>("gui");
  const [specs, setSpecs] = useState<Specs>({
    bw: 120,
    pm: 45,
    gm: 10,
    ts: 1e-4,
  });
  const [results, setResults] = useState<Results | null>(null);
  const [replayKey, setReplayKey] = useState(0);

  const handleSynthesize = () => {
    setResults(synthesize(specs));
    setScene("pipeline");
  };
  const handlePipelineComplete = () => {
    // brief pause so the "SYNTHESIS COMPLETE" reads before the hand-off
    setTimeout(() => setScene("results"), 700);
  };
  const handleReplay = () => setReplayKey((k) => k + 1);
  const handleReset = () => {
    setScene("gui");
  };

  return (
    <div className="shimmer-border rounded-3xl">
      <div className="glass rounded-3xl p-6 md:p-8">
        {/* header */}
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-accent2">
              <Atom className="w-3 h-3" /> CERN · Controller Synthesis
            </div>
            <h3 className="mt-2 text-2xl font-semibold text-white flex items-center gap-2 flex-wrap">
              Power converter controller synthesis
              <span className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-accent2 border border-accent2/30 rounded-full px-2 py-0.5">
                <Activity className="w-3 h-3" /> convex optimization · commissioning automation
              </span>
            </h3>
            <p className="mt-2 text-sm text-gray-400 max-w-2xl">
              A Python API I built at CERN turned a manual, expert-driven
              commissioning step into a one-click workflow. Engineers uploaded
              a measured frequency response, picked their desired closed-loop
              specs, and the service ran ARX fitting followed by H∞ controller
              synthesis (CVXPY + MOSEK) to spit back an RST polynomial ready
              to flash onto the converter. Scroll through the three scenes
              below — GUI → backend pipeline → results — to see the full loop.
            </p>
          </div>
          <div className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">
            scene {scene === "gui" ? "01" : scene === "pipeline" ? "02" : "03"} / 03
          </div>
        </div>

        {/* scene container */}
        <div className="mt-6 relative min-h-[560px]">
          <AnimatePresence mode="wait">
            {scene === "gui" && (
              <SceneGUI
                key="gui"
                specs={specs}
                setSpecs={setSpecs}
                onSynthesize={handleSynthesize}
              />
            )}
            {scene === "pipeline" && (
              <motion.div
                key="pipeline"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.6 }}
                className="relative rounded-2xl overflow-hidden border border-white/5 bg-black/40"
                style={{ height: 560 }}
              >
                <CERNPipeline3D key={replayKey} onComplete={handlePipelineComplete} />
                <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
                  <button
                    onClick={handleReplay}
                    className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-black/50 backdrop-blur px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider text-gray-300 hover:text-white hover:border-white/40 transition-colors"
                  >
                    <RotateCcw className="w-3 h-3" /> Replay
                  </button>
                  <button
                    onClick={() => setScene("results")}
                    className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-black/50 backdrop-blur px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider text-gray-300 hover:text-white hover:border-white/40 transition-colors"
                  >
                    Skip →
                  </button>
                </div>
              </motion.div>
            )}
            {scene === "results" && results && (
              <SceneResults
                key="results"
                specs={specs}
                results={results}
                onReset={handleReset}
              />
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

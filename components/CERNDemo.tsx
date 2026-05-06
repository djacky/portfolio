"use client";
import { useState, useEffect, useMemo, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";
import {
  Atom,
  ChevronDown,
  Play,
  RotateCcw,
  CheckCircle2,
  Database,
  Waves,
  Sigma,
  AlertTriangle,
} from "lucide-react";
import katex from "katex";
import { PLANTS, plantById, Plant, TS_FIXED } from "@/lib/hinf-plants";
import {
  bandwidthBoundsHz,
  cAbs,
  logFreqGrid,
  SynthesisResult,
  SynthesisSpecs,
} from "@/lib/hinf-synthesis";
import {
  synthesizeInWorker,
  disposeHinfWorker,
  ProgressEvent,
} from "@/lib/hinf-worker-client";

const ACCENT_GREEN = "#34d399";
const ACCENT_CYAN = "#22d3ee";
const ACCENT_GOLD = "#fbbf24";

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
     1. GUI  — plant selector + H∞ spec sliders
     2. 3D   — backend pipeline visualization (Three.js via R3F)
                 · live γ / bandwidth streamed from the Web Worker
     3. GUI  — synthesized RST polynomials + closed-loop Bode
   The synthesis itself runs in a Web Worker (lib/hinf-worker.ts)
   which drives an SCS-WASM SOCP solve at each γ of a bisection.
------------------------------------------------------------------ */

type Scene = "gui" | "pipeline" | "results";
type Specs = SynthesisSpecs;

/* ---------------- Bode plot (scene 1: |G|, scene 3: |T_cl|) ---------------- */

function BodePlot({
  plant,
  specs,
  closedLoop,
  achievedBw,
  results,
  height,
}: {
  plant: Plant;
  specs: Specs;
  closedLoop?: boolean;
  achievedBw?: number;
  results?: SynthesisResult;
  height?: number;
}) {
  const W = 360;
  const H = height ?? 130;
  const pad = 28;

  const { fNyq, points } = useMemo(() => {
    // Ts comes from specs (user-sliderable) — not plant.Ts, so the
    // Nyquist cut-off moves when the user changes the sampling period.
    const Ts = specs.Ts;
    const nyq = 1 / (2 * Ts);

    // Closed-loop uses the same linear synthesis grid the SOCP solved
    // on; open-loop uses a log-spaced grid for a clean display shape.
    if (closedLoop && results) {
      const grid = plant.buildGrid(Ts, specs.desBw);
      const Rc = results.RFull;
      const Sc = results.SFull;
      const Tc = results.TFull;
      const pts: { f: number; dB: number }[] = [];
      for (let k = 0; k < grid.w.length; k++) {
        const wk = grid.w[k];
        let Rre = 0, Rim = 0, Sre = 0, Sim = 0, Tre = 0, Tim = 0;
        for (let i = 0; i < Rc.length; i++) {
          const ang = -wk * Ts * i;
          Rre += Rc[i] * Math.cos(ang);
          Rim += Rc[i] * Math.sin(ang);
        }
        for (let i = 0; i < Sc.length; i++) {
          const ang = -wk * Ts * i;
          Sre += Sc[i] * Math.cos(ang);
          Sim += Sc[i] * Math.sin(ang);
        }
        for (let i = 0; i < Tc.length; i++) {
          const ang = -wk * Ts * i;
          Tre += Tc[i] * Math.cos(ang);
          Tim += Tc[i] * Math.sin(ang);
        }
        const Gk = grid.G[k];
        const MAk = grid.MA[k];
        const GTre = Gk.re * Tre - Gk.im * Tim;
        const GTim = Gk.re * Tim + Gk.im * Tre;
        const GMARe_r = Gk.re * MAk.re - Gk.im * MAk.im;
        const GMAIm_r = Gk.re * MAk.im + Gk.im * MAk.re;
        const GMAR_re = GMARe_r * Rre - GMAIm_r * Rim;
        const GMAR_im = GMARe_r * Rim + GMAIm_r * Rre;
        const Dre = GMAR_re + Sre;
        const Dim = GMAR_im + Sim;
        const dmag2 = Dre * Dre + Dim * Dim;
        const num = Math.hypot(GTre, GTim);
        const mag = dmag2 > 0 ? num / Math.sqrt(dmag2) : 0;
        pts.push({ f: wk / (2 * Math.PI), dB: 20 * Math.log10(Math.max(mag, 1e-20)) });
      }
      return { fNyq: nyq, points: pts };
    }

    // Open-loop: evaluate |G(jω)| on 200 log-spaced points.
    const wInit = plant.dominantPoleRad / 100;
    const wLog = logFreqGrid(wInit, Ts, 200);
    const pts = wLog.map((w) => ({
      f: w / (2 * Math.PI),
      dB: 20 * Math.log10(Math.max(cAbs(plant.frf(w)), 1e-20)),
    }));
    return { fNyq: nyq, points: pts };
  }, [plant, specs.Ts, closedLoop, results]);

  // Log x-axis: closed-loop centres the achieved f_c on the plot —
  // fMin = f_c² / fNyq makes log10(f_c) the midpoint of [log10(fMin),
  // log10(fNyq)].  Open-loop keeps its natural grid start.
  const fMin =
    closedLoop && achievedBw && achievedBw > 0 && achievedBw < fNyq
      ? Math.max((achievedBw * achievedBw) / fNyq, 1e-6)
      : (points[0]?.f ?? fNyq / 1000);
  // Clip points to the displayed x-range so the path doesn't leak
  // out the left edge of the plot box.
  const visible = points.filter((p) => p.f >= fMin && p.f <= fNyq);

  // Auto-scale the y-axis to the visible curve: 5 dB margin above the
  // peak and below the trough.  Fall back to a sensible default range
  // if `visible` happens to be empty.
  const dBVals = visible.map((p) => p.dB);
  const peak = dBVals.length ? Math.max(...dBVals) : 5;
  const trough = dBVals.length ? Math.min(...dBVals) : -40;
  const dBmax = peak + 5;
  const dBmin = trough - 5;

  const xOf = (f: number) =>
    pad + ((Math.log10(f) - Math.log10(fMin)) / (Math.log10(fNyq) - Math.log10(fMin))) * (W - 2 * pad);
  const yOf = (dB: number) => {
    const clamped = Math.max(dBmin, Math.min(dBmax, dB));
    return pad + ((dBmax - clamped) / (dBmax - dBmin)) * (H - 2 * pad);
  };
  const toPath = (pts: { f: number; dB: number }[]) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(p.f).toFixed(1)} ${yOf(p.dB).toFixed(1)}`).join(" ");

  const fmtHz = (f: number) => {
    if (f >= 1000) return `${(f / 1000).toFixed(1)} kHz`;
    if (f >= 10) return `${f.toFixed(0)} Hz`;
    if (f >= 1) return `${f.toFixed(1)} Hz`;
    return `${f.toFixed(2)} Hz`;
  };

  // Decade gridlines inside [fMin, fNyq].
  const decades: number[] = [];
  for (let e = Math.ceil(Math.log10(fMin)); e <= Math.floor(Math.log10(fNyq)); e++) {
    const f = Math.pow(10, e);
    if (f >= fMin && f <= fNyq) decades.push(f);
  }
  // dB gridlines — choose a round step that gives ~3-5 interior ticks
  // across the auto-scaled range.
  const dBRange = dBmax - dBmin;
  const dBStep =
    dBRange > 60 ? 20 : dBRange > 30 ? 10 : dBRange > 12 ? 5 : dBRange > 4 ? 2 : 1;
  const dBTicks: number[] = [];
  for (let t = Math.ceil(dBmin / dBStep) * dBStep; t <= dBmax; t += dBStep) {
    if (t > dBmin && t < dBmax) dBTicks.push(t);
  }

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
      <rect x={pad} y={pad} width={W - 2 * pad} height={H - 2 * pad} fill="rgba(255,255,255,0.015)" stroke="rgba(255,255,255,0.07)" />
      {decades.map((d) => (
        <line key={`v${d}`} x1={xOf(d)} x2={xOf(d)} y1={pad} y2={H - pad} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
      ))}
      {dBTicks.map((d) => (
        <g key={`h${d}`}>
          <line x1={pad} x2={W - pad} y1={yOf(d)} y2={yOf(d)} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
          <text x={pad - 3} y={yOf(d) + 2.5} fill="#6b7280" fontSize={7} fontFamily="monospace" textAnchor="end">{d}</text>
        </g>
      ))}
      {0 >= dBmin && 0 <= dBmax && (
        <line x1={pad} x2={W - pad} y1={yOf(0)} y2={yOf(0)} stroke="#6b7280" strokeWidth={0.6} />
      )}
      {-3 >= dBmin && -3 <= dBmax && (
        <line x1={pad} x2={W - pad} y1={yOf(-3)} y2={yOf(-3)} stroke={closedLoop ? "#34d399" : "#6b7280"} strokeWidth={0.6} strokeDasharray="2 2" />
      )}

      <path d={toPath(visible)} fill="none" stroke={closedLoop ? "#34d399" : "#22d3ee"} strokeWidth={1.4} />

      {closedLoop && achievedBw && achievedBw >= fMin && achievedBw <= fNyq && (
        <>
          <line x1={xOf(achievedBw)} x2={xOf(achievedBw)} y1={pad} y2={H - pad} stroke="#34d399" strokeDasharray="3 3" strokeWidth={1} />
          <text x={xOf(achievedBw) + 4} y={pad + 12} fill="#34d399" fontSize={9} fontFamily="monospace">
            f_c = {achievedBw.toFixed(0)} Hz ✓
          </text>
        </>
      )}
      <text x={pad} y={H - 6} fill="#6b7280" fontSize={8} fontFamily="monospace">{fmtHz(fMin)}</text>
      <text x={W - pad - 40} y={H - 6} fill="#6b7280" fontSize={8} fontFamily="monospace">{fmtHz(fNyq)}</text>
      <text x={4} y={pad + 4} fill="#6b7280" fontSize={8} fontFamily="monospace">{closedLoop ? "|T_cl| dB" : "|G| dB"}</text>
    </svg>
  );
}

/* ---------------- scene 1: GUI ---------------- */

function SceneGUI({
  plantId,
  setPlantId,
  specs,
  setSpecs,
  onSynthesize,
}: {
  plantId: string;
  setPlantId: (id: string) => void;
  specs: Specs;
  setSpecs: (s: Specs) => void;
  onSynthesize: () => void;
}) {
  const [pulsing, setPulsing] = useState(false);
  const plant = useMemo(() => plantById(plantId), [plantId]);
  const bwBounds = useMemo(() => bandwidthBoundsHz(specs.Ts), [specs.Ts]);

  const handleClick = () => {
    setPulsing(true);
    setTimeout(onSynthesize, 400);
  };

  const handlePlantChange = (id: string) => {
    const p = plantById(id);
    setPlantId(id);
    setSpecs({
      desMm: p.defaults.desMm,
      desBw: p.defaults.desBw,
      desZeta: p.defaults.desZeta,
      order: p.defaults.order,
      Ts: TS_FIXED,
    });
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
          Plant · frequency response
        </div>
        <PlantPicker plantId={plantId} onChange={handlePlantChange} />

        <div className="mt-3 rounded-lg bg-black/30 border border-white/5 p-3">
          <Tex expr={plant.latex} block className="text-gray-200 text-[11.5px] overflow-x-auto" />
        </div>
        <div className="mt-2 text-[11px] text-gray-500 leading-relaxed">
          {plant.description}
        </div>

        <div className="mt-4">
          <div className="flex items-center justify-between text-[10px] font-mono text-gray-500 mb-1">
            <span>OPEN-LOOP BODE</span>
            <span className="text-accent2 inline-flex items-center gap-1">
              <Waves className="w-3 h-3" />
              {plant.id}
            </span>
          </div>
          <div className="rounded-lg border border-white/5 bg-black/30 p-1">
            <BodePlot plant={plant} specs={specs} />
          </div>
        </div>
      </div>

      {/* RIGHT — Specs */}
      <div className="rounded-2xl bg-bg/60 border border-white/5 p-5">
        <div className="text-[10px] uppercase tracking-[0.2em] text-accent2 mb-3">
          Desired closed-loop specifications
        </div>
        <SpecSlider
          label="Modulus margin"
          sym="ΔM_ib"
          value={specs.desMm}
          unit=""
          min={0.1}
          max={0.9}
          step={0.05}
          fmt={(v) => v.toFixed(2)}
          onChange={(v) => setSpecs({ ...specs, desMm: v })}
        />
        <SpecSlider
          label="Closed-loop bandwidth"
          sym="f_c"
          value={specs.desBw}
          unit="Hz"
          min={bwBounds.min}
          max={bwBounds.max}
          step={Math.max(1, (bwBounds.max - bwBounds.min) / 100)}
          fmt={(v) => v.toFixed(0)}
          onChange={(v) => setSpecs({ ...specs, desBw: v })}
          hint={`F_s/25 .. F_s/8  (${bwBounds.min.toFixed(0)}–${bwBounds.max.toFixed(0)} Hz)`}
        />
        <SpecSlider
          label="Damping ratio"
          sym="ζ"
          value={specs.desZeta}
          unit=""
          min={0.5}
          max={1.0}
          step={0.05}
          fmt={(v) => v.toFixed(2)}
          onChange={(v) => setSpecs({ ...specs, desZeta: v })}
        />

        <div className="mt-4">
          <div className="flex items-baseline justify-between text-xs">
            <span className="text-gray-400">Sampling period</span>
            <span className="font-mono text-white tabular-nums">
              <span className="text-accent2">T_s</span> = {(specs.Ts * 1e6).toFixed(0)} µs
            </span>
          </div>
          <div className="text-[10px] font-mono text-gray-600 mt-1">
            F_s = {(1 / specs.Ts).toFixed(0)} Hz · fixed
          </div>
        </div>

        <SpecSlider
          label="Controller order"
          sym="n"
          value={specs.order ?? 6}
          unit=""
          min={3}
          max={10}
          step={1}
          fmt={(v) => `${v}`}
          onChange={(v) => setSpecs({ ...specs, order: Math.round(v) })}
          hint={`RST · ${(specs.order ?? 6) + 1} coefficients · 1 integrator`}
        />

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

function PlantPicker({
  plantId,
  onChange,
}: {
  plantId: string;
  onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const current = PLANTS.find((p) => p.id === plantId) ?? PLANTS[0];
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full rounded-lg bg-white/[0.04] border border-white/10 hover:border-accent2/50 px-3 py-2 text-sm text-gray-200 flex items-center justify-between transition-colors"
      >
        <span className="flex flex-col items-start text-left">
          <span className="text-gray-100">{current.label}</span>
          <span className="text-[10px] font-mono text-gray-500 mt-0.5">{current.sub}</span>
        </span>
        <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="absolute z-20 mt-1 w-full rounded-lg bg-[#0b0f1a] border border-white/15 shadow-xl overflow-hidden">
          {PLANTS.map((p) => (
            <button
              key={p.id}
              onClick={() => {
                onChange(p.id);
                setOpen(false);
              }}
              className={`w-full text-left px-3 py-2 hover:bg-white/[0.06] flex flex-col gap-0.5 transition-colors ${p.id === plantId ? "bg-accent2/[0.08]" : ""}`}
            >
              <span className="text-sm text-gray-100">{p.label}</span>
              <span className="text-[10px] font-mono text-gray-500">{p.sub}</span>
            </button>
          ))}
        </div>
      )}
    </div>
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
  fmt,
  hint,
  onChange,
}: {
  label: string;
  sym: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  step: number;
  fmt?: (v: number) => string;
  hint?: string;
  onChange: (v: number) => void;
}) {
  const display = fmt ? fmt(value) : value.toFixed(step < 1 ? 1 : 0);
  return (
    <div className="mt-3 first:mt-0">
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="font-mono text-white tabular-nums">
          <span className="text-accent2">{sym}</span> = {display} {unit}
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
      {hint && <div className="text-[10px] font-mono text-gray-600 mt-0.5">{hint}</div>}
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
  // Render coefficients as a polynomial in z⁻¹.  Synthesis coefficients
  // can span several decades; use a fixed 4-sig-fig scientific form for
  // anything below 0.01 so tiny numerator terms don't print as "0.000".
  const fmt = (c: number) => {
    const abs = Math.abs(c);
    if (abs === 0) return "0";
    if (abs >= 0.01 && abs < 1000) return abs.toFixed(4);
    return abs.toExponential(2);
  };
  const full = coeffs
    .map((c, i) => {
      const sign = i === 0 ? (c < 0 ? "−" : "") : c >= 0 ? " + " : " − ";
      const val = fmt(c);
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
  plantId,
  specs,
  results,
  onReset,
}: {
  plantId: string;
  specs: Specs;
  results: SynthesisResult;
  onReset: () => void;
}) {
  const plant = useMemo(() => plantById(plantId), [plantId]);
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
          <BodePlot plant={plant} specs={specs} closedLoop achievedBw={results.achievedBw} results={results} height={220} />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-good/30 bg-good/5 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-gray-500">Achieved bandwidth</div>
            <div className="font-mono text-sm text-good">
              f_c = {results.achievedBw.toFixed(1)} Hz ✓
            </div>
          </div>
          <div className="rounded-xl border border-good/30 bg-good/5 px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-gray-500">H∞ norm</div>
            <div className="font-mono text-sm text-good">
              ‖T_zw‖∞ = {results.gammaOpt.toFixed(3)}
            </div>
          </div>
        </div>
        <div className="mt-3 flex items-center gap-2 text-xs text-gray-400">
          <Database className="w-3.5 h-3.5 text-good" />
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-good animate-pulse" />
            {results.iterations.length} bisection iterations · SCS-WASM
          </span>
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
          <TypedPoly label="Numerator — feedforward" prefix="T(z⁻¹)" coeffs={results.TFull} delay={0} />
          <TypedPoly label="Numerator — feedback"  prefix="R(z⁻¹)" coeffs={results.RFull} delay={400} />
          <TypedPoly label="Denominator · has (1−z⁻¹) integrator"   prefix="S(z⁻¹)" coeffs={results.SFull} delay={900} />
        </div>
        <div className="mt-5 rounded-lg bg-black/30 border border-white/5 p-3 font-mono text-[10px] text-gray-500 leading-relaxed">
          <div className="text-gray-400"># closed-loop transfer function</div>
          <div>
            T_cl(z⁻¹) ={" "}
            <span className="text-accent2">G(z)·T(z⁻¹)</span>{" "}
            / <span className="text-fuchsia-300">(G(z)·MA(z)·R(z⁻¹) + S(z⁻¹))</span>
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
  const [plantId, setPlantId] = useState<string>(PLANTS[0].id);
  const [specs, setSpecs] = useState<Specs>({
    desMm: PLANTS[0].defaults.desMm,
    desBw: PLANTS[0].defaults.desBw,
    desZeta: PLANTS[0].defaults.desZeta,
    order: PLANTS[0].defaults.order,
    Ts: TS_FIXED,
  });
  const [results, setResults] = useState<SynthesisResult | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [gammaHistory, setGammaHistory] = useState<ProgressEvent[]>([]);
  const [infeasibleHint, setInfeasibleHint] = useState<string | null>(null);
  const [replayKey, setReplayKey] = useState(0);
  const [pipelineDone, setPipelineDone] = useState(false);
  const solveHandle = useRef<{ cancel: () => void } | null>(null);
  const solveTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Terminate the worker when the component unmounts so dev-HMR doesn't
  // accumulate orphaned WASM instances.
  useEffect(() => () => disposeHinfWorker(), []);

  const handleSynthesize = () => {
    setResults(null);
    setProgress(null);
    setGammaHistory([]);
    setInfeasibleHint(null);
    setPipelineDone(false);
    setScene("pipeline");
    if (solveTimeout.current) clearTimeout(solveTimeout.current);
    const handle = synthesizeInWorker(plantId, specs, (p) => {
      setProgress(p);
      setGammaHistory((prev) =>
        prev.length && prev[prev.length - 1].iter === p.iter ? prev : [...prev, p],
      );
    });
    solveHandle.current = handle;
    // Wall-clock timeout: if the worker can't finish in 45 s, cancel and
    // surface an infeasibility hint.  The 3D scene is already hovering
    // on the H∞ block so the user just sees a graceful transition.
    solveTimeout.current = setTimeout(() => {
      handle.cancel();
      setInfeasibleHint(
        "Synthesis timed out. Try relaxing the modulus margin, lowering the bandwidth, or raising the damping ratio.",
      );
    }, 45000);
    const clear = () => {
      if (solveTimeout.current) {
        clearTimeout(solveTimeout.current);
        solveTimeout.current = null;
      }
    };
    handle.promise
      .then((res) => {
        clear();
        if (res.feasible) {
          setResults(res);
        } else {
          setInfeasibleHint(res.infeasibilityHint ?? "No feasible controller for these specs.");
        }
      })
      .catch((err) => {
        clear();
        setInfeasibleHint(err?.message ?? "Synthesis failed");
      });
  };

  const handlePipelineComplete = () => {
    // Mark pipeline done; a useEffect below handles the transition
    // once the solver also resolves (avoids a stale-closure race where
    // the poll reads a snapshot of `results` taken when the pipeline
    // finished, instead of the live state).
    setPipelineDone(true);
  };

  // Transition to results / back to GUI once BOTH the pipeline animation
  // finished and the solver Promise has resolved.
  useEffect(() => {
    if (scene !== "pipeline" || !pipelineDone) return;
    if (infeasibleHint) {
      setScene("gui");
    } else if (results) {
      setScene("results");
    }
  }, [scene, pipelineDone, results, infeasibleHint]);

  const handleReplay = () => setReplayKey((k) => k + 1);
  const handleReset = () => {
    setInfeasibleHint(null);
    setScene("gui");
  };

  return (
    <div className="shimmer-border rounded-3xl">
      <div className="glass rounded-3xl p-6 md:p-8">
        {/* header */}
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-5">
          <div>
            <div className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.22em] text-accent2">
              <Atom className="w-3.5 h-3.5" />
              CERN · Controller Synthesis
            </div>
            <h3 className="mt-2 text-2xl md:text-3xl font-semibold text-gradient">
              Power converter controller synthesis
            </h3>
            <p className="mt-2 text-sm text-gray-400 max-w-2xl leading-relaxed">
              The LHC&apos;s magnets need controllers tuned to parts per million
              precision, but each power converter behaves a little differently.
              Pick a plant, pick your specs — the H∞ data-driven RST solve runs
              in your browser via SCS-WASM, off the UI thread.
            </p>
          </div>
          <div className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">
            scene {scene === "gui" ? "01" : scene === "pipeline" ? "02" : "03"} / 03
          </div>
        </div>

        {/* infeasibility toast */}
        <AnimatePresence>
          {infeasibleHint && scene === "gui" && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-4 rounded-xl border border-amber-500/40 bg-amber-500/5 p-3 flex items-start gap-2"
            >
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="text-[11px] font-mono uppercase tracking-wider text-amber-400 mb-0.5">
                  Synthesis infeasible
                </div>
                <div className="text-[12px] text-gray-300 leading-relaxed">{infeasibleHint}</div>
              </div>
              <button
                onClick={() => setInfeasibleHint(null)}
                className="text-gray-500 hover:text-gray-300 text-xs font-mono px-2"
              >
                dismiss
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* scene container */}
        <div className="mt-6 relative">
          <AnimatePresence mode="wait">
            {scene === "gui" && (
              <SceneGUI
                key="gui"
                plantId={plantId}
                setPlantId={setPlantId}
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
                <CERNPipeline3D
                  key={replayKey}
                  onComplete={handlePipelineComplete}
                  progress={progress}
                  gammaHistory={gammaHistory}
                  infeasible={!!infeasibleHint}
                  solverDone={!!results || !!infeasibleHint}
                />
                <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
                  <button
                    onClick={handleReplay}
                    className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-black/50 backdrop-blur px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider text-gray-300 hover:text-white hover:border-white/40 transition-colors"
                  >
                    <RotateCcw className="w-3 h-3" /> Replay
                  </button>
                  {results && (
                    <button
                      onClick={() => setScene("results")}
                      className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-black/50 backdrop-blur px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider text-gray-300 hover:text-white hover:border-white/40 transition-colors"
                    >
                      Skip →
                    </button>
                  )}
                </div>
              </motion.div>
            )}
            {scene === "results" && results && (
              <SceneResults
                key="results"
                plantId={plantId}
                specs={specs}
                results={results}
                onReset={handleReset}
              />
            )}
          </AnimatePresence>
        </div>

        <HInfFormulation specs={specs} />
      </div>
    </div>
  );
}

/* ==========================================================
   KaTeX helper + H∞ optimization formulation
   ========================================================== */

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

function HInfFormulation({ specs }: { specs: Specs }) {
  const plantExpr = String.raw`G(e^{-j\omega T_s}) \;=\; \text{measured frequency response}`;
  const desiredExpr = String.raw`\mathcal{S}_{ry}^{\,d}(s) \;=\; \frac{\omega_d^{2}}{s^{2} + 2\zeta\,\omega_d\,s + \omega_d^{2}}\,e^{-s\,d_r}, \quad W \;=\; \bigl(1 - \mathcal{S}_{ry}^{\,d}\bigr)^{-1}`;
  const rstExpr = String.raw`\begin{aligned} R(z^{-1},\rho) &= r_0 + r_1 z^{-1} + \cdots + r_{n_r} z^{-n_r} \\ S(z^{-1},\rho) &= 1 + s_1 z^{-1} + \cdots + s_{n_s} z^{-n_s} \\ T(z^{-1},\rho) &= t_0 + t_1 z^{-1} + \cdots + t_{n_t} z^{-n_t} \end{aligned}`;
  const decisionExpr = String.raw`\rho^{\!\top} \;=\; \bigl[\,r_0,\ldots,r_{n_r},\; s_1,\ldots,s_{n_s},\; t_0,\ldots,t_{n_t}\,\bigr] \in \mathbb{R}^{n_{rst}}`;
  const psiExpr = String.raw`\psi(\rho) \;=\; G\,R(\rho) + S(\rho), \qquad \mathcal{S}_{ry}(\rho) \;=\; \dfrac{G_f\,T(\rho)}{\psi(\rho)}`;

  const hinfObjExpr = String.raw`\min_{\rho}\;\bigl\lVert\, W\bigl(\,1 - \mathcal{S}_{ry}(\rho)\,\bigr)\bigr\rVert_{\infty}`;
  const epigraphExpr = String.raw`\min_{\rho,\,\gamma}\;\gamma \quad \text{s.t.}\quad \bigl[W\bigl(1-\mathcal{S}_{ry}(\rho)\bigr)\bigr]^{\!\star}\bigl[W\bigl(1-\mathcal{S}_{ry}(\rho)\bigr)\bigr] \;<\; \gamma`;
  const convexExpr = String.raw`\gamma^{-1}\,\bigl\lvert\,W\bigl(\psi(\rho) - G_f\,T(\rho)\bigr)\bigr\rvert^{2} \;<\; 2\,\Re\bigl\{\psi(\rho)\,\psi^{\star}(\rho_0)\bigr\} - \bigl\lvert\psi(\rho_0)\bigr\rvert^{2}`;

  const mmExpr = String.raw`\bigl\lvert\,\Delta M_{ib}\,S(\rho)\bigr\rvert^{2} \;<\; 2\,\Re\bigl\{\psi(\rho)\,\psi^{\star}(\rho_0)\bigr\} - \bigl\lvert\psi(\rho_0)\bigr\rvert^{2}`;
  const noiseExpr = String.raw`\bigl\lvert\,A_k^{-1}\,G_f^{\,k}\,S^{\,k}(\rho)\bigr\rvert^{2} \;<\; 2\,\Re\bigl\{\psi^{k}(\rho)\,\psi^{\star,k}(\rho_0)\bigr\} - \bigl\lvert\psi^{k}(\rho_0)\bigr\rvert^{2}`;
  const ctrlStabExpr = String.raw`\Re\{\,S(\rho)\,\} \;>\; 0 \qquad \forall\,\omega\in\Omega := \bigl[\,0,\,\pi/T_s\,\bigr]`;

  const Ts_us = (specs.Ts * 1e6).toFixed(0);

  return (
    <div className="mt-6 rounded-2xl border border-white/10 bg-black/30 p-5 md:p-6">
      <div className="flex items-center gap-2 mb-1">
        <Sigma className="w-4 h-4" style={{ color: ACCENT_GREEN }} />
        <div
          className="text-[11px] font-mono uppercase tracking-[0.22em]"
          style={{ color: ACCENT_GREEN }}
        >
          Optimization problem · H∞ controller synthesis
        </div>
      </div>
      <div className="text-[11px] text-gray-500 leading-relaxed max-w-3xl mb-4">
        PyFresco minimises the <Tex expr={String.raw`\mathcal{H}_\infty`} /> norm
        of the model-matching error between the desired sensitivity{" "}
        <Tex expr={String.raw`\mathcal{S}_{ry}^{\,d}`} /> and the one produced
        by the RST controller. The raw constraint is convex–concave; linearising{" "}
        <Tex expr={String.raw`\psi^{\star}(\rho)\,\psi(\rho)`} /> around a
        stabilising iterate <Tex expr={String.raw`\rho_0`} /> turns each inner
        problem into a convex SDP solved per-frequency by CVXPY + MOSEK. A
        bisection over <Tex expr={String.raw`\gamma`} /> and an outer
        sequential-convex step converge to a local optimum.
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {/* ---- Plant + controller ---- */}
        <FormSection title="Plant · identified FRF + RST structure">
          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-1">Frequency response</div>
          <Tex expr={plantExpr} block className="text-gray-200 text-[11.5px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Desired closed-loop</div>
          <Tex expr={desiredExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">
            RST controller · orders <Tex expr={String.raw`(n_r, n_s, n_t)`} />
          </div>
          <Tex expr={rstExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Decision variable</div>
          <Tex expr={decisionExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Closed-loop sensitivity</div>
          <Tex expr={psiExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />
        </FormSection>

        {/* ---- Objective · epigraph · convexified · hard constraints ---- */}
        <FormSection title="Objective · H∞ model matching">
          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-1">Original criterion</div>
          <Tex expr={hinfObjExpr} block className="text-gray-200 text-[13px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Epigraph form</div>
          <Tex expr={epigraphExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">
            Convexified · linearised about <Tex expr={String.raw`\rho_0`} />
          </div>
          <Tex expr={convexExpr} block className="text-gray-200 text-[11.5px] overflow-x-auto" />

          <div
            className="text-[10px] font-mono uppercase tracking-[0.18em] mt-4 mb-2"
            style={{ color: ACCENT_GREEN }}
          >
            Subject to · ∀ ω ∈ Ω, k = 1…a
          </div>

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mb-1">
            Modulus margin <Tex expr={String.raw`\Delta M_{ib}`} />
          </div>
          <Tex expr={mmExpr} block className="text-gray-200 text-[11px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">
            Noise attenuation at <Tex expr={String.raw`\omega_k`} />
          </div>
          <Tex expr={noiseExpr} block className="text-gray-200 text-[11px] overflow-x-auto" />

          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-3 mb-1">Open-loop controller stability</div>
          <Tex expr={ctrlStabExpr} block className="text-gray-200 text-[12px] overflow-x-auto" />
        </FormSection>
      </div>

      <div className="mt-5 rounded-xl border border-white/5 bg-black/20 p-4">
        <div
          className="text-[10.5px] font-mono uppercase tracking-[0.18em] mb-2"
          style={{ color: ACCENT_GOLD }}
        >
          What each constraint buys you
        </div>
        <ul className="text-[11px] text-gray-400 leading-relaxed space-y-1.5">
          <li>
            <span className="text-gray-200">Modulus margin</span> —
            keeps <Tex expr={String.raw`|\mathcal{S}_{d_y y}|`} /> below{" "}
            <Tex expr={String.raw`1/\Delta M_{ib}`} />, guaranteeing a minimum
            distance from the Nyquist locus to the −1 point. Robustness to
            plant mismatch.
          </li>
          <li>
            <span className="text-gray-200">Noise attenuation at</span>{" "}
            <Tex expr={String.raw`\omega_k`} /> — bounds the sensitivity from
            voltage-loop disturbance to output at each measured frequency by{" "}
            <Tex expr={String.raw`A_k`} />. Prevents amplification of
            known-noisy bands (e.g. rectifier ripple).
          </li>
          <li>
            <span className="text-gray-200">Open-loop controller stability</span>{" "}
            — sufficient condition for the zeros of{" "}
            <Tex expr={String.raw`S(z^{-1})`} /> to lie inside the unit
            circle, so the implemented controller is itself stable (matters
            for anti-windup back-calculation).
          </li>
        </ul>
      </div>

      <div className="mt-4 text-[10px] font-mono text-gray-500 leading-relaxed">
        <span style={{ color: ACCENT_CYAN }}>W</span> shapes the error toward
        the user-chosen bandwidth <Tex expr={String.raw`f_c`} /> and damping{" "}
        <Tex expr={String.raw`\zeta`} />;{" "}
        <span style={{ color: ACCENT_GREEN }}>γ</span> is driven down by
        bisection. Constraints are evaluated on the measured FRF grid —{" "}
        <Tex expr={String.raw`k = 1,\ldots,a`} /> indexes the frequency
        samples. Controller sample time <Tex expr={String.raw`T_s`} /> ={" "}
        {Ts_us} µs. Outer loop: bisect{" "}
        <Tex expr={String.raw`\gamma`} /> → solve the inner convex SDP → set{" "}
        <Tex expr={String.raw`\psi(\rho_0) \leftarrow \psi(\rho^{\star})`} />{" "}
        → repeat until{" "}
        <Tex expr={String.raw`|\gamma_{i+1} - \gamma_i|`} /> drops below
        tolerance. Output: the RST polynomials shown above.
      </div>
    </div>
  );
}

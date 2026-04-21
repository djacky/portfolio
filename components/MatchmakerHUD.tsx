"use client";

/* ------------------------------------------------------------------
   MatchmakerHUD — side panel. Metrics + controls + rolling loss plot.

   The attention inspector moved into the scene's sequence tape (that
   view shows the tokens and the attention together — they're easier
   to read side-by-side than separately).
------------------------------------------------------------------ */

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { Pause, Play, RefreshCw, Bug, Brain, Sliders } from "lucide-react";
import { getEngine } from "@/lib/matchmaker-engine";

const LOSS_HISTORY = 120;

// Dev-only diagnostics gate.  Webpack splits MatchmakerHUDDev into its own
// chunk; in prod, DEV folds to false so `DevDiagnostics` stays null and the
// chunk is never fetched by the browser (the `import()` inside `dynamic()`
// is only evaluated when dynamic() is called).
const DEV = process.env.NODE_ENV !== "production";
const DevDiagnostics = DEV
  ? dynamic(() => import("./MatchmakerHUDDev"), { ssr: false })
  : null;

export default function MatchmakerHUD() {
  const engine = getEngine();
  const [snap, setSnap] = useState(() => engine.snapshot());
  const [paused, setPaused] = useState(engine.trainingPaused);
  const [lr, setLr] = useState(engine.lr);
  const [speed, setSpeed] = useState<1 | 3 | 5>(engine.speedMultiplier as 1 | 3 | 5);

  const lossHistRef = useRef({
    skill: [] as number[],
    contrast: [] as number[],
    anomaly: [] as number[],
    // Dev-only: held-out PL NLL track for the overfitting-gap sparkline.
    heldOut: [] as number[],
  });

  useEffect(() => {
    const unsub = engine.subscribe(() => {
      const s = engine.snapshot();
      setSnap(s);
      const lh = lossHistRef.current;
      lh.skill.push(s.skillLoss);
      lh.contrast.push(s.contrastLoss);
      lh.anomaly.push(s.anomalyLoss);
      if (lh.skill.length > LOSS_HISTORY) lh.skill.shift();
      if (lh.contrast.length > LOSS_HISTORY) lh.contrast.shift();
      if (lh.anomaly.length > LOSS_HISTORY) lh.anomaly.shift();
      if (DEV) {
        lh.heldOut.push(s.heldOutNll ?? 0);
        if (lh.heldOut.length > LOSS_HISTORY) lh.heldOut.shift();
      }
    });
    return unsub;
  }, [engine]);

  const phaseLabel =
    snap.phase === "idle"      ? "queue" :
    snap.phase === "forming"   ? "lobby forming" :
    snap.phase === "playing"   ? "match live" :
                                 "resolving";

  return (
    <div className="flex flex-col gap-4">
      {/* Metric strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        <Metric label="rank ρ" value={snap.rankCorrelation.toFixed(2)} hint="Spearman(μ̂, trueSkill)" tint="#7c5cff" />
        <Metric
          label="lobby σ"
          value={snap.lobbySpread.toFixed(3)}
          hint={`vs random ${snap.randomBaseline.toFixed(3)}`}
          tint="#22d3ee"
          good={snap.lobbySpread > 0 && snap.lobbySpread < snap.randomBaseline}
        />
        <Metric label="matches" value={snap.matchesPlayed.toString()} hint={phaseLabel} tint="#a78bfa" />
        <Metric label="smurf P" value={(snap.smurfPrecision * 100).toFixed(0) + "%"} hint="precision" tint="#ef4444" />
        <Metric label="smurf R" value={(snap.smurfRecall * 100).toFixed(0) + "%"} hint="recall" tint="#f59e0b" />
        <Metric label="total loss" value={snap.loss.toFixed(2)} hint="skill + anom + 0.3·σ" tint="#34d399" />
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2 p-3 rounded-xl bg-black/30 border border-white/5">
        <button
          onClick={() => { setPaused((v) => { engine.setPaused(!v); return !v; }); }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-white/10 bg-white/5 hover:bg-white/10 text-xs font-mono"
          title="pause/resume training"
        >
          {paused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
          <span>{paused ? "resume" : "pause"} training</span>
        </button>

        <button
          onClick={() => engine.resetModel()}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-white/10 bg-white/5 hover:bg-white/10 text-xs font-mono"
          title="reset transformer weights"
        >
          <Brain className="w-3.5 h-3.5" />
          <span>reset model</span>
        </button>

        <button
          onClick={() => engine.reseed(160)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-white/10 bg-white/5 hover:bg-white/10 text-xs font-mono"
          title="spawn a fresh 160-player queue"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>reseed queue</span>
        </button>

        <button
          onClick={() => engine.injectSmurf()}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-red-400/40 bg-red-500/10 hover:bg-red-500/20 text-xs font-mono text-red-200"
          title="inject a fresh high-skill smurf account"
        >
          <Bug className="w-3.5 h-3.5" />
          <span>inject smurf</span>
        </button>

        <div className="flex items-center gap-1 text-[10px] font-mono text-gray-400"
             title="scales both training and match-generation timers">
          <span className="uppercase tracking-wider">speed</span>
          {([1, 3, 5] as const).map((m) => (
            <button
              key={m}
              onClick={() => { setSpeed(m); engine.setSpeed(m); }}
              className={
                "px-2 py-1 rounded-md border text-[11px] " +
                (speed === m
                  ? "border-amber-300/50 bg-amber-300/10 text-amber-200"
                  : "border-white/10 bg-white/5 hover:bg-white/10 text-gray-300")
              }
            >
              {m}×
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 ml-auto text-xs font-mono text-gray-400">
          <Sliders className="w-3.5 h-3.5" />
          <span>lr</span>
          <input
            type="range"
            min={0.0005}
            max={0.01}
            step={0.0005}
            value={lr}
            onChange={(e) => { const v = parseFloat(e.target.value); setLr(v); engine.setLr(v); }}
            className="w-24 accent-accent"
          />
          <span className="text-gray-300 w-12 tabular-nums">{lr.toFixed(4)}</span>
        </div>
      </div>

      {/* Loss panel */}
      <div className="p-3 rounded-xl bg-black/30 border border-white/5">
        <div className="flex items-center justify-between mb-2">
          <div className="text-[10px] font-mono uppercase tracking-wider text-gray-400">
            training losses · last {LOSS_HISTORY} ticks
          </div>
          <div className="flex gap-3 text-[10px] font-mono">
            <span className="text-[#7c5cff]">● skill {snap.skillLoss.toFixed(3)}</span>
            <span className="text-[#22d3ee]">● contrast {snap.contrastLoss.toFixed(3)}</span>
            <span className="text-[#ef4444]">● anomaly {snap.anomalyLoss.toFixed(3)}</span>
          </div>
        </div>
        <LossSparkline
          lines={[
            { color: "#7c5cff", data: lossHistRef.current.skill },
            { color: "#22d3ee", data: lossHistRef.current.contrast },
            { color: "#ef4444", data: lossHistRef.current.anomaly },
          ]}
        />
      </div>

      {DEV && DevDiagnostics && snap.dev && (
        <DevDiagnostics
          calibration1={snap.calibration1 ?? 0}
          calibration2={snap.calibration2 ?? 0}
          baselineRho={snap.baselineRho ?? 0}
          rho={snap.rankCorrelation}
          skillLoss={snap.skillLoss}
          heldOutNll={snap.heldOutNll ?? 0}
          sigmaLoss={snap.sigmaLoss}
          skillHist={lossHistRef.current.skill}
          heldOutHist={lossHistRef.current.heldOut}
          lossHistoryLen={LOSS_HISTORY}
        />
      )}

      {/* Event toast */}
      <AnimatePresence>
        {snap.lastEvent && Date.now() - snap.lastEventAt < 4000 && (
          <motion.div
            key={snap.lastEventAt}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="text-[11px] font-mono px-3 py-2 rounded-lg border border-accent2/40 bg-accent2/5 text-accent2"
          >
            → {snap.lastEvent}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Metric({
  label, value, hint, tint, good,
}: {
  label: string;
  value: string;
  hint: string;
  tint: string;
  good?: boolean;
}) {
  return (
    <div
      className="rounded-xl p-2.5 border"
      style={{
        background: "rgba(11,15,26,0.55)",
        borderColor: good ? "#34d39966" : "rgba(255,255,255,0.06)",
      }}
    >
      <div className="text-[9px] font-mono uppercase tracking-wider text-gray-500">{label}</div>
      <div className="text-lg font-semibold tabular-nums" style={{ color: tint }}>{value}</div>
      <div className="text-[9px] font-mono text-gray-500 truncate">{hint}</div>
    </div>
  );
}

function LossSparkline({ lines }: { lines: { color: string; data: number[] }[] }) {
  const W = 320, H = 60;
  const all = lines.flatMap((l) => l.data);
  const maxV = Math.max(0.001, ...all);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
      <line x1={0} y1={H - 0.5} x2={W} y2={H - 0.5} stroke="#1f2937" />
      {lines.map((l, li) => {
        if (l.data.length < 2) return null;
        const pts = l.data.map((v, i) => {
          const x = (i / (LOSS_HISTORY - 1)) * W;
          const y = H - (v / maxV) * (H - 4) - 2;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(" ");
        return <polyline key={li} fill="none" stroke={l.color} strokeWidth={1.4} points={pts} opacity={0.9} />;
      })}
    </svg>
  );
}

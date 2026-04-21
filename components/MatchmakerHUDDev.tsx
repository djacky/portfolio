"use client";

/* ------------------------------------------------------------------
   MatchmakerHUDDev — dev-only convergence diagnostics panel.

   This entire module is pulled in via a conditional require() from
   MatchmakerHUD that is only reachable when NODE_ENV !== "production",
   so webpack tree-shakes the whole file (including the FlaskConical
   icon import) out of production bundles.

   Renders three convergence checks that the top-line ρ alone can hide:
     (1) σ̂ calibration at ±1σ / ±2σ — is the uncertainty head honest?
     (2) baseline-beat ρ — is the transformer outperforming a dumb
         EMA-of-past-placements heuristic?
     (3) held-out vs train PL NLL — is skill loss generalizing or
         just overfitting the completed-lobby ring buffer?
------------------------------------------------------------------ */

import { FlaskConical } from "lucide-react";

export interface DevDiagnosticsProps {
  calibration1: number;
  calibration2: number;
  baselineRho: number;
  rho: number;
  skillLoss: number;
  heldOutNll: number;
  sigmaLoss: number;
  skillHist: number[];
  heldOutHist: number[];
  lossHistoryLen: number;
}

export default function DevDiagnostics({
  calibration1, calibration2, baselineRho, rho,
  skillLoss, heldOutNll, sigmaLoss, skillHist, heldOutHist, lossHistoryLen,
}: DevDiagnosticsProps) {
  const lift = rho - baselineRho;
  const cal1Good = Math.abs(calibration1 - 0.68) < 0.08;
  const cal2Good = Math.abs(calibration2 - 0.95) < 0.06;
  const liftGood = lift > 0.05;
  // σ tracks Glicko teacher well if MSE(log σ̂, log σ_target) < 0.04 —
  // i.e. σ̂ within ~20% of target on average.
  const sigmaGood = sigmaLoss < 0.04;
  const overfitGap = heldOutNll - skillLoss;
  return (
    <div className="p-3 rounded-xl bg-black/30 border border-amber-300/15">
      <div className="flex items-center gap-1.5 mb-2">
        <FlaskConical className="w-3 h-3 text-amber-300/80" />
        <div className="text-[10px] font-mono uppercase tracking-wider text-amber-300/80">
          dev diagnostics · convergence checks
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mb-3">
        <DevTile label="cal 1σ" value={(calibration1 * 100).toFixed(0) + "%"}
                 hint="target 68%" tint="#fbbf24" good={cal1Good} />
        <DevTile label="cal 2σ" value={(calibration2 * 100).toFixed(0) + "%"}
                 hint="target 95%" tint="#fbbf24" good={cal2Good} />
        <DevTile label="σ MSE" value={sigmaLoss.toFixed(3)}
                 hint="log σ̂ ↔ Glicko teacher" tint="#f472b6" good={sigmaGood} />
        <DevTile label="baseline ρ" value={baselineRho.toFixed(2)}
                 hint="EMA-winMag heuristic" tint="#94a3b8" />
        <DevTile label="lift" value={(lift >= 0 ? "+" : "") + lift.toFixed(2)}
                 hint="model − baseline" tint="#34d399" good={liftGood} />
      </div>
      <div className="flex items-center justify-between mb-1">
        <div className="text-[10px] font-mono uppercase tracking-wider text-gray-400">
          train vs held-out skill NLL · overfitting gap
        </div>
        <div className="flex gap-3 text-[10px] font-mono">
          <span className="text-[#7c5cff]">● train {skillLoss.toFixed(3)}</span>
          <span className="text-[#f472b6]">● held-out {heldOutNll.toFixed(3)}</span>
          <span className={overfitGap > 0.05 ? "text-red-300" : "text-gray-500"}>
            Δ {(overfitGap >= 0 ? "+" : "") + overfitGap.toFixed(3)}
          </span>
        </div>
      </div>
      <DevSparkline
        histLen={lossHistoryLen}
        lines={[
          { color: "#7c5cff", data: skillHist },
          { color: "#f472b6", data: heldOutHist },
        ]}
      />
    </div>
  );
}

function DevTile({
  label, value, hint, tint, good,
}: { label: string; value: string; hint: string; tint: string; good?: boolean }) {
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

function DevSparkline({
  histLen, lines,
}: { histLen: number; lines: { color: string; data: number[] }[] }) {
  const W = 320, H = 60;
  const all = lines.flatMap((l) => l.data);
  const maxV = Math.max(0.001, ...all);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
      <line x1={0} y1={H - 0.5} x2={W} y2={H - 0.5} stroke="#1f2937" />
      {lines.map((l, li) => {
        if (l.data.length < 2) return null;
        const pts = l.data.map((v, i) => {
          const x = (i / (histLen - 1)) * W;
          const y = H - (v / maxV) * (H - 4) - 2;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(" ");
        return <polyline key={li} fill="none" stroke={l.color} strokeWidth={1.4} points={pts} opacity={0.9} />;
      })}
    </svg>
  );
}

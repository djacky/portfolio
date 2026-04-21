"use client";

/* ------------------------------------------------------------------
   MatchmakerScene2D — convergence-focused visualization.

   Four stacked panels:

     1. CONVERGENCE SCATTER (canvas, left)
        x = trueSkill (hidden ground truth, [0,1])
        y = μ̂         (learned posterior mean, [0,1])
        vertical whisker = μ̂ ± σ̂  (Bayesian uncertainty)
        color = true tier; smurfs = red core + ring
        dashed y = x diagonal = perfect calibration
        live R² · cal1 · medErr readouts in the header
        hover / click to inspect (feeds LobbyStrip + SequenceTape)

     2. TIER CONFUSION MATRIX (html grid, right)
        4 × 4 of predicted tier (rows) vs true tier (cols).
        Diagonal cells glow gold (correct), off-diagonal cells
        purple (confusions).  At start the counts smear across all
        cells; at convergence mass concentrates on the diagonal.

     3. LOBBY STRIP (unchanged)
     4. SEQUENCE TAPE (unchanged)
------------------------------------------------------------------ */

import React, { useEffect, useMemo, useRef, useState } from "react";
import { getEngine, EngineSnapshot } from "@/lib/matchmaker-engine";
import { Player, LOBBY_SIZE, tierOf, Tier } from "@/lib/matchmaker-sim";

const W = 520;
const H = 440;
const MARGIN = { top: 40, right: 22, bottom: 44, left: 54 };
const PLOT_W = W - MARGIN.left - MARGIN.right;
const PLOT_H = H - MARGIN.top - MARGIN.bottom;

const TIERS: Tier[] = ["Bronze", "Silver", "Gold", "Diamond"];
const TIER_COLOR: Record<Tier, string> = {
  Bronze:  "#c2843f",
  Silver:  "#cbd5e1",
  Gold:    "#fbbf24",
  Diamond: "#22d3ee",
};
const TIER_BANDS: Array<{ lo: number; hi: number; tier: Tier }> = [
  { lo: 0.0,  hi: 0.25, tier: "Bronze"  },
  { lo: 0.25, hi: 0.5,  tier: "Silver"  },
  { lo: 0.5,  hi: 0.8,  tier: "Gold"    },
  { lo: 0.8,  hi: 1.0,  tier: "Diamond" },
];
const TIER_IDX: Record<Tier, number> = { Bronze: 0, Silver: 1, Gold: 2, Diamond: 3 };

function toX(trueSkill: number): number {
  const c = trueSkill < 0 ? 0 : trueSkill > 1 ? 1 : trueSkill;
  return MARGIN.left + c * PLOT_W;
}
function toY(mu: number): number {
  const c = mu < 0 ? 0 : mu > 1 ? 1 : mu;
  return MARGIN.top + (1 - c) * PLOT_H;
}

function hexA(hex: string, a: number): string {
  const h = hex.startsWith("#") ? hex.slice(1) : hex;
  if (h.length !== 6) return hex;
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

interface Props {
  hoveredId: number | null;
  pinnedId: number | null;
  onHover: (id: number | null) => void;
  onPin: (id: number | null) => void;
}

export default function MatchmakerScene2D({ hoveredId, pinnedId, onHover, onPin }: Props) {
  const inspectId = pinnedId ?? hoveredId;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] gap-3">
        <ConvergenceScatter
          hoveredId={hoveredId}
          pinnedId={pinnedId}
          onHover={onHover}
          onPin={onPin}
        />
        <ConfusionMatrix />
      </div>
      <LobbyStrip />
      <SequenceTape inspectId={inspectId} />
    </div>
  );
}

/* ---------------- convergence scatter ---------------- */

function ConvergenceScatter({ hoveredId, pinnedId, onHover, onPin }: Props) {
  const engine = getEngine();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Snapshot + smoothed dot positions live in refs so the RAF loop never
  // triggers a React re-render.
  const snapRef = useRef<EngineSnapshot>(engine.snapshot());
  const smoothPosRef = useRef<Map<number, { x: number; y: number; sigma: number }>>(new Map());
  const lastFrameRef = useRef(performance.now());
  const hoveredRef = useRef(hoveredId);
  const pinnedRef = useRef(pinnedId);

  // Live KPIs displayed in the header — recomputed every frame from snap.
  const kpiRef = useRef({ r2: 0, cal1: 0, medErr: 0 });
  // React state purely for the header pills (numbers update at ~2 Hz, not
  // every frame — that avoids noisy tens-of-hertz flicker on the text).
  const [kpi, setKpi] = useState(kpiRef.current);
  const lastKpiFlushRef = useRef(0);

  useEffect(() => { hoveredRef.current = hoveredId; }, [hoveredId]);
  useEffect(() => { pinnedRef.current = pinnedId; }, [pinnedId]);

  useEffect(() => {
    return engine.subscribe(() => {
      snapRef.current = engine.snapshot();
    });
  }, [engine]);

  useEffect(() => {
    let raf = 0;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);

    // Exponential smoothing — half-life ≈ 126ms.  Independent of frame rate.
    const K = 5.5;

    const loop = () => {
      raf = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = Math.min(0.05, Math.max(0.001, (now - lastFrameRef.current) / 1000));
      lastFrameRef.current = now;
      const alpha = 1 - Math.exp(-K * dt);

      const smooth = smoothPosRef.current;
      const snap = snapRef.current;
      // Running sums for the KPI pills (reuses the main dot loop for speed).
      let n = 0, sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
      let within1 = 0;
      const errs: number[] = [];

      for (const p of snap.players) {
        if (!Number.isFinite(p.muHatEma) || !Number.isFinite(p.trueSkill)) continue;
        const tx = toX(p.trueSkill);
        const ty = toY(p.muHatEma);
        const cur = smooth.get(p.id);
        if (!cur) {
          smooth.set(p.id, { x: tx, y: ty, sigma: p.sigmaHat });
        } else {
          cur.x += (tx - cur.x) * alpha;
          cur.y += (ty - cur.y) * alpha;
          cur.sigma += (p.sigmaHat - cur.sigma) * alpha;
        }
        // KPIs — computed on raw (not smoothed) values so the readouts
        // match the snapshot, not the animation.  Require some history so
        // brand-new players don't poison the pool with their μ̂ = 0.5 prior.
        if (p.history.length >= 2) {
          n++;
          sumX += p.trueSkill;
          sumY += p.muHatEma;
          sumXX += p.trueSkill * p.trueSkill;
          sumYY += p.muHatEma * p.muHatEma;
          sumXY += p.trueSkill * p.muHatEma;
          const err = Math.abs(p.trueSkill - p.muHatEma);
          errs.push(err);
          if (err <= p.sigmaHat) within1++;
        }
      }

      // Pearson² — well-defined, scale-aware measure of linear convergence.
      // (Spearman ρ was already in the HUD; R² complements it by penalizing
      // scale drift, which the Spearman doesn't see.)
      let r2 = 0;
      if (n > 1) {
        const denomX = Math.sqrt(Math.max(1e-9, n * sumXX - sumX * sumX));
        const denomY = Math.sqrt(Math.max(1e-9, n * sumYY - sumY * sumY));
        const cov = n * sumXY - sumX * sumY;
        const r = cov / (denomX * denomY + 1e-9);
        r2 = r * r;
      }
      const cal1 = n > 0 ? within1 / n : 0;
      errs.sort((a, b) => a - b);
      const medErr = errs.length > 0 ? errs[Math.floor(errs.length / 2)] : 0;
      kpiRef.current = { r2, cal1, medErr };
      if (now - lastKpiFlushRef.current > 500) {
        lastKpiFlushRef.current = now;
        setKpi({ r2, cal1, medErr });
      }

      drawScatter(ctx, snap, smooth, hoveredRef.current, pinnedRef.current);
    };
    loop();
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pickPlayer = (clientX: number, clientY: number): number | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const cx = (clientX - rect.left) * (W / rect.width);
    const cy = (clientY - rect.top) * (H / rect.height);
    const smooth = smoothPosRef.current;
    let best: number | null = null;
    let bestD2 = 16 * 16;
    for (const p of snapRef.current.players) {
      const pp = smooth.get(p.id);
      if (!pp) continue;
      const d2 = (pp.x - cx) ** 2 + (pp.y - cy) ** 2;
      if (d2 < bestD2) { bestD2 = d2; best = p.id; }
    }
    return best;
  };

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) =>
    onHover(pickPlayer(e.clientX, e.clientY));
  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const id = pickPlayer(e.clientX, e.clientY);
    onPin(id === pinnedRef.current ? null : id);
  };

  // cal1 target is 0.68 (1σ Gaussian); color-code the pill by proximity
  const calGood = kpi.cal1 >= 0.6 && kpi.cal1 <= 0.85;
  const r2Good = kpi.r2 >= 0.85;

  return (
    <div className="rounded-2xl overflow-hidden bg-gradient-to-br from-[#05070e] to-[#04060c] border border-white/5 flex flex-col">
      <div className="flex items-center justify-between gap-2 px-3 pt-2.5 pb-2 flex-wrap">
        <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 whitespace-nowrap">
          convergence · μ̂ vs trueSkill
        </div>
        <div className="flex gap-1.5 flex-wrap justify-end">
          <KpiPill label="R²"     value={kpi.r2.toFixed(2)}                good={r2Good}  tint="#a78bfa" />
          <KpiPill label="cal 1σ" value={`${(kpi.cal1 * 100).toFixed(0)}%`} good={calGood} tint="#22d3ee" />
          <KpiPill label="med|e|" value={kpi.medErr.toFixed(3)}             tint="#34d399" />
        </div>
      </div>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", aspectRatio: `${W} / ${H}`, display: "block", cursor: "crosshair" }}
        onPointerMove={onMove}
        onPointerLeave={() => onHover(null)}
        onPointerDown={onDown}
      />
    </div>
  );
}

function KpiPill({ label, value, good, tint }: { label: string; value: string; good?: boolean; tint: string }) {
  return (
    <div
      className="px-2 py-0.5 rounded-md border text-[10px] font-mono tabular-nums"
      style={{
        background: good ? "rgba(52,211,153,0.10)" : "rgba(11,15,26,0.78)",
        borderColor: good ? "rgba(52,211,153,0.45)" : "rgba(255,255,255,0.08)",
        color: good ? "#86efac" : "#cbd5e1",
      }}
    >
      <span className="uppercase tracking-wider mr-1" style={{ color: tint, opacity: 0.85 }}>{label}</span>
      <span>{value}</span>
    </div>
  );
}

function drawScatter(
  ctx: CanvasRenderingContext2D,
  snap: EngineSnapshot,
  pos: Map<number, { x: number; y: number; sigma: number }>,
  hoveredId: number | null,
  pinnedId: number | null,
) {
  // ---------- background ----------
  ctx.fillStyle = "#04060c";
  ctx.fillRect(0, 0, W, H);

  // Tier bands along the trueSkill (X) axis — subtle vertical strips so
  // you can read the ground-truth tier of any dot by its x position.
  for (const b of TIER_BANDS) {
    const xL = toX(b.lo);
    const xR = toX(b.hi);
    ctx.fillStyle = TIER_COLOR[b.tier] + "0c"; // ~5% alpha
    ctx.fillRect(xL, MARGIN.top, xR - xL, PLOT_H);
  }

  // plot-area soft tint
  ctx.fillStyle = "rgba(124, 92, 255, 0.02)";
  ctx.fillRect(MARGIN.left, MARGIN.top, PLOT_W, PLOT_H);

  // grid — 5 × 5
  ctx.strokeStyle = "rgba(255,255,255,0.045)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = MARGIN.top + (i / 5) * PLOT_H;
    ctx.beginPath(); ctx.moveTo(MARGIN.left, y); ctx.lineTo(MARGIN.left + PLOT_W, y); ctx.stroke();
    const x = MARGIN.left + (i / 5) * PLOT_W;
    ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, MARGIN.top + PLOT_H); ctx.stroke();
  }

  // tier-boundary dashed rules (both axes — the 4×4 grid of tier cells is
  // the same lens as the confusion matrix, just laid over the scatter)
  ctx.setLineDash([3, 5]);
  for (const v of [0.25, 0.5, 0.8]) {
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(MARGIN.left, y); ctx.lineTo(MARGIN.left + PLOT_W, y); ctx.stroke();
    const x = toX(v);
    ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, MARGIN.top + PLOT_H); ctx.stroke();
  }
  ctx.setLineDash([]);

  // ---------- y = x diagonal (perfect calibration) ----------
  // Glow layer first (wide, faint) then the sharp dashed line on top.
  const x0 = toX(0), y0 = toY(0);
  const x1 = toX(1), y1 = toY(1);
  ctx.save();
  ctx.strokeStyle = "rgba(167, 139, 250, 0.18)";
  ctx.lineWidth = 6;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.strokeStyle = "rgba(245, 245, 255, 0.55)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 6]);
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // "y = x · perfect" label, sitting on the diagonal near the top-right
  ctx.save();
  const lblX = toX(0.82), lblY = toY(0.82);
  ctx.translate(lblX, lblY);
  ctx.rotate(-Math.PI / 4);
  ctx.fillStyle = "rgba(226, 232, 240, 0.65)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("y = x  ·  perfect calibration", 0, -5);
  ctx.restore();

  // ---------- axis labels ----------
  ctx.fillStyle = "#64748b";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of [0, 0.25, 0.5, 0.8, 1]) {
    ctx.fillText(v.toFixed(2), MARGIN.left - 6, toY(v));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of [0, 0.25, 0.5, 0.8, 1]) {
    ctx.fillText(v.toFixed(2), toX(v), MARGIN.top + PLOT_H + 6);
  }

  // tier labels on x-axis (small chips above the ticks, in tier color)
  ctx.font = "9px ui-monospace, monospace";
  ctx.textBaseline = "bottom";
  for (const b of TIER_BANDS) {
    const cx = toX((b.lo + b.hi) / 2);
    ctx.fillStyle = TIER_COLOR[b.tier] + "aa";
    ctx.fillText(b.tier.toUpperCase(), cx, MARGIN.top - 8);
  }

  // Axis titles
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "11px ui-monospace, monospace";
  ctx.save();
  ctx.translate(14, MARGIN.top + PLOT_H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("predicted  μ̂  →", 0, 0);
  ctx.restore();
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("true skill (hidden)  →", MARGIN.left + PLOT_W / 2, H - 8);

  // ---------- error-bar whiskers (first layer — behind dots) ----------
  // Drawn as thin vertical lines of length ±σ̂ in μ̂-units.  Low alpha so
  // 160 whiskers don't overwhelm the scatter; the aggregate gives a sense
  // of pool-wide uncertainty shrinking as σ̂ tightens.
  for (const p of snap.players) {
    const pp = pos.get(p.id);
    if (!pp) continue;
    if (p.history.length < 1) continue;
    const yTop = toY(Math.min(1, p.muHatEma + pp.sigma));
    const yBot = toY(Math.max(0, p.muHatEma - pp.sigma));
    ctx.strokeStyle = hexA(p.color, 0.22);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pp.x, yTop); ctx.lineTo(pp.x, yBot);
    ctx.stroke();
    // tiny caps — make the whisker read as an error bar, not just a line
    ctx.beginPath();
    ctx.moveTo(pp.x - 2, yTop); ctx.lineTo(pp.x + 2, yTop);
    ctx.moveTo(pp.x - 2, yBot); ctx.lineTo(pp.x + 2, yBot);
    ctx.stroke();
  }

  // ---------- dots ----------
  for (const p of snap.players) {
    const pp = pos.get(p.id);
    if (!pp) continue;
    const isHovered = p.id === hoveredId;
    const isPinned = p.id === pinnedId;
    const isInLobby = snap.lobbyIds.has(p.id);
    // All dots the same base size — this plot is about calibration, not
    // confidence rendering (the whisker already encodes σ̂).
    const r = 3.5 * (isInLobby ? 1.15 : 1) * (isHovered || isPinned ? 1.4 : 1);

    // Outer glow for hovered/pinned/lobby members
    if (isHovered || isPinned || isInLobby) {
      ctx.fillStyle = hexA(isInLobby ? "#22d3ee" : p.color, 0.18);
      ctx.beginPath();
      ctx.arc(pp.x, pp.y, r + 4.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // core
    ctx.fillStyle = p.anomalyP > 0.5 ? "#ef4444" : p.color;
    ctx.beginPath();
    ctx.arc(pp.x, pp.y, r, 0, Math.PI * 2);
    ctx.fill();

    // smurf outer ring
    if (p.anomalyP > 0.5) {
      ctx.strokeStyle = "#fca5a5";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(pp.x, pp.y, r + 2.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (isHovered || isPinned) {
      ctx.strokeStyle = isPinned ? "#ffffff" : "#e2e8f0";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pp.x, pp.y, r + 5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // ---------- hover tooltip ----------
  const tipId = hoveredId ?? pinnedId;
  if (tipId != null) {
    const p = snap.players.find((q) => q.id === tipId);
    if (p) {
      const pp = pos.get(p.id)!;
      const err = p.trueSkill - p.muHatEma;
      const l1 = p.name;
      const l2 = `${tierOf(p.trueSkill)} · true ${p.trueSkill.toFixed(2)} · μ̂ ${p.muHatEma.toFixed(2)} ± ${p.sigmaHat.toFixed(2)}`;
      const extras: string[] = [`err ${err >= 0 ? "+" : ""}${err.toFixed(2)}`, `${p.matchesPlayed} matches`];
      if (p.anomalyP > 0.5) extras.push(`smurf ${(p.anomalyP * 100).toFixed(0)}%`);
      const l3 = extras.join(" · ");

      ctx.font = "bold 11px ui-monospace, monospace";
      const w1 = ctx.measureText(l1).width;
      ctx.font = "10px ui-monospace, monospace";
      const w2 = ctx.measureText(l2).width;
      const w3 = ctx.measureText(l3).width;
      const tw = Math.max(w1, w2, w3) + 14;
      const th = 50;
      let tx = pp.x + 12;
      let ty = pp.y - th - 8;
      if (tx + tw > W - 4) tx = pp.x - tw - 12;
      if (ty < 4) ty = pp.y + 12;

      ctx.fillStyle = "rgba(5,7,13,0.95)";
      ctx.strokeStyle = p.anomalyP > 0.5 ? "#ef4444" : p.color;
      ctx.lineWidth = 1;
      roundRect(ctx, tx, ty, tw, th, 6);
      ctx.fill(); ctx.stroke();

      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#fff";
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.fillText(l1, tx + 7, ty + 6);
      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(l2, tx + 7, ty + 21);
      ctx.fillText(l3, tx + 7, ty + 34);
    }
  }
}

/* ---------------- tier confusion matrix ---------------- */

function ConfusionMatrix() {
  const engine = getEngine();
  const [snap, setSnap] = useState(() => engine.snapshot());
  useEffect(() => engine.subscribe(() => setSnap(engine.snapshot())), [engine]);

  const { counts, rowTotals, diagonal, total } = useMemo(() => {
    const m: number[][] = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
    let tot = 0, diag = 0;
    const rows = [0, 0, 0, 0];
    for (const p of snap.players) {
      if (p.history.length < 2) continue; // skip brand-new players
      const tr = TIER_IDX[tierOf(p.trueSkill)];
      const pr = TIER_IDX[tierOf(p.muHatEma)];
      m[pr][tr]++;
      rows[pr]++;
      tot++;
      if (pr === tr) diag++;
    }
    return { counts: m, rowTotals: rows, diagonal: diag, total: tot };
  }, [snap]);

  const accuracy = total > 0 ? diagonal / total : 0;

  return (
    <div className="rounded-2xl bg-gradient-to-br from-[#05070e] to-[#04060c] border border-white/5 p-4 flex flex-col min-h-0">
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500">
          tier confusion
        </div>
        <div
          className="text-xs font-mono tabular-nums px-2 py-0.5 rounded-md border"
          style={{
            color: accuracy >= 0.65 ? "#86efac" : "#fbbf24",
            borderColor: accuracy >= 0.65 ? "rgba(52,211,153,0.45)" : "rgba(251,191,36,0.4)",
            background: accuracy >= 0.65 ? "rgba(52,211,153,0.10)" : "rgba(251,191,36,0.08)",
          }}
        >
          {(accuracy * 100).toFixed(0)}% diagonal
        </div>
      </div>

      {/* grid: label column + 4 data cols; label row + 4 data rows */}
      <div
        className="grid gap-1.5 flex-1 min-h-0"
        style={{
          gridTemplateColumns: "auto repeat(4, minmax(0, 1fr))",
          gridTemplateRows: "auto repeat(4, minmax(0, 1fr))",
        }}
      >
        {/* top-left corner: tiny truth arrow */}
        <div className="flex items-end justify-end text-[8px] font-mono text-gray-600 pr-1 pb-0.5">
          ↘
        </div>
        {/* column headers — true tier */}
        {TIERS.map((t) => (
          <div
            key={`col-${t}`}
            className="text-[9px] font-mono uppercase tracking-wider text-center pb-0.5"
            style={{ color: TIER_COLOR[t] }}
          >
            {t.slice(0, 3)}
          </div>
        ))}

        {/* each row: label + 4 cells */}
        {TIERS.map((predT, i) => (
          <React.Fragment key={`row-${predT}`}>
            <div
              className="text-[9px] font-mono uppercase tracking-wider flex items-center justify-end pr-1"
              style={{ color: TIER_COLOR[predT] }}
            >
              {predT.slice(0, 3)}
            </div>
            {TIERS.map((trueT, j) => (
              <ConfusionCell
                key={`cell-${i}-${j}`}
                count={counts[i][j]}
                rowTotal={rowTotals[i]}
                diagonal={i === j}
              />
            ))}
          </React.Fragment>
        ))}
      </div>

      <div className="mt-3 flex items-center justify-between text-[9px] font-mono text-gray-500">
        <span>rows: predicted</span>
        <span>cols: truth</span>
      </div>
    </div>
  );
}

function ConfusionCell({
  count, rowTotal, diagonal,
}: { count: number; rowTotal: number; diagonal: boolean }) {
  // Cell intensity scales on the ROW total, not the global max — that keeps
  // a heavily-populated Bronze row from making Diamond cells invisible.
  const frac = rowTotal > 0 ? count / rowTotal : 0;
  // Gamma-compress so small non-zero counts are still visible.
  const intensity = Math.pow(frac, 0.6);

  // Two color ramps: green for diagonal (correct), purple for off-diagonal
  // (confusion).  Empty cells fade to a dark border-only state.
  const [r, g, b] = diagonal ? [52, 211, 153] : [167, 139, 250];
  const fillA = 0.08 + intensity * 0.82;
  const textA = count === 0 ? 0.25 : 0.55 + intensity * 0.45;

  return (
    <div
      className="rounded-lg flex items-center justify-center relative overflow-hidden"
      style={{
        background: count === 0
          ? "rgba(255,255,255,0.025)"
          : `rgba(${r}, ${g}, ${b}, ${fillA})`,
        border: diagonal
          ? `1px solid rgba(${r}, ${g}, ${b}, ${0.35 + intensity * 0.35})`
          : "1px solid rgba(255,255,255,0.045)",
        boxShadow: diagonal && intensity > 0.4
          ? `0 0 ${6 + intensity * 14}px rgba(${r}, ${g}, ${b}, ${0.15 + intensity * 0.2})`
          : undefined,
        transition: "background 320ms ease, box-shadow 320ms ease, border-color 320ms ease",
        aspectRatio: "1",
      }}
    >
      <span
        className="font-mono tabular-nums text-[13px]"
        style={{
          color: count === 0 ? "#475569" : diagonal ? "#f0fdf4" : "#ede9fe",
          opacity: textA,
          fontWeight: diagonal ? 600 : 400,
        }}
      >
        {count}
      </span>
    </div>
  );
}

/* ---------------- lobby strip ---------------- */

function LobbyStrip() {
  const engine = getEngine();
  const [snap, setSnap] = useState(() => engine.snapshot());
  useEffect(() => engine.subscribe(() => setSnap(engine.snapshot())), [engine]);

  const hasLobby = snap.lobbyIds.size > 0;

  let lobby: Player[] = [];
  if (hasLobby) {
    lobby = snap.players.filter((p) => snap.lobbyIds.has(p.id));
    if (snap.lastOrdering && (snap.phase === "finishing" || snap.phase === "playing")) {
      const orderedIds = new Set(snap.lastOrdering.map((p) => p.id));
      lobby = snap.lastOrdering.filter((p) => orderedIds.has(p.id) && snap.lobbyIds.has(p.id));
    }
  }

  // Same skeleton for both states (header row + 30-cell grid).  Empty
  // placeholder cells when there's no lobby prevent the vertical snap
  // when a lobby forms — the box is sized by 30 aspect-square cells
  // regardless of content.
  return (
    <div className="p-2 rounded-xl bg-black/30 border border-white/5">
      <div className="flex items-center justify-between mb-1.5 px-1">
        <div className={`text-[10px] font-mono uppercase tracking-wider ${hasLobby ? "text-cyan-400" : "text-gray-500"}`}>
          lobby · {hasLobby ? (phaseTag(snap.phase).toLowerCase() || "queue") : "waiting"}
        </div>
        <div className="text-[10px] font-mono text-gray-500">
          {hasLobby ? (
            <>
              true-skill σ <span className={snap.lobbySpread < snap.randomBaseline ? "text-good" : "text-gray-400"}>{snap.lobbySpread.toFixed(3)}</span>
              {" · "}random {snap.randomBaseline.toFixed(3)}
            </>
          ) : (
            <span className="italic">30 eligible players needed</span>
          )}
        </div>
      </div>
      <div
        className="grid gap-1"
        style={{ gridTemplateColumns: "repeat(30, minmax(0, 1fr))" }}
      >
        {hasLobby
          ? lobby.map((p, i) => (
              <div
                key={p.id}
                className="aspect-square rounded-full relative flex items-center justify-center"
                style={{
                  background: p.anomalyP > 0.5 ? "#ef4444" : p.color,
                  boxShadow: `0 0 6px ${p.anomalyP > 0.5 ? "#ef4444" : p.color}55`,
                }}
                title={`${i + 1}. ${p.name} (${tierOf(p.trueSkill)})`}
              >
                {snap.phase === "finishing" && (
                  <span className="text-[8px] font-bold text-black/70">{i + 1}</span>
                )}
              </div>
            ))
          : Array.from({ length: 30 }).map((_, i) => (
              <div
                key={`ph-${i}`}
                className="aspect-square rounded-full border border-white/5 bg-white/[0.02]"
              />
            ))}
      </div>
    </div>
  );
}

function phaseTag(phase: string): string {
  if (phase === "forming") return "FORMING";
  if (phase === "playing") return "MATCH LIVE";
  if (phase === "finishing") return "RESOLVING";
  return "";
}

/* ---------------- sequence tape ---------------- */

function SequenceTape({ inspectId }: { inspectId: number | null }) {
  const engine = getEngine();
  const [tick, setTick] = useState(0);
  useEffect(() => engine.subscribe(() => setTick((t) => t + 1)), [engine]);

  const data = useMemo(() => {
    if (inspectId == null) return null;
    const p = engine.players.find((q) => q.id === inspectId);
    if (!p) return null;
    if (p.history.length === 0) return { p, avgAttn: null as Float32Array | null };
    const attn = engine.attentionFor(p.id);
    let avgAttn: Float32Array | null = null;
    if (attn) {
      const T = attn.tokens, H = attn.heads;
      const lastLayer = attn.weights[attn.weights.length - 1];
      avgAttn = new Float32Array(T);
      for (let i = 0; i < T; i++) {
        let s = 0;
        for (let h = 0; h < H; h++) s += lastLayer[h * T + i];
        avgAttn[i] = s / H;
      }
      let mx = 0;
      for (let i = 0; i < T; i++) if (avgAttn[i] > mx) mx = avgAttn[i];
      if (mx > 0) for (let i = 0; i < T; i++) avgAttn[i] /= mx;
    }
    return { p, avgAttn };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inspectId, tick]);

  if (!data) {
    return (
      <div className="h-[96px] flex flex-col items-center justify-center rounded-xl bg-black/30 border border-white/5 gap-1">
        <div className="text-[11px] font-mono text-gray-400">hover a player on the scatter</div>
        <div className="text-[10px] font-mono text-gray-500 italic">
          you&apos;ll see their last 20 matches + which ones the model is looking at
        </div>
      </div>
    );
  }

  const { p, avgAttn } = data;

  return (
    <div className="p-3 rounded-xl bg-black/30 border border-white/5">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: p.color }} />
          <span className="text-xs font-semibold text-white">{p.name}</span>
          <span className="text-[10px] font-mono text-gray-400">
            {tierOf(p.trueSkill)} · μ̂ {p.muHat.toFixed(2)} ± {p.sigmaHat.toFixed(2)} · {p.matchesPlayed} matches
          </span>
          {p.anomalyP > 0.5 && (
            <span className="text-[10px] font-mono text-red-400">⚠ smurf {(p.anomalyP * 100).toFixed(0)}%</span>
          )}
          {p.tilt > 0.3 && (
            <span className="text-[10px] font-mono text-amber-400">⚡ tilt {(p.tilt * 100).toFixed(0)}%</span>
          )}
        </div>
        <div className="text-[9px] font-mono uppercase tracking-wider text-gray-500">
          sequence · oldest → newest · bar = attention
        </div>
      </div>
      <TapeRow player={p} avgAttn={avgAttn} />
    </div>
  );
}

function TapeRow({ player, avgAttn }: { player: Player; avgAttn: Float32Array | null }) {
  const T = Math.min(player.history.length, 20);
  const emptySlots = 20 - T;
  const cells: React.ReactNode[] = [];
  for (let i = 0; i < emptySlots; i++) {
    cells.push(
      <div key={`e-${i}`} className="flex-1 flex flex-col items-stretch gap-0.5 min-w-0">
        <div className="h-10 rounded border border-dashed border-white/5" />
        <div className="h-1.5 rounded-b bg-white/[0.02]" />
      </div>,
    );
  }
  for (let i = 0; i < T; i++) {
    const hist = player.history[player.history.length - T + i];
    const placement = hist.placement;
    const winMag = 1 - placement / LOBBY_SIZE;
    const hue = 46 - (1 - winMag) * 46;
    const a = avgAttn ? avgAttn[i] : 0;
    cells.push(
      <div key={i} className="flex-1 flex flex-col items-stretch gap-0.5 min-w-0">
        <div
          className="h-10 rounded flex items-center justify-center"
          style={{
            background: `hsl(${hue}, 85%, ${25 + winMag * 22}%)`,
            boxShadow: a > 0.35 ? `0 0 ${5 + a * 8}px hsl(${hue}, 85%, 55%)` : undefined,
            opacity: 0.55 + winMag * 0.45,
          }}
          title={`match ${i + 1}: placed ${placement}/30`}
        >
          <span className="text-[8px] font-bold text-black/60">{placement}</span>
        </div>
        <div
          className="rounded-b"
          style={{
            height: `${3 + a * 14}px`,
            background: `hsl(258, 89%, ${45 + a * 22}%)`,
            opacity: 0.35 + a * 0.65,
          }}
          title={avgAttn ? `attention: ${(a * 100).toFixed(0)}%` : ""}
        />
      </div>,
    );
  }
  return <div className="flex gap-1 items-end">{cells}</div>;
}

"use client";

/* ------------------------------------------------------------------
   MPCCircuitPanel — animated 3-phase grid-tied VSC schematic.

   Topology:
                    Cf (star, neutral floating)
                      │   │   │
       grid ─Lg─┬─Lf──┴───┴───┴──┐  ┌─ S1 S3 S5 ─┐
       a   Ea  │                 │  │            │   ┌─── DC+ ──┐
       grid ─Lg─┼─Lf──────────...─┼──┤  3-phase   │              │
       b   Eb  │                 │  │  bridge     │   C_DC  Z_LOAD
       grid ─Lg─┴─Lf─────────────┘  │            │              │
       c   Ec                       └─ S2 S4 S6 ─┘   └─── DC- ──┘

   Rendering: Canvas2D on a DPR-scaled buffer, ticked by rAF.  Each
   frame reads engineRef.current.{i_a,i_b,i_c,V_dc,u_d,u_q} and:
     • advances current-particle positions along each phase (signed
       speed by phase current; DC particles by power into the bus),
     • modulates IGBT conduction glow from per-leg averaged duty,
     • pulses inductor coils with |i_g_x|, the bus cap with |V_dc-Vref|.
   Particle speed is gated by slowMo so they always move at a watchable
   pace regardless of how fast the sim is being driven.
------------------------------------------------------------------ */

import { useEffect, useRef, type MutableRefObject } from "react";
import { MPCEngine, PARAMS, dqToAbc } from "@/lib/mpc-sim";

const W_ABS = 380;
const H_ABS = 110;

const COL_WIRE = "#475569";
const COL_AC_A = "#22d3ee";
const COL_AC_B = "#a78bfa";
const COL_AC_C = "#fb7185";
const COL_AC_LABEL = "#cbd5e1";
const COL_L = "#22d3ee";
const COL_C = "#fbbf24";
const COL_IGBT = "#94a3b8";
const COL_IGBT_ON = "#f472b6";
const COL_DIODE = "#64748b";
const COL_DCPLUS = "#fbbf24";
const COL_DCMINUS = "#94a3b8";
const COL_LABEL = "#94a3b8";
const COL_VALUE = "#e2e8f0";

// ----------------------------------------------------------
//  Phase color map
// ----------------------------------------------------------
const PHASE_COL: [string, string, string] = [COL_AC_A, COL_AC_B, COL_AC_C];

// ----------------------------------------------------------
//  Layout (abstract coords).  Three horizontal phase rails.
// ----------------------------------------------------------
const PHASE_Y: [number, number, number] = [22, 47, 72];

const X = {
  src: 22,           // grid source center x
  src_r: 7,
  src_tap: 32,       // wire begins
  Lg_in: 40,
  Lg_out: 68,
  Cf_x: 84,          // capacitor branch x
  Lf_in: 100,
  Lf_out: 128,
  // bridge (legs A, B, C)
  legA: 154,
  legB: 182,
  legC: 210,
  bridge_top_y: 8,
  bridge_bot_y: 86,
  // dc
  bus_cap: 248,
  dcdc: 294,          // DC/DC converter block center x
  car: 346,           // EV glyph center x
  // capacitor star point
  star_y: 99,
};

// ----------------------------------------------------------
//  Particle system (one set per signed wire segment)
// ----------------------------------------------------------

type SegKind =
  | "ac_a" | "ac_b" | "ac_c"
  | "dc_plus" | "dc_minus"
  | "dcdc_out"
  | "cap_a" | "cap_b" | "cap_c";
type Seg = { kind: SegKind; pts: { x: number; y: number }[] };

const SEGS: Seg[] = [
  // Phase A: src → Lg → Cf-tap → Lf → leg A midpoint
  {
    kind: "ac_a",
    pts: [
      { x: X.src_tap, y: PHASE_Y[0] },
      { x: X.Lg_in, y: PHASE_Y[0] },
      { x: X.Lg_out, y: PHASE_Y[0] },
      { x: X.Cf_x, y: PHASE_Y[0] },
      { x: X.Lf_in, y: PHASE_Y[0] },
      { x: X.Lf_out, y: PHASE_Y[0] },
      { x: X.legA, y: PHASE_Y[0] },
      { x: X.legA, y: 47 },
    ],
  },
  // Phase B
  {
    kind: "ac_b",
    pts: [
      { x: X.src_tap, y: PHASE_Y[1] },
      { x: X.Lg_in, y: PHASE_Y[1] },
      { x: X.Lg_out, y: PHASE_Y[1] },
      { x: X.Cf_x, y: PHASE_Y[1] },
      { x: X.Lf_in, y: PHASE_Y[1] },
      { x: X.Lf_out, y: PHASE_Y[1] },
      { x: X.legB, y: PHASE_Y[1] },
      { x: X.legB, y: 47 },
    ],
  },
  // Phase C
  {
    kind: "ac_c",
    pts: [
      { x: X.src_tap, y: PHASE_Y[2] },
      { x: X.Lg_in, y: PHASE_Y[2] },
      { x: X.Lg_out, y: PHASE_Y[2] },
      { x: X.Cf_x, y: PHASE_Y[2] },
      { x: X.Lf_in, y: PHASE_Y[2] },
      { x: X.Lf_out, y: PHASE_Y[2] },
      { x: X.legC, y: PHASE_Y[2] },
      { x: X.legC, y: 47 },
    ],
  },
  // DC+ rail: leg tops → bus cap → DC/DC converter input
  {
    kind: "dc_plus",
    pts: [
      { x: X.legA, y: X.bridge_top_y },
      { x: X.legB, y: X.bridge_top_y },
      { x: X.legC, y: X.bridge_top_y },
      { x: X.bus_cap, y: X.bridge_top_y },
      { x: X.dcdc - 11, y: X.bridge_top_y },
    ],
  },
  // DC- rail
  {
    kind: "dc_minus",
    pts: [
      { x: X.legA, y: X.bridge_bot_y },
      { x: X.legB, y: X.bridge_bot_y },
      { x: X.legC, y: X.bridge_bot_y },
      { x: X.bus_cap, y: X.bridge_bot_y },
      { x: X.dcdc - 11, y: X.bridge_bot_y },
    ],
  },
  // DC/DC → car charge port (DC+ across the top, drops into port;
  // DC- across the bottom, rises into port).  Particles on these wires
  // directly visualize the EV charging current.
  {
    kind: "dcdc_out",
    pts: [
      { x: X.dcdc + 11, y: X.bridge_top_y },
      { x: X.car - 11, y: X.bridge_top_y },
      { x: X.car - 11, y: 41 },
    ],
  },
  {
    kind: "dcdc_out",
    pts: [
      { x: X.dcdc + 11, y: X.bridge_bot_y },
      { x: X.car - 11, y: X.bridge_bot_y },
      { x: X.car - 11, y: 56 },
    ],
  },
  // Filter cap branches (each phase tap → star point)
  { kind: "cap_a", pts: [{ x: X.Cf_x, y: PHASE_Y[0] }, { x: X.Cf_x, y: X.star_y }] },
  { kind: "cap_b", pts: [{ x: X.Cf_x, y: PHASE_Y[1] }, { x: X.Cf_x, y: X.star_y }] },
  { kind: "cap_c", pts: [{ x: X.Cf_x, y: PHASE_Y[2] }, { x: X.Cf_x, y: X.star_y }] },
];

type Particle = { segIdx: number; t: number };
function makeParticles(count: number, segIdx: number): Particle[] {
  const out: Particle[] = [];
  for (let i = 0; i < count; i++) out.push({ segIdx, t: i / count });
  return out;
}
function segLength(seg: Seg): number {
  let len = 0;
  for (let i = 1; i < seg.pts.length; i++) {
    len += Math.hypot(seg.pts[i].x - seg.pts[i - 1].x, seg.pts[i].y - seg.pts[i - 1].y);
  }
  return len;
}
function pointAlong(seg: Seg, t: number): { x: number; y: number } {
  const total = segLength(seg);
  let target = t * total;
  for (let i = 1; i < seg.pts.length; i++) {
    const dx = seg.pts[i].x - seg.pts[i - 1].x;
    const dy = seg.pts[i].y - seg.pts[i - 1].y;
    const l = Math.hypot(dx, dy);
    if (target <= l) {
      const f = l === 0 ? 0 : target / l;
      return { x: seg.pts[i - 1].x + dx * f, y: seg.pts[i - 1].y + dy * f };
    }
    target -= l;
  }
  const last = seg.pts[seg.pts.length - 1];
  return { x: last.x, y: last.y };
}

// ----------------------------------------------------------
//  Drawing primitives
// ----------------------------------------------------------

function drawPolyline(
  ctx: CanvasRenderingContext2D,
  pts: { x: number; y: number }[],
  color: string,
  width: number,
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.stroke();
}

function drawInductor(
  ctx: CanvasRenderingContext2D,
  x0: number,
  x1: number,
  y: number,
  color: string,
  glow: number,
  label?: string,
) {
  const loops = 4;
  const span = x1 - x0;
  const r = span / (loops * 2);
  ctx.save();
  ctx.shadowBlur = 3 + glow * 7;
  ctx.shadowColor = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.1;
  for (let i = 0; i < loops; i++) {
    ctx.beginPath();
    ctx.arc(x0 + r + i * 2 * r, y, r, Math.PI, 0, false);
    ctx.stroke();
  }
  ctx.restore();
  if (label) {
    ctx.fillStyle = COL_LABEL;
    ctx.font = "5.5px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText(label, (x0 + x1) / 2, y - 4.5);
  }
}

/** Vertical Y-cap: two plates near the bottom of the branch. */
function drawCapVerticalSmall(
  ctx: CanvasRenderingContext2D,
  x: number,
  y0: number,
  y1: number,
  color: string,
  glow: number,
) {
  // Plates near the lower (star) end so all three caps stack visibly.
  const midY = y1 - 6;
  const gap = 1.8;
  ctx.save();
  ctx.shadowBlur = 1.5 + glow * 4;
  ctx.shadowColor = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.1;
  // upper stub
  ctx.beginPath();
  ctx.moveTo(x, y0);
  ctx.lineTo(x, midY - gap);
  ctx.stroke();
  // lower stub
  ctx.beginPath();
  ctx.moveTo(x, midY + gap);
  ctx.lineTo(x, y1);
  ctx.stroke();
  // plates
  ctx.beginPath();
  ctx.moveTo(x - 4, midY - gap);
  ctx.lineTo(x + 4, midY - gap);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 4, midY + gap);
  ctx.lineTo(x + 4, midY + gap);
  ctx.stroke();
  ctx.restore();
}

function drawCapHorizontalBig(
  ctx: CanvasRenderingContext2D,
  x: number,
  yTop: number,
  yBot: number,
  color: string,
  glow: number,
  label: string,
) {
  const midY = (yTop + yBot) / 2;
  const gap = 3;
  ctx.save();
  ctx.shadowBlur = 4 + glow * 10;
  ctx.shadowColor = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(x, yTop);
  ctx.lineTo(x, midY - gap);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, midY + gap);
  ctx.lineTo(x, yBot);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 7, midY - gap);
  ctx.lineTo(x + 7, midY - gap);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 7, midY + gap);
  ctx.lineTo(x + 7, midY + gap);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = COL_LABEL;
  ctx.font = "6px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText(label, x, midY + 18);
}

function drawImpedanceVertical(
  ctx: CanvasRenderingContext2D,
  x: number,
  y0: number,
  y1: number,
  color: string,
  label: string,
) {
  const boxH = (y1 - y0) * 0.55;
  const boxY0 = (y0 + y1) / 2 - boxH / 2;
  const boxY1 = boxY0 + boxH;
  const boxW = 9;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x, y0);
  ctx.lineTo(x, boxY0);
  ctx.moveTo(x, boxY1);
  ctx.lineTo(x, y1);
  ctx.stroke();
  ctx.fillStyle = "rgba(148,163,184,0.06)";
  ctx.fillRect(x - boxW / 2, boxY0, boxW, boxH);
  ctx.strokeRect(x - boxW / 2, boxY0, boxW, boxH);
  ctx.restore();
  ctx.fillStyle = COL_LABEL;
  ctx.font = "6px ui-monospace, monospace";
  ctx.textAlign = "left";
  ctx.fillText(label, x + boxW / 2 + 2, (y0 + y1) / 2 + 2);
}

function drawIGBT(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  facingUp: boolean,
  conduction: number,
  label: string,
) {
  const w = 7;
  const h = 11;
  const col = conduction > 0.05 ? COL_IGBT_ON : COL_IGBT;
  ctx.save();
  ctx.shadowBlur = 2 + conduction * 9;
  ctx.shadowColor = COL_IGBT_ON;
  ctx.strokeStyle = col;
  ctx.lineWidth = 0.9;
  ctx.fillStyle = conduction > 0.05 ? "rgba(244,114,182,0.12)" : "rgba(255,255,255,0.03)";
  ctx.fillRect(x - w / 2, y - h / 2, w, h);
  ctx.strokeRect(x - w / 2, y - h / 2, w, h);
  ctx.beginPath();
  if (facingUp) {
    ctx.moveTo(x - 1.3, y - 1.5);
    ctx.lineTo(x + 1.3, y - 1.5);
    ctx.lineTo(x, y - 4);
  } else {
    ctx.moveTo(x - 1.3, y + 1.5);
    ctx.lineTo(x + 1.3, y + 1.5);
    ctx.lineTo(x, y + 4);
  }
  ctx.closePath();
  ctx.fillStyle = col;
  ctx.fill();
  // gate stub
  ctx.strokeStyle = "#64748b";
  ctx.beginPath();
  ctx.moveTo(x - w / 2 - 3, y);
  ctx.lineTo(x - w / 2, y);
  ctx.stroke();
  // antiparallel diode
  ctx.strokeStyle = COL_DIODE;
  ctx.beginPath();
  if (facingUp) {
    ctx.moveTo(x + w / 2 + 1.5, y + 3);
    ctx.lineTo(x + w / 2 + 4, y);
    ctx.lineTo(x + w / 2 + 1.5, y - 3);
    ctx.closePath();
  } else {
    ctx.moveTo(x + w / 2 + 1.5, y - 3);
    ctx.lineTo(x + w / 2 + 4, y);
    ctx.lineTo(x + w / 2 + 1.5, y + 3);
    ctx.closePath();
  }
  ctx.fillStyle = COL_DIODE;
  ctx.fill();
  ctx.beginPath();
  if (facingUp) {
    ctx.moveTo(x + w / 2 + 1, y - 3);
    ctx.lineTo(x + w / 2 + 5, y - 3);
  } else {
    ctx.moveTo(x + w / 2 + 1, y + 3);
    ctx.lineTo(x + w / 2 + 5, y + 3);
  }
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = col;
  ctx.font = "5px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText(label, x, facingUp ? y - 7 : y + 9);
}

/** DC/DC converter — wide, short block spanning the DC rail with a
 *  single centered "DC/DC" label.  Rail stubs run vertically from the
 *  bridge rails to the block edges so particles flow through cleanly. */
function drawDCDCBlock(
  ctx: CanvasRenderingContext2D,
  x: number,
  yTop: number,
  yBot: number,
  utilisation: number,
  _label: string,
) {
  const w = 22;
  const midY = (yTop + yBot) / 2;
  const h = 14;
  const bodyY0 = midY - h / 2;
  const bodyY1 = midY + h / 2;
  const active = utilisation > 0.05;
  const mainCol = active ? COL_IGBT_ON : "#94a3b8";

  // Body
  ctx.save();
  ctx.shadowBlur = 3 + utilisation * 10;
  ctx.shadowColor = COL_IGBT_ON;
  ctx.fillStyle = active ? "rgba(244,114,182,0.12)" : "rgba(148,163,184,0.08)";
  ctx.strokeStyle = mainCol;
  ctx.lineWidth = 1.3;
  ctx.fillRect(x - w / 2, bodyY0, w, h);
  ctx.strokeRect(x - w / 2, bodyY0, w, h);
  ctx.restore();

  // Rail stubs (input from bridge, output to car)
  ctx.save();
  ctx.strokeStyle = COL_WIRE;
  ctx.lineWidth = 0.9;
  ctx.beginPath();
  ctx.moveTo(x - w / 2, yTop); ctx.lineTo(x - w / 2, bodyY0);
  ctx.moveTo(x - w / 2, bodyY1); ctx.lineTo(x - w / 2, yBot);
  ctx.moveTo(x + w / 2, bodyY0); ctx.lineTo(x + w / 2, yTop);
  ctx.moveTo(x + w / 2, bodyY1); ctx.lineTo(x + w / 2, yBot);
  ctx.stroke();
  ctx.restore();

  // Centered "DC/DC" label
  ctx.fillStyle = COL_VALUE;
  ctx.font = "bold 7px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("DC/DC", x, midY);
  ctx.textBaseline = "alphabetic";
}

/** EV glyph — fastback silhouette with gradient body, tinted greenhouse,
 *  LED light strips, 5-spoke alloy wheels, and an underbody charging
 *  glow that pulses with the load current.  The charge port on the left
 *  fender catches the DC cable drops from the DC/DC output. */
function drawCar(
  ctx: CanvasRenderingContext2D,
  x: number,
  _y: number,
  chargeFrac: number,
  label: string,
) {
  const bodyW = 26;
  const bodyY0 = 42;          // beltline (body meets greenhouse)
  const bodyY1 = 58;          // bottom of body shell (above wheels)
  const roofY = 31;           // top of the curved roof
  const active = chargeFrac > 0.05;
  const stroke = active ? COL_DCPLUS : "#cbd5e1";

  // ---- Underbody charging glow ----
  if (active) {
    ctx.save();
    ctx.shadowBlur = 10;
    ctx.shadowColor = COL_DCPLUS;
    ctx.fillStyle = `rgba(251,191,36,${0.22 * chargeFrac})`;
    ctx.beginPath();
    ctx.ellipse(x, bodyY1 + 3.5, bodyW / 2 + 2.5, 1.6, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // ---- Body silhouette (one smooth fastback outline) ----
  ctx.save();
  ctx.shadowBlur = 0.8 + chargeFrac * 5;
  ctx.shadowColor = COL_DCPLUS;
  const grad = ctx.createLinearGradient(0, roofY, 0, bodyY1);
  grad.addColorStop(0, "rgba(203,213,225,0.28)");
  grad.addColorStop(0.55, "rgba(100,116,139,0.17)");
  grad.addColorStop(1, "rgba(51,65,85,0.10)");
  ctx.fillStyle = grad;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1.1;
  ctx.lineJoin = "round";
  ctx.beginPath();
  // bottom-rear corner
  ctx.moveTo(x - bodyW / 2, bodyY1);
  // rear bumper up
  ctx.lineTo(x - bodyW / 2, bodyY0 + 3);
  // rear fender tuck
  ctx.quadraticCurveTo(x - bodyW / 2, bodyY0, x - bodyW / 2 + 1, bodyY0 - 0.3);
  // trunk + rear glass sweep up to roof
  ctx.quadraticCurveTo(x - bodyW / 2 + 4, roofY + 1.8, x - 3.5, roofY + 0.3);
  // roof apex (slightly forward of center for a cab-forward look)
  ctx.quadraticCurveTo(x, roofY - 1.4, x + 3.5, roofY + 0.3);
  // windshield + hood down to front
  ctx.quadraticCurveTo(x + bodyW / 2 - 4, roofY + 1.8, x + bodyW / 2 - 1, bodyY0 - 0.3);
  // front fender curl
  ctx.quadraticCurveTo(x + bodyW / 2, bodyY0, x + bodyW / 2, bodyY0 + 3);
  // front bumper down
  ctx.lineTo(x + bodyW / 2, bodyY1);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  // ---- Greenhouse window fill (tinted glass) ----
  ctx.save();
  ctx.fillStyle = "rgba(14,25,45,0.78)";
  ctx.beginPath();
  ctx.moveTo(x - bodyW / 2 + 2, bodyY0 - 0.1);
  ctx.quadraticCurveTo(x - bodyW / 2 + 4.2, roofY + 1.8, x - 3.5, roofY + 0.4);
  ctx.quadraticCurveTo(x, roofY - 0.9, x + 3.5, roofY + 0.4);
  ctx.quadraticCurveTo(x + bodyW / 2 - 4.2, roofY + 1.8, x + bodyW / 2 - 2, bodyY0 - 0.1);
  ctx.closePath();
  ctx.fill();
  // B-pillar
  ctx.strokeStyle = "rgba(51,65,85,0.95)";
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  ctx.moveTo(x, roofY + 0.6);
  ctx.lineTo(x, bodyY0);
  ctx.stroke();
  // Window shine (thin highlight along top of glass)
  ctx.strokeStyle = "rgba(203,213,225,0.35)";
  ctx.lineWidth = 0.4;
  ctx.beginPath();
  ctx.moveTo(x - 3.2, roofY + 0.6);
  ctx.quadraticCurveTo(x, roofY - 0.4, x + 3.2, roofY + 0.6);
  ctx.stroke();
  ctx.restore();

  // ---- Character line (door crease) ----
  ctx.strokeStyle = "rgba(148,163,184,0.28)";
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(x - bodyW / 2 + 2.5, bodyY1 - 3.5);
  ctx.lineTo(x + bodyW / 2 - 2.5, bodyY1 - 3.5);
  ctx.stroke();

  // ---- LED headlight strip (front, amber glow) ----
  ctx.save();
  ctx.shadowBlur = 4;
  ctx.shadowColor = "#fef3c7";
  ctx.strokeStyle = "#fde68a";
  ctx.lineWidth = 1.4;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x + bodyW / 2 - 3, bodyY0 + 3.3);
  ctx.lineTo(x + bodyW / 2 - 0.8, bodyY0 + 3.3);
  ctx.stroke();
  ctx.restore();

  // ---- LED taillight strip (rear, red glow) ----
  ctx.save();
  ctx.shadowBlur = 3;
  ctx.shadowColor = "rgba(248,113,113,0.9)";
  ctx.strokeStyle = "rgba(248,113,113,0.9)";
  ctx.lineWidth = 1.4;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x - bodyW / 2 + 0.8, bodyY0 + 3.3);
  ctx.lineTo(x - bodyW / 2 + 3, bodyY0 + 3.3);
  ctx.stroke();
  ctx.restore();

  // ---- Alloy wheels with 5-spoke turbine pattern ----
  const wheelR = 3.7;
  const wheelY = bodyY1 + 1;
  for (const wx of [x - 7.5, x + 7.5]) {
    // Tire
    ctx.fillStyle = "#0b1120";
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    ctx.arc(wx, wheelY, wheelR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    // Silver rim
    ctx.fillStyle = "#94a3b8";
    ctx.beginPath();
    ctx.arc(wx, wheelY, wheelR * 0.6, 0, Math.PI * 2);
    ctx.fill();
    // 5 spokes (turbine)
    ctx.strokeStyle = "#0b1120";
    ctx.lineWidth = 0.55;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let s = 0; s < 5; s++) {
      const ang = (s / 5) * Math.PI * 2 - Math.PI / 2;
      const rInner = wheelR * 0.18;
      const rOuter = wheelR * 0.56;
      ctx.moveTo(wx + Math.cos(ang) * rInner, wheelY + Math.sin(ang) * rInner);
      ctx.lineTo(wx + Math.cos(ang) * rOuter, wheelY + Math.sin(ang) * rOuter);
    }
    ctx.stroke();
    // Hub center
    ctx.fillStyle = "#475569";
    ctx.beginPath();
    ctx.arc(wx, wheelY, wheelR * 0.17, 0, Math.PI * 2);
    ctx.fill();
  }

  // ---- Charge port (left fender) ----
  const portX = x - bodyW / 2;
  ctx.save();
  if (active) {
    ctx.shadowBlur = 6;
    ctx.shadowColor = COL_DCPLUS;
  }
  ctx.fillStyle = active ? COL_DCPLUS : "#475569";
  ctx.fillRect(portX - 2.4, 44, 2.4, 10);
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 0.5;
  ctx.strokeRect(portX - 2.4, 44, 2.4, 10);
  // Inlet pins
  ctx.fillStyle = "#0b1120";
  ctx.beginPath();
  ctx.arc(portX - 1.2, 47, 0.6, 0, Math.PI * 2);
  ctx.arc(portX - 1.2, 51, 0.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // ---- Text labels ----
  ctx.fillStyle = COL_DCPLUS;
  ctx.font = "bold 6px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText(label, x, 27);
  ctx.fillStyle = COL_LABEL;
  ctx.font = "5.5px ui-monospace, monospace";
  ctx.fillText("EV charge", x, bodyY1 + 12);
}

function drawACSource(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  color: string,
  label: string,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 4, y);
  ctx.quadraticCurveTo(x - 1.5, y - 3, x, y);
  ctx.quadraticCurveTo(x + 1.5, y + 3, x + 4, y);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = COL_AC_LABEL;
  ctx.font = "5.5px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillText(label, x - r - 1, y + 1.5);
}

// ----------------------------------------------------------
//  Component
// ----------------------------------------------------------

export default function MPCCircuitPanel({
  engineRef,
  slowMo = 0.05,
}: {
  engineRef: MutableRefObject<MPCEngine | null>;
  slowMo?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const lastTRef = useRef<number>(0);
  const slowMoRef = useRef(slowMo);
  slowMoRef.current = slowMo;

  if (particlesRef.current.length === 0) {
    const all: Particle[] = [];
    all.push(...makeParticles(5, 0)); // ac_a
    all.push(...makeParticles(5, 1)); // ac_b
    all.push(...makeParticles(5, 2)); // ac_c
    all.push(...makeParticles(5, 3)); // dc_plus
    all.push(...makeParticles(5, 4)); // dc_minus
    all.push(...makeParticles(4, 5)); // dcdc_out (DC+)
    all.push(...makeParticles(4, 6)); // dcdc_out (DC-)
    all.push(...makeParticles(2, 7)); // cap_a
    all.push(...makeParticles(2, 8)); // cap_b
    all.push(...makeParticles(2, 9)); // cap_c
    particlesRef.current = all;
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = (tNow: number) => {
      const engine = engineRef.current;
      const dt = lastTRef.current === 0 ? 1 / 60 : Math.min(0.1, (tNow - lastTRef.current) / 1000);
      lastTRef.current = tNow;

      const pxW = canvas.width;
      const pxH = canvas.height;
      const scale = Math.min(pxW / W_ABS, pxH / H_ABS);
      const ox = (pxW - W_ABS * scale) / 2;
      const oy = (pxH - H_ABS * scale) / 2;

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, pxW, pxH);
      ctx.fillStyle = "#070a12";
      ctx.fillRect(0, 0, pxW, pxH);
      ctx.setTransform(scale, 0, 0, scale, ox, oy);

      // ----- Read engine signals -----
      const igd = engine ? engine.i_gd_meas : 0;
      const igq = engine ? engine.i_gq_meas : 0;
      const theta = engine ? engine.theta : 0;
      const Vdc = engine ? engine.V_dc : PARAMS.V_dc_nom;
      const ud = engine ? engine.u_d : 0;
      const uq = engine ? engine.u_q : 0;
      const uMag = Math.hypot(ud, uq);
      const uBound = engine ? engine.uBound : (Vdc / Math.sqrt(3)) * PARAMS.U_MAX_FRAC;
      const uNorm = Math.min(1, uMag / Math.max(1, uBound));

      const [ia, ib, ic] = dqToAbc(igd, igq, theta);
      const [ua, ub, uc] = dqToAbc(ud, uq, theta);
      const phaseI: [number, number, number] = [ia, ib, ic];
      const phaseU: [number, number, number] = [ua, ub, uc];
      const iNorm = (i: number) => Math.min(1, Math.abs(i) / PARAMS.I_G_MAX);

      // Per-leg duty (averaged): d_x = 0.5 + 0.5 · u_x / V_dc
      const duty = (u: number) => {
        const d = 0.5 + 0.5 * Math.max(-1, Math.min(1, u / (Vdc / Math.sqrt(3))));
        return Math.max(0, Math.min(1, d));
      };
      const dA = duty(ua);
      const dB = duty(ub);
      const dC = duty(uc);

      // ----- DRAW -----
      // 1. Wires
      for (const seg of SEGS) {
        let col: string = COL_WIRE;
        if (seg.kind === "ac_a") col = "#1f3a44";
        else if (seg.kind === "ac_b") col = "#2c2740";
        else if (seg.kind === "ac_c") col = "#3a232c";
        else if (seg.kind === "dc_plus") col = "#5b4216";
        else if (seg.kind === "dc_minus") col = "#363c47";
        else if (seg.kind === "dcdc_out") col = "#4a3a16";
        else if (seg.kind.startsWith("cap")) col = "#3d3318";
        drawPolyline(
          ctx,
          seg.pts,
          col,
          seg.kind.startsWith("dc") || seg.kind === "dcdc_out" ? 1.0 : 0.9,
        );
      }
      // Cap star bus (small dot at star point)
      ctx.fillStyle = COL_LABEL;
      ctx.beginPath();
      ctx.arc(X.Cf_x, X.star_y, 0.9, 0, Math.PI * 2);
      ctx.fill();

      // 2. Inductors per phase
      for (let p = 0; p < 3; p++) {
        const y = PHASE_Y[p];
        const glow = iNorm(phaseI[p]);
        drawInductor(ctx, X.Lg_in, X.Lg_out, y, COL_L, glow, p === 0 ? "L_g" : undefined);
        drawInductor(ctx, X.Lf_in, X.Lf_out, y, COL_L, glow, p === 0 ? "L_f" : undefined);
      }

      // 3. Y-connected filter caps
      for (let p = 0; p < 3; p++) {
        drawCapVerticalSmall(ctx, X.Cf_x, PHASE_Y[p] + 1.5, X.star_y - 0.5, COL_C, uNorm * 0.5 + iNorm(phaseI[p]) * 0.5);
      }
      ctx.fillStyle = COL_LABEL;
      ctx.font = "5.5px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText("C_f (Y, star)", X.Cf_x + 5, X.star_y + 4);

      // 4. 6-IGBT bridge shell + devices + bridge stubs
      ctx.save();
      ctx.strokeStyle = "rgba(148,163,184,0.28)";
      ctx.setLineDash([2, 2]);
      ctx.lineWidth = 0.5;
      ctx.strokeRect(X.legA - 8, X.bridge_top_y - 2, X.legC - X.legA + 16, X.bridge_bot_y - X.bridge_top_y + 4);
      ctx.restore();

      const drawBridgeStubs = (legX: number) => {
        ctx.save();
        ctx.strokeStyle = COL_WIRE;
        ctx.lineCap = "round";
        ctx.lineWidth = 0.9;
        ctx.beginPath();
        // DC+ → top of upper IGBT
        ctx.moveTo(legX, X.bridge_top_y);
        ctx.lineTo(legX, 16);
        // bottom of upper IGBT → AC midpoint (47)
        ctx.moveTo(legX, 27);
        ctx.lineTo(legX, 47);
        // AC midpoint → top of lower IGBT
        ctx.moveTo(legX, 47);
        ctx.lineTo(legX, 67);
        // bottom of lower IGBT → DC-
        ctx.moveTo(legX, 78);
        ctx.lineTo(legX, X.bridge_bot_y);
        ctx.stroke();
        ctx.restore();
      };
      drawBridgeStubs(X.legA);
      drawBridgeStubs(X.legB);
      drawBridgeStubs(X.legC);

      // Top switches (S1, S3, S5) and bottom (S2, S4, S6).  Conduction by duty.
      drawIGBT(ctx, X.legA, 22, true, dA, "S1");
      drawIGBT(ctx, X.legA, 73, false, 1 - dA, "S2");
      drawIGBT(ctx, X.legB, 22, true, dB, "S3");
      drawIGBT(ctx, X.legB, 73, false, 1 - dB, "S4");
      drawIGBT(ctx, X.legC, 22, true, dC, "S5");
      drawIGBT(ctx, X.legC, 73, false, 1 - dC, "S6");

      // 5. DC bus cap + DC/DC converter + EV car
      drawCapHorizontalBig(ctx, X.bus_cap, X.bridge_top_y, X.bridge_bot_y, COL_DCPLUS, uNorm, "C_DC");

      const iCar = engine ? engine.i_load : 0;
      // Utilisation heuristic — full EV-demand rating is ~55 A on DC side
      // (matches the 55 A peak on the EV demand slider).
      const carFrac = Math.max(0, Math.min(1, iCar / 55));
      drawDCDCBlock(ctx, X.dcdc, X.bridge_top_y, X.bridge_bot_y, carFrac, "buck");
      drawCar(ctx, X.car, 47, carFrac, `i_car = ${iCar.toFixed(1)} A`);

      // 6. AC sources (3 sinusoidal sources, labeled e_a/e_b/e_c)
      for (let p = 0; p < 3; p++) {
        drawACSource(ctx, X.src, PHASE_Y[p], X.src_r, COL_AC_LABEL, ["e_a", "e_b", "e_c"][p]);
        // tap stub from circle to phase rail
        ctx.strokeStyle = COL_WIRE;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(X.src + X.src_r, PHASE_Y[p]);
        ctx.lineTo(X.src_tap, PHASE_Y[p]);
        ctx.stroke();
      }

      // 7. Particles — speed scales with phase-current sign·magnitude, gated
      //    by slowMo so they're always watchable.  We also fold a base trickle
      //    so the schematic never looks frozen.
      // Slow-mo factor: at slowMo=0.05 we want lazy flow; at 0.2 brisker.
      const slowMoGate = Math.max(0.05, Math.min(0.4, slowMoRef.current * 4));
      const baseSpeed = 0.012;

      const iCarRaw = engine ? engine.i_load : 0;
      const iCarNorm = Math.max(0, Math.min(1, iCarRaw / 55));
      for (const p of particlesRef.current) {
        const seg = SEGS[p.segIdx];
        let v = 0;
        if (seg.kind === "ac_a" || seg.kind === "ac_b" || seg.kind === "ac_c") {
          const phaseIdx = seg.kind === "ac_a" ? 0 : seg.kind === "ac_b" ? 1 : 2;
          const cur = phaseI[phaseIdx];
          const sign = cur >= 0 ? 1 : -1;
          const mag = iNorm(cur);
          v = baseSpeed * sign * (0.3 + 0.7 * mag) * slowMoGate;
        } else if (seg.kind === "dc_plus") {
          v = baseSpeed * (0.4 + 0.6 * uNorm) * slowMoGate;
        } else if (seg.kind === "dc_minus") {
          v = -baseSpeed * (0.4 + 0.6 * uNorm) * slowMoGate;
        } else if (seg.kind === "dcdc_out") {
          // Particle direction follows positive charging convention:
          // DC+ segment (first) flows toward the car; DC- (second) returns.
          const isPlus = p.segIdx === 5;
          v = baseSpeed * (isPlus ? 1 : -1) * (0.25 + 0.75 * iCarNorm) * slowMoGate;
        } else if (seg.kind === "cap_a" || seg.kind === "cap_b" || seg.kind === "cap_c") {
          // Cap branch current ≈ d/dt(v_cap) · C_f; rough proxy from i_g
          const phaseIdx = seg.kind === "cap_a" ? 0 : seg.kind === "cap_b" ? 1 : 2;
          v = baseSpeed * 0.5 * Math.sign(phaseI[phaseIdx]) * iNorm(phaseI[phaseIdx]) * slowMoGate;
        }
        p.t += v * dt * 60;
        while (p.t > 1) p.t -= 1;
        while (p.t < 0) p.t += 1;
        const edgeT = Math.min(p.t, 1 - p.t);
        const fade = Math.min(1, edgeT / 0.08);

        const pos = pointAlong(seg, p.t);
        let color: string = COL_WIRE;
        let radius = 0.85;
        if (seg.kind === "ac_a") {
          color = PHASE_COL[0];
          radius = 0.9 + iNorm(ia) * 0.5;
        } else if (seg.kind === "ac_b") {
          color = PHASE_COL[1];
          radius = 0.9 + iNorm(ib) * 0.5;
        } else if (seg.kind === "ac_c") {
          color = PHASE_COL[2];
          radius = 0.9 + iNorm(ic) * 0.5;
        } else if (seg.kind === "dc_plus") {
          color = COL_DCPLUS;
          radius = 0.85 + uNorm * 0.6;
        } else if (seg.kind === "dc_minus") {
          color = COL_DCMINUS;
          radius = 0.85 + uNorm * 0.6;
        } else if (seg.kind === "dcdc_out") {
          color = COL_DCPLUS;
          radius = 0.85 + iCarNorm * 0.7;
        } else if (seg.kind.startsWith("cap")) {
          color = COL_C;
          radius = 0.7;
        }

        ctx.save();
        ctx.globalAlpha = 0.55 + 0.4 * fade;
        ctx.shadowBlur = 1.5 + radius * 1.8;
        ctx.shadowColor = color;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // 8. HUD labels
      ctx.textAlign = "left";
      ctx.font = "6px ui-monospace, monospace";
      ctx.fillStyle = COL_VALUE;
      ctx.fillText(`i_a = ${ia.toFixed(2)} A`, 150, 100);
      ctx.fillText(`V_dc = ${Vdc.toFixed(0)} V`, 230, 100);
      ctx.fillText(`|u_dq| = ${uMag.toFixed(0)} V`, 150, 107);
      ctx.fillText(`|u| / (V_dc/√3) = ${(uMag / Math.max(1, Vdc / Math.sqrt(3)) * 100).toFixed(0)}%`, 230, 107);

      ctx.textAlign = "right";
      ctx.fillStyle = COL_LABEL;
      ctx.fillText(`f_grid = ${PARAMS.f_grid} Hz · f_sw = ${(PARAMS.f_sw / 1000).toFixed(0)} kHz · averaged model`, W_ABS - 2, 5);

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, [engineRef]);

  return (
    <div className="rounded-2xl border border-white/10 bg-[#060912] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-white/5">
        <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-gray-500">
          Plant · 3-phase active rectifier + LCL filter
        </div>
        <div className="text-[9px] font-mono text-gray-500">
          400 V<sub>LL,rms</sub> · 50 Hz · V<sub>dc</sub> ≈ 750 V · averaged bridge
        </div>
      </div>
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          height: 170,
          display: "block",
        }}
      />
    </div>
  );
}

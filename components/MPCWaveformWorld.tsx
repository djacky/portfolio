"use client";

/* ------------------------------------------------------------------
   MPCWaveformWorld — R3F scene for the 3-phase grid-tied VSC MPC.

   Two stacked plots in one canvas:
     • Top    : i_a, i_b, i_c (grid-side phase currents) over time,
                with a phase-A reference ghost and ±I_G_MAX rails.
                The MPC's prediction arc is rotated back into the
                phase-A frame via inverse Park and shown in gold.
     • Bottom : V_dc trace + V_dc_ref reference, with ±tolerance
                bands.  Lets the recruiter watch the outer loop fight
                the load step in real time.

   The scene is driven from MPCEngine (lib/mpc-sim.ts).  Each useFrame:
     1. Advance the engine N MPC ticks (N depends on slow-mo & dt),
     2. Pull ring buffers + latest predicted trajectory,
     3. Update Line2 geometry positions in place (no geom rebuild),
     4. Update prediction arc through inverse Park rotation,
     5. Modulate constraint-slab opacity by proximity.

   Slow-mo (slowMo prop): 1 = roughly real-time grid (way too fast to
   see), 0.05 = good for watching the 50 Hz cycles, 0.01 = inspect.
   The slowMo factor scales sim-time-per-render-frame.
------------------------------------------------------------------ */

import { useMemo, useRef, type MutableRefObject } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import type { Line2 } from "three-stdlib";

import { MPCEngine, PARAMS, dqToAbc } from "@/lib/mpc-sim";

// ----------------------------------------------------------
//  Scene constants
// ----------------------------------------------------------
const N_HIST = 240;
const MAX_N = 20;
const X_PAST_MIN = -5.2;
const X_PAST_MAX = 0;
const X_FUT_EXT = 2.4;

// Top (current) panel
const Y_I_CENTER = 0.85;
const Y_I_HALF = 0.85;          // visual half-height of the current panel
const I_SCALE = Y_I_HALF / PARAMS.I_G_MAX;
// Bottom (V_dc) panel — DUAL-AXIS:
//   LEFT  axis: V_dc − V_DC_REF        ±80 V   → ±Y_V_HALF   (gold trace)
//   RIGHT axis: i_gd (d-axis current)  ±10 A   → ±Y_V_HALF   (green trace)
// Both share Y_V_CENTER so the zero crossings line up visually.  The
// legend + axis labels on the overlay disambiguate units.
const Y_V_CENTER = -1.35;
const Y_V_HALF = 0.6;
const V_DC_RANGE = 80;          // ±V around V_DC_REF mapped to ±Y_V_HALF
const V_SCALE = Y_V_HALF / V_DC_RANGE;
const I_DC_RANGE = 80;          // ±A around 0 mapped to ±Y_V_HALF
const I_DC_SCALE = Y_V_HALF / I_DC_RANGE;

// Z layers
const Z_BG = -0.5;
const Z_SLAB = -0.3;
const Z_REF = -0.05;
const Z_ACTUAL = 0.0;
const Z_NOW = 0.06;
const Z_PRED = 0.1;
const Z_DOT = 0.12;

const COL_REF = "#e8ecf3";
const COL_PHASE_A = "#22d3ee";   // cyan
const COL_PHASE_B = "#a78bfa";   // violet
const COL_PHASE_C = "#fb7185";   // rose
// Per-phase prediction colours — lighter tints of each phase so the
// forecast visibly belongs to its own trace (sky-blue for cyan phase A,
// pale violet for phase B, pale rose for phase C).
const COL_PRED_A = "#7dd3fc";    // sky-300 (light blue for phase A)
const COL_PRED_B = "#c4b5fd";    // violet-300 (light violet for phase B)
const COL_PRED_C = "#fda4af";    // rose-300 (light rose for phase C)
const COL_VDC = "#fbbf24";
const COL_VDC_REF = "#94a3b8";
const COL_IGD = "#4ade80";       // green — dq d-axis current trace
const COL_IGD_REF = "#bbf7d0";   // faint green for i_gd* ghost
const COL_PRED_IGD = "#86efac";  // green-300 — predicted i_gd arc on V_dc panel
const COL_SLAB = "#ef4444";
const COL_NOW = "#ffffff";

// ----------------------------------------------------------
//  Helpers
// ----------------------------------------------------------

/** Fill a flat Float32Array (3 * N_HIST) from a ring-buffer slice, mapping
 *  values through `valueToY` and stamping the constant z. */
function fillHistoryPositions(
  out: Float32Array,
  buf: Float32Array,
  head: number,
  filled: number,
  z: number,
  valueToY: (v: number) => number,
) {
  const start = filled < N_HIST ? 0 : head;
  const span = X_PAST_MAX - X_PAST_MIN;
  for (let i = 0; i < N_HIST; i++) {
    const frac = i / (N_HIST - 1);
    const x = X_PAST_MIN + frac * span;
    const valid = i >= N_HIST - Math.max(filled, 1);
    const idx = (start + (i - (N_HIST - filled))) % N_HIST;
    const v = valid ? buf[idx] : 0;
    out[3 * i] = x;
    out[3 * i + 1] = valueToY(v);
    out[3 * i + 2] = z;
  }
}

const iToY = (v: number) => Y_I_CENTER + v * I_SCALE;
const vToY = (v: number) => {
  const e = (v - PARAMS.V_DC_REF) * V_SCALE;
  return Y_V_CENTER + Math.max(-Y_V_HALF, Math.min(Y_V_HALF, e));
};
/** Right-axis mapping on the V_dc panel: i_gd (A) → canvas y. */
const idToY = (v: number) => {
  const e = v * I_DC_SCALE;
  return Y_V_CENTER + Math.max(-Y_V_HALF, Math.min(Y_V_HALF, e));
};

// ----------------------------------------------------------
//  Scene
// ----------------------------------------------------------

interface SceneProps {
  engineRef: MutableRefObject<MPCEngine | null>;
  playing: boolean;
  slowMo: number;
}

function Scene({ engineRef, playing, slowMo }: SceneProps) {
  const refLine = useRef<Line2 | null>(null);
  const aLine = useRef<Line2 | null>(null);
  const bLine = useRef<Line2 | null>(null);
  const cLine = useRef<Line2 | null>(null);
  const vdcLine = useRef<Line2 | null>(null);
  const vdcRefLine = useRef<Line2 | null>(null);
  const igdLine = useRef<Line2 | null>(null);
  const igdRefLine = useRef<Line2 | null>(null);
  const predLineA = useRef<Line2 | null>(null);
  const predLineB = useRef<Line2 | null>(null);
  const predLineC = useRef<Line2 | null>(null);
  const predLineGd = useRef<Line2 | null>(null);
  const predDots = useRef<THREE.InstancedMesh | null>(null);

  const topSlab = useRef<THREE.MeshBasicMaterial | null>(null);
  const botSlab = useRef<THREE.MeshBasicMaterial | null>(null);
  const nowLineMat = useRef<THREE.MeshBasicMaterial | null>(null);
  const nowVMat = useRef<THREE.MeshBasicMaterial | null>(null);

  // Scratch buffers
  const refPos = useMemo(() => new Float32Array(N_HIST * 3), []);
  const aPos = useMemo(() => new Float32Array(N_HIST * 3), []);
  const bPos = useMemo(() => new Float32Array(N_HIST * 3), []);
  const cPos = useMemo(() => new Float32Array(N_HIST * 3), []);
  const vdcPos = useMemo(() => new Float32Array(N_HIST * 3), []);
  const vdcRefPos = useMemo(() => new Float32Array(N_HIST * 3), []);
  const igdPos = useMemo(() => new Float32Array(N_HIST * 3), []);
  const igdRefPos = useMemo(() => new Float32Array(N_HIST * 3), []);
  /** Per-phase prediction polylines — one gold arc per phase (A/B/C),
   *  each (MAX_N + 1) points.  Anchor = current measured phase current at
   *  x = X_PAST_MAX, then one point per horizon step.  Points past the
   *  active horizon are collapsed onto the last valid point so geometry
   *  length never shrinks. */
  const predPosA = useMemo(() => new Float32Array((MAX_N + 1) * 3), []);
  const predPosB = useMemo(() => new Float32Array((MAX_N + 1) * 3), []);
  const predPosC = useMemo(() => new Float32Array((MAX_N + 1) * 3), []);
  const predPosGd = useMemo(() => new Float32Array((MAX_N + 1) * 3), []);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Initial polyline points (stable refs)
  const initHistPts = useMemo(() => {
    const a: [number, number, number][] = [];
    for (let i = 0; i < N_HIST; i++) {
      const frac = i / (N_HIST - 1);
      const x = X_PAST_MIN + frac * (X_PAST_MAX - X_PAST_MIN);
      a.push([x, 0, 0]);
    }
    return a;
  }, []);
  const initPredPts = useMemo(() => {
    const a: [number, number, number][] = [];
    for (let i = 0; i <= MAX_N; i++) {
      const frac = i / MAX_N;
      a.push([X_PAST_MAX + frac * X_FUT_EXT, 0, 0]);
    }
    return a;
  }, []);

  useFrame((_, delta) => {
    const engine = engineRef.current;
    if (!engine || !playing) return;

    // Sim time per frame  =  delta · slowMo
    const dt = Math.min(delta, 1 / 15);
    const simSeconds = dt * slowMo;
    const steps = Math.max(1, Math.min(80, Math.round(simSeconds / PARAMS.T_s)));

    const tStart = typeof performance !== "undefined" ? performance.now() : 0;
    for (let i = 0; i < steps; i++) engine.step();
    if (engine.mode === "mpc") {
      const tEnd = typeof performance !== "undefined" ? performance.now() : 0;
      const perSolveUs = ((tEnd - tStart) * 1000) / steps;
      engine.reportSolveUs(perSolveUs);
    }

    // ---- top panel: phase currents ----
    fillHistoryPositions(aPos, engine.i_a_buf, engine.head, engine.filled, Z_ACTUAL, iToY);
    fillHistoryPositions(bPos, engine.i_b_buf, engine.head, engine.filled, Z_ACTUAL - 0.001, iToY);
    fillHistoryPositions(cPos, engine.i_c_buf, engine.head, engine.filled, Z_ACTUAL - 0.002, iToY);
    fillHistoryPositions(refPos, engine.i_a_ref_buf, engine.head, engine.filled, Z_REF, iToY);
    if (aLine.current) aLine.current.geometry.setPositions(aPos);
    if (bLine.current) bLine.current.geometry.setPositions(bPos);
    if (cLine.current) cLine.current.geometry.setPositions(cPos);
    if (refLine.current) refLine.current.geometry.setPositions(refPos);

    // ---- bottom panel: V_dc (left axis) + i_gd (right axis) ----
    fillHistoryPositions(vdcPos, engine.v_dc_buf, engine.head, engine.filled, Z_ACTUAL, vToY);
    fillHistoryPositions(vdcRefPos, engine.v_dc_ref_buf, engine.head, engine.filled, Z_REF, vToY);
    fillHistoryPositions(igdPos, engine.i_gd_buf, engine.head, engine.filled, Z_ACTUAL - 0.003, idToY);
    fillHistoryPositions(igdRefPos, engine.i_gd_ref_buf, engine.head, engine.filled, Z_REF - 0.001, idToY);
    if (vdcLine.current) vdcLine.current.geometry.setPositions(vdcPos);
    if (vdcRefLine.current) vdcRefLine.current.geometry.setPositions(vdcRefPos);
    if (igdLine.current) igdLine.current.geometry.setPositions(igdPos);
    if (igdRefLine.current) igdRefLine.current.geometry.setPositions(igdRefPos);

    // ---- prediction arcs (one per phase, rotated through inverse Park) ----
    const N = engine.horizon;
    const isMPC = engine.mode === "mpc" && engine.lastXPred;

    if (isMPC && engine.lastXPred) {
      // Anchor each arc to its phase's current measured value at x = X_PAST_MAX
      // so the prediction fuses seamlessly with the corresponding history trace.
      const [iaNow, ibNow, icNow] = dqToAbc(engine.i_gd_meas, engine.i_gq_meas, engine.theta);
      predPosA[0] = X_PAST_MAX; predPosA[1] = iToY(iaNow); predPosA[2] = Z_PRED;
      predPosB[0] = X_PAST_MAX; predPosB[1] = iToY(ibNow); predPosB[2] = Z_PRED;
      predPosC[0] = X_PAST_MAX; predPosC[1] = iToY(icNow); predPosC[2] = Z_PRED;
      predPosGd[0] = X_PAST_MAX; predPosGd[1] = idToY(engine.i_gd_meas); predPosGd[2] = Z_PRED;

      // Predicted state stride per horizon step — 6 when running the
      // plain MPC, 8 when the IMP augmentation appends two resonator
      // states.  Plant indices i_gd / i_gq are always 4 / 5 of each block.
      const nxStride = engine.mpc.nx;
      for (let k = 0; k < N; k++) {
        // Predicted state at horizon index k (1-indexed):
        //   i_gd, i_gq are at offsets 4, 5 of each block.
        const igd = engine.lastXPred[k * nxStride + 4];
        const igq = engine.lastXPred[k * nxStride + 5];
        const theta_k = engine.theta + (k + 1) * PARAMS.T_s * 2 * Math.PI * PARAMS.f_grid;
        const [ia_k, ib_k, ic_k] = dqToAbc(igd, igq, theta_k);
        const x = X_PAST_MAX + (X_FUT_EXT * (k + 1)) / N;
        const base = (k + 1) * 3;
        predPosA[base] = x; predPosA[base + 1] = iToY(ia_k); predPosA[base + 2] = Z_PRED;
        predPosB[base] = x; predPosB[base + 1] = iToY(ib_k); predPosB[base + 2] = Z_PRED;
        predPosC[base] = x; predPosC[base + 1] = iToY(ic_k); predPosC[base + 2] = Z_PRED;
        predPosGd[base] = x; predPosGd[base + 1] = idToY(igd); predPosGd[base + 2] = Z_PRED;
      }
      // Collapse the tail (when horizon < MAX_N) onto the last valid
      // point so the geometry length stays constant per arc.
      const collapse = (buf: Float32Array) => {
        const lastX = buf[N * 3];
        const lastY = buf[N * 3 + 1];
        for (let k = N + 1; k <= MAX_N; k++) {
          buf[k * 3] = lastX;
          buf[k * 3 + 1] = lastY;
          buf[k * 3 + 2] = Z_PRED;
        }
      };
      collapse(predPosA); collapse(predPosB); collapse(predPosC); collapse(predPosGd);
      if (predLineA.current) predLineA.current.geometry.setPositions(predPosA);
      if (predLineB.current) predLineB.current.geometry.setPositions(predPosB);
      if (predLineC.current) predLineC.current.geometry.setPositions(predPosC);
      if (predLineGd.current) predLineGd.current.geometry.setPositions(predPosGd);

      // Prediction dots stay on phase A only (visual anchor / marker of the
      // preview step) so we don't get 3× dot clutter across all phases.
      if (predDots.current) {
        for (let k = 0; k < MAX_N; k++) {
          if (k < N) {
            const x = X_PAST_MAX + (X_FUT_EXT * (k + 1)) / N;
            const igd = engine.lastXPred[k * nxStride + 4];
            const igq = engine.lastXPred[k * nxStride + 5];
            const theta_k = engine.theta + (k + 1) * PARAMS.T_s * 2 * Math.PI * PARAMS.f_grid;
            const [ia_k] = dqToAbc(igd, igq, theta_k);
            const s = 1 - 0.55 * (k / N);
            dummy.position.set(x, iToY(ia_k), Z_DOT);
            dummy.scale.set(s, s, s);
          } else {
            dummy.position.set(0, 0, 0);
            dummy.scale.set(0, 0, 0);
          }
          dummy.updateMatrix();
          predDots.current.setMatrixAt(k, dummy.matrix);
        }
        predDots.current.instanceMatrix.needsUpdate = true;
      }
    } else {
      // PI mode → collapse the prediction artifacts.
      const [iaNow, ibNow, icNow] = dqToAbc(engine.i_gd_meas, engine.i_gq_meas, engine.theta);
      const collapseTo = (buf: Float32Array, y: number) => {
        for (let k = 0; k <= MAX_N; k++) {
          buf[k * 3] = X_PAST_MAX;
          buf[k * 3 + 1] = y;
          buf[k * 3 + 2] = Z_PRED;
        }
      };
      collapseTo(predPosA, iToY(iaNow));
      collapseTo(predPosB, iToY(ibNow));
      collapseTo(predPosC, iToY(icNow));
      collapseTo(predPosGd, idToY(engine.i_gd_meas));
      if (predLineA.current) predLineA.current.geometry.setPositions(predPosA);
      if (predLineB.current) predLineB.current.geometry.setPositions(predPosB);
      if (predLineC.current) predLineC.current.geometry.setPositions(predPosC);
      if (predLineGd.current) predLineGd.current.geometry.setPositions(predPosGd);
      if (predDots.current) {
        for (let k = 0; k < MAX_N; k++) {
          dummy.position.set(0, 0, 0);
          dummy.scale.set(0, 0, 0);
          dummy.updateMatrix();
          predDots.current.setMatrixAt(k, dummy.matrix);
        }
        predDots.current.instanceMatrix.needsUpdate = true;
      }
    }

    // Constraint slab proximity glow — top panel uses |i_a| / I_G_MAX
    const [iaCur] = dqToAbc(engine.i_gd_meas, engine.i_gq_meas, engine.theta);
    const yI = iaCur * I_SCALE;
    const proxTop = Math.max(0, Math.min(1, (yI / Y_I_HALF - 0.78) * 4));
    const proxBot = Math.max(0, Math.min(1, (-yI / Y_I_HALF - 0.78) * 4));
    const clipGlow = engine.clipFlashT > 0 ? 1 : 0;
    if (topSlab.current) topSlab.current.opacity = 0.16 + proxTop * 0.55 + clipGlow * 0.25;
    if (botSlab.current) botSlab.current.opacity = 0.16 + proxBot * 0.55 + clipGlow * 0.25;
    if (nowLineMat.current) nowLineMat.current.opacity = 0.55 + (isMPC ? 0.15 : 0);
    if (nowVMat.current) nowVMat.current.opacity = 0.4;
  });

  return (
    <>
      {/* Backdrop */}
      <mesh position={[0, 0, Z_BG]}>
        <planeGeometry args={[20, 6]} />
        <meshBasicMaterial color="#070a12" toneMapped={false} />
      </mesh>

      {/* Panel separators / zero axes ----------------------------------- */}
      {/* current zero axis */}
      <mesh position={[-1.4, Y_I_CENTER, -0.35]}>
        <planeGeometry args={[8, 0.005]} />
        <meshBasicMaterial color="#334155" transparent opacity={0.45} toneMapped={false} />
      </mesh>
      {/* V_dc baseline (reference line is drawn separately as an actual data trace) */}
      <mesh position={[-1.4, Y_V_CENTER, -0.35]}>
        <planeGeometry args={[8, 0.004]} />
        <meshBasicMaterial color="#334155" transparent opacity={0.35} toneMapped={false} />
      </mesh>
      {/* divider between current and V_dc panels */}
      <mesh position={[-1.4, (Y_I_CENTER - Y_I_HALF + Y_V_CENTER + Y_V_HALF) / 2, -0.4]}>
        <planeGeometry args={[8, 0.003]} />
        <meshBasicMaterial color="#1e293b" transparent opacity={0.5} toneMapped={false} />
      </mesh>

      {/* Constraint slabs at ±I_G_MAX */}
      <mesh position={[-1.4, Y_I_CENTER + Y_I_HALF, Z_SLAB]}>
        <planeGeometry args={[8, 0.045]} />
        <meshBasicMaterial
          ref={topSlab}
          color={COL_SLAB}
          transparent
          opacity={0.2}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[-1.4, Y_I_CENTER - Y_I_HALF, Z_SLAB]}>
        <planeGeometry args={[8, 0.045]} />
        <meshBasicMaterial
          ref={botSlab}
          color={COL_SLAB}
          transparent
          opacity={0.2}
          toneMapped={false}
        />
      </mesh>
      {/* Soft halos behind slabs */}
      <mesh position={[-1.4, Y_I_CENTER + Y_I_HALF, Z_SLAB - 0.02]}>
        <planeGeometry args={[8, 0.22]} />
        <meshBasicMaterial color={COL_SLAB} transparent opacity={0.07} toneMapped={false} />
      </mesh>
      <mesh position={[-1.4, Y_I_CENTER - Y_I_HALF, Z_SLAB - 0.02]}>
        <planeGeometry args={[8, 0.22]} />
        <meshBasicMaterial color={COL_SLAB} transparent opacity={0.07} toneMapped={false} />
      </mesh>

      {/* "Now" verticals on each panel */}
      <mesh position={[0, Y_I_CENTER, Z_NOW]}>
        <planeGeometry args={[0.012, 2 * Y_I_HALF + 0.05]} />
        <meshBasicMaterial
          ref={nowLineMat}
          color={COL_NOW}
          transparent
          opacity={0.55}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[0, Y_V_CENTER, Z_NOW]}>
        <planeGeometry args={[0.012, 2 * Y_V_HALF + 0.05]} />
        <meshBasicMaterial
          ref={nowVMat}
          color={COL_NOW}
          transparent
          opacity={0.4}
          toneMapped={false}
        />
      </mesh>

      {/* Phase A reference (silver, faint) */}
      <Line
        ref={refLine}
        points={initHistPts}
        color={COL_REF}
        lineWidth={1.4}
        transparent
        opacity={0.55}
        toneMapped={false}
      />

      {/* Three phase ribbons */}
      <Line ref={aLine} points={initHistPts} color={COL_PHASE_A} lineWidth={2.6} toneMapped={false} />
      <Line ref={bLine} points={initHistPts} color={COL_PHASE_B} lineWidth={2.2} transparent opacity={0.85} toneMapped={false} />
      <Line ref={cLine} points={initHistPts} color={COL_PHASE_C} lineWidth={2.2} transparent opacity={0.85} toneMapped={false} />

      {/* V_dc reference (dashed-feel via faint silver, drawn behind actual) */}
      <Line
        ref={vdcRefLine}
        points={initHistPts}
        color={COL_VDC_REF}
        lineWidth={1.2}
        transparent
        opacity={0.55}
        toneMapped={false}
      />
      {/* V_dc actual */}
      <Line
        ref={vdcLine}
        points={initHistPts}
        color={COL_VDC}
        lineWidth={2.4}
        toneMapped={false}
      />

      {/* i_gd reference ghost (right axis) */}
      <Line
        ref={igdRefLine}
        points={initHistPts}
        color={COL_IGD_REF}
        lineWidth={1.2}
        transparent
        opacity={0.55}
        toneMapped={false}
      />
      {/* i_gd actual (right axis, green) */}
      <Line
        ref={igdLine}
        points={initHistPts}
        color={COL_IGD}
        lineWidth={2.2}
        transparent
        opacity={0.95}
        toneMapped={false}
      />

      {/* Per-phase prediction arcs — one preview per phase current. */}
      <Line ref={predLineA} points={initPredPts} color={COL_PRED_A} lineWidth={3.5} toneMapped={false} />
      <Line ref={predLineB} points={initPredPts} color={COL_PRED_B} lineWidth={3.5} transparent opacity={0.9} toneMapped={false} />
      <Line ref={predLineC} points={initPredPts} color={COL_PRED_C} lineWidth={3.5} transparent opacity={0.9} toneMapped={false} />

      {/* dq d-axis prediction arc on the V_dc panel (right axis, green) */}
      <Line
        ref={predLineGd}
        points={initPredPts}
        color={COL_PRED_IGD}
        lineWidth={3.0}
        transparent
        opacity={0.9}
        toneMapped={false}
      />

      {/* Prediction dots (anchor on phase A) */}
      <instancedMesh ref={predDots} args={[undefined, undefined, MAX_N]} frustumCulled={false}>
        <sphereGeometry args={[0.04, 14, 14]} />
        <meshBasicMaterial color={COL_PRED_A} toneMapped={false} />
      </instancedMesh>

      {/* Tick markers at ±I_G_MAX */}
      <mesh position={[X_PAST_MIN - 0.08, Y_I_CENTER + Y_I_HALF, 0.0]}>
        <planeGeometry args={[0.05, 0.05]} />
        <meshBasicMaterial color="#fca5a5" toneMapped={false} />
      </mesh>
      <mesh position={[X_PAST_MIN - 0.08, Y_I_CENTER - Y_I_HALF, 0.0]}>
        <planeGeometry args={[0.05, 0.05]} />
        <meshBasicMaterial color="#fca5a5" toneMapped={false} />
      </mesh>
    </>
  );
}

// ----------------------------------------------------------
//  Wrapper
// ----------------------------------------------------------

export default function MPCWaveformWorld({
  engineRef,
  playing,
  slowMo = 0.05,
}: {
  engineRef: MutableRefObject<MPCEngine | null>;
  playing: boolean;
  /** Sim-seconds per render-second. 0.05 ≈ five 50-Hz cycles per real second
   *  (looks brisk but legible).  0.01 = inspect mode, 0.2 = real-time-ish. */
  slowMo?: number;
}) {
  return (
    <Canvas
      orthographic
      camera={{ position: [0, 0, 10], zoom: 88, near: 0.1, far: 100 }}
      gl={{ antialias: true, alpha: true }}
      style={{ width: "100%", height: "100%" }}
    >
      <Scene engineRef={engineRef} playing={playing} slowMo={slowMo} />
    </Canvas>
  );
}

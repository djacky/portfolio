/* ------------------------------------------------------------------
   Implicit MPC — three-phase grid-tied voltage source converter (VSC)
   with an LCL output filter, working in the synchronous dq frame.

   Plant (continuous, dq, PLL locked at grid angle θ = ω·t):
     L_f · di_f/dt   = u  − v_cf − R_f·i_f  + ω·L_f·J·i_f
     C_f · dv_cf/dt  = i_f − i_g            + ω·C_f·J·v_cf
     L_g · di_g/dt   = v_cf − v_g − R_g·i_g + ω·L_g·J·i_g
     C_dc·dV_dc/dt   = (3/2)(u_d·i_fd + u_q·i_fq) / V_dc  −  i_load

   where J = [[0, −1], [1, 0]] (90° rotation in dq).

   State vector (6):  x = [i_fd, i_fq, v_cfd, v_cfq, i_gd, i_gq]
   Input  vector (2): u = [u_d, u_q]                  (averaged bridge)
   Disturbance     :  v_g = [v_gd, v_gq]              (grid voltage)
   Auxiliary state :  V_dc                            (DC-bus voltage)

   The MPC discretises the 6-state LCL by ZOH (matrix exponential via
   scaling-and-squaring + Padé) at Ts = 250 µs, then solves at every
   tick the condensed QP

       min_U   ½ Uᵀ H U + fᵀ U
       s.t.    A_con U ≤ b(x₀, u_prev, r[·], v_g, wHat)

   with U = [u_d(0), u_q(0), …, u_d(N−1), u_q(N−1)] ∈ R^{2N},
   tracking a current reference R = [i_fd*, i_fq*, …] over the
   horizon (the controlled outputs are the converter-side currents).

   Constraints enforced inside the QP (all hard, projected by Hildreth's
   dual coordinate descent — small dense problem, no factorisations
   per iter, warm-started from the previous multiplier):
     • input box      :  |u_d|, |u_q| ≤ U_MAX                (≈ V_dc/√3)
     • input slew     :  |Δu_d|, |Δu_q| ≤ dU_MAX
     • current rails  :  |i_fd|, |i_fq|, |i_gd|, |i_gq| ≤ I_MAX_*
   These produce ~16N rows per solve (160 for N=10).

   An outer cascaded loop sits above: a PI on V_dc generates a P_ref
   active-power command, which becomes i_gd* (at unity power factor
   i_gq* = 0).  Reference and load disturbance enter at their physical
   locations (P_ref = grid-side current set-point, i_load = DC-bus
   current draw).

   For comparison the file ships a naive per-axis cascaded PI baseline
   that is intentionally NOT decoupled — clear visual deficit when the
   user enables a load step or aggressive ref step.

   Visualisation: dq states are rotated back into abc by the inverse
   amplitude-invariant Park transform for the waveform plot.
------------------------------------------------------------------ */

// ============================================================
//  Plant parameters — 3-phase, 5 kW class, European LV grid
// ============================================================

export const PARAMS = {
  // Grid
  V_grid_LL_rms: 400,              // V    line-to-line RMS
  f_grid: 50,                      // Hz   grid frequency

  // LCL filter
  L_f: 2.5e-3,                     // H    converter-side inductor
  R_f: 0.05,                       // Ω    L_f winding loss
  L_g: 1.5e-3,                     // H    grid-side inductor
  R_g: 0.05,                       // Ω    L_g winding loss
  C_f: 8e-6,                       // F    filter capacitor (Y-connected)

  // DC link
  C_dc: 1.5e-3,                    // F    DC-link capacitor
  V_dc_nom: 750,                   // V    nominal DC bus voltage

  // Operating envelope — sized for a 40 kW-class DC fast charger feeding
  // a 55 A-max EV battery.  At V_dc = 750 V, 55 A load ≈ 41 kW, which at
  // V_gd = 326 V maps to i_gd ≈ 84 A — well above I_G_MAX, so a full-demand
  // slider combined with a ref step is guaranteed to make the MPC's
  // grid-side current rail bind.
  P_rated: 40_000,                 // W    rated active power
  I_F_MAX: 75,                     // A    converter-side current rail (peak, per axis)
  I_G_MAX: 65,                     // A    grid-side current rail (peak, per axis)

  // Sampling
  T_s: 250e-6,                     // s    MPC sample time (4 kHz) — see notes
  T_plant: 50e-6,                  // s    inner plant integration step (20 kHz)
  f_sw: 10_000,                    // Hz   physical switching frequency (display only)

  // MPC limits (tightened relative to physical limits to leave headroom)
  // U_MAX is the linear SVPWM bound: |u_dq| ≤ V_dc/√3.  We use 0.92·V_dc/√3
  // as an axis-wise box to over-bound the inscribed circle conservatively.
  // dU_MAX is per-sample-per-axis slew; tight enough that aggressive ref
  // steps make the slew constraint bite.
  U_MAX_FRAC: 0.92,
  dU_MAX: 60,                      // V    per-sample input slew, per axis

  // Outer V_dc loop
  V_DC_REF: 750,
  V_DC_KP: 0.6,                    // A / V    proportional
  V_DC_KI: 18,                     // A / (V·s) integral
  P_REF_MAX: 50_000,               // W    saturation on outer loop
} as const;

// -------- grid harmonic disturbance --------
// User-held 5th-harmonic pollution on the grid voltage — the dominant
// distortion on real LV grids (6-pulse rectifier loads, VFDs, arc furnaces).
// The 5th is a NEGATIVE-sequence harmonic in abc: v_a at 5ω, v_b shifted
// +120°, v_c shifted −120°.  After the Park transform at the fundamental ω
// it appears as a 6× ripple rotating backward in the dq frame:
//     v_gd^(5)(t) = V_5 · cos(6ωt)
//     v_gq^(5)(t) = −V_5 · sin(6ωt)                    (6·50 Hz = 300 Hz)
// Magnitude 8 % of the phase peak — inside IEEE 519 / IEC 61000-2-2
// compatibility envelopes but large enough that the MPC's Ed·v_g feedforward
// visibly attenuates what a blind controller would pass through to i_g.
// Unlike the sag, this does not change average power demand, so it stays
// stable under any slider setting including the 55 A rail-binding extreme.
const HARMONIC_AMP_FRAC = 0.08;
const HARMONIC_ORDER = 5;

// -------- IMP harmonic rejection (augmented MPC state) --------
// The 5th-harmonic shows up in the dq frame at H_DQ = 6× fundamental.  To
// reject it asymptotically, the textbook Francis–Wonham internal-model
// principle says we must embed a copy of the disturbance generator in the
// controller's state space.  We do that by augmenting the MPC model with a
// 2-state resonator at 6ω driven by the grid-side tracking error:
//     ξ ∈ R^2,       dξ/dt = A_r · ξ  +  (y − r)
//     A_r = [[−σ, +H_DQ·ω], [−H_DQ·ω, −σ]]              (damped rotor)
// Augmented state becomes x_aug = [x_plant; ξ] ∈ R^8.  The MPC predicts,
// constrains, AND costs the full 8-state trajectory — weighting ξ in the
// objective forces the optimal U to drive ξ→0, which by the IMP theorem
// equals zero steady-state error at ω_r.  Because ξ is part of the state
// space, the SVPWM circle and current rails are respected by construction
// — no post-MPC "companion correction" that can saturate and destabilise.
const HARMONIC_REJECTION_ORDER_DQ = 6;
// Light damping on the rotor so the poles sit at r·e^(±jω_r·Ts) with r<1
// instead of exactly on the unit circle.  A pure integrator at 6ω windups
// catastrophically during rail-binding (no horizon is long enough for ξ to
// decay before the QP becomes infeasible).  σ ≈ 15 rad/s gives a rotor
// time-constant of ~65 ms — fast enough that the MPC can see ξ grow and
// react within a couple of grid cycles, slow enough that the internal model
// still has ~45 dB of loop gain at resonance.
const RESONATOR_DAMPING = 15;
// Cost weight on the two resonator states.  With β tuned so |ξ|≈|e| at
// resonance, the weight should track Q_y so the MPC trades tracking and
// rejection symmetrically.  Terminal weight moderately higher for
// steady-state emphasis at the horizon end.
const RESONATOR_Q = 1500;
const RESONATOR_Q_F = 4500;
// Reference low-pass cut-off used by the engine's ξ driver (Hz).  The outer
// V_dc PI leaks a 6ω ripple into i_gd_ref via V_dc power-balance (V_DC_KP·
// V_dc·ΔV_dc ≈ 5 A ripple in the d-axis reference).  Driving ξ off raw
// (y − r) then has the rotor chase its own reference ripple, not the grid
// disturbance the IMP is supposed to kill.  We therefore feed ξ with
// (y − LPF(r)) at ~15 Hz cutoff so only DC-slow reference content counts
// as "the operating point"; any 6ω in r is treated as an unrejectable
// component the controller shouldn't fight.
const RESONATOR_REF_LPF_HZ = 15;

// Derived constants
export const OMEGA = 2 * Math.PI * PARAMS.f_grid;          // rad/s
// Rotor input gain (applied to the continuous-time B_r matrix before ZOH).
// With B_r = β·I₂, the rotor's closed-loop amplitude ratio from ripple e
// to state ξ is roughly β/(2σω_r) in open loop.  β=1 gives |ξ/e|≈1.8e-5
// at σ=15, ω_r≈1884 — ξ 5 orders of magnitude smaller than the ripple,
// and no finite Q_ξ can make the MPC care.  Empirically, β≈5000 lands
// |ξ|/|e|≈1 under the MPC's closed loop so Q_ξ comparable to Qy creates
// meaningful cost tension between tracking and rejection without having
// ξ dominate (which detunes the MPC against its own rail constraints).
const RESONATOR_B_GAIN = 5000;
export const V_GRID_PEAK = (PARAMS.V_grid_LL_rms * Math.SQRT2) / Math.sqrt(3); // phase-peak
// Amplitude-invariant Park: with PLL locked, v_gd = V_GRID_PEAK, v_gq = 0
export const V_GD = V_GRID_PEAK;
export const V_GQ = 0;
// Cycles per simulated second; useful for tagging plots.

// ============================================================
//  Linear algebra primitives
// ============================================================

type Mat = number[][];
type Vec = number[];

function zeros(n: number, m?: number): Mat {
  const M: Mat = [];
  const cols = m ?? n;
  for (let i = 0; i < n; i++) M.push(new Array(cols).fill(0));
  return M;
}
function eye(n: number): Mat {
  const M = zeros(n);
  for (let i = 0; i < n; i++) M[i][i] = 1;
  return M;
}
function copyMat(A: Mat): Mat {
  return A.map((r) => r.slice());
}
function addMat(A: Mat, B: Mat): Mat {
  const n = A.length;
  const m = A[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) C[i][j] = A[i][j] + B[i][j];
  return C;
}
function scaleMat(A: Mat, s: number): Mat {
  const n = A.length;
  const m = A[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) C[i][j] = A[i][j] * s;
  return C;
}
function matMul(A: Mat, B: Mat): Mat {
  const n = A.length;
  const k = B.length;
  const m = B[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const Ai = A[i];
    const Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < m; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}
function matVec(A: Mat, x: Vec): Vec {
  const n = A.length;
  const m = A[0].length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const Ai = A[i];
    for (let j = 0; j < m; j++) s += Ai[j] * x[j];
    out[i] = s;
  }
  return out;
}
function infNorm(A: Mat): number {
  let nm = 0;
  for (let i = 0; i < A.length; i++) {
    let r = 0;
    for (let j = 0; j < A[i].length; j++) r += Math.abs(A[i][j]);
    if (r > nm) nm = r;
  }
  return nm;
}
function invert(A: Mat): Mat {
  const n = A.length;
  const M: Mat = [];
  for (let i = 0; i < n; i++) {
    const r = new Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) r[j] = A[i][j];
    r[n + i] = 1;
    M.push(r);
  }
  for (let k = 0; k < n; k++) {
    let p = k;
    let maxV = Math.abs(M[k][k]);
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(M[i][k]);
      if (v > maxV) {
        maxV = v;
        p = i;
      }
    }
    if (maxV < 1e-14) throw new Error("singular matrix");
    if (p !== k) {
      const tmp = M[k];
      M[k] = M[p];
      M[p] = tmp;
    }
    const piv = M[k][k];
    for (let j = 0; j < 2 * n; j++) M[k][j] /= piv;
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = M[i][k];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[i][j] -= f * M[k][j];
    }
  }
  const out: Mat = [];
  for (let i = 0; i < n; i++) {
    const r = new Array(n);
    for (let j = 0; j < n; j++) r[j] = M[i][n + j];
    out.push(r);
  }
  return out;
}

/** Matrix exponential expm(A) by scaling-and-squaring with Taylor (deg 12).
 *  We first scale A → A_s = A / 2^s so that ||A_s||_∞ < 0.5, evaluate the
 *  Taylor series to degree 12 (well-conditioned in that region), then
 *  square s times.  Standard, robust, more than enough accuracy for the
 *  6×6 / 10×10 matrices we feed it. */
function expm(A: Mat): Mat {
  const n = A.length;
  const norm = infNorm(A);
  const s = norm < 0.5 ? 0 : Math.ceil(Math.log2(norm / 0.5));
  const scale = 1 / Math.pow(2, s);
  const As = scaleMat(A, scale);

  let term = eye(n);
  let result = eye(n);
  for (let k = 1; k <= 12; k++) {
    term = matMul(term, As);
    term = scaleMat(term, 1 / k);
    result = addMat(result, term);
  }
  for (let i = 0; i < s; i++) result = matMul(result, result);
  return result;
}

/** Zero-order-hold discretisation of  dx/dt = A·x + B·u + E·v
 *  with constant u(t) = u_k and v(t) = v_k over [k·Ts, (k+1)·Ts].
 *
 *  Uses the augmented-matrix trick:
 *      M = [[A_c, B_c, E_c], [0,   0,   0]] · Ts
 *      expm(M) = [[A_d, B_d, E_d], [0, I, 0]]
 *  yielding A_d, B_d_zoh, E_d_zoh in a single expm call. */
function zoh(
  Ac: Mat,
  Bc: Mat,
  Ec: Mat,
  Ts: number,
): { Ad: Mat; Bd: Mat; Ed: Mat } {
  const nx = Ac.length;
  const nu = Bc[0].length;
  const nv = Ec[0].length;
  const N = nx + nu + nv;
  const M = zeros(N, N);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nx; j++) M[i][j] = Ac[i][j] * Ts;
    for (let j = 0; j < nu; j++) M[i][nx + j] = Bc[i][j] * Ts;
    for (let j = 0; j < nv; j++) M[i][nx + nu + j] = Ec[i][j] * Ts;
  }
  const eM = expm(M);
  const Ad: Mat = [];
  const Bd: Mat = [];
  const Ed: Mat = [];
  for (let i = 0; i < nx; i++) {
    const arow = new Array(nx);
    const brow = new Array(nu);
    const erow = new Array(nv);
    for (let j = 0; j < nx; j++) arow[j] = eM[i][j];
    for (let j = 0; j < nu; j++) brow[j] = eM[i][nx + j];
    for (let j = 0; j < nv; j++) erow[j] = eM[i][nx + nu + j];
    Ad.push(arow);
    Bd.push(brow);
    Ed.push(erow);
  }
  return { Ad, Bd, Ed };
}

// ============================================================
//  LCL continuous-time state-space (dq, with ω cross-coupling)
// ============================================================

/** Build the 6×6 continuous-time A_c matrix for the LCL plant. */
function buildAc(): Mat {
  const A = zeros(6);
  const { L_f, R_f, L_g, R_g, C_f } = PARAMS;
  const w = OMEGA;
  // Row 0: di_fd/dt
  A[0][0] = -R_f / L_f;
  A[0][1] = w;
  A[0][2] = -1 / L_f;
  // Row 1: di_fq/dt
  A[1][0] = -w;
  A[1][1] = -R_f / L_f;
  A[1][3] = -1 / L_f;
  // Row 2: dv_cfd/dt
  A[2][0] = 1 / C_f;
  A[2][3] = w;
  A[2][4] = -1 / C_f;
  // Row 3: dv_cfq/dt
  A[3][1] = 1 / C_f;
  A[3][2] = -w;
  A[3][5] = -1 / C_f;
  // Row 4: di_gd/dt
  A[4][2] = 1 / L_g;
  A[4][4] = -R_g / L_g;
  A[4][5] = w;
  // Row 5: di_gq/dt
  A[5][3] = 1 / L_g;
  A[5][4] = -w;
  A[5][5] = -R_g / L_g;
  return A;
}

/** B_c (6×2): converter voltage enters through L_f. */
function buildBc(): Mat {
  const B = zeros(6, 2);
  B[0][0] = 1 / PARAMS.L_f;
  B[1][1] = 1 / PARAMS.L_f;
  return B;
}

/** E_c (6×2): grid voltage enters through L_g, opposing i_g. */
function buildEc(): Mat {
  const E = zeros(6, 2);
  E[4][0] = -1 / PARAMS.L_g;
  E[5][1] = -1 / PARAMS.L_g;
  return E;
}

// ============================================================
//  Park transforms (amplitude-invariant)
// ============================================================

/** Inverse Park: dq → abc.  θ is the grid angle (rad). */
export function dqToAbc(d: number, q: number, theta: number): [number, number, number] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const c1 = Math.cos(theta - (2 * Math.PI) / 3);
  const s1 = Math.sin(theta - (2 * Math.PI) / 3);
  const c2 = Math.cos(theta + (2 * Math.PI) / 3);
  const s2 = Math.sin(theta + (2 * Math.PI) / 3);
  return [d * c - q * s, d * c1 - q * s1, d * c2 - q * s2];
}

// ============================================================
//  Three-phase LCL plant simulator
// ============================================================

export class LCLPlant {
  /** State: [i_fd, i_fq, v_cfd, v_cfq, i_gd, i_gq]. */
  public x: Float64Array = new Float64Array(6);
  /** DC-bus voltage (separate state — non-linear coupling to inverter). */
  public V_dc: number = PARAMS.V_dc_nom;
  /** Grid angle (PLL locked, monotonically increasing). */
  public theta = 0;
  /** Inverter input load current (DC side). Disturbance input. */
  public i_load = 0;

  private readonly dt: number;
  // ZOH discrete-time matrices for the linear LCL block, evaluated at dt.
  // Forward-Euler at 50 µs is *unconditionally unstable* for the LCL
  // resonance (ω_res ≈ 11.5 krad/s, ζ ≈ 0.0014) — the discretisation error
  // pumps the resonance and the plant blows up regardless of the controller.
  // ZOH is A-stable and exact for piecewise-constant u, v_g.
  private readonly Ad: number[][];
  private readonly Bd: number[][];
  private readonly Ed: number[][];

  constructor(dt = PARAMS.T_plant) {
    this.dt = dt;
    const Ac = buildAc();
    const Bc = buildBc();
    const Ec = buildEc();
    const { Ad, Bd, Ed } = zoh(Ac, Bc, Ec, dt);
    this.Ad = Ad;
    this.Bd = Bd;
    this.Ed = Ed;
    // Pre-charge filter cap to grid voltage so we start in electrical
    // equilibrium — without this, di_g/dt ≈ −v_g/L_g ≈ −217 kA/s on sample 0
    // and the grid current blows past I_G_MAX before the MPC can react.
    this.x[2] = V_GD;
  }

  reset(): void {
    this.x.fill(0);
    this.x[2] = V_GD;
    this.V_dc = PARAMS.V_dc_nom;
    this.theta = 0;
    this.i_load = 0;
  }

  /** One ZOH step at dt.  u is converter dq voltage commanded by the
   *  controller (already saturated in the linear SVPWM region). */
  step(u_d: number, u_q: number, v_gd = V_GD, v_gq = V_GQ): void {
    const { dt, Ad, Bd, Ed } = this;
    // Saturate u_d, u_q to the linear SVPWM circle |u| ≤ V_dc/√3.
    const u_lim = this.V_dc / Math.sqrt(3);
    const u_mag = Math.hypot(u_d, u_q);
    let ud = u_d;
    let uq = u_q;
    if (u_mag > u_lim) {
      const k = u_lim / u_mag;
      ud *= k;
      uq *= k;
    }

    // x[k+1] = Ad·x[k] + Bd·u + Ed·v_g
    const xNext = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      const Adi = Ad[i];
      let s =
        Adi[0] * this.x[0] +
        Adi[1] * this.x[1] +
        Adi[2] * this.x[2] +
        Adi[3] * this.x[3] +
        Adi[4] * this.x[4] +
        Adi[5] * this.x[5];
      s += Bd[i][0] * ud + Bd[i][1] * uq;
      s += Ed[i][0] * v_gd + Ed[i][1] * v_gq;
      xNext[i] = s;
    }

    // DC-bus dynamics — non-linear, slow, well-behaved with forward Euler:
    //   i_dc_in = (3/2)·(u_d·i_fd + u_q·i_fq) / V_dc   (power balance)
    //   C_dc · dV_dc/dt = i_dc_in − i_load
    // Use the average i_f across the interval to stay second-order accurate.
    const i_fd_avg = 0.5 * (this.x[0] + xNext[0]);
    const i_fq_avg = 0.5 * (this.x[1] + xNext[1]);
    const p_in = 1.5 * (ud * i_fd_avg + uq * i_fq_avg);
    const i_dc_in = p_in / Math.max(this.V_dc, 1);
    const dVdc = (i_dc_in - this.i_load) / PARAMS.C_dc;
    this.V_dc = Math.max(50, this.V_dc + dt * dVdc);

    for (let i = 0; i < 6; i++) this.x[i] = xNext[i];

    // Advance grid angle.
    this.theta += OMEGA * dt;
    if (this.theta > 2 * Math.PI) this.theta -= 2 * Math.PI;
  }
}

// ============================================================
//  MPC — condensed multi-state, multi-input, with disturbance
// ============================================================

/** Live-classifiable active constraint families. */
export type ActiveConstraint =
  | "u_box"
  | "u_slew"
  | "i_f_max"
  | "i_g_max";

export interface MPCSolve {
  u_d: number;
  u_q: number;
  uSeq: Float64Array;          // [u_d(0), u_q(0), …]   length 2N
  xPred: Float64Array;         // nx·N — full predicted state trajectory (nx=6 or 8 depending on IMP)
  solveUs: number;
  iters: number;
  activeCount: number;
  active: ActiveConstraint[];
  jTrack: number;
  jEffort: number;
  /** SVPWM input bound used when the QP was assembled. */
  uBound: number;
}

export class MPCController {
  readonly N: number;
  /** Plant state dim (6) — always carried at indices 0..5 of x_aug. */
  readonly nxPlant = 6;
  readonly nu = 2;

  /** Full augmented state dim: 6 without reject, 8 with reject (appends 2
   *  resonator states ξ_d, ξ_q at indices 6, 7).  Rebuilt on toggle. */
  public nx: number = 6;

  // Discrete-time plant matrices (for the 6-state LCL) — rebuilt only when
  // params/Ts change.  Public so the engine's state observer reuses the
  // same ZOH model without recomputing its own expm.
  public Ad!: Mat;
  public Bd!: Mat;
  public Ed!: Mat;

  /** Augmented-state discrete matrices — used by prediction/constraint
   *  builders.  When harmonicReject is off these alias Ad/Bd/Ed (nx=6);
   *  when on they are the 8×8 / 8×2 block matrices that include the rotor
   *  dynamics and reference feedforward. */
  private Aaug!: Mat;
  private Baug!: Mat;
  private Eaug!: Mat;
  /** F_aug — drives ξ from the reference r[k].  nx_aug × 2.  Only non-zero
   *  when harmonicReject is on; needed so the MPC's predictions see the
   *  same (y−r)-driven rotor dynamics that the engine will apply. */
  private Faug!: Mat;

  /** Whether the augmented rotor states are live in this controller. */
  public harmonicReject = false;

  // Condensed prediction matrices.
  private Phi!: Mat;            // (nx·N) × nx
  private Gamma!: Mat;          // (nx·N) × (nu·N)   (lower-block-triangular)
  private Psi!: Mat;            // (nx·N) × nv       (cumulative E_aug)
  /** Reference feedforward: Omega[k][j] = A_aug^(k-j) · F_aug, blocked
   *  analogously to Gamma.  (nx·N) × (2·N).  Empty (0-row) when reject off. */
  private Omega!: Mat;

  // Cost weights (selecting the controlled outputs i_g_d, i_g_q).
  // We track grid-side current — that's what feeds the bus / load.
  private readonly Qy = 60;     // weight on (i_g − i_g*)^2 per axis
  private readonly Qf = 200;    // terminal weight (same axes)
  // Rw = 0.003 — raised from the old 5e-4.  At that level the closed-loop
  // gain from measurement noise onto u_dq was ≈ 1/Γ (Γ = Gamma[i_gd][u_d]),
  // which is enormous for the LCL; every 50 mA of sensor noise turned into
  // ~5 V of commanded voltage and the plant state picked it straight back
  // up.  6× R still leaves Q >> R (tracking stays tight) but kills the
  // noise-to-u amplification.
  private readonly Rw = 0.003;

  private H!: Mat;
  private Hinv!: Mat;
  private Acon!: Mat;
  private bBase!: Float64Array;
  private AH!: Mat;
  private D!: Mat;
  private m!: number;
  private lambda!: Float64Array;
  /** Last assembled SVPWM input bound (axis box magnitude). */
  private uBoundCached = 0;

  constructor(N = 10, harmonicReject = false) {
    this.N = N;
    this.discretise();
    this.configureAugmentation(harmonicReject);
    this.lambda = new Float64Array(this.m);
  }

  private discretise(): void {
    const Ac = buildAc();
    const Bc = buildBc();
    const Ec = buildEc();
    const { Ad, Bd, Ed } = zoh(Ac, Bc, Ec, PARAMS.T_s);
    this.Ad = Ad;
    this.Bd = Bd;
    this.Ed = Ed;
  }

  /** Assemble A_aug / B_aug / E_aug / F_aug for the requested augmentation
   *  mode, then rebuild prediction/cost/constraint stacks against them.
   *  When `on` is false, the rotor block is dropped and the augmented
   *  matrices alias the 6-state plant.  When `on` is true we prepend the
   *  ZOH'd plant with a 2-state damped rotor driven by (y−r):
   *      ξ[k+1] = A_r_d · ξ[k] + B_r_d · (C_out·x_plant[k] − r[k])
   *  giving the block structure
   *      A_aug = [[Ad, 0], [B_r_d · C_out, A_r_d]]
   *      B_aug = [Bd; 0]    E_aug = [Ed; 0]    F_aug = [0; −B_r_d]
   *  All subsequent builders loop over this.nx and blindly consume the
   *  augmented matrices, so turning IMP on truly extends the state space. */
  private configureAugmentation(on: boolean): void {
    this.harmonicReject = on;
    if (!on) {
      this.nx = this.nxPlant;
      this.Aaug = this.Ad;
      this.Baug = this.Bd;
      this.Eaug = this.Ed;
      this.Faug = zeros(this.nx, this.nu);
    } else {
      const nxp = this.nxPlant;
      const nxi = 2;
      const nx = nxp + nxi;
      this.nx = nx;
      const { Ar_d, Br_d } = buildRotorDiscrete(PARAMS.T_s);
      // Top block: plant dynamics zero-padded for the ξ columns.
      const A = zeros(nx, nx);
      for (let i = 0; i < nxp; i++)
        for (let j = 0; j < nxp; j++) A[i][j] = this.Ad[i][j];
      // Bottom-left B_r_d · C_out.  C_out picks x_plant indices 4,5 = [i_gd, i_gq].
      //   (B_r_d · C_out)[i][j] = B_r_d[i][j−4]   for j ∈ {4,5}, else 0.
      for (let i = 0; i < nxi; i++) {
        A[nxp + i][4] = Br_d[i][0];
        A[nxp + i][5] = Br_d[i][1];
      }
      // Bottom-right A_r_d.
      for (let i = 0; i < nxi; i++)
        for (let j = 0; j < nxi; j++) A[nxp + i][nxp + j] = Ar_d[i][j];
      this.Aaug = A;

      const B = zeros(nx, this.nu);
      for (let i = 0; i < nxp; i++)
        for (let j = 0; j < this.nu; j++) B[i][j] = this.Bd[i][j];
      this.Baug = B;

      const E = zeros(nx, this.nu);
      for (let i = 0; i < nxp; i++)
        for (let j = 0; j < this.nu; j++) E[i][j] = this.Ed[i][j];
      this.Eaug = E;

      // F_aug — reference enters ξ dynamics as −B_r_d·r (since ξ̇ grows on y−r).
      const F = zeros(nx, this.nu);
      for (let i = 0; i < nxi; i++)
        for (let j = 0; j < this.nu; j++) F[nxp + i][j] = -Br_d[i][j];
      this.Faug = F;
    }
    this.buildPredictionMatrices();
    this.buildCostMatrices();
    this.buildConstraintMatrix(PARAMS.V_dc_nom);
    this.lambda = new Float64Array(this.m);
  }

  /** Flip the harmonic-rejection augmentation on/off.  Rebuilds all the
   *  condensed QP matrices; expected to be called rarely (user toggle). */
  setHarmonicReject(on: boolean): void {
    if (on === this.harmonicReject) return;
    this.configureAugmentation(on);
  }

  private buildPredictionMatrices(): void {
    const { N, nx, nu, Aaug, Baug, Eaug, Faug } = this;
    const Phi = zeros(nx * N, nx);
    const Gamma = zeros(nx * N, nu * N);
    const Psi = zeros(nx * N, nu); // disturbance is 2-D too (v_g)
    // Omega: reference feedforward, block-triangular like Gamma.  Only
    // non-trivial rows are the ξ rows (when augmented); plant rows are
    // zero since F_aug's top 6 rows are zero.
    const Omega = zeros(nx * N, nu * N);

    // A_pow[k] = A_aug^{k+1}
    const Apow: Mat[] = [];
    let Acur = Aaug;
    for (let k = 0; k < N; k++) {
      Apow.push(Acur);
      if (k < N - 1) Acur = matMul(Aaug, Acur);
    }

    // Row block k (predicted x_aug[k+1]) :
    //   x[k+1] = A^{k+1} x_0
    //          + Σ_{j=0..k} A^{k-j} B u[j]
    //          + Σ_{j=0..k} A^{k-j} E v_g        (v_g constant → collapses into Psi)
    //          + Σ_{j=0..k} A^{k-j} F r[j]       (r varies, Omega block-triangular)
    for (let k = 0; k < N; k++) {
      // Phi
      const Apk = Apow[k];
      for (let i = 0; i < nx; i++)
        for (let c = 0; c < nx; c++) Phi[k * nx + i][c] = Apk[i][c];
      // Gamma + Omega blocks (B / F preceded by A^{k-j})
      for (let j = 0; j <= k; j++) {
        const power = k - j; // 0..k
        const Ajk: Mat = power === 0 ? eye(nx) : Apow[power - 1];
        const gBlock = matMul(Ajk, Baug);
        const oBlock = matMul(Ajk, Faug);
        for (let i = 0; i < nx; i++) {
          for (let c = 0; c < nu; c++) {
            Gamma[k * nx + i][j * nu + c] = gBlock[i][c];
            Omega[k * nx + i][j * nu + c] = oBlock[i][c];
          }
        }
      }
      // Psi  =  Σ_{j=0..k} A^{k-j} E_aug      (constant-v_g assumption)
      const cumA = zeros(nx);
      for (let j = 0; j <= k; j++) {
        const power = k - j;
        const Ajk: Mat = power === 0 ? eye(nx) : Apow[power - 1];
        for (let r = 0; r < nx; r++)
          for (let c = 0; c < nx; c++) cumA[r][c] += Ajk[r][c];
      }
      const psiBlock = matMul(cumA, Eaug);
      for (let i = 0; i < nx; i++)
        for (let c = 0; c < nu; c++) Psi[k * nx + i][c] = psiBlock[i][c];
    }

    this.Phi = Phi;
    this.Gamma = Gamma;
    this.Psi = Psi;
    this.Omega = Omega;
  }

  private buildCostMatrices(): void {
    const { N, nx, nu, Gamma } = this;
    // Qbar — diagonal.  Weighted entries:
    //   indices 4, 5  → grid current tracking error (always on)
    //   indices 6, 7  → rotor states ξ_d, ξ_q (only when augmented)
    // Putting Q_ξ > 0 on the rotor states is the whole point of state
    // augmentation: the MPC now believes ξ matters, so it plans u so that
    // ξ stays small, which (by the rotor dynamics) means killing any
    // sustained 6ω component in (y − r).  Without a Q_ξ weight the MPC
    // would happily let ξ grow — tracking cost alone, over a finite
    // horizon, cannot hit asymptotic sinusoidal rejection.
    const Qd = new Float64Array(nx * N);
    for (let k = 0; k < N; k++) {
      const isTerm = k === N - 1;
      const w = isTerm ? this.Qf : this.Qy;
      Qd[k * nx + 4] = w;
      Qd[k * nx + 5] = w;
      if (this.harmonicReject) {
        const wXi = isTerm ? RESONATOR_Q_F : RESONATOR_Q;
        Qd[k * nx + 6] = wXi;
        Qd[k * nx + 7] = wXi;
      }
    }
    // Rbar — diagonal, length nu·N
    const Rd = new Float64Array(nu * N);
    for (let i = 0; i < nu * N; i++) Rd[i] = this.Rw;

    // H = 2·(Gammaᵀ · Q · Gamma + R)
    const H = zeros(nu * N);
    for (let i = 0; i < nu * N; i++) {
      for (let j = 0; j < nu * N; j++) {
        let s = 0;
        for (let p = 0; p < nx * N; p++) s += Gamma[p][i] * Qd[p] * Gamma[p][j];
        H[i][j] = 2 * s;
      }
      H[i][i] += 2 * Rd[i];
    }
    this.H = H;
    this.Hinv = invert(H);
  }

  /** Build the box and slew constraint matrix.  State-bound rows are also
   *  here but their RHS is patched per solve (they depend on x₀ + Psi·v_g). */
  private buildConstraintMatrix(V_dc: number): void {
    const { N, nu, nx, Gamma } = this;
    const uBound = (V_dc / Math.sqrt(3)) * PARAMS.U_MAX_FRAC;
    this.uBoundCached = uBound;

    const rows: Mat = [];
    const b: number[] = [];

    // Input box: |u_d(k)| ≤ uBound, |u_q(k)| ≤ uBound  →  2·2·N = 4N rows
    for (let k = 0; k < N; k++) {
      for (let a = 0; a < nu; a++) {
        const idx = k * nu + a;
        const r1 = new Array(nu * N).fill(0);
        r1[idx] = 1;
        rows.push(r1);
        b.push(uBound);
        const r2 = new Array(nu * N).fill(0);
        r2[idx] = -1;
        rows.push(r2);
        b.push(uBound);
      }
    }
    // Slew: |u(k) − u(k−1)| ≤ dU  per axis  → 4N rows
    //   for k=0, "u(−1)" = u_prev → patched into bEff at solve time
    for (let k = 0; k < N; k++) {
      for (let a = 0; a < nu; a++) {
        const idx = k * nu + a;
        const idxPrev = (k - 1) * nu + a;
        const r1 = new Array(nu * N).fill(0);
        r1[idx] = 1;
        if (k > 0) r1[idxPrev] = -1;
        rows.push(r1);
        b.push(PARAMS.dU_MAX);
        const r2 = new Array(nu * N).fill(0);
        r2[idx] = -1;
        if (k > 0) r2[idxPrev] = 1;
        rows.push(r2);
        b.push(PARAMS.dU_MAX);
      }
    }
    // State rails on i_fd, i_fq, i_gd, i_gq  → 4·2·N = 8N rows
    //   (Gamma·U)[k][s] ≤ I_MAX − (Phi·x₀)[k][s] − (Psi·v_g)[k][s]
    //   indices into x: i_fd=0, i_fq=1, i_gd=4, i_gq=5
    const stateRows = [
      { idx: 0, lim: PARAMS.I_F_MAX },
      { idx: 1, lim: PARAMS.I_F_MAX },
      { idx: 4, lim: PARAMS.I_G_MAX },
      { idx: 5, lim: PARAMS.I_G_MAX },
    ];
    for (let k = 0; k < N; k++) {
      const blockBase = k * nx;
      for (const sr of stateRows) {
        const grow = Gamma[blockBase + sr.idx]; // length nu·N
        const rPos = grow.slice();
        rows.push(rPos);
        b.push(sr.lim);
        const rNeg = grow.map((v) => -v);
        rows.push(rNeg);
        b.push(sr.lim);
      }
    }

    this.Acon = rows;
    this.bBase = Float64Array.from(b);
    this.m = rows.length;

    // AH = Acon · Hinv  (m × nuN), constant given H/Acon
    const nuN = nu * N;
    const AH = zeros(this.m, nuN);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < nuN; j++) {
        let s = 0;
        const Ari = this.Acon[i];
        for (let p = 0; p < nuN; p++) s += Ari[p] * this.Hinv[p][j];
        AH[i][j] = s;
      }
    }
    this.AH = AH;

    // D = AH · Aconᵀ  (m × m), symmetric PSD
    const D = zeros(this.m);
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.m; j++) {
        let s = 0;
        const AHi = AH[i];
        const Aj = this.Acon[j];
        for (let p = 0; p < nuN; p++) s += AHi[p] * Aj[p];
        D[i][j] = s;
      }
    }
    this.D = D;

    this.lambda = new Float64Array(this.m);
  }

  /** Refresh the SVPWM input bound when the DC bus drifts.  Only the
   *  RHS of the input-box rows depends on V_dc — Acon, AH, D are
   *  invariant — so we patch bBase in place rather than rebuilding the
   *  whole constraint stack.  O(N), called every solve. */
  refreshInputBound(V_dc: number): void {
    const uBound = (V_dc / Math.sqrt(3)) * PARAMS.U_MAX_FRAC;
    if (Math.abs(uBound - this.uBoundCached) < 0.05) return;
    this.uBoundCached = uBound;
    // Input-box rows occupy indices [0, 4N).  Per k: +u_d, -u_d, +u_q, -u_q.
    const rows = 4 * this.N;
    for (let i = 0; i < rows; i++) this.bBase[i] = uBound;
  }

  /** Solve the QP.
   *
   *  @param x0       current augmented state estimate, length `this.nx`.
   *                  When harmonicReject is off: the 6-state plant estimate.
   *                  When on: 6-state plant + 2 resonator states ξ_d, ξ_q.
   *  @param uPrev    last applied input [u_d, u_q] (V)
   *  @param refSeq   reference grid current over the horizon, length 2N:
   *                  [i_gd*(1), i_gq*(1), …, i_gd*(N), i_gq*(N)]
   *  @param V_dc     measured DC bus (sets the SVPWM input bound)
   *  @param vG       grid voltage [v_gd, v_gq] in dq (constant over horizon)
   */
  solve(
    x0: Float64Array,
    uPrev: Float64Array,
    refSeq: Float64Array,
    V_dc: number,
    vG: [number, number] = [V_GD, V_GQ],
    /** Optional separate reference driving the ξ rotor (augmented mode
     *  only).  Typically a LPF'd version of refSeq, so the rotor is not
     *  driven by the 6ω ripple the outer V_dc PI leaks into i_gd_ref.
     *  Defaults to refSeq. */
    refSeqXi?: Float64Array,
  ): MPCSolve {
    const t0 = performance.now();
    this.refreshInputBound(V_dc);

    const { N, nx, nu } = this;
    const nuN = nu * N;
    const m = this.m;

    // Phi · x0  (length nx·N)
    const Phix = new Float64Array(nx * N);
    for (let i = 0; i < nx * N; i++) {
      let s = 0;
      const Pi = this.Phi[i];
      for (let j = 0; j < nx; j++) s += Pi[j] * x0[j];
      Phix[i] = s;
    }
    // Psi · v_g  (length nx·N)
    const Psv = new Float64Array(nx * N);
    for (let i = 0; i < nx * N; i++) {
      Psv[i] = this.Psi[i][0] * vG[0] + this.Psi[i][1] * vG[1];
    }
    // Omega · R_xi — reference-feedforward component of the free trajectory
    // for the augmented rotor.  MUST match what the engine actually applies
    // to ξ each tick (same rail-clamp on the reference), otherwise engine
    // and controller run on different augmented plants and the closed loop
    // destabilises.  R_xi clamps to ±I_G_MAX·0.95 so that an unreachable
    // outer-loop demand (i_gd*→104 A at peak EV charge vs 65 A rail) can't
    // pretend to be rejectable 6ω content and wind ξ into instability.
    const omR = new Float64Array(nx * N);
    if (this.harmonicReject) {
      const refCap = PARAMS.I_G_MAX * 0.95;
      const baseXi = refSeqXi ?? refSeq;
      const refXi = new Float64Array(nuN);
      for (let j = 0; j < nuN; j++) {
        const r = baseXi[j];
        refXi[j] = r > refCap ? refCap : r < -refCap ? -refCap : r;
      }
      for (let i = 0; i < nx * N; i++) {
        let s = 0;
        const Oi = this.Omega[i];
        for (let j = 0; j < nuN; j++) s += Oi[j] * refXi[j];
        omR[i] = s;
      }
    }
    // X_free = Phi·x0 + Psi·v_g + Omega·R_xi   (open-loop trajectory at U=0)
    const Xfree = new Float64Array(nx * N);
    for (let i = 0; i < nx * N; i++) Xfree[i] = Phix[i] + Psv[i] + omR[i];

    // y_free − r,  but only for indices 4 and 5 (i_g_d, i_g_q) at each block.
    // Build e_y: length 2N (the controlled outputs over horizon).
    const ey = new Float64Array(2 * N);
    for (let k = 0; k < N; k++) {
      ey[k * 2] = Xfree[k * nx + 4] - refSeq[k * 2];
      ey[k * 2 + 1] = Xfree[k * nx + 5] - refSeq[k * 2 + 1];
    }

    // Build f = 2·Gammaᵀ·Q·(Xfree − Rfull).
    //   - For indices 4,5 (i_g d/q): Q·ey term.
    //   - For indices 6,7 (ξ, when augmented): Q_ξ · Xfree term (ξ target = 0).
    // f[i] = 2·Σ_{k} Q_k · Gamma[k·nx + idx][i] · residual
    const f = new Float64Array(nuN);
    for (let i = 0; i < nuN; i++) {
      let s = 0;
      for (let k = 0; k < N; k++) {
        const isTerm = k === N - 1;
        const w = isTerm ? this.Qf : this.Qy;
        s += w * this.Gamma[k * nx + 4][i] * ey[k * 2];
        s += w * this.Gamma[k * nx + 5][i] * ey[k * 2 + 1];
        if (this.harmonicReject) {
          const wXi = isTerm ? RESONATOR_Q_F : RESONATOR_Q;
          s += wXi * this.Gamma[k * nx + 6][i] * Xfree[k * nx + 6];
          s += wXi * this.Gamma[k * nx + 7][i] * Xfree[k * nx + 7];
        }
      }
      f[i] = 2 * s;
    }

    // Build bEff: copy bBase, patch slew rows for k=0 and state rows.
    const bEff = new Float64Array(m);
    for (let i = 0; i < m; i++) bEff[i] = this.bBase[i];

    // Slew rows for k=0:
    //   layout: rows 0..4N-1 = input box; rows 4N..8N-1 = slew.
    //   slew block ordering: per k  (4 rows: +ud, -ud, +uq, -uq)
    //   For k=0 the +1 row needs uPrev added; the −1 row needs uPrev subtracted.
    const slewBase = 4 * N;
    bEff[slewBase + 0] = PARAMS.dU_MAX + uPrev[0]; // +u_d(0) − u_d(−1) ≤ dU
    bEff[slewBase + 1] = PARAMS.dU_MAX - uPrev[0]; // −u_d(0) + u_d(−1) ≤ dU
    bEff[slewBase + 2] = PARAMS.dU_MAX + uPrev[1];
    bEff[slewBase + 3] = PARAMS.dU_MAX - uPrev[1];

    // State rows (after 8N).  Each block of 8 rows (4 axes × 2 signs):
    //   row order: +i_fd, -i_fd, +i_fq, -i_fq, +i_gd, -i_gd, +i_gq, -i_gq
    //   bEff = limit  ∓ Xfree[k·nx + idx]
    const stateBase = 8 * N;
    const stateMap: Array<{ off: number; idx: number }> = [
      { off: 0, idx: 0 },
      { off: 1, idx: 0 },
      { off: 2, idx: 1 },
      { off: 3, idx: 1 },
      { off: 4, idx: 4 },
      { off: 5, idx: 4 },
      { off: 6, idx: 5 },
      { off: 7, idx: 5 },
    ];
    for (let k = 0; k < N; k++) {
      for (const sm of stateMap) {
        // Even off → row b is +Xfree subtracted; odd off → +Xfree added
        const sign = sm.off % 2 === 0 ? -1 : +1;
        bEff[stateBase + k * 8 + sm.off] += sign * Xfree[k * nx + sm.idx];
      }
    }

    // d = bEff + AH·f
    const d = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = bEff[i];
      const AHi = this.AH[i];
      for (let j = 0; j < nuN; j++) s += AHi[j] * f[j];
      d[i] = s;
    }

    // Hildreth dual coordinate descent (warm-started).
    const lambda = this.lambda;
    for (let i = 0; i < m; i++) if (lambda[i] < 0) lambda[i] = 0;
    const MAX_ITERS = 60;
    const TOL = 1e-6;
    let iter = 0;
    let maxDelta = Infinity;
    for (; iter < MAX_ITERS; iter++) {
      maxDelta = 0;
      for (let i = 0; i < m; i++) {
        const Drow = this.D[i];
        let acc = d[i];
        for (let j = 0; j < m; j++) {
          if (j !== i) acc += Drow[j] * lambda[j];
        }
        const Dii = Drow[i] > 1e-12 ? Drow[i] : 1e-12;
        let wi = -acc / Dii;
        if (wi < 0) wi = 0;
        const delta = Math.abs(wi - lambda[i]);
        if (delta > maxDelta) maxDelta = delta;
        lambda[i] = wi;
      }
      if (maxDelta < TOL) break;
    }

    // Primal recovery: U = −Hinv · (f + Aconᵀ λ)
    const AtL = new Float64Array(nuN);
    for (let j = 0; j < nuN; j++) {
      let s = 0;
      for (let i = 0; i < m; i++) s += this.Acon[i][j] * lambda[i];
      AtL[j] = s;
    }
    const U = new Float64Array(nuN);
    for (let i = 0; i < nuN; i++) {
      let s = 0;
      const Hi = this.Hinv[i];
      for (let j = 0; j < nuN; j++) s -= Hi[j] * (f[j] + AtL[j]);
      U[i] = s;
    }

    // Predicted state trajectory: X = Xfree + Gamma·U
    const xPred = new Float64Array(nx * N);
    for (let i = 0; i < nx * N; i++) {
      let s = Xfree[i];
      const Gi = this.Gamma[i];
      for (let j = 0; j < nuN; j++) s += Gi[j] * U[j];
      xPred[i] = s;
    }

    // Cost split
    let jTrack = 0;
    let jEffort = 0;
    for (let k = 0; k < N; k++) {
      const w = k === N - 1 ? this.Qf : this.Qy;
      const ed = xPred[k * nx + 4] - refSeq[k * 2];
      const eq = xPred[k * nx + 5] - refSeq[k * 2 + 1];
      jTrack += w * (ed * ed + eq * eq);
    }
    for (let i = 0; i < nuN; i++) jEffort += this.Rw * U[i] * U[i];

    // Active-set classification
    const active: ActiveConstraint[] = [];
    const seen = new Set<ActiveConstraint>();
    let activeCount = 0;
    const THRESH = 1e-4;
    const add = (n: ActiveConstraint) => {
      if (!seen.has(n)) {
        seen.add(n);
        active.push(n);
      }
    };
    for (let i = 0; i < m; i++) {
      if (lambda[i] <= THRESH) continue;
      activeCount++;
      if (i < 4 * N) add("u_box");
      else if (i < 8 * N) add("u_slew");
      else {
        const off = (i - 8 * N) % 8;
        if (off < 4) add("i_f_max");
        else add("i_g_max");
      }
    }

    return {
      u_d: U[0],
      u_q: U[1],
      uSeq: U,
      xPred,
      solveUs: (performance.now() - t0) * 1000,
      iters: iter + 1,
      activeCount,
      active,
      jTrack,
      jEffort,
      uBound: this.uBoundCached,
    };
  }

  resetWarmStart(): void {
    this.lambda.fill(0);
  }
}

// ============================================================
//  Naive cascaded PI baseline (per-axis, NOT decoupled).  This is
//  the textbook deficient design — undergraduate code.  It has no
//  knowledge of cross-coupling, no constraint awareness, no
//  feed-forward of grid voltage.  Will visibly clip the rails on
//  load steps and leak between d and q axes.
// ============================================================

export class PIController {
  private integ_d = 0;
  private integ_q = 0;
  // Loop-shaping against a lumped-inductor approximation of the LCL
  // (L_eff ≈ L_f + L_g = 4 mH).  For a plant gain 1/(L_eff·s):
  //     closed-loop BW ω_c  ⇒  Kp ≈ L_eff·ω_c
  //     integrator zero ω_i ⇒  Ki ≈ Kp·ω_i
  // At 40 kW / 65 A scale we detune to ω_c ≈ 250 rad/s and ω_i ≈ 25 rad/s —
  // Kp = 1.0 V/A, Ki = 25 V/(A·s).  The integrator is *not* responsible for
  // building the ~326 V steady-state command: that comes from a grid-voltage
  // feed-forward (v_gd added to u_d, v_gq=0 to u_q) — a totally standard
  // move in any real VSC PI.  Without FF the integrator had to climb 326 V
  // on every cold start / reset / mode-switch and the closed loop rang the
  // LCL resonance to death.
  //
  // Measurement LPF: without active damping the LCL's 1.6 kHz resonance
  // (ζ ≈ 1e-3) feeds straight back through the PI, which "corrects" it,
  // pumps another transient into u, and the ringing slowly grows —
  // exactly the creeping instability a naive dq-PI has on any LCL plant.
  // A first-order LPF on the measurement at ω_f ≈ 800 rad/s sits well
  // above the closed-loop BW (250 rad/s) and well below resonance
  // (10 krad/s), rejecting enough of the resonant mode to keep the
  // baseline running.  It's still naive: no dq decoupling (ωL cross-term
  // leaks), no constraint awareness (rails pierce on disturbance), no
  // preview — all the levers the MPC actually exploits.
  // Cascade sanity:  inner current BW must be ≥ 5× outer V_dc BW or the
  // cascade inverts and V_dc oscillates load changes into the reference.
  // Outer V_dc PI crossover ≈ V_DC_KP/C_dc = 400 rad/s.  Target inner
  // crossover: 1500 rad/s (3.8× faster).  LCL resonance sits at 11.5 krad/s,
  // so 1500 rad/s is also ≈ ω_res/8 — well below resonance with margin,
  // and the active-damping i_cf feedback below keeps the resonance pole
  // tamed inside that margin.
  //   Kp = ω_c · L_total = 1500 · 4 mH = 6 V/A
  //   Ki = Kp · ω_i,  ω_i = ω_c/10 = 150 rad/s  →  Ki = 900 V/(A·s)
  //   Kbc = Ki/Kp = 150  (standard back-calculation tracking gain)
  private readonly Kp = 6;
  private readonly Ki = 900;
  private readonly Kbc = 150;
  private readonly INTEG_CAP = 300;   // V — loose safety net, back-calc does the work
  // α = 1 − exp(−ω_f · T_s).  ω_f = 3000 rad/s → α ≈ 0.528.
  // Sits 2× above the closed-loop BW (1500 rad/s) so it doesn't eat phase
  // margin, and ≈ 4× below the LCL resonance at 11.5 krad/s so it still
  // attenuates the resonant mode appreciably.
  private readonly MEAS_ALPHA = 0.528;
  private i_d_filt = 0;
  private i_q_filt = 0;
  private filtPrimed = false;

  // ---- Active damping: direct capacitor-current feedback ----
  // The LCL resonance (~1.84 kHz here, ζ ≈ 0.2%) has to be damped by
  // feedback on i_cf, NOT on v_cf — the two are 90° apart at resonance, so
  // v_cf feedback is wrong-phase and destabilizes.  Tried estimating i_cf
  // by backward-differencing v_cf, but at this resonance ω·Ts ≈ 2.87 rad
  // (near Nyquist), and the difference operator no longer has the +90°
  // phase of a true derivative — it collapses toward proportional v_cf
  // feedback, which is exactly the wrong phase.
  //
  // KCL at the cap node gives i_cf directly:  i_cf = i_f − i_g.  Both
  // currents are measured states, so we feed back i_f − i_g without any
  // derivative approximation — correct phase at every frequency, zero at
  // DC steady state (because i_f = i_g at equilibrium).  Subtracting
  // K_AD·i_cf from u behaves like a virtual resistance K_AD Ω in series
  // with L_f, which adds damping at resonance.
  //
  // Damping math: ζ_res ≈ K_AD / (2 · ω_res · L_f) = K_AD / 57.7
  //   K_AD =  5 Ω → ζ ≈ 0.087  (under-damped, visible ring)
  //   K_AD = 10 Ω → ζ ≈ 0.173  (acceptable, no visible ring)
  //   K_AD = 18 Ω → ζ ≈ 0.31   (well-damped but the mode-entry kick from
  //                             any residual i_cf ≠ 0 is large enough to
  //                             excite V_dc overshoot and wind up the
  //                             outer integrator negative → bus crash)
  // 10 Ω is the sweet spot here — enough damping for fast transients
  // without giving the damping term so much authority that it slams u
  // on any transfer-related mismatch.
  private readonly K_AD = 10.0;

  reset(): void {
    this.integ_d = 0;
    this.integ_q = 0;
    this.i_d_filt = 0;
    this.i_q_filt = 0;
    this.filtPrimed = false;
  }

  /** Bumpless transfer — seed the integrator so u_d(0) = u_seed_d given the
   *  current tracking error AND the grid feed-forward.  Used on MPC→PI
   *  switch so the PI picks up at the MPC's last command without a jump. */
  primeBumpless(u_seed_d: number, u_seed_q: number, i_d: number, i_q: number,
                iref_d: number, iref_q: number, v_gd: number, v_gq: number): void {
    // Prime the LPF to the current measurement so it doesn't ring up from 0.
    this.i_d_filt = i_d;
    this.i_q_filt = i_q;
    this.filtPrimed = true;
    this.integ_d = u_seed_d - this.Kp * (iref_d - i_d) - v_gd;
    this.integ_q = u_seed_q - this.Kp * (iref_q - i_q) - v_gq;
    const cap = this.INTEG_CAP;
    if (this.integ_d > cap) this.integ_d = cap;
    else if (this.integ_d < -cap) this.integ_d = -cap;
    if (this.integ_q > cap) this.integ_q = cap;
    else if (this.integ_q < -cap) this.integ_q = -cap;
  }

  step(
    i_d: number,
    i_q: number,
    iref_d: number,
    iref_q: number,
    v_gd: number,
    v_gq: number,
    i_cfd: number,
    i_cfq: number,
    u_lim: number,
    Ts: number,
  ): [number, number] {
    // First-order LPF on the measurement — prevents LCL resonance feedback
    // and attenuates sensor dither before it reaches Kp/Ki.
    if (!this.filtPrimed) {
      this.i_d_filt = i_d;
      this.i_q_filt = i_q;
      this.filtPrimed = true;
    } else {
      this.i_d_filt += this.MEAS_ALPHA * (i_d - this.i_d_filt);
      this.i_q_filt += this.MEAS_ALPHA * (i_q - this.i_q_filt);
    }

    const ed = iref_d - this.i_d_filt;
    const eq = iref_q - this.i_q_filt;
    const cap = this.INTEG_CAP;

    // Regular integral update (no conditional freeze — back-calc handles
    // wind-up more gracefully by relaxing the integrator toward the
    // consistent value instead of hard-freezing it).
    this.integ_d += this.Ki * ed * Ts;
    this.integ_q += this.Ki * eq * Ts;

    // Safety cap — in normal operation back-calc keeps the integrator well
    // inside this envelope; the cap only matters for pathological setpoints.
    if (this.integ_d > cap) this.integ_d = cap;
    else if (this.integ_d < -cap) this.integ_d = -cap;
    if (this.integ_q > cap) this.integ_q = cap;
    else if (this.integ_q < -cap) this.integ_q = -cap;

    // Grid-voltage feed-forward: the bulk of u_d is the grid EMF itself.
    // The PI only has to correct the residual inductor drop + load disturb.
    // Active-damping term: subtract K_AD·i_cf — acts as a virtual resistor
    // in series with L_f, adding damping at the LCL resonance.  i_cf
    // is measured directly (i_f − i_g by KCL at the cap node), which has
    // correct phase at every frequency and is zero at DC steady state.
    const u_d_uc = v_gd + this.Kp * ed + this.integ_d - this.K_AD * i_cfd;
    const u_q_uc = v_gq + this.Kp * eq + this.integ_q - this.K_AD * i_cfq;

    // Circle clip to the SVPWM bound |u| ≤ u_lim (= V_dc/√3 · U_MAX_FRAC).
    let u_d = u_d_uc;
    let u_q = u_q_uc;
    const mag = Math.hypot(u_d_uc, u_q_uc);
    if (mag > u_lim && mag > 1e-9) {
      const s = u_lim / mag;
      u_d = u_d_uc * s;
      u_q = u_q_uc * s;
      // Back-calculation anti-windup — bleed the integrator toward the
      // value consistent with the actually-applied u.  Stops the classic
      // "reference unreachable, integrator ramps forever" collapse that
      // detonates a naive PI on an over-rail EV-demand ramp.
      this.integ_d += this.Kbc * (u_d - u_d_uc) * Ts;
      this.integ_q += this.Kbc * (u_q - u_q_uc) * Ts;
      if (this.integ_d > cap) this.integ_d = cap;
      else if (this.integ_d < -cap) this.integ_d = -cap;
      if (this.integ_q > cap) this.integ_q = cap;
      else if (this.integ_q < -cap) this.integ_q = -cap;
    }
    return [u_d, u_q];
  }
}

// ============================================================
//  Outer V_dc PI loop — generates active power reference
// ============================================================

class VdcOuterPI {
  private integ = 0;

  reset(): void {
    this.integ = 0;
  }

  /** Returns active-power reference P_ref (W).  In PI mode this is just a
   *  residual trim on top of the load feedforward computed in the engine;
   *  in MPC mode this carries the full V_dc regulation task. */
  step(V_dc: number, V_dc_ref: number, Ts: number): number {
    const e = V_dc_ref - V_dc;
    this.integ += PARAMS.V_DC_KI * e * Ts;
    if (this.integ > PARAMS.P_REF_MAX) this.integ = PARAMS.P_REF_MAX;
    else if (this.integ < -PARAMS.P_REF_MAX) this.integ = -PARAMS.P_REF_MAX;
    let P = PARAMS.V_DC_KP * V_dc * e + this.integ;
    if (P > PARAMS.P_REF_MAX) P = PARAMS.P_REF_MAX;
    else if (P < -PARAMS.P_REF_MAX) P = -PARAMS.P_REF_MAX;
    return P;
  }
}

// ============================================================
//  IMP state-augmentation — 2D damped rotor at 6ω
// ============================================================
//
// Helpers that build the discrete rotor matrices used when the MPC runs in
// its augmented-state configuration.  The rotor lives inside the MPC's state
// vector (not as a post-controller companion) so the QP's predictions,
// constraints, and cost all see ξ natively.
//
// Continuous-time form (real 2×2 damped rotor):
//     dξ/dt = A_r · ξ  +  B_r · (y − r)
//     A_r   = [[−σ, +ω_r], [−ω_r, −σ]],   B_r = I₂
// The eigenvalues are −σ ± jω_r; the ±jω_r imaginary part is the internal
// model that makes ξ resonate with the 6ω error (negative-sequence → dq
// appears at ω_r = H_DQ·OMEGA in the "clockwise" sense matched by this real
// 2×2 block).  Adding Q_ξ·ξᵀξ to the MPC cost makes the optimal input
// actively suppress any sustained ω_r content in (y − r); Francis–Wonham
// IMP then guarantees asymptotic zero error at ω_r.  Unlike the former
// "PR companion" approach, ξ sits inside the constraint-aware QP so the
// SVPWM circle and current rails cannot be violated by the harmonic loop.
// ============================================================

/** Closed-form ZOH discretisation of
 *      dξ/dt = A_r · ξ + β·I · e
 *  with β = RESONATOR_B_GAIN.  Returns A_r_d = exp(A_r·Ts) and
 *  B_r_d = ∫₀^Ts exp(A_r·τ) dτ · β·I.  β is chosen so |ξ/e| at resonance
 *  is O(1), making ξ numerically comparable to the tracking error so a
 *  sensibly-sized Q_ξ actually causes the MPC to plan ξ-suppression. */
function buildRotorDiscrete(Ts: number): { Ar_d: Mat; Br_d: Mat } {
  const wr = HARMONIC_REJECTION_ORDER_DQ * OMEGA;
  const s = RESONATOR_DAMPING;
  const c = Math.cos(wr * Ts);
  const sn = Math.sin(wr * Ts);
  const decay = Math.exp(-s * Ts);
  // A_r_d = e^(−σ·Ts) · [[cos, +sin], [−sin, cos]]
  const Ar_d: Mat = [
    [decay * c,  decay * sn],
    [-decay * sn, decay * c],
  ];
  // B_r_d = A_r⁻¹ · (A_r_d − I) · (β·I).  For A_r = [[−σ, ω_r],
  // [−ω_r, −σ]], A_r⁻¹ = (1/(σ²+ω_r²)) · [[−σ, −ω_r], [+ω_r, −σ]].
  const det = s * s + wr * wr;
  const beta = RESONATOR_B_GAIN;
  const Ainv: Mat = [
    [-s / det, -wr / det],
    [ wr / det, -s / det],
  ];
  const AmI: Mat = [
    [Ar_d[0][0] - 1, Ar_d[0][1]    ],
    [Ar_d[1][0],     Ar_d[1][1] - 1],
  ];
  const AinvAmI = matMul(Ainv, AmI);
  const Br_d: Mat = [
    [AinvAmI[0][0] * beta, AinvAmI[0][1] * beta],
    [AinvAmI[1][0] * beta, AinvAmI[1][1] * beta],
  ];
  return { Ar_d, Br_d };
}

// ============================================================
//  Engine — wraps plant + both controllers + ring buffers.
//  Owns sim state and is driven by the render loop.
// ============================================================

export type ControllerMode = "mpc" | "pi";

export class MPCEngine {
  // -------- ring-buffer history (plotted) --------
  public readonly N_HISTORY = 240;
  public readonly i_a_buf = new Float32Array(this.N_HISTORY);
  public readonly i_b_buf = new Float32Array(this.N_HISTORY);
  public readonly i_c_buf = new Float32Array(this.N_HISTORY);
  public readonly i_a_ref_buf = new Float32Array(this.N_HISTORY);
  public readonly v_dc_buf = new Float32Array(this.N_HISTORY);
  public readonly v_dc_ref_buf = new Float32Array(this.N_HISTORY);
  /** Direct d-axis grid current — plotted on the V_dc panel's right axis
   *  so the viewer can see the dq-side set-point the MPC is actually
   *  regulating against (ref-step → i_gd tracks instantly, V_dc barely
   *  nudges because the power imbalance is integrated through C_dc). */
  public readonly i_gd_buf = new Float32Array(this.N_HISTORY);
  public readonly i_gd_ref_buf = new Float32Array(this.N_HISTORY);
  public head = 0;
  public filled = 0;
  /** Wall-clock seconds-of-sim per buffer slot — used by waveform world to
   *  scale the X axis correctly when slow-mo changes how many MPC steps
   *  fit into one render frame. */
  public readonly buf_dt = PARAMS.T_s;

  // -------- physical signals (for HUDs) --------
  public V_dc: number = PARAMS.V_dc_nom;
  public i_gd_meas = 0;
  public i_gq_meas = 0;
  public i_fd_meas = 0;
  public i_fq_meas = 0;
  public theta = 0;
  public iref_d = 0;
  public iref_q = 0;
  public P_ref = 0;

  /** EV-side DC current demand (the load sitting behind the DC/DC buck).
   *  Exposed so the schematic can draw a live `i_car` label and the
   *  Disturbance Console slider can mutate it.  Backed by plant.i_load. */
  public get i_load(): number {
    return this.plant.i_load;
  }

  // -------- state for controllers --------
  // Seed u_prev to the steady-state command (v_cfd ≈ V_GD).  Otherwise the
  // 60 V/sample slew constraint pins the first solve to ±60 V while the real
  // operating point sits at ≈ 326 V, causing a multi-millisecond startup
  // transient even with the cap pre-charged.
  public u_d = V_GD;
  public u_q = 0;
  public mode: ControllerMode = "mpc";
  public horizon = 10;
  public mpc: MPCController;
  private pi: PIController;
  private outerPI: VdcOuterPI;
  /** Rotor state when harmonic rejection is on.  Carried in the engine so
   *  it can be seeded at toggle time and then handed to the MPC as the
   *  augmented x₀ on every solve.  Zero when reject is off. */
  private xi_d = 0;
  private xi_q = 0;
  /** LPF'd reference used as the "DC operating point" for ξ-error input
   *  (see RESONATOR_REF_LPF_HZ).  Initialised to track iref on first use. */
  private xi_ref_d_lp = 0;
  private xi_ref_q_lp = 0;
  private xi_ref_primed = false;
  /** Discrete rotor matrices — cached here so we can propagate ξ at the
   *  end of each tick using the same ZOH'd dynamics that the MPC's
   *  prediction assumes.  Built once; the MPC re-builds its own copy. */
  private readonly rotor = buildRotorDiscrete(PARAMS.T_s);
  private plant: LCLPlant;
  /** Timestep counter (in MPC ticks). */
  public t = 0;

  // Luenberger state observer — the "extra states" answer to the noise
  // question.  Instead of a per-axis IIR (which throws away the model), we
  // carry a full 6-state estimate x̂ propagated through the plant's own ZOH
  // dynamics and corrected by a diagonal gain L on the measurement
  // innovation:
  //     x̂[k+1|k]   = Ad·x̂[k|k] + Bd·u[k] + Ed·v_g
  //     x̂[k+1|k+1] = x̂[k+1|k] + L · (y[k+1] − x̂[k+1|k])
  //  L tuned by hand — small on filter-cap voltages (slow, model-trustworthy),
  //  larger on currents (where the load step really does kick them).  This
  //  is what any textbook MPC actually feeds: state estimates, not raw sensor
  //  values.  The V_dc loop is non-linear (power balance), so it gets a
  //  separate 1-state IIR instead of joining the linear observer.
  private readonly L_gain = [0.22, 0.22, 0.10, 0.10, 0.25, 0.25];
  private readonly L_vdc = 0.15;
  private xHat = new Float64Array(6);
  private vDcHat: number = PARAMS.V_dc_nom;
  private obsPrimed = false;

  // -------- disturbance toggles --------
  public gridHarmonicActive = false;
  // IMP state augmentation: when on, the MPC runs with an 8-state model
  // that includes a 2-state damped rotor at 6ω driven by the tracking
  // error; cost Q_ξ > 0 on the rotor states then forces the MPC to plan
  // control that drives ξ → 0 → zero steady-state error at 6ω (Francis–
  // Wonham internal-model principle).  A/B-compares plain-MPC feedforward
  // against a truly augmented-state MPC so the viewer can see the 6ω
  // residual collapse when IMP is on.
  public harmonicRejectOn = false;
  /** Whether IMP is *effectively* augmenting the MPC this tick.  Follows
   *  harmonicRejectOn when the inner rails aren't binding, but is latched
   *  OFF whenever the previous solve reported i_g_max / i_f_max / u_box
   *  active — the augmented-state MPC's prediction model has no usable
   *  degree of freedom against an unreachable reference, and trying to
   *  optimise ξ in that regime destabilises the rail-bound operating
   *  point.  When rails release, we flip back to the augmented controller
   *  and rejection resumes on the next tick. */
  private impEffective = false;
  /** Rolling count of consecutive ticks the rail constraints have been
   *  inactive — used to hysterise the swap back from 6-state to 8-state so
   *  brief rail taps don't flutter the MPC rebuild. */
  private railClearTicks = 0;
  public loadStepOn = false;
  public noiseOn = false;
  private noiseSigma_i = 0;       // A   per-axis current noise
  private noiseSigma_v = 0;       // V   DC voltage noise
  // Scaled against the 65 A grid-side rail: 0.15 A ≈ 0.2% of rail, mild
  // baseline sensor dither.  User noise toggle adds 2.5 A (≈ 4% of rail)
  // which is the realistic envelope for shunt sensors at this class.
  private readonly baseNoiseSigma_i = 0.15;
  private readonly baseNoiseSigma_v = 0.5;

  // -------- diagnostics --------
  public solveUs = 0;
  public solveUsEma = 0;
  public solveUsBatch = 0;
  public iters = 0;
  public itersMax = 1;
  private itersWindow = new Int16Array(60);
  private itersWinHead = 0;
  private itersWinFilled = 0;
  public active: ActiveConstraint[] = [];
  public activeCount = 0;
  public jTrack = 0;
  public jEffort = 0;
  public lastUSeq: Float64Array | null = null;
  public lastXPred: Float64Array | null = null;
  public lastRefPred: Float64Array | null = null;
  public clipFlashT = 0;          // sample-counts-remaining of red flash for rail violation
  public uBound = (PARAMS.V_dc_nom / Math.sqrt(3)) * PARAMS.U_MAX_FRAC;

  // -------- prediction reference buffers --------
  private refFuture: Float64Array;

  constructor(horizon = 10) {
    this.horizon = horizon;
    this.mpc = new MPCController(horizon);
    this.pi = new PIController();
    this.outerPI = new VdcOuterPI();
    this.plant = new LCLPlant(PARAMS.T_plant);
    // Nominal DC load — 30 A so the demo opens with a mid-envelope charging
    // session (~22 kW) already in progress.  The EV-demand slider sweeps
    // above and below this, letting the viewer push into rail-binding and
    // drop back to idle without having to re-seed from zero.
    this.plant.i_load = 30;
    this.refFuture = new Float64Array(2 * Math.max(horizon, 20));
  }

  // -------- top-level UI hooks --------

  setHorizon(N: number): void {
    if (N === this.horizon) return;
    this.horizon = N;
    // Preserve the augmentation flag across horizon rebuilds so a user who
    // turned IMP on doesn't silently lose it when they drag the horizon slider.
    this.mpc = new MPCController(N, this.harmonicRejectOn);
    if (this.refFuture.length < 2 * N) this.refFuture = new Float64Array(2 * N);
  }

  setMode(m: ControllerMode): void {
    if (m === this.mode) return;
    this.mode = m;
    this.mpc.resetWarmStart();
    this.clipFlashT = 0;
    // Re-prime the observer on the next tick.  If PI just finished a wild
    // ride, x̂ is stale; latch to the true measurement rather than letting
    // the MPC plan from an observer that's chasing a diverged trajectory.
    this.obsPrimed = false;

    if (m === "pi") {
      // Bumpless transfer: seed the PI integrator so its first command
      // equals the last MPC command.  Without this the PI snaps from
      // ~326 V (u_d at steady state) to 0 V on switch and the LCL rings.
      this.pi.reset();
      this.pi.primeBumpless(
        this.u_d, this.u_q,
        this.i_gd_meas, this.i_gq_meas,
        this.iref_d, this.iref_q,
        V_GD, V_GQ,
      );
      // With load feedforward providing the bulk of P_ref, the outer PI
      // is only a trim loop — zero its integrator so we start with no
      // residual from the MPC-mode history.
      this.outerPI.reset();
    } else {
      this.pi.reset();
      this.outerPI.reset();
      // Neutralise u_prev so the slew-rate constraint doesn't pin the
      // first MPC solve to a PI-commanded overshoot.
      this.u_d = V_GD;
      this.u_q = 0;
    }
  }

  /** User-held grid 5th-harmonic injection.  While on, overlays a 6× ripple
   *  on (v_gd, v_gq) representing the dominant distortion on polluted LV
   *  grids.  Held indefinitely so the operator can compare the steady-state
   *  grid-current ripple with the harmonic on vs. off; visible i_g deviation
   *  shows the MPC's v_g feedforward + model-based prediction at work. */
  toggleGridHarmonic(): void {
    this.gridHarmonicActive = !this.gridHarmonicActive;
  }

  /** Toggle the IMP state-augmentation.  When ON the MPC is rebuilt against
   *  an 8-state model that embeds a damped 2-state rotor at 6ω driven by
   *  the grid-current tracking error; the rotor states ξ are carried in
   *  the QP's prediction, constraints, and cost, so the MPC plans u so as
   *  to drive ξ → 0 (which = zero sustained 6ω error by the internal-
   *  model principle).  When OFF the MPC falls back to its 6-state plant
   *  model.  ξ is zeroed on every toggle so the first post-toggle solve
   *  doesn't inherit stale rotor content that belongs to the previous
   *  operating point. */
  toggleHarmonicReject(): void {
    this.harmonicRejectOn = !this.harmonicRejectOn;
    this.xi_d = 0;
    this.xi_q = 0;
    this.xi_ref_primed = false;
    this.railClearTicks = 0;
    if (!this.harmonicRejectOn) {
      // Turning IMP off — always revert the MPC to its 6-state form.
      this.mpc.setHarmonicReject(false);
      this.impEffective = false;
    } else {
      // Turning IMP on — arm the augmentation only if rails aren't
      // currently binding.  If they are, the step()-time gate will
      // flip it on when they release.
      const railActive = this.active.includes("i_g_max")
        || this.active.includes("i_f_max")
        || this.active.includes("u_box");
      if (!railActive) {
        this.mpc.setHarmonicReject(true);
        this.impEffective = true;
      } else {
        this.mpc.setHarmonicReject(false);
        this.impEffective = false;
      }
    }
  }

  toggleLoadStep(): void {
    this.loadStepOn = !this.loadStepOn;
    // Step the DC-side load.  Nominal V_dc·i_load ≈ 15 kW (20 A); the step
    // triples it to ~45 kW (45 A), which is enough to make i_g approach
    // I_G_MAX and force the rails to bind before the outer loop catches up.
    this.plant.i_load = this.loadStepOn ? 45 : 20;
  }

  /** Set the EV demand directly — used by the charge-demand slider so the
   *  user can sweep through the full load envelope and watch the outer V_dc
   *  PI + inner MPC re-arbitrate in real time.  Clamped to the physically
   *  sensible range. */
  setLoadCurrent(i: number): void {
    this.plant.i_load = Math.max(0, Math.min(60, i));
    // Collapse the old toggle state so the button isn't showing "active"
    // when the slider is driving the load.
    this.loadStepOn = false;
  }

  toggleNoise(): void {
    this.noiseOn = !this.noiseOn;
    this.noiseSigma_i = this.noiseOn ? 2.5 : 0;
    this.noiseSigma_v = this.noiseOn ? 6 : 0;
  }

  reportSolveUs(us: number): void {
    if (us <= 0) return;
    this.solveUsBatch = us;
    this.solveUsEma = this.solveUsEma === 0 ? us : this.solveUsEma * 0.85 + us * 0.15;
  }

  reset(): void {
    this.plant.reset();
    this.plant.i_load = 30;
    this.V_dc = PARAMS.V_dc_nom;
    this.i_gd_meas = 0;
    this.i_gq_meas = 0;
    this.i_fd_meas = 0;
    this.i_fq_meas = 0;
    this.theta = 0;
    this.iref_d = 0;
    this.iref_q = 0;
    this.P_ref = 0;
    this.u_d = V_GD;
    this.u_q = 0;
    this.t = 0;
    this.gridHarmonicActive = false;
    this.harmonicRejectOn = false;
    this.loadStepOn = false;
    this.noiseOn = false;
    this.noiseSigma_i = 0;
    this.noiseSigma_v = 0;
    this.i_a_buf.fill(0);
    this.i_b_buf.fill(0);
    this.i_c_buf.fill(0);
    this.i_a_ref_buf.fill(0);
    this.v_dc_buf.fill(PARAMS.V_dc_nom);
    this.v_dc_ref_buf.fill(PARAMS.V_dc_nom);
    this.i_gd_buf.fill(0);
    this.i_gd_ref_buf.fill(0);
    this.head = 0;
    this.filled = 0;
    this.mpc.resetWarmStart();
    this.pi.reset();
    this.outerPI.reset();
    this.xi_d = 0;
    this.xi_q = 0;
    this.xi_ref_primed = false;
    this.impEffective = false;
    this.railClearTicks = 0;
    this.mpc.setHarmonicReject(false);
    this.solveUs = 0;
    this.solveUsEma = 0;
    this.solveUsBatch = 0;
    this.iters = 0;
    this.itersMax = 1;
    this.itersWindow.fill(0);
    this.itersWinHead = 0;
    this.itersWinFilled = 0;
    this.active = [];
    this.activeCount = 0;
    this.jTrack = 0;
    this.jEffort = 0;
    this.lastUSeq = null;
    this.lastXPred = null;
    this.lastRefPred = null;
    this.clipFlashT = 0;
    this.uBound = (PARAMS.V_dc_nom / Math.sqrt(3)) * PARAMS.U_MAX_FRAC;
    this.xHat.fill(0);
    this.vDcHat = PARAMS.V_dc_nom;
    this.obsPrimed = false;
  }

  // -------- main step (1 MPC tick = T_s seconds of sim) --------

  step(): void {
    const N = this.horizon;

    // ---- 0. Grid voltage seen this tick ----
    // Computed once and fed through observer predict, MPC solve, PI step,
    // and plant integration.  When the harmonic toggle is on, overlay the
    // 5th-harmonic dq components — this is a pure voltage disturbance on the
    // plant, while the MPC only feeds forward the *current* sample (predictions
    // over the horizon still assume v_g constant), so the controller visibly
    // attenuates but does not perfectly cancel the ripple.  The IMP toggle
    // lifts that limit by wrapping a PR companion (step 3b below) around the
    // loop — infinite gain at 6ω → asymptotic rejection.
    let vGdNow = V_GD;
    let vGqNow = V_GQ;
    if (this.gridHarmonicActive) {
      // 5th-harmonic (negative sequence) → 6× backward ripple in dq.
      const phi = (HARMONIC_ORDER + 1) * OMEGA * (this.t * PARAMS.T_s);
      const V_5 = V_GD * HARMONIC_AMP_FRAC;
      vGdNow += V_5 * Math.cos(phi);
      vGqNow += -V_5 * Math.sin(phi);
    }

    // ---- 1. Outer loop ----
    // In both modes the outer V_dc PI runs.  In PI mode we ALSO add a
    // DC-side load feedforward — without it the outer has to integrate
    // every load step into a new P_ref, which a low-BW PI cascade simply
    // cannot do in time.  Symptom without FF: user drops EV demand and
    // i_gd* drifts the *wrong* way while the outer integrator unwinds,
    // because the inner PI is still being commanded the pre-change
    // reference.  MPC doesn't need this because its prediction + state
    // feedback accomplish the same decoupling.
    const P_trim = this.outerPI.step(this.V_dc, PARAMS.V_DC_REF, PARAMS.T_s);
    const P_ff = this.mode === "pi"
      ? PARAMS.V_DC_REF * this.plant.i_load
      : 0;
    this.P_ref = P_ff + P_trim;
    // Convert P_ref → i_gd*  via amplitude-invariant Park: P = (3/2)·v_gd·i_gd
    // (we set i_gq* = 0 for unity power factor).  Use *nominal* V_GD here
    // even during a sag, otherwise the reference would chase the disturbance
    // and mask the transient we're trying to show.
    let igd_ref = (2 / 3) * this.P_ref / V_GD;
    // Clamp to a generous outer envelope so the outer loop can't ask for the
    // impossible.  The MPC will still enforce the *true* hard rail.
    if (igd_ref > PARAMS.I_G_MAX * 1.6) igd_ref = PARAMS.I_G_MAX * 1.6;
    else if (igd_ref < -PARAMS.I_G_MAX * 1.6) igd_ref = -PARAMS.I_G_MAX * 1.6;
    this.iref_d = igd_ref;
    this.iref_q = 0;

    // ---- 2. Measurement (with noise if on) ----
    const sigmaI = this.baseNoiseSigma_i + this.noiseSigma_i;
    const sigmaV = this.baseNoiseSigma_v + this.noiseSigma_v;
    const xMeas = new Float64Array(6);
    for (let i = 0; i < 6; i++) {
      xMeas[i] = this.plant.x[i] + (sigmaI > 0 ? gaussian(sigmaI) : 0);
    }
    const V_dc_meas = this.plant.V_dc + (sigmaV > 0 ? gaussian(sigmaV) : 0);

    // ---- 2b. State observer (Luenberger) ----
    // Prime on the first tick so we don't drag a 6-state ringdown from zero.
    if (!this.obsPrimed) {
      for (let i = 0; i < 6; i++) this.xHat[i] = xMeas[i];
      this.vDcHat = V_dc_meas;
      this.obsPrimed = true;
    } else {
      // Predict: x̂_pred = Ad·x̂ + Bd·u_prev + Ed·v_g
      const Ad = this.mpc.Ad;
      const Bd = this.mpc.Bd;
      const Ed = this.mpc.Ed;
      const xp = new Float64Array(6);
      for (let i = 0; i < 6; i++) {
        const Ai = Ad[i];
        let s =
          Ai[0] * this.xHat[0] + Ai[1] * this.xHat[1] +
          Ai[2] * this.xHat[2] + Ai[3] * this.xHat[3] +
          Ai[4] * this.xHat[4] + Ai[5] * this.xHat[5];
        s += Bd[i][0] * this.u_d + Bd[i][1] * this.u_q;
        s += Ed[i][0] * vGdNow + Ed[i][1] * vGqNow;
        xp[i] = s;
      }
      // Correct with innovation
      for (let i = 0; i < 6; i++) {
        this.xHat[i] = xp[i] + this.L_gain[i] * (xMeas[i] - xp[i]);
      }
      this.vDcHat = this.L_vdc * V_dc_meas + (1 - this.L_vdc) * this.vDcHat;
    }
    // Expose the filtered V_dc for HUD + outer loop; MPC consumes the full
    // observer state.  PI (baseline) intentionally does NOT read the
    // observer — it gets raw noisy measurements so the recruiter sees what
    // "just pipe the sensor into a PI" actually looks like.
    this.V_dc = this.vDcHat;
    const xCtrl = this.xHat;
    const V_dc_ctrl = this.vDcHat;

    // ---- 3. Inner controller ----
    let ud: number, uq: number;
    if (this.mode === "mpc") {
      // Rail-bound gating for IMP augmentation.  The augmented-state MPC
      // plans against an internal model that integrates (y − r) into ξ,
      // and when a rail constraint binds there's no feasible u that
      // drives ξ to zero — the QP picks a pathological trajectory that
      // destabilises the rail-bound operating point.  So: when the last
      // solve reported an active state or input rail, drop back to the
      // plain 6-state MPC for a few ticks.  Matrix rebuild only fires on
      // the transition edge, so this doesn't tax per-tick compute.
      if (this.harmonicRejectOn) {
        // Gate IMP off when the grid-current rail is active — that
        // signals an unreachable outer-loop demand (e.g. 55 A EV peak
        // driving i_gd*→100 A vs a 65 A rail) and the augmented IMP
        // can't achieve rejection against an impossible target.  We do
        // NOT gate on u_box here: the MPC intermittently taps the
        // SVPWM circle while shaping its voltage command against the
        // 300 Hz grid ripple — that's a SIGN OF WORKING rejection, not
        // a failure mode.  The fast-ramp transient (ξ winding up on a
        // stale LPF'd reference) is handled separately by the ξ-LPF
        // snap-reset below when a large reference step is detected.
        const railActive = this.active.includes("i_g_max");
        if (railActive) {
          this.railClearTicks = 0;
          if (this.impEffective) {
            this.mpc.setHarmonicReject(false);
            this.impEffective = false;
            this.xi_d = 0;
            this.xi_q = 0;
          }
        } else {
          // Require a short quiet window before re-arming IMP so a brief
          // rail tap doesn't thrash the swap.
          if (++this.railClearTicks > 20 && !this.impEffective) {
            this.mpc.setHarmonicReject(true);
            this.impEffective = true;
            this.xi_d = 0;
            this.xi_q = 0;
          }
        }
      }
      // Build reference sequence — assume constant over horizon.
      for (let k = 0; k < N; k++) {
        this.refFuture[k * 2] = igd_ref;
        this.refFuture[k * 2 + 1] = 0;
      }
      const uPrev = new Float64Array([this.u_d, this.u_q]);
      // Build augmented x0 — 6-state plant estimate with ξ appended when
      // IMP is effective this tick.  Matches MPCController.nx; the MPC's
      // prediction then correctly propagates ξ using (y − r) over the
      // horizon.
      const nxAug = this.mpc.nx;
      const x0Aug = new Float64Array(nxAug);
      for (let i = 0; i < 6; i++) x0Aug[i] = xCtrl[i];
      let refSeqXi: Float64Array | undefined;
      if (nxAug === 8) {
        // CRITICAL — the Luenberger observer has an ~180 Hz effective
        // corner frequency, so xHat[4,5] arrives at the MPC with the
        // 300 Hz grid-current ripple already attenuated by ≥5 dB.  Feeding
        // that into the augmented prediction would hide the disturbance
        // the IMP is supposed to reject.  So when IMP is active we
        // overwrite the two grid-current slots with the raw measurement,
        // preserving the ripple content the augmentation needs.  The
        // other four plant states stay on the observer (that's where the
        // noise-rejection benefit lives); only the directly-measured
        // grid currents bypass it.
        x0Aug[4] = xMeas[4];
        x0Aug[5] = xMeas[5];
        x0Aug[6] = this.xi_d;
        x0Aug[7] = this.xi_q;
        // Separate ξ-driving reference — the LPF'd d-axis setpoint so the
        // MPC's rotor prediction matches the engine's rotor propagation.
        refSeqXi = new Float64Array(2 * N);
        const r_d = this.xi_ref_primed ? this.xi_ref_d_lp : igd_ref;
        const r_q = this.xi_ref_primed ? this.xi_ref_q_lp : 0;
        for (let k = 0; k < N; k++) {
          refSeqXi[k * 2] = r_d;
          refSeqXi[k * 2 + 1] = r_q;
        }
      }
      const sol = this.mpc.solve(x0Aug, uPrev, this.refFuture, V_dc_ctrl, [vGdNow, vGqNow], refSeqXi);
      ud = sol.u_d;
      uq = sol.u_q;

      this.solveUs = sol.solveUs;
      if (sol.solveUs > 0) {
        this.solveUsEma = this.solveUsEma === 0 ? sol.solveUs : this.solveUsEma * 0.9 + sol.solveUs * 0.1;
      }
      this.iters = sol.iters;
      this.itersWindow[this.itersWinHead] = sol.iters;
      this.itersWinHead = (this.itersWinHead + 1) % this.itersWindow.length;
      if (this.itersWinFilled < this.itersWindow.length) this.itersWinFilled++;
      let mx = 1;
      for (let i = 0; i < this.itersWinFilled; i++) {
        if (this.itersWindow[i] > mx) mx = this.itersWindow[i];
      }
      this.itersMax = mx;
      this.active = sol.active;
      this.activeCount = sol.activeCount;
      this.jTrack = sol.jTrack;
      this.jEffort = sol.jEffort;
      this.lastUSeq = sol.uSeq;
      this.lastXPred = sol.xPred;
      const rp = new Float64Array(2 * N);
      for (let k = 0; k < 2 * N; k++) rp[k] = this.refFuture[k];
      this.lastRefPred = rp;
      this.uBound = sol.uBound;
    } else {
      // PI baseline: per-axis on grid current.  Uses the RAW measurement
      // (no observer) and grid-voltage feed-forward (standard practice);
      // PI internally clips to the SVPWM circle bound and uses back-calc
      // anti-windup so the integrator doesn't detonate on unreachable
      // references.  Still NO dq decoupling and NO constraint-aware
      // preview — the two levers the MPC actually exploits.
      const uLim = (V_dc_meas / Math.sqrt(3)) * PARAMS.U_MAX_FRAC;
      const i_cfd_meas = xMeas[0] - xMeas[4];
      const i_cfq_meas = xMeas[1] - xMeas[5];
      [ud, uq] = this.pi.step(
        xMeas[4], xMeas[5],
        igd_ref, 0,
        vGdNow, vGqNow,
        i_cfd_meas, i_cfq_meas,
        uLim, PARAMS.T_s,
      );
      this.solveUs = 0;
      this.iters = 0;
      this.active = [];
      this.activeCount = 0;
      this.jTrack = 0;
      this.jEffort = 0;
      this.lastUSeq = null;
      this.lastXPred = null;
      this.lastRefPred = null;
      this.uBound = uLim;
    }

    // ---- 3b. IMP rotor-state propagation ----
    // No more post-MPC correction: when IMP is on, the augmentation lives
    // inside the MPC's state space and the QP's u is already the IMP-
    // aware command.  We just have to advance ξ one tick forward using
    // the EXACT same discrete dynamics the MPC's prediction assumes:
    //   ξ[k+1] = A_r_d · ξ[k] + B_r_d · (y[k] − r[k])
    // where y[k] / r[k] are the current-tick grid-current / reference
    // values fed into the solve (xCtrl[4,5] and igd_ref / 0).  Engine-side
    // consistency is critical: if the engine propagates ξ differently
    // than the MPC predicts, the controller would plan against a rotor
    // state that never materialises, and rejection would decay.
    //
    // Anti-windup: when the MPC reports an active i_g_max rail (55 A EV
    // peak drives i_gd* above the 65 A rail so the MPC clamps i_gd), the
    // error (y−r) picks up an unrejectable DC offset.  Propagating ξ on
    // that would drive it into a positive-feedback blowup where the MPC
    // tries harder and harder against an impossible reference.  We
    // therefore propagate only the homogeneous part A_r_d·ξ (damping bleed)
    // while a rail is binding — standard resonant-controller back-calc.
    if (this.impEffective) {
      // ξ[k+1] = A_r_d·ξ[k] + B_r_d·(y − LPF(r_xi)), matching the MPC's
      // internal augmented-state model.  LPF(r) strips the 6ω ripple the
      // outer V_dc PI leaks into i_gd_ref so the rotor sees only the
      // grid-side 6ω residual, not its own reference chasing the same
      // disturbance.  r_xi also clamps unreachable demands.
      const Ar = this.rotor.Ar_d;
      const Br = this.rotor.Br_d;
      let xi0 = this.xi_d;
      let xi1 = this.xi_q;
      const refCap = PARAMS.I_G_MAX * 0.95;
      const igd_ref_xi = Math.max(-refCap, Math.min(refCap, igd_ref));
      // LPF alpha: α = T_s / (τ + T_s), τ = 1/(2π·f_c).
      const lpAlpha = PARAMS.T_s / (1 / (2 * Math.PI * RESONATOR_REF_LPF_HZ) + PARAMS.T_s);
      if (!this.xi_ref_primed) {
        this.xi_ref_d_lp = igd_ref_xi;
        this.xi_ref_q_lp = 0;
        this.xi_ref_primed = true;
      } else if (Math.abs(igd_ref_xi - this.xi_ref_d_lp) > 5) {
        // Large reference jump detected — snap the LPF to the new setpoint
        // and zero the rotor state.  A 15 Hz LPF takes ~70 ms to track a
        // step, which during a fast slider move (20→55 A) leaves a 30+ A
        // apparent "error" feeding the rotor, blowing up ξ into cost that
        // the rail-bound QP can't relieve.  Snap-and-reset kills that
        // transient so IMP resumes from a clean operating point once
        // tracking settles.
        this.xi_ref_d_lp = igd_ref_xi;
        this.xi_ref_q_lp = 0;
        this.xi_d = 0;
        this.xi_q = 0;
        xi0 = 0;
        xi1 = 0;
      } else {
        this.xi_ref_d_lp += lpAlpha * (igd_ref_xi - this.xi_ref_d_lp);
        this.xi_ref_q_lp += lpAlpha * (0 - this.xi_ref_q_lp);
      }
      // Raw measurement (not xHat!) so the rotor sees the full 300 Hz
      // ripple — same reasoning as the x0Aug override above.
      const ed = xMeas[4] - this.xi_ref_d_lp;
      const eq = xMeas[5] - this.xi_ref_q_lp;
      this.xi_d = Ar[0][0] * xi0 + Ar[0][1] * xi1 + Br[0][0] * ed + Br[0][1] * eq;
      this.xi_q = Ar[1][0] * xi0 + Ar[1][1] * xi1 + Br[1][0] * ed + Br[1][1] * eq;
    } else if (this.xi_d !== 0 || this.xi_q !== 0) {
      this.xi_d = 0;
      this.xi_q = 0;
      this.xi_ref_primed = false;
    }

    this.u_d = ud;
    this.u_q = uq;

    // ---- 4. Plant integration: T_s of sim, in T_plant sub-steps ----
    const subSteps = Math.max(1, Math.round(PARAMS.T_s / PARAMS.T_plant));
    for (let s = 0; s < subSteps; s++) this.plant.step(ud, uq, vGdNow, vGqNow);

    // Copy back true state for next-tick measurement.
    this.i_fd_meas = this.plant.x[0];
    this.i_fq_meas = this.plant.x[1];
    this.i_gd_meas = this.plant.x[4];
    this.i_gq_meas = this.plant.x[5];
    this.theta = this.plant.theta;

    // Rail violation flash (any current axis touching its rail).
    const railHit =
      Math.abs(this.i_fd_meas) > PARAMS.I_F_MAX * 0.99 ||
      Math.abs(this.i_fq_meas) > PARAMS.I_F_MAX * 0.99 ||
      Math.abs(this.i_gd_meas) > PARAMS.I_G_MAX * 0.99 ||
      Math.abs(this.i_gq_meas) > PARAMS.I_G_MAX * 0.99;
    if (railHit) this.clipFlashT = 24;
    else if (this.clipFlashT > 0) this.clipFlashT--;

    // ---- 5. Buffers — stash phase currents, V_dc, and the abc reference ----
    // IMPORTANT: plot the NOISY sensor signal, not the true plant state.
    // Otherwise the "Inject noise" toggle does nothing visible, and the
    // viewer can't see what the controller is actually fighting.  The
    // observer-filtered value (i_gd_meas) is what the MPC consumes
    // internally; the plot tells the story of the raw measurement.
    const sigmaIvis = this.baseNoiseSigma_i + this.noiseSigma_i;
    const igd_vis = this.plant.x[4] + (sigmaIvis > 0 ? gaussian(sigmaIvis) : 0);
    const igq_vis = this.plant.x[5] + (sigmaIvis > 0 ? gaussian(sigmaIvis) : 0);
    const [i_a, i_b, i_c] = dqToAbc(igd_vis, igq_vis, this.theta);
    const [i_a_ref] = dqToAbc(igd_ref, 0, this.theta);

    const sigmaVvis = this.baseNoiseSigma_v + this.noiseSigma_v;
    const vdc_vis = this.plant.V_dc + (sigmaVvis > 0 ? gaussian(sigmaVvis) : 0);

    this.i_a_buf[this.head] = i_a;
    this.i_b_buf[this.head] = i_b;
    this.i_c_buf[this.head] = i_c;
    this.i_a_ref_buf[this.head] = i_a_ref;
    this.v_dc_buf[this.head] = vdc_vis;
    this.v_dc_ref_buf[this.head] = PARAMS.V_DC_REF;
    this.i_gd_buf[this.head] = igd_vis;
    this.i_gd_ref_buf[this.head] = igd_ref;
    this.head = (this.head + 1) % this.N_HISTORY;
    if (this.filled < this.N_HISTORY) this.filled++;

    this.t += 1;
  }
}

function gaussian(sigma: number): number {
  const u1 = Math.max(Math.random(), 1e-9);
  const u2 = Math.random();
  return sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

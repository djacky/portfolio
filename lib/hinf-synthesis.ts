/* ------------------------------------------------------------------
   H∞ data-driven RST controller synthesis — port of pyfresco's
   `rst_init` (obcd/OptAlgoIB.py). The feasibility SOCP at each γ is
   solved by SCS (Splitting Conic Solver, WASM build); the outer loop
   bisects on γ to find the smallest H∞ norm that keeps every
   frequency-wise SOCP feasible.

   Fixed structure for v1:
     • Controller order 5 (n_r = n_s = n_t = 6)
     • 1 integrator factor (S has (1 - z⁻¹))
     • ρ = [r_0..r_{N_R-1}, s_0..s_{N_S_PRE-1}, t_0..t_{N_T-1}]
   ------------------------------------------------------------------ */

export type Complex = { re: number; im: number };

export const c = (re: number, im = 0): Complex => ({ re, im });
export const cMul = (a: Complex, b: Complex): Complex => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});
export const cAdd = (a: Complex, b: Complex): Complex => ({
  re: a.re + b.re,
  im: a.im + b.im,
});
export const cSub = (a: Complex, b: Complex): Complex => ({
  re: a.re - b.re,
  im: a.im - b.im,
});
export const cScale = (a: Complex, s: number): Complex => ({
  re: a.re * s,
  im: a.im * s,
});
export const cAbs = (a: Complex): number => Math.hypot(a.re, a.im);
export const cExp = (theta: number): Complex => ({
  re: Math.cos(theta),
  im: Math.sin(theta),
});

// Problem dimensions.  These are parameterised by the user-selected
// controller *order* (polynomial degree): N_R = N_S = N_T = order + 1
// coefficients each.  The setter below keeps all derived sizes in sync;
// synthesizeController / synthesizeAtGamma call it on entry.  Default
// order is 5 (six coefficients per polynomial).
export const N_INT = 1;
export const DEFAULT_ORDER = 5;

export let N_R = DEFAULT_ORDER + 1;
export let N_S = DEFAULT_ORDER + 1;
export let N_T = DEFAULT_ORDER + 1;
export let N_S_PRE = N_S - 1 - N_INT; // So_pre coefficients
export let N_RHO = N_R + N_S_PRE + N_T;
export const OFFSET_R = 0;
export let OFFSET_S = N_R;
export let OFFSET_T = N_R + N_S_PRE;
// Largest z⁻ⁱ index touched by any block: r-block uses 0..N_R−1,
// s-block uses 1..N_S_PRE, t-block uses 0..N_T−1.
export let Z_LEN = Math.max(N_R, N_T, N_S_PRE + 1);

export function setControllerOrder(order: number) {
  const ord = Math.max(1, Math.floor(order));
  const n = ord + 1; // number of coefficients
  N_R = n;
  N_S = n;
  N_T = n;
  N_S_PRE = N_S - 1 - N_INT;
  N_RHO = N_R + N_S_PRE + N_T;
  OFFSET_S = N_R;
  OFFSET_T = N_R + N_S_PRE;
  Z_LEN = Math.max(N_R, N_T, N_S_PRE + 1);
}

// pyfresco constants.
const BIS_TOL = 1e-3;
const LMI_TOL = 1e-3;
const G_MAX = 10;
const G_MIN = 0.1;
const CONSTRAINT_SLACK = 1e-6; // pyfresco uses F1 <= -1e-6, F2 <= -1e-6
const SNI_LOWER = 5e-3;        // Re(1 + So_pre) >= 5e-3

/* ------------------------------------------------------------------
   SCS settings override — lets sweep scripts patch the solver knobs
   (eps_*, scale, alpha, etc.) without forking the production path.
   Production runs leave this at null and use the curated defaults
   inside innerSolveSCS.
   ------------------------------------------------------------------ */
export type ScsSettingsOverride = Partial<{
  epsAbs: number;
  epsRel: number;
  epsInfeas: number;
  maxIters: number;
  timeLimitSecs: number;
  scale: number;
  adaptiveScale: boolean;
  normalize: boolean;
  alpha: number;
  rhoX: number;
}>;

let scsSettingsOverride: ScsSettingsOverride | null = null;
export function setScsSettingsOverride(o: ScsSettingsOverride | null) {
  scsSettingsOverride = o;
}

export interface SynthesisSpecs {
  desMm: number;      // modulus margin ∈ (0, 1)
  desBw: number;      // closed-loop bandwidth in Hz
  desZeta: number;    // damping ratio for desired 2nd-order CL transfer
  Ts: number;         // sampling period (s)
  order?: number;     // controller order (polynomial degree); defaults to DEFAULT_ORDER
}

export interface FrequencyGrid {
  w: number[];       // rad/s
  G: Complex[];      // plant FRF (used for closed-loop evaluation)
  MA: Complex[];     // pre-filter FRF (typically all 1)
  // Coprime factorization G = N / M, both stable.  For stable plants
  // the factorization is trivial (N = G, M = 1).  For unstable or
  // integrator-containing plants we pick F(s) = (s+a)^n with a chosen
  // at the desired closed-loop bandwidth and form N = B/F, M = A/F so
  // the H∞ SOCP acts on a stable, proper problem.  See Karimi & Kammer
  // (Automatica 2017) "Data-driven design of robust controllers for
  // unstable systems".
  N: Complex[];
  M: Complex[];
}

export interface SynthesisProgress {
  iter: number;
  gamma: number;
  bw: number | null;
  feasible: boolean;
  innerLoss: number;
}

export interface SynthesisResult {
  feasible: boolean;
  gammaOpt: number;       // H∞-style norm = 1/γ_pyfresco
  gammaPyfresco: number;  // the γ from the pyfresco formulation (large = good)
  rhoR: number[];
  rhoS: number[];         // So_pre, before integrator
  rhoT: number[];         // raw (un-normalized) T coefficients
  RFull: number[];        // R(z⁻¹) coefficients
  SFull: number[];        // S(z⁻¹) with (1−z⁻¹) integrator factored in
  TFull: number[];        // T(z⁻¹) coefficients, normalized by GAIN so CL(DC)=1
  gain: number;           // ΣT_raw / ΣR — applied as T := T/GAIN (pyfresco rst_ilc_obj)
  achievedBw: number;     // −3 dB of |G·T/(G·R + S)| (Hz); 0 if no crossing
  iterations: SynthesisProgress[];
  infeasibilityHint?: string;
}

// Desired 2nd-order sensitivity weighting Wd = 1/(1 - Td).  pyfresco
// also multiplies Td by exp(-j·ω·d_r); per user directive we neglect
// the reference-delay term.  Kept uncapped to match pyfresco — SCS's
// equilibration handles the DC-side magnitude on its own.
function desiredWeighting(
  w: number[],
  bwHz: number,
  zeta: number,
): { Td: Complex[]; Wd: Complex[] } {
  const wd =
    (2 * Math.PI * bwHz) /
    Math.sqrt(
      1 -
        2 * zeta * zeta +
        Math.sqrt(2 - 4 * zeta * zeta + 4 * Math.pow(zeta, 4)),
    );
  const Td: Complex[] = new Array(w.length);
  const Wd: Complex[] = new Array(w.length);
  for (let k = 0; k < w.length; k++) {
    const wk = w[k];
    // Td = wd² / (wd² − ω² + j·2ζ·wd·ω)
    const denomRe = wd * wd - wk * wk;
    const denomIm = 2 * zeta * wd * wk;
    const num = wd * wd;
    const mag2 = denomRe * denomRe + denomIm * denomIm;
    Td[k] = { re: (num * denomRe) / mag2, im: (-num * denomIm) / mag2 };
    // Wd = 1/(1 − Td)
    const one_minus_Td = { re: 1 - Td[k].re, im: -Td[k].im };
    const m2 = one_minus_Td.re * one_minus_Td.re + one_minus_Td.im * one_minus_Td.im;
    Wd[k] = { re: one_minus_Td.re / m2, im: -one_minus_Td.im / m2 };
  }
  return { Td, Wd };
}

/* ------------------------------------------------------------------
   Constraint-matrix precomputation.

   Every per-frequency expression is affine in ρ:
     PSI_k(ρ)     = c_PSI,k · ρ  + d_PSI,k
     u_k(ρ)       = c_u,k   · ρ  + d_u,k       (= Wd_k·(PSI_k − G_k·T_k))
     So_k(ρ)     = c_So,k  · ρ  + d_So,k      (= (1−z⁻¹)·(1 + So_pre))
     So_NI_k(ρ)  = c_SNI,k · ρ  + 1            (= 1 + So_pre)

   We store real and imag parts as separate Float64Arrays of length
   N_FREQ × N_RHO to keep inner-loop ops vectorised and cheap.
   ------------------------------------------------------------------ */

interface CoeffMats {
  nFreq: number;
  // Affine offsets (d_*).
  dPsiRe: Float64Array;
  dPsiIm: Float64Array;
  dURe: Float64Array;
  dUIm: Float64Array;
  dSoRe: Float64Array;
  dSoIm: Float64Array;
  // Linear coefficients — flat row-major [freq × rho].
  cPsiRe: Float64Array;
  cPsiIm: Float64Array;
  cURe: Float64Array;
  cUIm: Float64Array;
  cSoRe: Float64Array;
  cSoIm: Float64Array;
  cSniRe: Float64Array; // only Re needed (linear constraint on Re)
}

function precomputeCoeffs(
  grid: FrequencyGrid,
  Ts: number,
  Wd: Complex[],
): CoeffMats {
  const nFreq = grid.w.length;
  const dPsiRe = new Float64Array(nFreq);
  const dPsiIm = new Float64Array(nFreq);
  const dURe = new Float64Array(nFreq);
  const dUIm = new Float64Array(nFreq);
  const dSoRe = new Float64Array(nFreq);
  const dSoIm = new Float64Array(nFreq);
  const cPsiRe = new Float64Array(nFreq * N_RHO);
  const cPsiIm = new Float64Array(nFreq * N_RHO);
  const cURe = new Float64Array(nFreq * N_RHO);
  const cUIm = new Float64Array(nFreq * N_RHO);
  const cSoRe = new Float64Array(nFreq * N_RHO);
  const cSoIm = new Float64Array(nFreq * N_RHO);
  const cSniRe = new Float64Array(nFreq * N_RHO);

  for (let k = 0; k < nFreq; k++) {
    // Coprime factorization: G = N/M.  The SOCP operates on the
    // identity PSI = S·M + N·MA·R = M·(S + G·MA·R), which rescales the
    // original problem by |M| but keeps the closed-loop transfer
    // function unchanged.  For stable plants N = G and M = 1, and this
    // collapses back to pyfresco's rst_init formulation.
    const N = grid.N[k];
    const M = grid.M[k];
    const MA = grid.MA[k];
    const Wdk = Wd[k];
    const w = grid.w[k];

    // z_neg[i] = e^(−j·ω·Ts·i)
    const zNeg: Complex[] = new Array(Z_LEN);
    for (let i = 0; i < Z_LEN; i++) {
      zNeg[i] = cExp(-w * Ts * i);
    }
    const oneMinusZInv: Complex = cSub(c(1), zNeg[1]);
    const oneMinusZInvM: Complex = cMul(oneMinusZInv, M);
    const NMA: Complex = cMul(N, MA);
    const WdN: Complex = cMul(Wdk, N);

    // PSI constants: (1 − z⁻¹)·M   (S_full's leading "1" times M)
    dPsiRe[k] = oneMinusZInvM.re;
    dPsiIm[k] = oneMinusZInvM.im;
    // u constants: Wd·(1 − z⁻¹)·M
    const dUk = cMul(Wdk, oneMinusZInvM);
    dURe[k] = dUk.re;
    dUIm[k] = dUk.im;
    // So constants (used for |ΔM·S·M| modulus-margin): (1 − z⁻¹)·M
    dSoRe[k] = oneMinusZInvM.re;
    dSoIm[k] = oneMinusZInvM.im;

    const rowBase = k * N_RHO;

    // --- r-block: PSI += N·MA·z^(-i)·r_i, u += Wd·N·MA·z^(-i)·r_i
    for (let i = 0; i < N_R; i++) {
      const coef = cMul(NMA, zNeg[i]);
      cPsiRe[rowBase + OFFSET_R + i] = coef.re;
      cPsiIm[rowBase + OFFSET_R + i] = coef.im;
      const uCoef = cMul(Wdk, coef);
      cURe[rowBase + OFFSET_R + i] = uCoef.re;
      cUIm[rowBase + OFFSET_R + i] = uCoef.im;
      // So and So_NI are 0 on r-block
    }

    // --- s-block: PSI/So += (1-z⁻¹)·M·z^(-(i+1))·s_i,
    //              So_NI += z^(-(i+1))·s_i    (no M — it's S_pre, pre-integrator),
    //              u += Wd·(1-z⁻¹)·M·z^(-(i+1))·s_i
    for (let i = 0; i < N_S_PRE; i++) {
      const zi1 = zNeg[i + 1];
      const sCoef = cMul(oneMinusZInvM, zi1);
      cPsiRe[rowBase + OFFSET_S + i] = sCoef.re;
      cPsiIm[rowBase + OFFSET_S + i] = sCoef.im;
      cSoRe[rowBase + OFFSET_S + i] = sCoef.re;
      cSoIm[rowBase + OFFSET_S + i] = sCoef.im;
      cSniRe[rowBase + OFFSET_S + i] = zi1.re;
      const uCoef = cMul(Wdk, sCoef);
      cURe[rowBase + OFFSET_S + i] = uCoef.re;
      cUIm[rowBase + OFFSET_S + i] = uCoef.im;
    }

    // --- t-block: u += -Wd·N·z^(-i)·t_i
    for (let i = 0; i < N_T; i++) {
      const tCoef = cScale(cMul(WdN, zNeg[i]), -1);
      cURe[rowBase + OFFSET_T + i] = tCoef.re;
      cUIm[rowBase + OFFSET_T + i] = tCoef.im;
    }
  }

  return {
    nFreq,
    dPsiRe,
    dPsiIm,
    dURe,
    dUIm,
    dSoRe,
    dSoIm,
    cPsiRe,
    cPsiIm,
    cURe,
    cUIm,
    cSoRe,
    cSoIm,
    cSniRe,
  };
}

/* ------------------------------------------------------------------
   Inner solver — SCS conic program.

   Per-frequency structure — three inequality constraints each:
     v1_k  : γ·|u_k(ρ)|        ≤  Re(PSI_k(ρ)) − CONSTRAINT_SLACK   (SOC)
     v2_k  : des_mm·|So_k(ρ)|   ≤  Re(PSI_k(ρ)) − CONSTRAINT_SLACK   (SOC)
     v3_k  : SNI_LOWER           ≤  Re(1 + So_pre(ρ))                (linear)

   SCS standard form:   min ½ ρᵀ P ρ + cᵀ ρ   s.t.  A ρ + s = b,  s ∈ 𝒦.
   Pure feasibility → c = 0, P = εI (tiny regulariser just to pin ρ).

   Row layout in A / b (m = 7·N_freq):
     rows 0..N_freq−1           : v3_k as positive orthant
     rows N_freq + 6k  + 0..2   : v1_k as SOC of size 3
     rows N_freq + 6k  + 3..5   : v2_k as SOC of size 3
   ------------------------------------------------------------------ */

interface InnerResult {
  rho: Float64Array;
  residual: number;   // max individual violation (0 ↔ feasible)
  loss: number;       // unused with SCS; kept for API compatibility
  worstV1: number;
  worstV2: number;
  worstV3: number;
  status: string;
  // Full primal/dual/slack — saved to warm-start the next SOCP in the
  // bisection, where (A, b, cones) are nearly identical between iterates.
  warm?: { x: Float64Array; y: Float64Array; s: Float64Array };
}

// Lazily-loaded SCS module (WASM).  We memoise the promise so the
// WASM blob is compiled exactly once per process / page load.
type SCSModule = Awaited<ReturnType<typeof import("scs-solver")["default"]>>;
let scsPromise: Promise<SCSModule> | null = null;
export async function getSCS(): Promise<SCSModule> {
  if (!scsPromise) {
    const { default: createSCS } = await import("scs-solver");
    scsPromise = createSCS();
  }
  return scsPromise;
}

interface ScsProblem {
  data: {
    m: number;
    n: number;
    A_x: number[];
    A_i: number[];
    A_p: number[];
    P_x: number[];
    P_i: number[];
    P_p: number[];
    b: number[];
    c: number[];
  };
  cone: { l: number; q: number[]; qsize: number };
}

// Build the CSC sparse SOCP for a given (γ, des_mm).
function buildScsProblem(
  mats: CoeffMats,
  gamma: number,
  desMm: number,
): ScsProblem {
  const n = mats.nFreq;
  const m = 7 * n;

  const b = new Array<number>(m).fill(0);
  // v3_k rows: b = 1 − SNI_LOWER ; A row = −cSniRe[k,:]
  for (let k = 0; k < n; k++) {
    b[k] = 1 - SNI_LOWER;
  }
  // v1_k rows (SOC of 3), v2_k rows (SOC of 3)
  for (let k = 0; k < n; k++) {
    const rowV1 = n + 6 * k;
    const rowV2 = rowV1 + 3;
    // v1 SOC: s0 = Re(PSI) − δ, s1 = γ·Re(u), s2 = γ·Im(u)
    b[rowV1 + 0] = mats.dPsiRe[k] - CONSTRAINT_SLACK;
    b[rowV1 + 1] = gamma * mats.dURe[k];
    b[rowV1 + 2] = gamma * mats.dUIm[k];
    // v2 SOC: s0 = Re(PSI) − δ, s1 = desMm·Re(So), s2 = desMm·Im(So)
    b[rowV2 + 0] = mats.dPsiRe[k] - CONSTRAINT_SLACK;
    b[rowV2 + 1] = desMm * mats.dSoRe[k];
    b[rowV2 + 2] = desMm * mats.dSoIm[k];
  }

  // CSC: iterate columns (16 ρ components).  For each column we emit
  // (row, value) pairs in increasing row order.
  const A_x: number[] = [];
  const A_i: number[] = [];
  const A_p: number[] = [0];

  for (let j = 0; j < N_RHO; j++) {
    const inR = j >= OFFSET_R && j < OFFSET_R + N_R;
    const inS = j >= OFFSET_S && j < OFFSET_S + N_S_PRE;
    // j is in T block otherwise.

    // v3 rows (0 … n−1): non-zero only for s-block columns.
    if (inS) {
      for (let k = 0; k < n; k++) {
        const val = -mats.cSniRe[k * N_RHO + j];
        if (val !== 0) {
          A_x.push(val);
          A_i.push(k);
        }
      }
    }

    // v1 / v2 SOC rows.
    for (let k = 0; k < n; k++) {
      const base = k * N_RHO + j;
      const rowV1 = n + 6 * k;
      const rowV2 = rowV1 + 3;

      // v1 s0 : −cPsiRe (r+s blocks only)
      if (inR || inS) {
        A_x.push(-mats.cPsiRe[base]);
        A_i.push(rowV1 + 0);
      }
      // v1 s1 : −γ·cURe (all blocks)
      A_x.push(-gamma * mats.cURe[base]);
      A_i.push(rowV1 + 1);
      // v1 s2 : −γ·cUIm (all blocks)
      A_x.push(-gamma * mats.cUIm[base]);
      A_i.push(rowV1 + 2);
      // v2 s0 : −cPsiRe (r+s blocks only)
      if (inR || inS) {
        A_x.push(-mats.cPsiRe[base]);
        A_i.push(rowV2 + 0);
      }
      // v2 s1 : −desMm·cSoRe (s-block only)
      if (inS) {
        A_x.push(-desMm * mats.cSoRe[base]);
        A_i.push(rowV2 + 1);
      }
      // v2 s2 : −desMm·cSoIm (s-block only)
      if (inS) {
        A_x.push(-desMm * mats.cSoIm[base]);
        A_i.push(rowV2 + 2);
      }
    }

    A_p.push(A_x.length);
  }

  // P = εI — tiny regulariser so the feasibility problem has a
  // unique interior optimum (least-norm feasible ρ).
  const regEps = 1e-8;
  const P_x = new Array<number>(N_RHO).fill(regEps);
  const P_i = Array.from({ length: N_RHO }, (_, i) => i);
  const P_p = Array.from({ length: N_RHO + 1 }, (_, i) => i);
  const cVec = new Array<number>(N_RHO).fill(0);

  const qArr = new Array<number>(2 * n).fill(3);

  return {
    data: {
      m,
      n: N_RHO,
      A_x,
      A_i,
      A_p,
      P_x,
      P_i,
      P_p,
      b,
      c: cVec,
    },
    cone: { l: n, q: qArr, qsize: 2 * n },
  };
}

// Evaluate the per-frequency violations at a given ρ.  Used to get a
// residual that is directly comparable to the pyfresco LMI_tol.
function evalResiduals(
  mats: CoeffMats,
  gamma: number,
  desMm: number,
  rho: ArrayLike<number>,
): { worstV1: number; worstV2: number; worstV3: number } {
  const n = mats.nFreq;
  let worstV1 = 0;
  let worstV2 = 0;
  let worstV3 = 0;
  for (let k = 0; k < n; k++) {
    const base = k * N_RHO;
    let psiRe = mats.dPsiRe[k];
    let uRe = mats.dURe[k];
    let uIm = mats.dUIm[k];
    let soRe = mats.dSoRe[k];
    let soIm = mats.dSoIm[k];
    let sniRe = 1;
    for (let j = 0; j < N_RHO; j++) {
      const rj = rho[j];
      psiRe += mats.cPsiRe[base + j] * rj;
      uRe += mats.cURe[base + j] * rj;
      uIm += mats.cUIm[base + j] * rj;
      soRe += mats.cSoRe[base + j] * rj;
      soIm += mats.cSoIm[base + j] * rj;
      sniRe += mats.cSniRe[base + j] * rj;
    }
    const v1 = gamma * Math.hypot(uRe, uIm) - psiRe + CONSTRAINT_SLACK;
    const v2 = desMm * Math.hypot(soRe, soIm) - psiRe + CONSTRAINT_SLACK;
    const v3 = SNI_LOWER - sniRe;
    if (v1 > worstV1) worstV1 = v1;
    if (v2 > worstV2) worstV2 = v2;
    if (v3 > worstV3) worstV3 = v3;
  }
  return { worstV1, worstV2, worstV3 };
}

async function innerSolveSCS(
  mats: CoeffMats,
  gamma: number,
  desMm: number,
  warm?: { x: Float64Array; y: Float64Array; s: Float64Array },
): Promise<InnerResult> {
  const SCS = await getSCS();
  const { data, cone } = buildScsProblem(mats, gamma, desMm);

  const settings = new SCS.ScsSettings();
  SCS.setDefaultSettings(settings);
  // One decade looser than before — still tight by SCS standards (docs
  // recommend 1e-6 for "accurate solutions") but keeps the solver from
  // stopping on SOLVED_INACCURATE at tight γ ≈ 1 SOCPs.
  settings.epsAbs = 1e-6;
  settings.epsRel = 1e-6;
  settings.epsInfeas = 1e-8;
  // On the torsional case SCS never actually reaches eps=1e-6 — it
  // returns SOLVED_INACCURATE whatever iter budget we give it — so
  // the 100k/10s ceiling just burns 3.7 s per SOCP on residual margin
  // the outer bisection doesn't consume (it judges feasibility at
  // LMI_TOL=1e-3).  20k/3s caps that waste at ~0.75 s per SOCP — 5×
  // faster end-to-end — and shifts γ by <5 % on the hard case.  Easy
  // plants converge in a few hundred iters and don't notice.
  settings.maxIters = 40000;
  settings.timeLimitSecs = 5;
  settings.verbose = 0;
  settings.warmStart = !!warm;

  // SCS's adaptive scaling heuristic drifts into a bad regime on some
  // of our SOCPs (notably the torsional chain at tight γ): it reports
  // SOLVED but the primal residual is actually ≫ tol, which the outer
  // bisection then reads as "infeasible" and bails out way above the
  // real optimum.  Disabling adaptive scaling (fixed scale=0.1) closes
  // the full γ = 7.98 → 1.10 gap on the torsional case and leaves the
  // other plants unchanged.
  settings.adaptiveScale = false;

  // Sweep harness can patch any of the above before the solve.
  if (scsSettingsOverride) Object.assign(settings, scsSettingsOverride);

  // SCS.solve's 4th argument is a prior ScsSolution used as warm start.
  // The README's own example shows it expects plain number[] arrays
  // (not Float64Array) and a full solution-shaped object.  We feed it
  // minimal info/status fields; SCS only reads x/y/s.
  let sol;
  if (warm) {
    const warmArg = {
      x: Array.from(warm.x),
      y: Array.from(warm.y),
      s: Array.from(warm.s),
      info: {
        iter: 0, pobj: 0, dobj: 0,
        resPri: 0, resDual: 0, resInfeas: 0, resUnbdd: 0,
        solveTime: 0, setupTime: 0,
      },
      status: "SOLVED",
    };
    sol = SCS.solve(data, cone, settings, warmArg);
  } else {
    sol = SCS.solve(data, cone, settings);
  }

  if (!sol) {
    return {
      rho: new Float64Array(N_RHO),
      residual: 1,
      loss: 0,
      worstV1: 0,
      worstV2: 0,
      worstV3: 0,
      status: "NULL",
    };
  }

  const status = sol.status;
  const primalOK = status === "SOLVED" || status === "SOLVED_INACCURATE";

  if (!primalOK || !sol.x) {
    // Infeasible / unbounded / failed: return dummy ρ and flag residual.
    return {
      rho: new Float64Array(N_RHO),
      residual: 1,
      loss: 0,
      worstV1: 0,
      worstV2: 0,
      worstV3: 0,
      status,
    };
  }

  const rho = new Float64Array(Array.from(sol.x));
  const nextWarm = sol.y && sol.s
    ? {
        x: new Float64Array(Array.from(sol.x)),
        y: new Float64Array(Array.from(sol.y)),
        s: new Float64Array(Array.from(sol.s)),
      }
    : undefined;

  const { worstV1, worstV2, worstV3 } = evalResiduals(mats, gamma, desMm, rho);
  const residual = Math.max(worstV1, worstV2, worstV3, 0);
  return { rho, residual, loss: 0, worstV1, worstV2, worstV3, status, warm: nextWarm };
}

// Diagnostic: solve the feasibility SOCP at a fixed γ.  Exported for
// the harness in scripts/.
export async function synthesizeAtGamma(
  grid: FrequencyGrid,
  specs: SynthesisSpecs,
  gamma: number,
  _maxIter?: number,
  _lr?: number,
): Promise<InnerResult> {
  void _maxIter;
  void _lr;
  setControllerOrder(specs.order ?? DEFAULT_ORDER);
  const { Wd } = desiredWeighting(grid.w, specs.desBw, specs.desZeta);
  const mats = precomputeCoeffs(grid, specs.Ts, Wd);
  return innerSolveSCS(mats, gamma, specs.desMm);
}

/* ------------------------------------------------------------------
   Closed-loop bandwidth: −3 dB point of the reference-to-output
   sensitivity  y/r = G·T / (G·MA·R + S).  Walks the grid (which is
   log-spaced) and linearly interpolates to hit −3 dB.
   ------------------------------------------------------------------ */

function achievedBandwidthHz(
  grid: FrequencyGrid,
  Ts: number,
  rhoR: number[],
  rhoSPre: number[],
  rhoT: number[],
): number {
  const n = grid.w.length;
  const fVals: number[] = [];
  const dbVals: number[] = [];
  for (let k = 0; k < n; k++) {
    const w = grid.w[k];
    const zNeg: Complex[] = new Array(Z_LEN);
    for (let i = 0; i < Z_LEN; i++) zNeg[i] = cExp(-w * Ts * i);
    let R = c(0);
    for (let i = 0; i < N_R; i++) R = cAdd(R, cScale(zNeg[i], rhoR[i]));
    let SoPre = c(0);
    for (let i = 0; i < N_S_PRE; i++)
      SoPre = cAdd(SoPre, cScale(zNeg[i + 1], rhoSPre[i]));
    const S = cMul(cSub(c(1), zNeg[1]), cAdd(c(1), SoPre));
    let T = c(0);
    for (let i = 0; i < N_T; i++) T = cAdd(T, cScale(zNeg[i], rhoT[i]));

    const GT = cMul(grid.G[k], T);
    const GMA_R = cMul(cMul(grid.G[k], grid.MA[k]), R);
    const denom = cAdd(GMA_R, S);
    const denomMag2 = denom.re * denom.re + denom.im * denom.im;
    const num = cAbs(GT);
    const mag = denomMag2 > 0 ? num / Math.sqrt(denomMag2) : 0;
    fVals.push(w / (2 * Math.PI));
    dbVals.push(20 * Math.log10(Math.max(mag, 1e-20)));
  }

  // First crossing from ≥ −3 dB down to < −3 dB (pyfresco bw_estimate
  // pattern: np.where(CL_db < -3)[0][0] then interp_x to hit -3 dB).
  // If CL starts below -3 dB (bad DC tracking) or never drops below -3,
  // return 0 so the UI can flag "could not compute bandwidth".
  if (dbVals[0] < -3) return 0;
  for (let k = 1; k < n; k++) {
    if (dbVals[k] < -3 && dbVals[k - 1] >= -3) {
      const lx0 = Math.log10(fVals[k - 1]);
      const lx1 = Math.log10(fVals[k]);
      const y0 = dbVals[k - 1];
      const y1 = dbVals[k];
      const lxc = lx0 + ((-3 - y0) / (y1 - y0)) * (lx1 - lx0);
      return Math.pow(10, lxc);
    }
  }
  return 0;
}

/* ------------------------------------------------------------------
   Outer bisection — the entry point.
   ------------------------------------------------------------------ */

export async function synthesizeController(
  grid: FrequencyGrid,
  specs: SynthesisSpecs,
  onProgress?: (p: SynthesisProgress) => void,
): Promise<SynthesisResult> {
  setControllerOrder(specs.order ?? DEFAULT_ORDER);
  const { Td, Wd } = desiredWeighting(grid.w, specs.desBw, specs.desZeta);
  void Td;
  const mats = precomputeCoeffs(grid, specs.Ts, Wd);

  let gMax = G_MAX;
  let gMin = G_MIN;
  let gamma = 0.2; // pyfresco initial

  let bestRho: Float64Array | null = null;
  let bestGamma = 0;

  const iterations: SynthesisProgress[] = [];
  let warm: { x: Float64Array; y: Float64Array; s: Float64Array } | undefined;

  const bisFlags: boolean[] = [];
  let bisIter = 0;

  while (gMax - gMin > BIS_TOL && bisIter < 30) {
    bisIter++;
    const { rho, residual, loss, warm: nextWarm } = await innerSolveSCS(
      mats,
      gamma,
      specs.desMm,
      warm,
    );

    const feasible = residual < LMI_TOL;

    let bw: number | null = null;
    if (feasible) {
      const rR: number[] = Array.from(rho.subarray(OFFSET_R, OFFSET_R + N_R));
      const rS: number[] = Array.from(rho.subarray(OFFSET_S, OFFSET_S + N_S_PRE));
      const rT: number[] = Array.from(rho.subarray(OFFSET_T, OFFSET_T + N_T));
      bw = achievedBandwidthHz(grid, specs.Ts, rR, rS, rT);
    }

    iterations.push({ iter: bisIter, gamma, bw, feasible, innerLoss: loss });
    onProgress?.({ iter: bisIter, gamma, bw, feasible, innerLoss: loss });

    // Warm-start the next SOCP from whichever solve succeeded (feasible
    // or not — SCS can still use an approximate previous iterate to
    // bootstrap the next problem since the A matrix only differs by the
    // γ-scaled rows).
    if (nextWarm) warm = nextWarm;

    if (feasible) {
      bestRho = new Float64Array(rho);
      bestGamma = gamma;
      gMin = gamma;
      gamma = (gamma + gMax) / 2;
      bisFlags.push(false);
    } else {
      gMax = gamma;
      gamma = (gamma + gMin) / 2;
      bisFlags.push(true);
    }
  }

  // Infeasible if every bisection step failed or residual huge.
  const allFailed = bisFlags.length > 0 && bisFlags.every((f) => f);
  const pyfrescoInfeasibleGamma =
    bestGamma === 0 ? (1 / gamma) : 1 / bestGamma;
  const infeasible = allFailed || !bestRho || pyfrescoInfeasibleGamma > 9;

  if (infeasible || !bestRho) {
    return {
      feasible: false,
      gammaOpt: Infinity,
      gammaPyfresco: bestGamma,
      rhoR: [],
      rhoS: [],
      rhoT: [],
      RFull: [],
      SFull: [],
      TFull: [],
      gain: 1,
      achievedBw: 0,
      iterations,
      infeasibilityHint:
        "Could not satisfy the modulus-margin / bandwidth / weighting constraints. Try a lower modulus margin, a lower closed-loop bandwidth, or a higher damping ratio.",
    };
  }

  // Build full polynomials.
  const rR: number[] = Array.from(bestRho.subarray(OFFSET_R, OFFSET_R + N_R));
  const rS: number[] = Array.from(bestRho.subarray(OFFSET_S, OFFSET_S + N_S_PRE));
  const rTraw: number[] = Array.from(bestRho.subarray(OFFSET_T, OFFSET_T + N_T));

  // Full S(z⁻¹) = (1 − z⁻¹)·(1 + s_0 z⁻¹ + … + s_{N_S_PRE−1} z⁻ᴺᴾ).
  // Expanded: c_0 = 1, c_1 = s_0 − 1, c_i = s_{i−1} − s_{i−2} for i in [2, N_S_PRE],
  // c_{N_S_PRE+1} = −s_{N_S_PRE−1}.  Length = N_S_PRE + 2 = N_S.
  const SFull: number[] = new Array(N_S_PRE + 2);
  SFull[0] = 1;
  SFull[1] = rS[0] - 1;
  for (let i = 2; i <= N_S_PRE; i++) SFull[i] = rS[i - 1] - rS[i - 2];
  SFull[N_S_PRE + 1] = -rS[N_S_PRE - 1];

  // T normalization: pyfresco's rst_ilc_obj (opt_select.py:26) exports T
  // as T_vec / GAIN where GAIN = ΣT/ΣR.  Because S has the integrator
  // factor (S(1)=0), CL(1) = T(1)/R(1) = ΣT/ΣR; dividing T by that
  // ratio forces CL(DC) = 1 for clean reference tracking.
  const sumR = rR.reduce((a, b) => a + b, 0);
  const sumT = rTraw.reduce((a, b) => a + b, 0);
  const gain = Math.abs(sumR) > 1e-12 ? sumT / sumR : 1;
  const rT = gain !== 0 ? rTraw.map((x) => x / gain) : rTraw;

  // Bandwidth on the normalized controller (matches what the plot shows).
  const normalizedBw = achievedBandwidthHz(grid, specs.Ts, rR, rS, rT);

  return {
    feasible: true,
    gammaOpt: 1 / bestGamma,
    gammaPyfresco: bestGamma,
    rhoR: rR,
    rhoS: rS,
    rhoT: rT,
    RFull: rR,
    SFull,
    TFull: rT,
    gain,
    achievedBw: normalizedBw,
    iterations,
  };
}

/* ------------------------------------------------------------------
   Utility: recommended bandwidth bounds per pyfresco docs —
   Fs/25 ≤ f_c ≤ Fs/8.
   ------------------------------------------------------------------ */

export function bandwidthBoundsHz(Ts: number): { min: number; max: number; default: number } {
  const Fs = 1 / Ts;
  return { min: Fs / 25, max: Fs / 8, default: Fs / 15 };
}

/* ------------------------------------------------------------------
   Utility: linearly-spaced frequency grid from w_init to π/Ts.
   200 points is a practical default for the SOCP and keeps the dense
   sampling near Nyquist where plant phase rolls hard.
   ------------------------------------------------------------------ */

export function linearFreqGrid(wInit: number, Ts: number, nPoints: number): number[] {
  const wMax = Math.PI / Ts;
  const out: number[] = new Array(nPoints);
  const step = (wMax - wInit) / (nPoints - 1);
  for (let i = 0; i < nPoints; i++) out[i] = wInit + i * step;
  return out;
}

// Log-spaced grid — used only for displaying open-loop plant Bodes.
export function logFreqGrid(wInit: number, Ts: number, nPoints: number): number[] {
  const wMax = Math.PI / Ts;
  const lo = Math.log10(wInit);
  const hi = Math.log10(wMax);
  const out: number[] = new Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    out[i] = Math.pow(10, lo + ((hi - lo) * i) / (nPoints - 1));
  }
  return out;
}

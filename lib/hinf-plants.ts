/* ------------------------------------------------------------------
   Preset plants for the H∞ synthesis demo.  Every plant exposes:

     • a human-readable label + description
     • a frequency-domain evaluator  G(jω)
     • LaTeX-rendered transfer function
     • default synthesis specs (bw/ζ/mm)
     • a dominant-pole frequency (rad/s) for adaptive grid w_init
     • a coprime-factor builder  (s+a)^n → N = B/F, M = A/F,
       with a = 2π·desBwHz.  Stable plants use the trivial
       factorization N = G, M = 1.

   The sampling period is fixed at TS_FIXED for the whole demo — the
   closed-loop bandwidth / modulus margin / damping sliders give plenty
   of room without the Nyquist constraint becoming its own puzzle.
   ------------------------------------------------------------------ */

import {
  c,
  cAdd,
  cMul,
  cExp,
  Complex,
  FrequencyGrid,
  linearFreqGrid,
} from "./hinf-synthesis";

/** Sampling period fixed for the whole demo. */
export const TS_FIXED = 5e-4;

export interface Plant {
  id: string;
  label: string;
  sub: string;         // short descriptor for the selector
  description: string; // longer helper text
  latex: string;       // transfer function
  Ts: number;
  /** Approximate dominant-pole frequency (rad/s) — used to set
   *  w_init = dominantPoleRad / 100 so the Bode grid shows ≥ 2
   *  decades of DC plateau before the first break. */
  dominantPoleRad: number;
  /** Continuous-time FRF evaluator. */
  frf: (w: number) => Complex;
  /** Coprime factors N(jω), M(jω) given the user's target
   *  closed-loop bandwidth. */
  buildCoprime: (w: number[], desBwHz: number) => { N: Complex[]; M: Complex[] };
  /** Build the 200-point synthesis grid. */
  buildGrid(Ts: number, desBwHz: number): FrequencyGrid;
  /** Default synthesis specs. */
  defaults: {
    desMm: number;
    desBw: number;
    desZeta: number;
    order: number;   // controller polynomial degree (N_R = N_S = N_T = order + 1)
  };
}

const N_GRID = 200;

interface PlantSpec {
  id: string;
  label: string;
  sub: string;
  description: string;
  latex: string;
  dominantPoleRad: number;
  frf: (w: number) => Complex;
  /** If omitted, the plant is treated as stable and we use the trivial
   *  factorization N = G, M = 1. */
  buildCoprime?: (w: number[], desBwHz: number) => { N: Complex[]; M: Complex[] };
  defaults: Plant["defaults"];
}

function trivialCoprime(w: number[], G: Complex[]): { N: Complex[]; M: Complex[] } {
  return { N: G, M: w.map(() => c(1)) };
}

function makePlant(spec: PlantSpec): Plant {
  const buildCoprime =
    spec.buildCoprime ??
    ((w: number[], _desBwHz: number) => trivialCoprime(w, w.map(spec.frf)));

  return {
    id: spec.id,
    label: spec.label,
    sub: spec.sub,
    description: spec.description,
    latex: spec.latex,
    Ts: TS_FIXED,
    dominantPoleRad: spec.dominantPoleRad,
    frf: spec.frf,
    buildCoprime,
    defaults: spec.defaults,
    buildGrid(Ts = TS_FIXED, desBwHz: number) {
      const wInit = spec.dominantPoleRad / 100;
      const w = linearFreqGrid(wInit, Ts, N_GRID);
      const G = w.map(spec.frf);
      const MA = w.map(() => c(1));
      const { N, M } = buildCoprime(w, desBwHz);
      return { w, G, MA, N, M };
    },
  };
}

// Shared default bandwidth — the single-Ts demo settles at 200 Hz as a
// reasonable midpoint of the Fs/25..Fs/8 = 80..250 Hz slider range.
const DEFAULT_BW_HZ = 200;
const DEFAULT_ZETA = 0.8;
const DEFAULT_ORDER = 6;

/* ------------------------------------------------------------------
   Plant 1 — First-order magnet (Rs + sL) with pure actuator delay.
   Classic dipole correction coil.
   ------------------------------------------------------------------ */

const PLANT_1: Plant = makePlant({
  id: "magnet-rl",
  label: "Dipole correction coil",
  sub: "Rs + sL · pure delay",
  description:
    "A small correction dipole: an R–L load with a 150 µs actuator delay. First-order roll-off, ~−20 dB/decade.",
  latex: String.raw`G(s) \;=\; \dfrac{1}{R_s + s\,L}\;e^{-s\,\tau_d}, \quad R_s = 0.1\,\Omega,\; L = 10\,\text{mH},\; \tau_d = 150\,\mu s`,
  dominantPoleRad: 0.1 / 10e-3, // Rs / L
  frf: (w) => {
    const Rs = 0.1, L = 10e-3, tauD = 150e-6;
    const denom = c(Rs, w * L);
    const denomMag2 = denom.re * denom.re + denom.im * denom.im;
    const invDenom = c(denom.re / denomMag2, -denom.im / denomMag2);
    return cMul(invDenom, cExp(-w * tauD));
  },
  defaults: {
    desMm: 0.5,
    desBw: DEFAULT_BW_HZ,
    desZeta: DEFAULT_ZETA,
    order: DEFAULT_ORDER,
  },
});

/* ------------------------------------------------------------------
   Plant 2 — Fractional-order eddy-current magnet.
     G(s) = K / (1 + (s·τ)^α),   α = 0.75 → −15 dB/decade
   Models the anomalous roll-off seen on iron-yoke magnets where
   eddy currents in the yoke laminations create a non-integer-order
   low-pass.
   ------------------------------------------------------------------ */

const PLANT_2: Plant = makePlant({
  id: "eddy-fractional",
  label: "Eddy-current magnet",
  sub: "fractional α = 0.75 · −15 dB/dec",
  description:
    "A large iron-yoke magnet whose flux sees the stator laminations as a fractional low-pass: −15 dB/dec roll-off from the non-integer pole order α = 0.75. pyfresco handles these because the algorithm only needs a complex FRF, not a rational model.",
  latex: String.raw`G(s) \;=\; \dfrac{K}{1 + (s\,\tau)^{\alpha}}\;e^{-s\,\tau_d}, \quad K = 1,\; \tau = 1/(2\pi\cdot 20),\; \alpha = 0.75,\; \tau_d = 120\,\mu s`,
  dominantPoleRad: 2 * Math.PI * 20, // 1/τ
  frf: (w) => {
    const tau = 1 / (2 * Math.PI * 20), alpha = 0.75, tauD = 120e-6;
    // (jωτ)^α via polar form
    const r = Math.abs(w * tau);
    const theta = (w >= 0 ? Math.PI / 2 : -Math.PI / 2);
    const rA = Math.pow(r, alpha);
    const tA = theta * alpha;
    const pow = { re: rA * Math.cos(tA), im: rA * Math.sin(tA) };
    const denom = cAdd(c(1), pow);
    const denomMag2 = denom.re * denom.re + denom.im * denom.im;
    const invDenom = c(denom.re / denomMag2, -denom.im / denomMag2);
    return cMul(invDenom, cExp(-w * tauD));
  },
  defaults: {
    desMm: 0.5,
    desBw: DEFAULT_BW_HZ,
    desZeta: DEFAULT_ZETA,
    order: DEFAULT_ORDER,
  },
});

/* ------------------------------------------------------------------
   Plant 3 — Reaction-wheel inverted pendulum, linearized at upright.
   Abstracted second-order form so the unstable pole sits around 80 Hz
   (≈ 2.5× below the 200 Hz closed-loop target); parameterized directly
   by ω_p rather than (m, L, g) since no physical rod matches this scale.
     G(s) = ω_p² / (s² − ω_p²)       [unit DC magnitude]
   UNSTABLE: twin real poles at ±ω_p.  Needs coprime factorization:
     F(s) = (s + a)²,  a = 2·ω_p    (anchored to the unstable-mode
                                    scale, not the closed-loop BW —
                                    keeps |N|, |M| O(1) across the band
                                    and is friendlier to SCS than the
                                    literal Karimi choice a = 2π·f_c).
   ------------------------------------------------------------------ */

const PEND_FP_HZ = 80;                               // unstable-mode frequency
const PEND_WP = 2 * Math.PI * PEND_FP_HZ;            // ≈ 502.65 rad/s
const PEND_WP2 = PEND_WP * PEND_WP;                  // ≈ 2.527e5
const PEND_K = PEND_WP2;                             // unit DC magnitude

const PLANT_3: Plant = makePlant({
  id: "pendulum-inverted",
  label: "Inverted pendulum",
  sub: "unstable · reaction-wheel · f_p ≈ 80 Hz",
  description:
    "A reaction-wheel inverted pendulum linearized about the upright equilibrium — a textbook unstable plant with a pair of real poles at ±ω_p ≈ ±503 rad/s (≈ 80 Hz). We synthesize against a stable coprime factorization G = N/M with F(s) = (s+a)² and a = 2·ω_p, so the H∞ SOCP is well-posed even though G itself is unstable.",
  latex: String.raw`G(s) \;=\; \dfrac{\omega_p^{2}}{s^{2} - \omega_p^{2}}, \quad \omega_p = 2\pi\cdot 80\,\text{rad/s}`,
  dominantPoleRad: PEND_WP,
  frf: (w) => {
    const denomRe = -(w * w) - PEND_WP2;
    return c(PEND_K / denomRe, 0);
  },
  buildCoprime: (w, _desBwHz) => {
    const a = 2 * PEND_WP;
    const N: Complex[] = new Array(w.length);
    const M: Complex[] = new Array(w.length);
    for (let k = 0; k < w.length; k++) {
      const wk = w[k];
      const fRe = a * a - wk * wk;
      const fIm = 2 * a * wk;
      const fMag2 = fRe * fRe + fIm * fIm;
      N[k] = {
        re: (PEND_K * fRe) / fMag2,
        im: (-PEND_K * fIm) / fMag2,
      };
      const aRe = -(wk * wk) - PEND_WP2;
      M[k] = {
        re: (aRe * fRe) / fMag2,
        im: (-aRe * fIm) / fMag2,
      };
    }
    return { N, M };
  },
  defaults: {
    desMm: 0.5,
    desBw: DEFAULT_BW_HZ,
    desZeta: DEFAULT_ZETA,
    order: DEFAULT_ORDER,
  },
});

/* ------------------------------------------------------------------
   Plant 4 — Three-mass torsional chain (motor → shaft → intermediate
   → shaft → load).  Standard 3-inertia / 2-shaft lumped model with
   viscous friction on each mass and shaft damping setting the mode Q.
   Two flexible modes sit near 55 Hz and 110 Hz with an anti-resonance
   between them.

   Equations of motion in Laplace:
     α₁ θ₁ − β₁ θ₂           = U
     −β₁ θ₁ + α₂ θ₂ − β₂ θ₃   = 0
              −β₂ θ₂ + α₃ θ₃  = 0
   with α₁ = J₁s² + (d₁+b)s + k₁,
        α₂ = J₂s² + (d₁+d₂+b)s + k₁+k₂,
        α₃ = J₃s² + (d₂+b)s + k₂,
        β₁ = d₁s + k₁,   β₂ = d₂s + k₂.
   Eliminating θ₁, θ₂ gives torque → ω₃:
     G_vel(s) = s·β₁β₂ / [α₁(α₂α₃ − β₂²) − β₁²α₃]
   Evaluating at s=0 gives 0/0 — the denominator has an exact rigid-
   body root that cancels the leading s in the numerator (verified
   analytically via α(0), β(0) = k's).  After cancellation, G_vel is a
   stable, proper, 5-pole / 2-zero rational function with finite DC
   gain 1/(3b).  Because it is already stable we use the trivial
   coprime factorization N = G, M = 1 — no (s+a)ⁿ denominators to
   pick.  The controller's forced integrator (N_INT = 1) supplies the
   low-frequency loop action.

   We pre-expand the symbolic cancellation at module load so the FRF
   we evaluate has no 0/0 fuzz near DC.
   ------------------------------------------------------------------ */

const TORS = {
  J1: 5e-3, J2: 3e-3, J3: 4e-3,     // inertias (kg·m²)
  k1: 800,  k2: 300,                 // shaft stiffnesses (N·m/rad)
  d1: 0.08, d2: 0.04,                // shaft damping (N·m·s/rad)
  b:  0.05,                          // ground viscous friction per mass
};

/** Pre-expand the torsional-chain polynomials once and drop the
 *  rigid-body s factor symbolically, so G = numPoly / denPoly is a
 *  minimal, stable realization with no DC indeterminacy. */
function buildTorsPolys() {
  const { J1, J2, J3, k1, k2, d1, d2, b } = TORS;
  const mul = (a: number[], bb: number[]) => {
    const r = new Array(a.length + bb.length - 1).fill(0);
    for (let i = 0; i < a.length; i++)
      for (let j = 0; j < bb.length; j++) r[i + j] += a[i] * bb[j];
    return r;
  };
  const sub = (a: number[], bb: number[]) => {
    const n = Math.max(a.length, bb.length);
    const r = new Array(n).fill(0);
    for (let i = 0; i < n; i++) r[i] = (a[i] ?? 0) - (bb[i] ?? 0);
    return r;
  };
  // α_i(s), β_i(s) as ascending-coefficient polynomials
  const a1  = [k1,       d1 + b,       J1];
  const a2  = [k1 + k2,  d1 + d2 + b,  J2];
  const a3  = [k2,       d2 + b,       J3];
  const be1 = [k1,       d1];
  const be2 = [k2,       d2];
  // num(s) = s · β₁ · β₂  →  after dropping the leading s: β₁·β₂
  const num = mul(be1, be2);
  // den(s) = α₁(α₂α₃ − β₂²) − β₁²α₃  →  drop the rigid-body s root
  const den = sub(
    mul(a1, sub(mul(a2, a3), mul(be2, be2))),
    mul(mul(be1, be1), a3),
  );
  return { num, den: den.slice(1) };
}

const TORS_POLYS = buildTorsPolys();

/** Evaluate an ascending-power real-coefficient polynomial at s = jω. */
function polyAtJw(p: number[], w: number): Complex {
  let re = 0, im = 0, jkRe = 1, jkIm = 0;
  for (let k = 0; k < p.length; k++) {
    re += p[k] * jkRe;
    im += p[k] * jkIm;
    // (jω)·(jkRe + j·jkIm) = −jkIm·ω + j·jkRe·ω
    const nextRe = -jkIm * w;
    const nextIm =  jkRe * w;
    jkRe = nextRe;
    jkIm = nextIm;
  }
  return c(re, im);
}

/** Torsional-chain velocity FRF — stable, proper, with the s/s
 *  cancellation baked into the polynomial representation. */
function torsionalFRF(w: number): Complex {
  const num = polyAtJw(TORS_POLYS.num, w);
  const den = polyAtJw(TORS_POLYS.den, w);
  const denMag2 = den.re * den.re + den.im * den.im;
  if (denMag2 < 1e-30) return c(1e30, 0);
  return c(
    (num.re * den.re + num.im * den.im) / denMag2,
    (num.im * den.re - num.re * den.im) / denMag2,
  );
}

const PLANT_4: Plant = makePlant({
  id: "torsional-3mass",
  label: "Torsional chain",
  sub: "3-mass · two resonances · stable",
  description:
    "A three-inertia torsional shaft: motor rotor J₁ coupled through compliant shaft k₁ to intermediate mass J₂, then through k₂ to load J₃. Light damping leaves two flexible modes near 55 Hz and 110 Hz with an anti-resonance between them. The rigid-body pole at the origin is exactly cancelled by a structural zero from the chain's ground damping, so G is a stable 5-pole / 2-zero transfer function with finite DC gain 1/(3b) — no coprime factorization needed (N = G, M = 1). The controller's forced integrator supplies the type-1 loop action.",
  latex: String.raw`\begin{aligned}
    G(s) \;&=\; \dfrac{\dot\theta_3(s)}{T_{\text{in}}(s)}
         \;=\; \dfrac{\beta_1(s)\,\beta_2(s)}{D(s)} \\[4pt]
    \alpha_i(s) \;&=\; J_i\,s^{2} + (d_i + b)\,s + k_i \\[2pt]
    \beta_i(s)  \;&=\; d_i\,s + k_i \\[2pt]
    D(s) \;&=\; \alpha_1\bigl(\alpha_2\alpha_3 - \beta_2^{\,2}\bigr)
              - \beta_1^{\,2}\,\alpha_3 \quad (\deg 5)
  \end{aligned}`,
  dominantPoleRad: 2 * Math.PI * 30, // low-frequency corner ≈ 30 Hz for w_init
  frf: torsionalFRF,
  // buildCoprime omitted → trivial N = G, M = 1 via makePlant default.
  defaults: {
    desMm: 0.5,
    desBw: DEFAULT_BW_HZ,
    desZeta: DEFAULT_ZETA,
    order: DEFAULT_ORDER,
  },
});

/* ------------------------------------------------------------------
   Plant 5 — DC motor with integrator, electrical pole, and delay.
     G(s) = Km · e^{-s τ_d} / [s · (τ_m s + 1) · (τ_e s + 1)]
   Models a digital servo drive: position output (integrator), mechanical
   inertia/friction pole at 1/τ_m ≈ 16 Hz, armature electrical pole at
   1/τ_e ≈ 160 Hz, and 150 µs of sample + ZOH + PWM + compute delay.

   For the coprime factorization we rewrite the plant in MONIC form by
   absorbing τ_m τ_e into the gain — this is critical for numerical
   conditioning of the SOCP:
     G(s) = K' · e^{-s τ_d} / [s (s + p_m)(s + p_e)],
            K' = Km / (τ_m τ_e),  p_m = 1/τ_m,  p_e = 1/τ_e
   Then with F(s) = (s+a)³, a = 2π·desBwHz,
     N(s) = K' e^{-s τ_d} / (s+a)³
     M(s) = s (s + p_m)(s + p_e) / (s+a)³
   The non-monic form A(s) = s(τ_m s+1)(τ_e s+1) has leading coefficient
   τ_m τ_e ≈ 1e-5, which makes |M| → 1e-5 at high frequency and |N(0)|
   ≈ Km/a³ ≈ 5e-8 — both are five orders of magnitude smaller than
   anything the SCS solver expects, and γ blows up.  The monic rewrite
   keeps |M| → 1 and |N(0)| ≈ K'/a³ ≈ 5e-3 — still small at DC but in
   the right ballpark.  Same plant, just numerically well-posed.

   The delay stays in N as a non-rational factor — synthesis only needs
   the complex FRF, so e^{-jω τ_d} comes along for free.
   ------------------------------------------------------------------ */

const MOTOR_KM = 100.0;
const MOTOR_TAU_M = 0.01;    // mechanical pole at 100 rad/s ≈ 16 Hz
const MOTOR_TAU_E = 1e-3;    // electrical pole at 1000 rad/s ≈ 160 Hz
const MOTOR_TAU_D = 150e-6;  // 150 µs sample + ZOH + PWM + compute delay
const MOTOR_PM = 1 / MOTOR_TAU_M;                  // 100 rad/s
const MOTOR_PE = 1 / MOTOR_TAU_E;                  // 1000 rad/s
const MOTOR_KP = MOTOR_KM * MOTOR_PM * MOTOR_PE;   // monic-form gain K' = 1e7

const PLANT_5: Plant = makePlant({
  id: "dc-motor",
  label: "DC motor · position",
  sub: "integrator + 2 poles · 150 µs delay",
  description:
    "Digital servo-driven DC motor with position output: integrator on the jω axis, mechanical pole at 1/τ_m ≈ 16 Hz, armature electrical pole at 1/τ_e ≈ 160 Hz (just below the closed-loop target), plus 150 µs of sample + ZOH + PWM + compute delay. Same Karimi coprime trick as the pendulum but bumped to F(s) = (s+a)³ for the third-order rational denominator; the delay rides along inside N as a non-rational e^{-jω τ_d} factor — pyfresco handles it because synthesis only needs the complex FRF, not a state-space realization.",
  latex: String.raw`G(s) \;=\; \dfrac{K_m\;e^{-s\,\tau_d}}{s\,(\tau_m\,s + 1)\,(\tau_e\,s + 1)}, \quad K_m = 100,\; \tau_m = 10\,\text{ms},\; \tau_e = 1\,\text{ms},\; \tau_d = 150\,\mu s`,
  dominantPoleRad: 1.0 / MOTOR_TAU_M, // mechanical pole sets the low-freq shape
  frf: (w) => {
    // G(jω) = Km · e^(-jωτ_d) / (jω · (jωτ_m + 1) · (jωτ_e + 1))
    const jw = c(0, w);
    const tauMJw1 = c(1, w * MOTOR_TAU_M);
    const tauEJw1 = c(1, w * MOTOR_TAU_E);
    const denom = cMul(cMul(jw, tauMJw1), tauEJw1);
    const denomMag2 = denom.re * denom.re + denom.im * denom.im;
    if (denomMag2 < 1e-30) return c(1e30, 0);
    const ratio = c(
      (MOTOR_KM * denom.re) / denomMag2,
      (-MOTOR_KM * denom.im) / denomMag2,
    );
    return cMul(ratio, cExp(-w * MOTOR_TAU_D));
  },
  buildCoprime: (w, desBwHz) => {
    const a = 2 * Math.PI * desBwHz;
    const N: Complex[] = new Array(w.length);
    const M: Complex[] = new Array(w.length);
    for (let k = 0; k < w.length; k++) {
      const wk = w[k];
      // F(jω) = (a + jω)³  →  Re = a(a² − 3ω²),  Im = ω(3a² − ω²)
      const fRe = a * (a * a - 3 * wk * wk);
      const fIm = wk * (3 * a * a - wk * wk);
      const fMag2 = fRe * fRe + fIm * fIm;
      // N(jω) = K' · e^(-jωτ_d) / F(jω) = K'·e^(-jωτ_d) · conj(F) / |F|²
      const nNumRe =  MOTOR_KP * Math.cos(wk * MOTOR_TAU_D);
      const nNumIm = -MOTOR_KP * Math.sin(wk * MOTOR_TAU_D);
      N[k] = {
        re: (nNumRe * fRe + nNumIm * fIm) / fMag2,
        im: (nNumIm * fRe - nNumRe * fIm) / fMag2,
      };
      // A(jω) = jω · (jω + p_m) · (jω + p_e)  [monic]
      //       = −(p_m + p_e)·ω² + j·ω·(p_m·p_e − ω²)
      const aRe = -(MOTOR_PM + MOTOR_PE) * wk * wk;
      const aIm = wk * (MOTOR_PM * MOTOR_PE - wk * wk);
      // M = A · conj(F) / |F|²
      M[k] = {
        re: (aRe * fRe + aIm * fIm) / fMag2,
        im: (aIm * fRe - aRe * fIm) / fMag2,
      };
    }
    return { N, M };
  },
  defaults: {
    desMm: 0.5,
    desBw: DEFAULT_BW_HZ,
    desZeta: DEFAULT_ZETA,
    order: DEFAULT_ORDER,
  },
});

export const PLANTS: Plant[] = [PLANT_1, PLANT_2, PLANT_3, PLANT_4, PLANT_5];

export function plantById(id: string): Plant {
  const p = PLANTS.find((p) => p.id === id);
  if (!p) throw new Error(`Unknown plant: ${id}`);
  return p;
}

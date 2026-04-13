// Q-learning for LQR via LSPI (Lagoudakis & Parr 2003).
//
// Q-function class:
//   Q(x, u) = [x; u]^T H [x; u]
// where H is symmetric (5×5 for x∈ℝ⁴, u∈ℝ¹). We learn the 15 unique
// upper-triangular entries of H from real (xₜ, uₜ, cₜ, xₜ₊₁) tuples
// via least-squares Bellman residual minimization (LSTDQ), then take
// the greedy policy K = H_uu⁻¹ H_xu^T (LSPI policy improvement).
//
// This is provably-convergent model-free RL on the LQR class. If it
// works in pure Node, the in-browser version will work too.
//
// Verification target: K = LQR_K = [-41.64, 101.62, -17.31, 29.14]
// from K = 0 init using ≤500 real-environment samples.

const L1=1, L2=1, M1=1, M2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;
const LQR_K = [-41.64, 101.62, -17.31, 29.14];
const LQR_NORM = Math.sqrt(LQR_K.reduce((a,b)=>a+b*b, 0));

const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };
function deriv(s,u){const{th1,th2,w1,w2}=s;const d=th1-th2,cd=Math.cos(d),sd=Math.sin(d);const M11=(M1+M2)*L1*L1,M12=M2*L1*L2*cd,M22=M2*L2*L2,det=M11*M22-M12*M12;const C1=M2*L1*L2*sd*w2*w2,C2=-M2*L1*L2*sd*w1*w1;const G1=(M1+M2)*G*L1*Math.sin(th1),G2=M2*G*L2*Math.sin(th2);const r1=u-DAMPING*w1-C1-G1,r2=-DAMPING*w2-C2-G2;return [w1,w2,(M22*r1-M12*r2)/det,(-M12*r1+M11*r2)/det];}
function stepEnv(s,u,dt=DT){const k1=deriv(s,u);const s2={th1:s.th1+k1[0]*dt/2,th2:s.th2+k1[1]*dt/2,w1:s.w1+k1[2]*dt/2,w2:s.w2+k1[3]*dt/2};const k2=deriv(s2,u);const s3={th1:s.th1+k2[0]*dt/2,th2:s.th2+k2[1]*dt/2,w1:s.w1+k2[2]*dt/2,w2:s.w2+k2[3]*dt/2};const k3=deriv(s3,u);const s4={th1:s.th1+k3[0]*dt,th2:s.th2+k3[1]*dt,w1:s.w1+k3[2]*dt,w2:s.w2+k3[3]*dt};const k4=deriv(s4,u);return{th1:s.th1+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,th2:s.th2+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6,w1:s.w1+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*dt/6,w2:s.w2+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*dt/6};}

const featuresOf = s => [s.th1, wrap(s.th2 - Math.PI), s.w1, s.w2];

// Feature normalization: rescale (x, u) so all entries are O(1).
// This dramatically improves LSTDQ conditioning. Gains are recovered
// in the UNNORMALIZED frame at the end.
const X_SCALE = [0.3, 0.3, 2.0, 2.0];  // typical magnitudes in linear regime
const U_SCALE = 15.0;                    // half of TORQUE_MAX
const normX = x => [x[0]/X_SCALE[0], x[1]/X_SCALE[1], x[2]/X_SCALE[2], x[3]/X_SCALE[3]];
const normU = u => u / U_SCALE;
// If K̃ is the gain in normalized space (ũ = -K̃·x̃), then in original space
// u = U_SCALE·ũ = -U_SCALE·K̃·(x/X_SCALE) = -(U_SCALE/X_SCALE[i])·K̃[i]·x[i]
const denormK = Kt => [Kt[0]*U_SCALE/X_SCALE[0], Kt[1]*U_SCALE/X_SCALE[1],
                       Kt[2]*U_SCALE/X_SCALE[2], Kt[3]*U_SCALE/X_SCALE[3]];

function gauss() {
  const u1 = Math.max(1e-9, Math.random()), u2 = Math.random();
  return Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2);
}

function randomBasinIC() {
  return { th1: (Math.random()-0.5)*0.2, th2: Math.PI + (Math.random()-0.5)*0.2,
           w1:  (Math.random()-0.5)*1,   w2:  (Math.random()-0.5)*1 };
}

// ================================================================
// Quadratic features φ(x, u) such that φ^T h = z^T H z when h is the
// vec of H's upper triangle (with off-diagonal entries doubled in φ).
// dim(z) = 5, dim(φ) = 15.
// ================================================================
const FEATURE_DIM = 15;

function phi(x, u) {
  const z = [x[0], x[1], x[2], x[3], u];
  const f = new Array(FEATURE_DIM);
  let k = 0;
  for (let i = 0; i < 5; i++) {
    for (let j = i; j < 5; j++) {
      f[k++] = (i === j ? 1 : 2) * z[i] * z[j];
    }
  }
  return f;
}

function unflattenH(h) {
  const H = Array.from({length: 5}, () => new Array(5).fill(0));
  let k = 0;
  for (let i = 0; i < 5; i++) {
    for (let j = i; j < 5; j++) {
      H[i][j] = h[k];
      H[j][i] = h[k];
      k++;
    }
  }
  return H;
}

// Greedy policy: K[i] = H_xu[i] / H_uu (since u is scalar).
// u* = -K · x.
function extractK(H) {
  const Huu = H[4][4];
  if (!Number.isFinite(Huu) || Math.abs(Huu) < 1e-9) return null;
  return [H[0][4]/Huu, H[1][4]/Huu, H[2][4]/Huu, H[3][4]/Huu];
}

// Gauss elimination with partial pivoting for the LSTDQ linear system.
function solve(A, b) {
  const n = b.length;
  const M = A.map(r => r.slice());
  const y = b.slice();
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[pivot][i])) pivot = k;
    }
    if (pivot !== i) { [M[i], M[pivot]] = [M[pivot], M[i]]; [y[i], y[pivot]] = [y[pivot], y[i]]; }
    if (Math.abs(M[i][i]) < 1e-12) return null;
    for (let k = i + 1; k < n; k++) {
      const c = M[k][i] / M[i][i];
      for (let j = i; j < n; j++) M[k][j] -= c * M[i][j];
      y[k] -= c * y[i];
    }
  }
  const h = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * h[j];
    h[i] = s / M[i][i];
  }
  return h;
}

// ================================================================
// Sample collection. Roll out under (current K + exploration noise),
// reset on out-of-regime states. Returns N tuples (x, u, c, x').
// ================================================================
const Q_DIAG = [10, 10, 1, 1];
const R_COST = 0.1;

function costOf(x, u) {
  return Q_DIAG[0]*x[0]*x[0] + Q_DIAG[1]*x[1]*x[1] +
         Q_DIAG[2]*x[2]*x[2] + Q_DIAG[3]*x[3]*x[3] +
         R_COST * u * u;
}

function inLinearRegime(x) {
  return Math.abs(x[0]) < 0.6 && Math.abs(x[1]) < 0.6 &&
         Math.abs(x[2]) < 4 && Math.abs(x[3]) < 4 &&
         Number.isFinite(x[0]);
}

function collectSamples(K, N, sigma) {
  const buf = [];
  let s = randomBasinIC();
  let stepCount = 0;
  let tries = 0;
  const maxTries = N * 20;
  while (buf.length < N && tries < maxTries) {
    tries++;
    const x = featuresOf(s);
    if (!inLinearRegime(x)) { s = randomBasinIC(); stepCount = 0; continue; }
    let u = -(K[0]*x[0] + K[1]*x[1] + K[2]*x[2] + K[3]*x[3]) + sigma * gauss();
    if (u > TORQUE_MAX) u = TORQUE_MAX;
    else if (u < -TORQUE_MAX) u = -TORQUE_MAX;
    const next = stepEnv(s, u);
    const xn = featuresOf(next);
    // Cost is charged at the state where the action is TAKEN: c(x, u).
    const c = costOf(x, u);
    if (!inLinearRegime(xn)) {
      // Reject — don't pollute LSTDQ with out-of-regime bootstraps.
      s = randomBasinIC();
      stepCount = 0;
      continue;
    }
    buf.push({ x, u, c, xn });
    s = next;
    stepCount += 1;
    if (stepCount > 30) { s = randomBasinIC(); stepCount = 0; }
  }
  return buf;
}

// ================================================================
// LSTDQ: solve for h such that h satisfies the Bellman equation
// under policy K, on the given sample buffer.
//   Σ φ(x,u) [φ(x,u) - γ φ(x', -K·x')]^T h ≈ Σ φ(x,u) c
// Tikhonov regularization with ridge λ for numerical stability.
// ================================================================
// LSTDQ in NORMALIZED coordinates. Kt is the gain in normalized space
// (ũ = -Kt·x̃). Returns h in normalized features.
function lstdq(samples, Kt, gamma, lambda) {
  const D = FEATURE_DIM;
  const A = Array.from({length: D}, () => new Array(D).fill(0));
  const b = new Array(D).fill(0);
  for (let i = 0; i < D; i++) A[i][i] = lambda;

  for (const { x, u, c, xn } of samples) {
    const xt  = normX(x);
    const ut  = normU(u);
    const xnt = normX(xn);
    // Target policy in normalized space: ũ' = -Kt · x̃'
    const unt = -(Kt[0]*xnt[0] + Kt[1]*xnt[1] + Kt[2]*xnt[2] + Kt[3]*xnt[3]);
    const f  = phi(xt, ut);
    const fn = phi(xnt, unt);
    for (let i = 0; i < D; i++) {
      const fi = f[i];
      for (let j = 0; j < D; j++) {
        A[i][j] += fi * (f[j] - gamma * fn[j]);
      }
      b[i] += fi * c;
    }
  }
  return solve(A, b);
}

// ================================================================
// LSPI outer loop: alternate LSTDQ (policy evaluation) and policy
// improvement K ← greedy(H) until K stops moving.
// ================================================================
// LSPI runs in normalized coordinates. Kt0 is the initial gain in that space.
// Returns K in original coordinates.
function lspi(samples, Kt0, gamma, maxIter, verbose = true) {
  let Kt = [...Kt0];
  for (let it = 1; it <= maxIter; it++) {
    const h = lstdq(samples, Kt, gamma, 1e-4);
    if (!h) { if (verbose) console.log(`  LSPI iter ${it}: solve failed`); return denormK(Kt); }
    const H = unflattenH(h);
    const Ktnew = extractK(H);
    if (!Ktnew) { if (verbose) console.log(`  LSPI iter ${it}: H_uu degenerate`); return denormK(Kt); }

    let diff = 0;
    for (let i = 0; i < 4; i++) diff += (Ktnew[i] - Kt[i]) ** 2;
    diff = Math.sqrt(diff);

    const K = denormK(Ktnew);
    let err = 0;
    for (let i = 0; i < 4; i++) err += (K[i] - LQR_K[i]) ** 2;
    err = Math.sqrt(err) / LQR_NORM;

    if (verbose) {
      const fmt = a => a.map(x => x.toFixed(2).padStart(8)).join(" ");
      console.log(`  LSPI iter ${String(it).padStart(2)}: K=[${fmt(K)}]  ΔK̃=${diff.toFixed(3).padStart(7)}  err=${(err*100).toFixed(1).padStart(5)}%  H̃uu=${H[4][4].toFixed(3)}`);
    }

    Kt = Ktnew;
    if (diff < 1e-4) break;
  }
  return denormK(Kt);
}

// ================================================================
// Test runs.
// ================================================================
console.log("=== Q-learning (LSPI) for LQR — pendubot ===");
console.log(`Target K_LQR: [${LQR_K.map(x => x.toFixed(2)).join(", ")}]`);
console.log(`||K_LQR|| = ${LQR_NORM.toFixed(2)}\n`);

// LQR gain expressed in normalized coordinates (what LSPI actually solves for).
// u = -K·x  →  ũ = u/U_SCALE = -(K[i]·X_SCALE[i]/U_SCALE) · x̃[i]
const LQR_Kt = [LQR_K[0]*X_SCALE[0]/U_SCALE, LQR_K[1]*X_SCALE[1]/U_SCALE,
                LQR_K[2]*X_SCALE[2]/U_SCALE, LQR_K[3]*X_SCALE[3]/U_SCALE];
console.log(`K̃_LQR (normalized): [${LQR_Kt.map(x => x.toFixed(3)).join(", ")}]\n`);

// ================================================================
// Sanity check 1: LSTDQ at π = K_LQR should recover a stable H, and
// the greedy policy from that H should be ≈ K_LQR (fixed point).
// ================================================================
console.log("-- Sanity: LSTDQ @ π=K_LQR (should be near-fixed-point) --");
{
  const samples = collectSamples(LQR_K, 2000, 3.0);
  console.log(`  collected ${samples.length} samples (behavior = LQR + σ=3)`);
  const h = lstdq(samples, LQR_Kt, 0.99, 1e-4);
  if (h) {
    const H = unflattenH(h);
    const Ktnew = extractK(H);
    const Knew = denormK(Ktnew);
    let err = 0;
    for (let i = 0; i < 4; i++) err += (Knew[i] - LQR_K[i]) ** 2;
    const fmt = a => a.map(x => x.toFixed(2).padStart(8)).join(" ");
    console.log(`  K_new = [${fmt(Knew)}]`);
    console.log(`  K_LQR = [${fmt(LQR_K)}]`);
    console.log(`  rel err: ${(Math.sqrt(err)/LQR_NORM*100).toFixed(1)}%, H̃uu=${H[4][4].toFixed(3)}\n`);
  } else {
    console.log("  LSTDQ solve failed\n");
  }
}

// ================================================================
// Online LSPI: between policy iterations, RE-COLLECT samples under
// the improved policy. This is the proper LSPI cycle and mirrors
// the in-browser streaming scenario. Data buffer grows over time.
// ================================================================
function runOnlineLSPI(totalSamples, perRound, sigma) {
  console.log(`-- total=${totalSamples}, per round=${perRound}, σ=${sigma}, init K=0 --`);
  let K = [0, 0, 0, 0];
  let Kt = [0, 0, 0, 0];
  const buf = [];
  let round = 0;
  while (buf.length < totalSamples) {
    round++;
    const batch = collectSamples(K, perRound, sigma);
    for (const s of batch) buf.push(s);
    const h = lstdq(buf, Kt, 0.99, 1e-4);
    if (!h) { console.log(`  round ${round}: LSTDQ failed`); break; }
    const H = unflattenH(h);
    const Ktnew = extractK(H);
    if (!Ktnew) { console.log(`  round ${round}: H_uu degenerate`); break; }

    // Damped policy update (prevents overshoot on noisy gradient).
    const alpha = 0.5;
    for (let i = 0; i < 4; i++) Kt[i] = (1 - alpha) * Kt[i] + alpha * Ktnew[i];
    K = denormK(Kt);

    let err = 0;
    for (let i = 0; i < 4; i++) err += (K[i] - LQR_K[i]) ** 2;
    err = Math.sqrt(err) / LQR_NORM;
    const fmt = a => a.map(x => x.toFixed(2).padStart(8)).join(" ");
    console.log(`  round ${String(round).padStart(3)} N=${String(buf.length).padStart(5)}  K=[${fmt(K)}]  err=${(err*100).toFixed(1).padStart(5)}%  H̃uu=${H[4][4].toFixed(2)}`);
  }
  let err = 0;
  for (let i = 0; i < 4; i++) err += (K[i] - LQR_K[i]) ** 2;
  console.log(`  >> final rel err: ${(Math.sqrt(err) / LQR_NORM * 100).toFixed(1)}%\n`);
  return K;
}

runOnlineLSPI(2000, 50, 15.0);
runOnlineLSPI(4000, 100, 15.0);
runOnlineLSPI(4000, 200, 10.0);
runOnlineLSPI(8000, 200, 10.0);

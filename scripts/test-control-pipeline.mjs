// Validate the full control pipeline:
//   alpha → linearize-at-inverted → DARE → K
// then run a closed-loop sim with the recovered K and verify the
// inverted equilibrium is stabilized.
//
// Expected: with TRUE alpha, the recovered K should be close to the
// hand-tuned LQR gains [-41.64, 101.62, -17.31, 29.14] used by the
// BC version. With LEARNED alpha (from short sim window), K should
// also stabilize, even if numerically different.

const M1=1, M2=1, L1=1, L2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;

const ALPHA_TRUE = [
  (M1+M2)*L1*L1,    // α₁ = 2
  M2*L2*L2,         // α₂ = 1
  M2*L1*L2,         // α₃ = 1
  (M1+M2)*G*L1,     // α₄ = 19.62
  M2*G*L2,          // α₅ = 9.81
  DAMPING,
];

// True simulator (same as PendulumScene).
function deriv(s, u) {
  const { th1, th2, w1, w2 } = s;
  const d = th1 - th2, cd = Math.cos(d), sd = Math.sin(d);
  const M11 = (M1+M2)*L1*L1, M12 = M2*L1*L2*cd, M22 = M2*L2*L2;
  const det = M11*M22 - M12*M12;
  const C1 = M2*L1*L2*sd*w2*w2;
  const C2 = -M2*L1*L2*sd*w1*w1;
  const G1 = (M1+M2)*G*L1*Math.sin(th1);
  const G2 = M2*G*L2*Math.sin(th2);
  const r1 = u - DAMPING*w1 - C1 - G1;
  const r2 = -DAMPING*w2 - C2 - G2;
  return [w1, w2, (M22*r1 - M12*r2)/det, (-M12*r1 + M11*r2)/det];
}
function stepEnv(s, u, dt=DT) {
  const k1 = deriv(s, u);
  const s2 = { th1:s.th1+k1[0]*dt/2, th2:s.th2+k1[1]*dt/2, w1:s.w1+k1[2]*dt/2, w2:s.w2+k1[3]*dt/2 };
  const k2 = deriv(s2, u);
  const s3 = { th1:s.th1+k2[0]*dt/2, th2:s.th2+k2[1]*dt/2, w1:s.w1+k2[2]*dt/2, w2:s.w2+k2[3]*dt/2 };
  const k3 = deriv(s3, u);
  const s4 = { th1:s.th1+k3[0]*dt, th2:s.th2+k3[1]*dt, w1:s.w1+k3[2]*dt, w2:s.w2+k3[3]*dt };
  const k4 = deriv(s4, u);
  return {
    th1: s.th1 + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])*dt/6,
    th2: s.th2 + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])*dt/6,
    w1:  s.w1  + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])*dt/6,
    w2:  s.w2  + (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])*dt/6,
  };
}
const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };

// Linearize the EOM around the inverted equilibrium θ₁=0, θ₂=π:
//   state x = [θ₁, δθ₂, ω₁, ω₂],  δθ₂ = θ₂ − π
// At equilibrium: cos(d)=cos(−π)=−1, sin terms vanish, ω=0.
//   M = [α₁, −α₃; −α₃, α₂], det = α₁α₂ − α₃²
//   G linearized: ∂G₁/∂θ₁ = α₄, ∂G₂/∂δθ₂ = −α₅
//   damping → diagonal D
// Solving M q̈ + Dq̇ + Gx = (u, 0):
//   θ̈₁ =  (1/det)·(−α₂α₄·θ₁ + α₃α₅·δθ₂ − α₂D·ω₁ − α₃D·ω₂ + α₂·u)
//   δ̈θ₂ = (1/det)·(−α₃α₄·θ₁ + α₁α₅·δθ₂ − α₃D·ω₁ − α₁D·ω₂ + α₃·u)
function linearizeAtInverted(alpha) {
  const [a1, a2, a3, a4, a5, D] = alpha;
  const det = a1*a2 - a3*a3;
  const A = [
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-a2*a4/det,  a3*a5/det, -a2*D/det, -a3*D/det],
    [-a3*a4/det,  a1*a5/det, -a3*D/det, -a1*D/det],
  ];
  const B = [0, 0, a2/det, a3/det];
  return { A, B };
}

// Continuous algebraic Riccati equation, solved via the differential
// Riccati equation integrated forward from P=0 until steady state:
//   dP/dτ = A'P + PA − PB R⁻¹ B'P + Q
// Steady-state solution P satisfies CARE; gain K = R⁻¹ B'P (1×4).
//
// For B as a 4×1 column and R scalar, the term PBR⁻¹B'P is the
// outer product (PB)(PB)'/R.
function solveCARE(A, B, Q, R) {
  const N = 4;
  const dt = 0.001;
  let P = Array.from({length:N}, () => new Array(N).fill(0));
  for (let iter = 0; iter < 60000; iter++) {
    // PB (4)
    const PB = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let k = 0; k < N; k++) s += P[i][k] * B[k];
      PB[i] = s;
    }
    // A'P (4×4)
    const AtP = Array.from({length:N}, () => new Array(N).fill(0));
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < N; k++) s += A[k][i] * P[k][j];
        AtP[i][j] = s;
      }
    // PA (4×4) = (A'P)' since P symmetric and (PA)_{ij}=Σ P_{ik}A_{kj}
    // We compute it directly from P, A:
    const PA = Array.from({length:N}, () => new Array(N).fill(0));
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        let s = 0;
        for (let k = 0; k < N; k++) s += P[i][k] * A[k][j];
        PA[i][j] = s;
      }
    // dP/dτ
    let maxDelta = 0;
    const Pnew = Array.from({length:N}, () => new Array(N).fill(0));
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const dpij = AtP[i][j] + PA[i][j] - (PB[i] * PB[j]) / R + Q[i][j];
        Pnew[i][j] = P[i][j] + dt * dpij;
        const d = Math.abs(dpij);
        if (d > maxDelta) maxDelta = d;
      }
    // Symmetrize each step (numerical drift can break symmetry)
    for (let i = 0; i < N; i++)
      for (let j = i+1; j < N; j++) {
        const m = 0.5*(Pnew[i][j] + Pnew[j][i]);
        Pnew[i][j] = m;
        Pnew[j][i] = m;
      }
    P = Pnew;
    if (maxDelta < 1e-9) break;
  }
  // K = R⁻¹ B'P
  const K = new Array(N).fill(0);
  for (let j = 0; j < N; j++) {
    let s = 0;
    for (let i = 0; i < N; i++) s += B[i] * P[i][j];
    K[j] = s / R;
  }
  for (const k of K) if (!Number.isFinite(k)) return null;
  return K;
}

function computeKFromAlpha(alpha) {
  const { A, B } = linearizeAtInverted(alpha);
  const Q = [
    [10, 0, 0, 0],
    [0, 10, 0, 0],
    [0, 0,  1, 0],
    [0, 0,  0, 1],
  ];
  const R = 0.1;
  return solveCARE(A, B, Q, R);
}

function lqrFn(K) {
  return s => {
    const dth2 = wrap(s.th2 - Math.PI);
    const u = -(K[0]*s.th1 + K[1]*dth2 + K[2]*s.w1 + K[3]*s.w2);
    return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
  };
}

function simulate(K, init, maxSec=4) {
  const ctrl = lqrFn(K);
  let s = { ...init };
  const steps = maxSec / DT;
  for (let i = 0; i < steps; i++) {
    s = stepEnv(s, ctrl(s));
    if (!Number.isFinite(s.th1)) return { stable: false, finalErr: Infinity };
  }
  const dth2 = Math.abs(wrap(s.th2 - Math.PI));
  const totErr = Math.abs(s.th1) + dth2 + Math.abs(s.w1)*0.1 + Math.abs(s.w2)*0.1;
  return { stable: dth2 < 0.02 && Math.abs(s.th1) < 0.05, finalErr: totErr };
}

console.log("=== test 1: K from TRUE alpha ===");
console.log(`hand-tuned BC LQR_K = [-41.64, 101.62, -17.31, 29.14]`);
const K_true = computeKFromAlpha(ALPHA_TRUE);
console.log(`derived K          = [${K_true.map(x => x.toFixed(2)).join(", ")}]`);

console.log("\n=== test 2: closed-loop balance from various ICs near inverted ===");
for (const ic of [
  { th1: 0,    th2: Math.PI,        w1: 0,   w2: 0 },
  { th1: 0.1,  th2: Math.PI - 0.1,  w1: 0,   w2: 0 },
  { th1: 0.2,  th2: Math.PI + 0.2,  w1: 0.5, w2: 0 },
  { th1: -0.3, th2: Math.PI - 0.4,  w1: 0,   w2: 0.5 },
  { th1: 0.4,  th2: Math.PI + 0.5,  w1: 1,   w2: 1 },
]) {
  const r = simulate(K_true, ic);
  console.log(`  ic=(${ic.th1.toFixed(2)}, π+${(ic.th2-Math.PI).toFixed(2)}, ${ic.w1.toFixed(1)}, ${ic.w2.toFixed(1)})  stable=${r.stable}  finalErr=${r.finalErr.toFixed(4)}`);
}

console.log("\n=== test 3: K from PERTURBED alpha (5% noise on each param) ===");
const ALPHA_NOISY = ALPHA_TRUE.map(a => a * (1 + (Math.random()-0.5)*0.1));
console.log(`alpha_noisy = [${ALPHA_NOISY.map(x => x.toFixed(3)).join(", ")}]`);
const K_noisy = computeKFromAlpha(ALPHA_NOISY);
console.log(`K_noisy     = [${K_noisy.map(x => x.toFixed(2)).join(", ")}]`);
const r_noisy = simulate(K_noisy, { th1: 0.2, th2: Math.PI + 0.2, w1: 0, w2: 0 });
console.log(`balance from (0.2, π+0.2, 0, 0):  stable=${r_noisy.stable}  finalErr=${r_noisy.finalErr.toFixed(4)}`);

// ---------- full end-to-end pipeline test ----------
console.log("\n=== test 4: end-to-end — identify α, derive K, balance ===");

function regressorRows(s, ddth1, ddth2, u) {
  const d = s.th1 - s.th2, cd = Math.cos(d), sd = Math.sin(d);
  const row1 = [ddth1, 0,       cd*ddth2 + sd*s.w2*s.w2, Math.sin(s.th1), 0,               s.w1];
  const row2 = [0,     ddth2,   cd*ddth1 - sd*s.w1*s.w1, 0,               Math.sin(s.th2), s.w2];
  return { row1, row2, b1: u, b2: 0 };
}
function solveLinear(A, b) {
  const N = A.length;
  const M = A.map((row,i) => [...row, b[i]]);
  for (let i = 0; i < N; i++) {
    let maxRow = i;
    for (let k = i+1; k < N; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];
    if (Math.abs(M[i][i]) < 1e-14) return null;
    for (let k = i+1; k < N; k++) {
      const f = M[k][i] / M[i][i];
      for (let j = i; j <= N; j++) M[k][j] -= f * M[i][j];
    }
  }
  const x = new Array(N).fill(0);
  for (let i = N-1; i >= 0; i--) {
    let s = M[i][N];
    for (let j = i+1; j < N; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}
function ridgeSolve(rows, rhs, lambda) {
  const D = 6;
  const YtY = Array.from({length:D}, () => new Array(D).fill(0));
  const Ytb = new Array(D).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i], b = rhs[i];
    for (let j = 0; j < D; j++) {
      Ytb[j] += r[j] * b;
      for (let k = 0; k < D; k++) YtY[j][k] += r[j] * r[k];
    }
  }
  for (let j = 0; j < D; j++) YtY[j][j] += lambda;
  return solveLinear(YtY, Ytb);
}

// Generate 200 samples with the same augmentation profile we expect
// in-browser: mostly free-swing, with occasional known torque pulses.
function collectIDSamples(nSteps, fracTorque=0.25) {
  const rows = [], rhs = [];
  let s = { th1: (Math.random()-0.5)*Math.PI, th2: (Math.random()-0.5)*Math.PI, w1: 0, w2: 0 };
  const traj = [s];
  const us = [];
  while (rows.length / 2 < nSteps) {
    const u = Math.random() < fracTorque ? (Math.random()-0.5)*20 : 0;
    us.push(u);
    s = stepEnv(s, u);
    traj.push(s);
    if (traj.length >= 3) {
      const t = traj.length - 2;
      const ddth1 = (traj[t+1].w1 - traj[t-1].w1) / (2*DT);
      const ddth2 = (traj[t+1].w2 - traj[t-1].w2) / (2*DT);
      const { row1, row2, b1, b2 } = regressorRows(traj[t], ddth1, ddth2, us[t]);
      rows.push(row1); rhs.push(b1);
      rows.push(row2); rhs.push(b2);
    }
    if (Math.abs(s.w1) > 40 || Math.abs(s.w2) > 40 || !Number.isFinite(s.th1)) {
      s = { th1:(Math.random()-0.5)*Math.PI, th2:(Math.random()-0.5)*Math.PI, w1:0, w2:0 };
      traj.length = 0; us.length = 0; traj.push(s);
    }
  }
  return { rows, rhs };
}

const E_TARGET_FROM_ALPHA = alpha => -alpha[3] + alpha[4];
function energyFromAlpha(s, alpha) {
  const [a1, a2, a3, a4, a5] = alpha;
  const KE = 0.5*a1*s.w1*s.w1 + 0.5*a2*s.w2*s.w2 + a3*s.w1*s.w2*Math.cos(s.th1 - s.th2);
  const PE = -a4*Math.cos(s.th1) - a5*Math.cos(s.th2);
  return KE + PE;
}
function swingUpFromAlpha(s, alpha, ke=24, kp=1, kd=0.5) {
  const Etilde = energyFromAlpha(s, alpha) - E_TARGET_FROM_ALPHA(alpha);
  const u = ke * (-Etilde) * s.w1 - kp*Math.sin(s.th1) - kd*s.w1;
  return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
}
function inBasinK(s, K) {
  let t = { ...s };
  const ctrl = lqrFn(K);
  for (let i = 0; i < 80; i++) {
    t = stepEnv(t, ctrl(t));
    if (!Number.isFinite(t.th1)) return false;
    if (Math.abs(wrap(t.th2 - Math.PI)) > 1.4) return false;
  }
  return Math.abs(wrap(t.th2 - Math.PI)) < 0.25 && Math.abs(t.w1) < 4 && Math.abs(t.w2) < 4;
}
function runFullControl(alpha, K, init, maxSec=15) {
  const ctrlLqr = lqrFn(K);
  let s = { ...init };
  const steps = maxSec / DT;
  for (let i = 0; i < steps; i++) {
    const u = inBasinK(s, K) ? ctrlLqr(s) : swingUpFromAlpha(s, alpha);
    s = stepEnv(s, u);
    if (!Number.isFinite(s.th1)) return null;
    const d = Math.abs(wrap(s.th2 - Math.PI));
    if (d < 0.05 && Math.abs(s.w1) < 0.3 && Math.abs(s.w2) < 0.3) return i * DT;
  }
  return null;
}

// Identify α from 200 samples (4s of data at 50 Hz)
for (const N of [100, 200, 500, 1500]) {
  const { rows, rhs } = collectIDSamples(N, 0.25);
  const alpha = ridgeSolve(rows, rhs, 1e-4);
  const errs = alpha.map((x, i) => Math.abs(x - ALPHA_TRUE[i]) / Math.abs(ALPHA_TRUE[i]) * 100);
  const K = computeKFromAlpha(alpha);
  let nSuccess = 0;
  const times = [];
  for (let trial = 0; trial < 20; trial++) {
    const ic = { th1:(Math.random()-0.5)*0.6, th2:(Math.random()-0.5)*0.6, w1:0, w2:0 };
    const t = runFullControl(alpha, K, ic);
    if (t !== null) { nSuccess++; times.push(t); }
  }
  times.sort((a,b)=>a-b);
  const median = times.length ? times[Math.floor(times.length/2)].toFixed(1)+"s" : "-";
  console.log(`N=${String(N).padStart(4)}:  α err avg=${(errs.slice(0,5).reduce((a,b)=>a+b)/5).toFixed(2)}%  K=[${K.map(x=>x.toFixed(1)).join(",")}]  swing-up rate=${(nSuccess/20*100).toFixed(0)}%  median=${median}`);
}

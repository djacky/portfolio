// System identification + Discrete Algebraic Riccati Equation (DARE).
//
// Pipeline:
//   1. Collect (x, u, x') transitions from the nonlinear pendubot near
//      the inverted equilibrium (θ₂ = π). In the browser these come
//      from the user-grabbed pendulum; here we simulate realistic grabs.
//   2. Fit linear discrete-time model  x' ≈ A x + B u  via ridge LS.
//   3. Solve DARE for (A, B, Q, R) to get K.
//   4. Verify K ≈ LQR_K.
//
// This is model-based only in that (A, B) are LEARNED FROM REAL DATA —
// no random-IC shadow rollouts. The learning signal is 100% real
// pendulum movement.

const L1=1, L2=1, M1=1, M2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;
const LQR_K = [-41.64, 101.62, -17.31, 29.14];
const LQR_NORM = Math.sqrt(LQR_K.reduce((a,b)=>a+b*b, 0));

const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };
function deriv(s,u){const{th1,th2,w1,w2}=s;const d=th1-th2,cd=Math.cos(d),sd=Math.sin(d);const M11=(M1+M2)*L1*L1,M12=M2*L1*L2*cd,M22=M2*L2*L2,det=M11*M22-M12*M12;const C1=M2*L1*L2*sd*w2*w2,C2=-M2*L1*L2*sd*w1*w1;const G1=(M1+M2)*G*L1*Math.sin(th1),G2=M2*G*L2*Math.sin(th2);const r1=u-DAMPING*w1-C1-G1,r2=-DAMPING*w2-C2-G2;return [w1,w2,(M22*r1-M12*r2)/det,(-M12*r1+M11*r2)/det];}
function stepEnv(s,u,dt=DT){const k1=deriv(s,u);const s2={th1:s.th1+k1[0]*dt/2,th2:s.th2+k1[1]*dt/2,w1:s.w1+k1[2]*dt/2,w2:s.w2+k1[3]*dt/2};const k2=deriv(s2,u);const s3={th1:s.th1+k2[0]*dt/2,th2:s.th2+k2[1]*dt/2,w1:s.w1+k2[2]*dt/2,w2:s.w2+k2[3]*dt/2};const k3=deriv(s3,u);const s4={th1:s.th1+k3[0]*dt,th2:s.th2+k3[1]*dt,w1:s.w1+k3[2]*dt,w2:s.w2+k3[3]*dt};const k4=deriv(s4,u);return{th1:s.th1+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,th2:s.th2+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6,w1:s.w1+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*dt/6,w2:s.w2+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*dt/6};}

// Local coordinates around inverted equilibrium: x = [θ₁, θ₂−π, ω₁, ω₂]
const featuresOf = s => [s.th1, wrap(s.th2 - Math.PI), s.w1, s.w2];

function gauss() {
  const u1 = Math.max(1e-9, Math.random()), u2 = Math.random();
  return Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2);
}

// ================================================================
// Matrix utilities
// ================================================================
const mat = (r, c) => Array.from({length: r}, () => new Array(c).fill(0));

function matMul(A, B) {
  const r = A.length, c = B[0].length, K = B.length;
  const C = mat(r, c);
  for (let i = 0; i < r; i++) {
    for (let k = 0; k < K; k++) {
      const a = A[i][k];
      if (a === 0) continue;
      for (let j = 0; j < c; j++) C[i][j] += a * B[k][j];
    }
  }
  return C;
}

function matT(A) {
  const r = A.length, c = A[0].length;
  const B = mat(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) B[j][i] = A[i][j];
  return B;
}

// Gauss elimination with partial pivoting. Solves A x = b.
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map(r => r.slice());
  const y = b.slice();
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[pivot][i])) pivot = k;
    }
    if (pivot !== i) { [M[i], M[pivot]] = [M[pivot], M[i]]; [y[i], y[pivot]] = [y[pivot], y[i]]; }
    if (Math.abs(M[i][i]) < 1e-14) return null;
    for (let k = i + 1; k < n; k++) {
      const c = M[k][i] / M[i][i];
      for (let j = i; j < n; j++) M[k][j] -= c * M[i][j];
      y[k] -= c * y[i];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// ================================================================
// System ID: fit x' ≈ A x + B u via ridge LS.
//
// Stack φ = [x; u] ∈ ℝ⁵. For each output dim k ∈ {0..3},
// solve  (Φ'Φ + λI) θ_k = Φ' y_k   where y_k = x'[k].
// Then A[k][:] = θ_k[0..3], B[k] = θ_k[4].
// ================================================================
function fitAB(samples, lambda) {
  const D = 5;
  const XtX = mat(D, D);
  const XtY = mat(D, 4);
  for (const { x, u, xn } of samples) {
    const z = [x[0], x[1], x[2], x[3], u];
    for (let i = 0; i < D; i++) {
      const zi = z[i];
      for (let j = 0; j < D; j++) XtX[i][j] += zi * z[j];
      for (let j = 0; j < 4; j++)  XtY[i][j] += zi * xn[j];
    }
  }
  for (let i = 0; i < D; i++) XtX[i][i] += lambda;
  const A = mat(4, 4);
  const B = new Array(4).fill(0);
  for (let k = 0; k < 4; k++) {
    const rhs = XtY.map(row => row[k]);
    const theta = solveLinear(XtX, rhs);
    if (!theta) return null;
    for (let j = 0; j < 4; j++) A[k][j] = theta[j];
    B[k] = theta[4];
  }
  return { A, B };
}

// ================================================================
// DARE via fixed-point Riccati iteration (scalar u).
//
//   P_{k+1} = Q + A'PA − A'Pb (R + b'Pb)⁻¹ b'PA
//   K       = (R + b'Pb)⁻¹ b'PA
//
// Stops when ‖P_{k+1} − P_k‖_max < tol.
// ================================================================
function dare(A, b, Q, R, maxIter = 10000, tol = 1e-11) {
  let P = Q.map(row => row.slice());
  for (let it = 0; it < maxIter; it++) {
    // BtPB = b'Pb (scalar)
    let BtPB = 0;
    const Pb = new Array(4).fill(0);
    for (let i = 0; i < 4; i++) {
      let s = 0;
      for (let j = 0; j < 4; j++) s += P[i][j] * b[j];
      Pb[i] = s;
      BtPB += b[i] * s;
    }
    // APb[i] = (A'Pb)[i] = Σ_k A[k][i] * Pb[k]
    const APb = new Array(4).fill(0);
    for (let i = 0; i < 4; i++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[k][i] * Pb[k];
      APb[i] = s;
    }
    // PA[i][j] = Σ_k P[i][k] A[k][j]
    const PA = mat(4, 4);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += P[i][k] * A[k][j];
        PA[i][j] = s;
      }
    }
    // APA[i][j] = (A'PA)[i][j] = Σ_k A[k][i] * PA[k][j]
    const APA = mat(4, 4);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += A[k][i] * PA[k][j];
        APA[i][j] = s;
      }
    }
    // BtPA[j] = b'PA[:,j] = Σ_i b[i] * PA[i][j]
    const BtPA = new Array(4).fill(0);
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let i = 0; i < 4; i++) s += b[i] * PA[i][j];
      BtPA[j] = s;
    }
    const inv = 1 / (R + BtPB);
    // P_new[i][j] = Q[i][j] + APA[i][j] − APb[i] * inv * BtPA[j]
    let maxDiff = 0;
    const Pnew = mat(4, 4);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        Pnew[i][j] = Q[i][j] + APA[i][j] - APb[i] * inv * BtPA[j];
        const d = Math.abs(Pnew[i][j] - P[i][j]);
        if (d > maxDiff) maxDiff = d;
      }
    }
    P = Pnew;
    if (maxDiff < tol) break;
  }
  // Final K = inv(R + b'Pb) * b'PA
  let BtPB = 0;
  const Pb = new Array(4).fill(0);
  for (let i = 0; i < 4; i++) {
    let s = 0;
    for (let j = 0; j < 4; j++) s += P[i][j] * b[j];
    Pb[i] = s;
    BtPB += b[i] * s;
  }
  const BtPA = new Array(4).fill(0);
  for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let i = 0; i < 4; i++) {
      let t = 0;
      for (let k = 0; k < 4; k++) t += P[i][k] * A[k][j];
      s += b[i] * t;
    }
    BtPA[j] = s;
  }
  const inv = 1 / (R + BtPB);
  return { K: BtPA.map(x => inv * x), P };
}

// ================================================================
// Sanity check: use the ANALYTICAL linearization of the pendubot
// at the inverted equilibrium and run DARE. Should match LQR_K.
// ================================================================
function analyticalAB() {
  // Compute numerical Jacobian of one Euler-forward step around
  // x = 0, u = 0. This mirrors what a perfect system ID would recover.
  const eq = { th1: 0, th2: Math.PI, w1: 0, w2: 0 };
  const base = featuresOf(stepEnv(eq, 0));
  const A = mat(4, 4);
  const B = new Array(4).fill(0);
  const eps = 1e-5;
  // ∂xn/∂x[j]
  for (let j = 0; j < 4; j++) {
    const s = { ...eq };
    if (j === 0) s.th1 += eps;
    else if (j === 1) s.th2 += eps;
    else if (j === 2) s.w1 += eps;
    else if (j === 3) s.w2 += eps;
    const xnp = featuresOf(stepEnv(s, 0));
    for (let i = 0; i < 4; i++) A[i][j] = (xnp[i] - base[i]) / eps;
  }
  // ∂xn/∂u
  const xnp = featuresOf(stepEnv(eq, eps));
  for (let i = 0; i < 4; i++) B[i] = (xnp[i] - base[i]) / eps;
  return { A, B };
}

// ================================================================
// Realistic data collection: simulates "user grabs pendulum at a
// random state in the linear-ish regime, lets it evolve with small
// random torque for a few steps, then regrabs". This mimics the
// stream of (x, u, x') tuples the browser demo will see.
// ================================================================
function randomGrabIC() {
  return {
    th1: (Math.random() - 0.5) * 0.8,
    th2: Math.PI + (Math.random() - 0.5) * 0.8,
    w1:  (Math.random() - 0.5) * 3,
    w2:  (Math.random() - 0.5) * 3,
  };
}

function inLinearRegime(x) {
  return Math.abs(x[0]) < 0.8 && Math.abs(x[1]) < 0.8 &&
         Math.abs(x[2]) < 4 && Math.abs(x[3]) < 4 && Number.isFinite(x[0]);
}

// Tighter grab region: the pendulum demo's user-grab behavior usually
// stays within ~0.3 rad of the upright config. And we keep rollouts short
// (2-3 steps per grab) so samples don't drift into the nonlinear regime.
function tightGrabIC() {
  return {
    th1: (Math.random() - 0.5) * 0.4,
    th2: Math.PI + (Math.random() - 0.5) * 0.4,
    w1:  (Math.random() - 0.5) * 2,
    w2:  (Math.random() - 0.5) * 2,
  };
}

function collectRealTransitions(N, sigma, stepsPerGrab = 3) {
  const buf = [];
  let s = tightGrabIC();
  let stepCount = 0;
  let tries = 0;
  while (buf.length < N && tries < N * 30) {
    tries++;
    const x = featuresOf(s);
    if (!inLinearRegime(x)) { s = tightGrabIC(); stepCount = 0; continue; }
    let u = sigma * gauss();
    if (u > TORQUE_MAX) u = TORQUE_MAX;
    else if (u < -TORQUE_MAX) u = -TORQUE_MAX;
    const next = stepEnv(s, u);
    const xn = featuresOf(next);
    if (!inLinearRegime(xn)) { s = tightGrabIC(); stepCount = 0; continue; }
    buf.push({ x, u, xn });
    s = next;
    stepCount++;
    if (stepCount >= stepsPerGrab) { s = tightGrabIC(); stepCount = 0; }
  }
  return buf;
}

function relErr(K, ref) {
  const t = ref || LQR_K;
  let e = 0, n = 0;
  for (let i = 0; i < 4; i++) { e += (K[i] - t[i]) ** 2; n += t[i] * t[i]; }
  return Math.sqrt(e) / Math.sqrt(n);
}

const fmt4 = a => a.map(x => x.toFixed(2).padStart(8)).join(" ");
const fmt44 = A => A.map(row => "  [" + row.map(x => x.toFixed(4).padStart(9)).join(" ") + "]").join("\n");

const Q = [
  [10, 0, 0, 0],
  [0, 10, 0, 0],
  [0, 0,  1, 0],
  [0, 0,  0, 1],
];
const R = 0.1;

console.log("=== System ID + DARE verification ===");
console.log(`LQR_K reference: [${fmt4(LQR_K)}]`);
console.log(`||K_LQR|| = ${LQR_NORM.toFixed(2)}\n`);

// ---------------------------------------------------------------
// Test 0: analytical linearization + DARE. The trick: to match
// CONTINUOUS-time LQR gains via discrete DARE, use Q_d = Q·DT and
// R_d = R·DT — this is the trapezoidal approximation of
// J = ∫ (x'Qx + u'Ru) dt as a discrete Σ · DT.
// ---------------------------------------------------------------
console.log("-- Test 0a: analytical + DARE, Q_d = Q, R_d = R (pure discrete) --");
const { A: A_true, B: B_true } = analyticalAB();
console.log("A_true =");
console.log(fmt44(A_true));
console.log(`B_true = [${B_true.map(x => x.toFixed(4)).join(", ")}]`);
{
  const { K } = dare(A_true, B_true, Q, R);
  console.log(`K    = [${fmt4(K)}]`);
  console.log(`vs LQR_K: ${(relErr(K, LQR_K)*100).toFixed(2)}%\n`);
}

console.log("-- Test 0b: analytical + DARE, Q_d = Q·DT, R_d = R·DT (continuous-equiv) --");
const Q_dt = Q.map(row => row.map(x => x * DT));
const R_dt = R * DT;
const { K: K_ref } = dare(A_true, B_true, Q_dt, R_dt);
console.log(`K_ref = [${fmt4(K_ref)}]`);
console.log(`K_LQR = [${fmt4(LQR_K)}]`);
console.log(`vs LQR_K: ${(relErr(K_ref, LQR_K)*100).toFixed(2)}%\n`);

// All subsequent tests use the Q·DT / R·DT convention and measure
// against K_ref (the analytical-limit DARE solution for this setup).

// ---------------------------------------------------------------
// Test 1: system ID on N real transitions, DARE on learned model.
// Averages over multiple seeds to smooth out stochastic noise.
// ---------------------------------------------------------------
function testTrial(N, sigma, seeds = 10) {
  const errs = [];
  const errsLQR = [];
  for (let s = 0; s < seeds; s++) {
    const samples = collectRealTransitions(N, sigma);
    const fit = fitAB(samples, 1e-6);
    if (!fit) continue;
    const { K } = dare(fit.A, fit.B, Q_dt, R_dt);
    if (!K.every(Number.isFinite)) continue;
    errs.push(relErr(K, K_ref));
    errsLQR.push(relErr(K, LQR_K));
  }
  errs.sort((a,b)=>a-b);
  errsLQR.sort((a,b)=>a-b);
  const median = errs[Math.floor(errs.length/2)];
  const medianLQR = errsLQR[Math.floor(errsLQR.length/2)];
  const worst = errs[errs.length - 1];
  const best = errs[0];
  return { median, medianLQR, best, worst, n: errs.length };
}

const trials = [
  { N: 20,   sigma: 10 },
  { N: 30,   sigma: 10 },
  { N: 50,   sigma: 10 },
  { N: 100,  sigma: 10 },
  { N: 200,  sigma: 10 },
  { N: 500,  sigma: 10 },
  { N: 50,   sigma: 5  },
  { N: 50,   sigma: 15 },
  { N: 50,   sigma: 20 },
];

console.log("-- Test 1: system ID + DARE on real transitions (10 seeds, reported vs K_ref) --");
for (const { N, sigma } of trials) {
  const r = testTrial(N, sigma, 10);
  console.log(`  N=${String(N).padStart(4)}  σ=${String(sigma).padStart(3)}  ` +
    `median=${(r.median*100).toFixed(1).padStart(5)}%  ` +
    `best=${(r.best*100).toFixed(1).padStart(5)}%  ` +
    `worst=${(r.worst*100).toFixed(1).padStart(5)}%  ` +
    `(vs LQR_K: ${(r.medianLQR*100).toFixed(1)}%)`);
}

// ---------------------------------------------------------------
// Test 2: streaming / online. Refit + DARE at milestones as new
// samples arrive. Mirrors exactly what the browser demo will do.
// Averaged over 10 seeds.
// ---------------------------------------------------------------
console.log("\n-- Test 2: streaming online system ID + DARE (10 seeds) --");
{
  const milestones = [10, 20, 30, 50, 75, 100, 150, 200, 300];
  const accum = milestones.map(() => []);
  for (let seed = 0; seed < 10; seed++) {
    const buf = [];
    let s = tightGrabIC();
    let stepCount = 0;
    let mi = 0;
    while (buf.length < 300) {
      const x = featuresOf(s);
      if (!inLinearRegime(x)) { s = tightGrabIC(); stepCount = 0; continue; }
      const u = Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, 10 * gauss()));
      const next = stepEnv(s, u);
      const xn = featuresOf(next);
      if (!inLinearRegime(xn)) { s = tightGrabIC(); stepCount = 0; continue; }
      buf.push({ x, u, xn });
      s = next;
      stepCount++;
      if (stepCount >= 3) { s = tightGrabIC(); stepCount = 0; }
      if (mi < milestones.length && buf.length === milestones[mi]) {
        const fit = fitAB(buf, 1e-6);
        if (fit) {
          const { K } = dare(fit.A, fit.B, Q_dt, R_dt);
          if (K.every(Number.isFinite)) accum[mi].push(relErr(K, K_ref));
        }
        mi++;
      }
    }
  }
  for (let i = 0; i < milestones.length; i++) {
    const arr = accum[i].slice().sort((a,b)=>a-b);
    if (arr.length === 0) continue;
    const med = arr[Math.floor(arr.length/2)];
    const best = arr[0], worst = arr[arr.length-1];
    console.log(`  N=${String(milestones[i]).padStart(4)}  ` +
      `median=${(med*100).toFixed(1).padStart(5)}%  ` +
      `best=${(best*100).toFixed(1).padStart(5)}%  ` +
      `worst=${(worst*100).toFixed(1).padStart(5)}%`);
  }
}

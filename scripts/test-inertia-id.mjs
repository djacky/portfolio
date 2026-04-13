// Linear-in-parameter inertia identification for the pendubot.
//
// The double-pendulum EOM
//     M(q) q̈ + C(q,q̇) + G(q) + D q̇ = (u, 0)
// is LINEAR in the inertia/gravity parameters
//     α = [α₁, α₂, α₃, α₄, α₅, D]
//   = [(m₁+m₂)L₁², m₂L₂², m₂L₁L₂, (m₁+m₂)gL₁, m₂gL₂, damping]
//
// Each sample (q, q̇, q̈, u) gives a 2-row block of a regressor matrix
// Y(q,q̇,q̈) and a 2-element RHS (u, 0). Stack and solve ridge LS:
//     θ̂ = (YᵀY + λI)⁻¹ Yᵀb
//
// Key wrinkle: free-swing samples (u = 0) give a homogeneous block,
// so they constrain the *direction* of α but not its scale. We need
// at least some samples with u ≠ 0 to pin the absolute magnitude of
// the inertia (otherwise masses are undetermined; lengths are still
// recoverable from gravity-term ratios).

const M1=1, M2=1, L1=1, L2=1, G=9.81, DAMPING=0.0015, DT=0.02;

// Ground truth: what we should recover.
const ALPHA_TRUE = [
  (M1+M2)*L1*L1,    // α₁ = 2
  M2*L2*L2,         // α₂ = 1
  M2*L1*L2,         // α₃ = 1
  (M1+M2)*G*L1,     // α₄ = 19.62
  M2*G*L2,          // α₅ = 9.81
  DAMPING,          // D  = 0.0015
];

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

// 2-row regressor block for one sample (q, q̇, q̈, u_applied).
//   Row 1 (joint 1):
//       α₁·θ̈₁ + α₃·(cd·θ̈₂ + sd·ω₂²) + α₄·sin θ₁ + D·ω₁ = u
//   Row 2 (joint 2):
//       α₂·θ̈₂ + α₃·(cd·θ̈₁ − sd·ω₁²) + α₅·sin θ₂ + D·ω₂ = 0
function regressorRows(s, ddth1, ddth2, u) {
  const d = s.th1 - s.th2, cd = Math.cos(d), sd = Math.sin(d);
  const row1 = [ddth1, 0,            cd*ddth2 + sd*s.w2*s.w2, Math.sin(s.th1), 0,             s.w1];
  const row2 = [0,     ddth2,        cd*ddth1 - sd*s.w1*s.w1, 0,               Math.sin(s.th2), s.w2];
  return { row1, row2, b1: u, b2: 0 };
}

// Standard Gauss elimination with partial pivoting for the 6×6 normal eqs.
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
    const r = rows[i];
    const b = rhs[i];
    for (let j = 0; j < D; j++) {
      Ytb[j] += r[j] * b;
      for (let k = 0; k < D; k++) YtY[j][k] += r[j] * r[k];
    }
  }
  for (let j = 0; j < D; j++) YtY[j][j] += lambda;
  return solveLinear(YtY, Ytb);
}

// Generate (q, q̇, q̈, u) samples by stepping the simulator. To break
// the homogeneous-ambiguity, every sample gets a non-zero u (random
// torque). q̈ is estimated by central finite difference of q̇.
function genData(nSamples, frac_u_zero = 0.0) {
  const rows = [];
  const rhs = [];
  let s = { th1:(Math.random()-0.5)*Math.PI, th2:(Math.random()-0.5)*Math.PI, w1:0, w2:0 };
  // Buffer 3 frames so central FD has neighbors.
  let traj = [s];
  let us = [];
  while (rows.length / 2 < nSamples) {
    // Random hand-applied torque (or zero if free-swing fraction wins).
    const u = Math.random() < frac_u_zero ? 0 : (Math.random()-0.5) * 20;
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
    // Reseed if numerics blow up (large angles still allowed; unbounded velocity rejected).
    if (Math.abs(s.w1) > 50 || Math.abs(s.w2) > 50 || !Number.isFinite(s.th1)) {
      s = { th1:(Math.random()-0.5)*Math.PI, th2:(Math.random()-0.5)*Math.PI, w1:0, w2:0 };
      traj = [s]; us = [];
    }
  }
  return { rows, rhs };
}

function physicalParams(alpha) {
  const [a1, a2, a3, a4, a5, D] = alpha;
  // L₁ = g·α₁/α₄, L₂ = g·α₂/α₅
  const L1f = G * a1 / a4;
  const L2f = G * a2 / a5;
  const m_total = a1 / (L1f * L1f);    // m₁ + m₂
  const m2f = a3 / (L1f * L2f);
  const m1f = m_total - m2f;
  return { m1: m1f, m2: m2f, L1: L1f, L2: L2f, D };
}

console.log("=== Pendubot inertia identification — synthetic data ===\n");
console.log(`True α = [${ALPHA_TRUE.map(x => x.toFixed(4)).join(", ")}]`);
console.log(`True m₁=${M1}  m₂=${M2}  L₁=${L1}  L₂=${L2}  D=${DAMPING}\n`);

for (const N of [50, 200, 500, 1500, 3000]) {
  const { rows, rhs } = genData(N);
  const theta = ridgeSolve(rows, rhs, 1e-6);
  if (!theta) { console.log(`N=${N}: solve failed`); continue; }
  const errAlpha = theta.map((x, i) => Math.abs(x - ALPHA_TRUE[i]) / Math.abs(ALPHA_TRUE[i]) * 100);
  const phys = physicalParams(theta);
  console.log(`N=${N}:`);
  console.log(`  α  = [${theta.map(x => x.toFixed(4).padStart(8)).join(", ")}]`);
  console.log(`  err = [${errAlpha.map(x => x.toFixed(2).padStart(6) + "%").join(", ")}]`);
  console.log(`  m₁=${phys.m1.toFixed(3)}  m₂=${phys.m2.toFixed(3)}  L₁=${phys.L1.toFixed(3)}  L₂=${phys.L2.toFixed(3)}  D=${phys.D.toFixed(5)}`);
  console.log("");
}

console.log("\n=== Robustness: 90% free-swing samples (u=0) ===");
console.log("(rank-deficient — only direction recoverable, not scale)\n");
for (const N of [200, 1500]) {
  const { rows, rhs } = genData(N, 0.9);
  const theta = ridgeSolve(rows, rhs, 1e-6);
  if (!theta) { console.log(`N=${N}: solve failed`); continue; }
  const phys = physicalParams(theta);
  console.log(`N=${N} (90% free):  m₁=${phys.m1.toFixed(3)}  m₂=${phys.m2.toFixed(3)}  L₁=${phys.L1.toFixed(3)}  L₂=${phys.L2.toFixed(3)}`);
}

// Map the LQR basin of attraction: try many (dth2, w2) initial
// conditions and report which ones stabilize. This tells us the
// REAL catch window for LQR, as opposed to the 0.6 we've been using.

const L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0, G = 9.81;
const DAMPING = 0.0015;
const DT = 0.02;
const TORQUE_MAX = 30;

const LQR_K = [-41.64, 101.62, -17.31, 29.14];

function wrap(a) {
  let x = (a + Math.PI) % (2 * Math.PI);
  if (x < 0) x += 2 * Math.PI;
  return x - Math.PI;
}

function deriv(s, u) {
  const { th1, th2, w1, w2 } = s;
  const d = th1 - th2;
  const cd = Math.cos(d), sd = Math.sin(d);
  const M11 = (M1 + M2) * L1 * L1;
  const M12 = M2 * L1 * L2 * cd;
  const M22 = M2 * L2 * L2;
  const det = M11 * M22 - M12 * M12;
  const C1 = M2 * L1 * L2 * sd * w2 * w2;
  const C2 = -M2 * L1 * L2 * sd * w1 * w1;
  const G1 = (M1 + M2) * G * L1 * Math.sin(th1);
  const G2 = M2 * G * L2 * Math.sin(th2);
  const Q1 = u - DAMPING * w1;
  const Q2 = -DAMPING * w2;
  const rhs1 = Q1 - C1 - G1;
  const rhs2 = Q2 - C2 - G2;
  return [w1, w2, (M22*rhs1 - M12*rhs2)/det, (-M12*rhs1 + M11*rhs2)/det];
}

function stepEnv(s, u, dt = DT) {
  const k1 = deriv(s, u);
  const s2 = { th1: s.th1 + k1[0]*dt/2, th2: s.th2 + k1[1]*dt/2, w1: s.w1 + k1[2]*dt/2, w2: s.w2 + k1[3]*dt/2 };
  const k2 = deriv(s2, u);
  const s3 = { th1: s.th1 + k2[0]*dt/2, th2: s.th2 + k2[1]*dt/2, w1: s.w1 + k2[2]*dt/2, w2: s.w2 + k2[3]*dt/2 };
  const k3 = deriv(s3, u);
  const s4 = { th1: s.th1 + k3[0]*dt, th2: s.th2 + k3[1]*dt, w1: s.w1 + k3[2]*dt, w2: s.w2 + k3[3]*dt };
  const k4 = deriv(s4, u);
  return {
    th1: s.th1 + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])*dt/6,
    th2: s.th2 + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])*dt/6,
    w1:  s.w1  + (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])*dt/6,
    w2:  s.w2  + (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])*dt/6,
  };
}

function lqrAction(s) {
  const dth2 = wrap(s.th2 - Math.PI);
  const u = -(LQR_K[0]*s.th1 + LQR_K[1]*dth2 + LQR_K[2]*s.w1 + LQR_K[3]*s.w2);
  return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
}

function stabilizes(init) {
  let s = { ...init };
  for (let i = 0; i < 500; i++) {
    s = stepEnv(s, lqrAction(s));
    const dth2 = Math.abs(wrap(s.th2 - Math.PI));
    if (!Number.isFinite(s.th1) || dth2 > 1.5) return false;
  }
  return Math.abs(wrap(s.th2 - Math.PI)) < 0.05;
}

// Grid of (dth2, w2) — scanning the catch window.
console.log("LQR basin: 'Y' = stabilizes, '.' = diverges");
console.log("Rows: dth2 from +0.6 to -0.6; Cols: w2 from -4 to +4");
console.log();
const dth2s = [];
for (let d = 0.6; d >= -0.6001; d -= 0.1) dth2s.push(Math.round(d*100)/100);
const w2s = [];
for (let w = -4; w <= 4.001; w += 0.5) w2s.push(Math.round(w*10)/10);

let header = "  dth2  |";
for (const w of w2s) header += ` ${w.toString().padStart(4)}`;
console.log(header);
console.log("--------+" + "-".repeat(w2s.length * 5));
for (const d of dth2s) {
  let row = `  ${d.toFixed(2).padStart(5)} |`;
  for (const w of w2s) {
    const ok = stabilizes({ th1: 0, th2: Math.PI + d, w1: 0, w2: w });
    row += `    ${ok ? "Y" : "."}`;
  }
  console.log(row);
}

// Standalone numerical test: does the LQR teacher from PendulumScene
// actually stabilize the pendubot at the (θ₁=0, θ₂=π) equilibrium?
// Mirrors the dynamics in components/PendulumScene.tsx exactly.

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
  const cd = Math.cos(d);
  const sd = Math.sin(d);
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
  const a1 = (M22 * rhs1 - M12 * rhs2) / det;
  const a2 = (-M12 * rhs1 + M11 * rhs2) / det;
  return [w1, w2, a1, a2];
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

// Start near the inverted equilibrium with a small perturbation on dth2.
function runTest(init, label) {
  let s = { ...init };
  let maxDth2 = 0;
  for (let i = 0; i < 500; i++) { // 10 sec
    const u = lqrAction(s);
    s = stepEnv(s, u);
    const dth2 = Math.abs(wrap(s.th2 - Math.PI));
    if (dth2 > maxDth2) maxDth2 = dth2;
    if (!Number.isFinite(s.th1) || dth2 > 1.5) {
      console.log(`${label}: DIVERGED at step ${i}, dth2=${dth2.toFixed(3)}, th1=${s.th1.toFixed(3)}`);
      return;
    }
  }
  const finalDth2 = Math.abs(wrap(s.th2 - Math.PI));
  const finalTh1 = Math.abs(s.th1);
  console.log(
    `${label}: final |dth2|=${finalDth2.toFixed(4)}, |th1|=${finalTh1.toFixed(4)}, ` +
    `|w1|=${Math.abs(s.w1).toFixed(4)}, |w2|=${Math.abs(s.w2).toFixed(4)}, ` +
    `maxDth2=${maxDth2.toFixed(3)}`,
  );
}

console.log("=== Test: LQR teacher stability at inverted equilibrium ===");
runTest({ th1: 0, th2: Math.PI + 0.05, w1: 0, w2: 0 }, "dth2=+0.05");
runTest({ th1: 0, th2: Math.PI - 0.05, w1: 0, w2: 0 }, "dth2=-0.05");
runTest({ th1: 0.1, th2: Math.PI, w1: 0, w2: 0 }, "th1=+0.1");
runTest({ th1: 0, th2: Math.PI + 0.2, w1: 0, w2: 0 }, "dth2=+0.2");
runTest({ th1: 0, th2: Math.PI, w1: 1, w2: 0 }, "w1=+1");
runTest({ th1: 0, th2: Math.PI, w1: 0, w2: 1 }, "w2=+1");
runTest({ th1: 0, th2: Math.PI + 0.3, w1: 0, w2: 0 }, "dth2=+0.3");
runTest({ th1: 0, th2: Math.PI + 0.5, w1: 0, w2: 0 }, "dth2=+0.5");

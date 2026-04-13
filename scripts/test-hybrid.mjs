// End-to-end test: energy-pump swing-up + LQR balance, starting
// from hanging rest. This is the classical pendubot recipe; if it
// doesn't work here, the dynamics model is wrong, and no learner
// will fix it. If it does work, we copy this EXACT controller into
// the inference path.

const L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0, G = 9.81;
const DAMPING = 0.0015;
const DT = 0.02;
const TORQUE_MAX = 30;

const LQR_K = [-41.64, 101.62, -17.31, 29.14];
const E_TARGET = -(M1 + M2) * G * L1 + M2 * G * L2;

function wrap(a) {
  let x = (a + Math.PI) % (2 * Math.PI);
  if (x < 0) x += 2 * Math.PI;
  return x - Math.PI;
}

function energy(s) {
  const KE =
    0.5 * (M1 + M2) * L1 * L1 * s.w1 * s.w1 +
    0.5 * M2 * L2 * L2 * s.w2 * s.w2 +
    M2 * L1 * L2 * s.w1 * s.w2 * Math.cos(s.th1 - s.th2);
  const PE = -(M1 + M2) * G * L1 * Math.cos(s.th1) - M2 * G * L2 * Math.cos(s.th2);
  return KE + PE;
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
  return [w1, w2, (M22*(Q1-C1-G1) - M12*(Q2-C2-G2))/det, (-M12*(Q1-C1-G1) + M11*(Q2-C2-G2))/det];
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

// ---- the hybrid controller we'll actually deploy ----
const LQR_DTH2 = 0.30;   // handoff window: must be inside LQR basin
const LQR_W1 = 3.0;
const LQR_W2 = 3.0;
const SWING_KE = 8;
const SWING_KP = 2;
const SWING_KD = 0.3;

function hybrid(s) {
  const dth2 = wrap(s.th2 - Math.PI);
  const inBasin =
    Math.abs(dth2) < LQR_DTH2 &&
    Math.abs(s.w1) < LQR_W1 &&
    Math.abs(s.w2) < LQR_W2;
  if (inBasin) {
    const u = -(LQR_K[0]*s.th1 + LQR_K[1]*dth2 + LQR_K[2]*s.w1 + LQR_K[3]*s.w2);
    return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
  }
  const Etilde = energy(s) - E_TARGET;
  const u = SWING_KE * (-Etilde) * s.w1 - SWING_KP * Math.sin(s.th1) - SWING_KD * s.w1;
  return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
}

// Run the full pipeline from hanging rest.
function runFull(label, init) {
  let s = { ...init };
  let caught = -1;
  for (let i = 0; i < 1500; i++) { // 30 sec
    s = stepEnv(s, hybrid(s));
    const dth2 = Math.abs(wrap(s.th2 - Math.PI));
    if (caught < 0 && dth2 < 0.05 && Math.abs(s.w1) < 0.3 && Math.abs(s.w2) < 0.3) {
      caught = i;
    }
    if (!Number.isFinite(s.th1)) {
      console.log(`${label}: BLEW UP at step ${i}`);
      return;
    }
  }
  const dth2 = Math.abs(wrap(s.th2 - Math.PI));
  const stabilized = dth2 < 0.1 && Math.abs(s.w1) < 0.5 && Math.abs(s.w2) < 0.5;
  console.log(
    `${label}: ${stabilized ? "STABILIZED" : "FAILED"} ` +
    `(caught@${caught < 0 ? "never" : (caught*DT).toFixed(1)+"s"}, ` +
    `final dth2=${dth2.toFixed(3)}, w1=${s.w1.toFixed(2)}, w2=${s.w2.toFixed(2)})`,
  );
}

console.log("=== Hybrid controller: swing-up + LQR, from hanging rest ===");
runFull("hanging",         { th1: 0, th2: 0, w1: 0, w2: 0 });
runFull("hanging nudged",  { th1: 0, th2: 0.02, w1: 0, w2: 0 });
runFull("th1=0.3 offset",  { th1: 0.3, th2: 0, w1: 0, w2: 0 });
runFull("random kick",     { th1: 0.1, th2: -0.2, w1: 0.5, w2: -0.3 });
runFull("pre-kick w1",     { th1: 0, th2: 0, w1: 2, w2: 0 });

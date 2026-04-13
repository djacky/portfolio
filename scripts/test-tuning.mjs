// Sweep handoff tunings and catch-window sizes to find a hybrid that
// converges fast from hanging rest. Goal: dth2 → 0 within ~10 seconds.

const L1=1, L2=1, M1=1, M2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;
const LQR_K = [-41.64, 101.62, -17.31, 29.14];
const E_TARGET = -(M1+M2)*G*L1 + M2*G*L2;

const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };
const energy = s => 0.5*(M1+M2)*L1*L1*s.w1*s.w1 + 0.5*M2*L2*L2*s.w2*s.w2 + M2*L1*L2*s.w1*s.w2*Math.cos(s.th1-s.th2) - (M1+M2)*G*L1*Math.cos(s.th1) - M2*G*L2*Math.cos(s.th2);

function deriv(s, u) {
  const { th1, th2, w1, w2 } = s;
  const d = th1 - th2, cd = Math.cos(d), sd = Math.sin(d);
  const M11=(M1+M2)*L1*L1, M12=M2*L1*L2*cd, M22=M2*L2*L2, det=M11*M22-M12*M12;
  const C1=M2*L1*L2*sd*w2*w2, C2=-M2*L1*L2*sd*w1*w1;
  const G1=(M1+M2)*G*L1*Math.sin(th1), G2=M2*G*L2*Math.sin(th2);
  const rhs1=u-DAMPING*w1-C1-G1, rhs2=-DAMPING*w2-C2-G2;
  return [w1, w2, (M22*rhs1-M12*rhs2)/det, (-M12*rhs1+M11*rhs2)/det];
}
function stepEnv(s, u, dt=DT) {
  const k1=deriv(s,u);
  const s2={th1:s.th1+k1[0]*dt/2,th2:s.th2+k1[1]*dt/2,w1:s.w1+k1[2]*dt/2,w2:s.w2+k1[3]*dt/2};
  const k2=deriv(s2,u);
  const s3={th1:s.th1+k2[0]*dt/2,th2:s.th2+k2[1]*dt/2,w1:s.w1+k2[2]*dt/2,w2:s.w2+k2[3]*dt/2};
  const k3=deriv(s3,u);
  const s4={th1:s.th1+k3[0]*dt,th2:s.th2+k3[1]*dt,w1:s.w1+k3[2]*dt,w2:s.w2+k3[3]*dt};
  const k4=deriv(s4,u);
  return { th1:s.th1+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6, th2:s.th2+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6, w1:s.w1+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*dt/6, w2:s.w2+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*dt/6 };
}

// Energy-error handoff: switch to LQR when the state lies inside the
// Lyapunov sublevel set of LQR (quadratic form). Uses LQR cost directly.
// Cheap alternative: use a rough ellipsoid defined by LQR gains.
function lqrCost(s) {
  const dth2 = wrap(s.th2 - Math.PI);
  // Empirical Lyapunov function matching the LQR basin diagonal from
  // test-teacher-velocity.mjs: the basin is w2 ≈ -5*dth2 ± 1.5, so
  // the Lyapunov ellipsoid is roughly (w2 + 5*dth2)² + dth2² + small terms.
  return (
    8   * dth2 * dth2 +
    0.5 * (s.w2 + 5*dth2) * (s.w2 + 5*dth2) +
    2   * s.th1 * s.th1 +
    0.3 * s.w1 * s.w1
  );
}

function hybrid(ke, kp, kd, costThresh) {
  return s => {
    if (lqrCost(s) < costThresh) {
      const dth2 = wrap(s.th2 - Math.PI);
      const u = -(LQR_K[0]*s.th1 + LQR_K[1]*dth2 + LQR_K[2]*s.w1 + LQR_K[3]*s.w2);
      return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
    }
    const Etilde = energy(s) - E_TARGET;
    const u = ke * (-Etilde) * s.w1 - kp * Math.sin(s.th1) - kd * s.w1;
    return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
  };
}

function runFull(label, ctrl, init, seconds=20) {
  let s = { ...init };
  const steps = seconds / DT;
  let catchStep = -1;
  let lqrActiveStep = -1;
  for (let i=0; i<steps; i++) {
    const cost = lqrCost(s);
    if (lqrActiveStep < 0 && cost < 15) lqrActiveStep = i;
    const u = ctrl(s);
    s = stepEnv(s, u);
    const dth2 = Math.abs(wrap(s.th2-Math.PI));
    if (catchStep < 0 && dth2 < 0.05 && Math.abs(s.w1) < 0.3 && Math.abs(s.w2) < 0.3) {
      catchStep = i;
    }
    if (!Number.isFinite(s.th1)) { console.log(`${label}: NaN at ${i}`); return; }
  }
  const dth2 = Math.abs(wrap(s.th2-Math.PI));
  const ok = dth2 < 0.1 && Math.abs(s.w1) < 0.5 && Math.abs(s.w2) < 0.5;
  console.log(
    `${label}: ${ok ? "OK" : "FAIL"}  ` +
    `first LQR entry=${lqrActiveStep<0?"never":(lqrActiveStep*DT).toFixed(1)+"s"}  ` +
    `caught@${catchStep<0?"never":(catchStep*DT).toFixed(1)+"s"}  ` +
    `final dth2=${dth2.toFixed(3)} w1=${s.w1.toFixed(2)} w2=${s.w2.toFixed(2)}`
  );
}

console.log("=== Tuning sweep: Lyapunov-gated LQR handoff ===");
const inits = [
  ["nudged th2=0.02", { th1:0, th2:0.02, w1:0, w2:0 }],
  ["nudged th2=0.05", { th1:0, th2:0.05, w1:0, w2:0 }],
  ["nudged th2=0.10", { th1:0, th2:0.10, w1:0, w2:0 }],
];
for (const [name, init] of inits) {
  console.log(`\n-- init: ${name} --`);
  for (const ke of [6, 10, 20]) {
    for (const costT of [15, 25, 40]) {
      runFull(`ke=${ke}, cost<${costT}`, hybrid(ke, 2, 0.3, costT), init);
    }
  }
}

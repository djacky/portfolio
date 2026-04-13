// Diagnose why swing-up never catches. Log energy, w1, dth2
// over time to see if energy pumping is reaching E_target.

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
  return {
    th1:s.th1+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,
    th2:s.th2+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6,
    w1:s.w1+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*dt/6,
    w2:s.w2+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*dt/6,
  };
}

// Hybrid: original Åström-Furuta swing-up + LQR. Test various tunings.
function makeHybrid({ ke, kp, kd, lqrDth2, lqrW }) {
  return s => {
    const dth2 = wrap(s.th2 - Math.PI);
    if (Math.abs(dth2)<lqrDth2 && Math.abs(s.w1)<lqrW && Math.abs(s.w2)<lqrW) {
      const u = -(LQR_K[0]*s.th1 + LQR_K[1]*dth2 + LQR_K[2]*s.w1 + LQR_K[3]*s.w2);
      return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
    }
    const Etilde = energy(s) - E_TARGET;
    const u = ke * (-Etilde) * s.w1 - kp * Math.sin(s.th1) - kd * s.w1;
    return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
  };
}

function runLogged(label, ctrl, init, seconds=60) {
  let s = { ...init };
  const steps = seconds / DT;
  let sampled = [];
  for (let i=0; i<steps; i++) {
    s = stepEnv(s, ctrl(s));
    if (i % Math.round(steps/12) === 0) {
      sampled.push({
        t: (i*DT).toFixed(1),
        dth2: wrap(s.th2-Math.PI).toFixed(2),
        w1: s.w1.toFixed(2),
        w2: s.w2.toFixed(2),
        E: energy(s).toFixed(2),
      });
    }
    if (!Number.isFinite(s.th1)) { console.log(`${label}: NaN at ${i}`); return; }
  }
  console.log(`${label}:`);
  for (const r of sampled) {
    console.log(`  t=${r.t.padStart(4)}s  dth2=${r.dth2.padStart(5)}  w1=${r.w1.padStart(5)}  w2=${r.w2.padStart(5)}  E=${r.E}  (Etarget=${E_TARGET.toFixed(2)})`);
  }
  const final = energy(s).toFixed(2);
  console.log(`  >> final E=${final}, dth2=${wrap(s.th2-Math.PI).toFixed(3)}`);
}

console.log("=== swing-up diagnostic: Åström-Furuta variants ===\n");

// Original tuning, nudged init
runLogged("orig ke=8, kp=2, kd=0.3, nudged",
  makeHybrid({ ke:8, kp:2, kd:0.3, lqrDth2:0.3, lqrW:3 }),
  { th1:0, th2:0.02, w1:0, w2:0 });

// Higher ke
runLogged("\nke=30, kp=2, kd=0.3, nudged",
  makeHybrid({ ke:30, kp:2, kd:0.3, lqrDth2:0.3, lqrW:3 }),
  { th1:0, th2:0.02, w1:0, w2:0 });

// Even higher ke, softer kp
runLogged("\nke=50, kp=5, kd=0.5, nudged",
  makeHybrid({ ke:50, kp:5, kd:0.5, lqrDth2:0.3, lqrW:3 }),
  { th1:0, th2:0.02, w1:0, w2:0 });

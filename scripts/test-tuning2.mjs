// Wider tuning sweep plus a try-before-commit basin check: at each
// step, the LQR handoff is ONLY taken if a 0.4-second rollout under
// LQR would keep the state close to the equilibrium. This is the
// robust "funnel" test and should only admit states truly in the basin.

const L1=1, L2=1, M1=1, M2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;
const LQR_K = [-41.64, 101.62, -17.31, 29.14];
const E_TARGET = -(M1+M2)*G*L1 + M2*G*L2;

const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };
const energy = s => 0.5*(M1+M2)*L1*L1*s.w1*s.w1 + 0.5*M2*L2*L2*s.w2*s.w2 + M2*L1*L2*s.w1*s.w2*Math.cos(s.th1-s.th2) - (M1+M2)*G*L1*Math.cos(s.th1) - M2*G*L2*Math.cos(s.th2);
function deriv(s,u){const{th1,th2,w1,w2}=s;const d=th1-th2,cd=Math.cos(d),sd=Math.sin(d);const M11=(M1+M2)*L1*L1,M12=M2*L1*L2*cd,M22=M2*L2*L2,det=M11*M22-M12*M12;const C1=M2*L1*L2*sd*w2*w2,C2=-M2*L1*L2*sd*w1*w1;const G1=(M1+M2)*G*L1*Math.sin(th1),G2=M2*G*L2*Math.sin(th2);const r1=u-DAMPING*w1-C1-G1,r2=-DAMPING*w2-C2-G2;return [w1,w2,(M22*r1-M12*r2)/det,(-M12*r1+M11*r2)/det];}
function stepEnv(s,u,dt=DT){const k1=deriv(s,u);const s2={th1:s.th1+k1[0]*dt/2,th2:s.th2+k1[1]*dt/2,w1:s.w1+k1[2]*dt/2,w2:s.w2+k1[3]*dt/2};const k2=deriv(s2,u);const s3={th1:s.th1+k2[0]*dt/2,th2:s.th2+k2[1]*dt/2,w1:s.w1+k2[2]*dt/2,w2:s.w2+k2[3]*dt/2};const k3=deriv(s3,u);const s4={th1:s.th1+k3[0]*dt,th2:s.th2+k3[1]*dt,w1:s.w1+k3[2]*dt,w2:s.w2+k3[3]*dt};const k4=deriv(s4,u);return{th1:s.th1+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,th2:s.th2+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6,w1:s.w1+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*dt/6,w2:s.w2+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*dt/6};}

function lqrU(s) {
  const dth2 = wrap(s.th2 - Math.PI);
  const u = -(LQR_K[0]*s.th1 + LQR_K[1]*dth2 + LQR_K[2]*s.w1 + LQR_K[3]*s.w2);
  return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
}

// Try-before-commit: does LQR, starting from s, reach the equilibrium
// within HORIZON steps without diverging? Used as a basin oracle.
const BASIN_HORIZON = 40; // 0.8 s lookahead
function inBasin(s) {
  let t = { ...s };
  for (let i = 0; i < BASIN_HORIZON; i++) {
    t = stepEnv(t, lqrU(t));
    const d = Math.abs(wrap(t.th2 - Math.PI));
    if (!Number.isFinite(t.th1) || d > 1.0) return false;
  }
  const d = Math.abs(wrap(t.th2 - Math.PI));
  return d < 0.15 && Math.abs(t.w1) < 2 && Math.abs(t.w2) < 2;
}

function hybrid(ke, kp, kd) {
  return s => {
    if (inBasin(s)) return lqrU(s);
    const Etilde = energy(s) - E_TARGET;
    const u = ke * (-Etilde) * s.w1 - kp * Math.sin(s.th1) - kd * s.w1;
    return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
  };
}

function runFull(label, ctrl, init, seconds=25) {
  let s = { ...init };
  let catchStep = -1, basinStep = -1;
  const steps = seconds/DT;
  for (let i=0; i<steps; i++) {
    if (basinStep < 0 && inBasin(s)) basinStep = i;
    s = stepEnv(s, ctrl(s));
    const d = Math.abs(wrap(s.th2-Math.PI));
    if (catchStep < 0 && d<0.05 && Math.abs(s.w1)<0.3 && Math.abs(s.w2)<0.3) catchStep = i;
    if (!Number.isFinite(s.th1)) { console.log(`${label}: NaN`); return; }
  }
  const d = Math.abs(wrap(s.th2-Math.PI));
  const ok = d<0.1 && Math.abs(s.w1)<0.5 && Math.abs(s.w2)<0.5;
  console.log(`${label}: ${ok?"OK":"FAIL"} basin@${basinStep<0?"never":(basinStep*DT).toFixed(1)+"s"} caught@${catchStep<0?"never":(catchStep*DT).toFixed(1)+"s"} final|dth2|=${d.toFixed(2)}`);
}

console.log("=== Funnel handoff + swing-up tuning ===");
const inits = [
  ["nudged 0.02", { th1:0, th2:0.02, w1:0, w2:0 }],
  ["nudged 0.10", { th1:0, th2:0.10, w1:0, w2:0 }],
  ["th1 kick",    { th1:0.3, th2:0, w1:0, w2:0 }],
  ["w2 kick",     { th1:0, th2:0, w1:0, w2:0.5 }],
];
for (const [name, init] of inits) {
  console.log(`\n-- init: ${name} --`);
  for (const ke of [10, 20, 40, 80]) {
    for (const kp of [2, 5]) {
      runFull(`ke=${ke.toString().padStart(2)}, kp=${kp}`, hybrid(ke, kp, 0.3), init);
    }
  }
}

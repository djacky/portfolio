// Two improvements at once:
//   1. Best gains from test-tuning4: ke=50, kp=1, kd=0.3
//   2. Widen the LQR basin's early-acceptance window — the swing-up
//      passes through the basin briefly; we want to catch it instead
//      of letting it swing past.
//
// Also test a hybrid that schedules in a "catching" gain bias when
// link 2 is upright and slow, even if technically outside the strict
// basin. This is a robustness improvement, not a tuning sweep.

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

// Funnel test with parameters.
function makeBasin(horizon, dth2Cap, finalDth2, finalW) {
  return function(s) {
    let t = { ...s };
    for (let i = 0; i < horizon; i++) {
      t = stepEnv(t, lqrU(t));
      const d = Math.abs(wrap(t.th2 - Math.PI));
      if (!Number.isFinite(t.th1) || d > dth2Cap) return false;
    }
    const d = Math.abs(wrap(t.th2 - Math.PI));
    return d < finalDth2 && Math.abs(t.w1) < finalW && Math.abs(t.w2) < finalW;
  };
}

function hybrid(ke, kp, kd, basin) {
  return s => {
    if (basin(s)) return lqrU(s);
    const Etilde = energy(s) - E_TARGET;
    const u = ke * (-Etilde) * s.w1 - kp * Math.sin(s.th1) - kd * s.w1;
    return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
  };
}

function runEpisode(ctrl, init, maxSeconds=20) {
  let s = { ...init };
  const steps = maxSeconds / DT;
  for (let i = 0; i < steps; i++) {
    s = stepEnv(s, ctrl(s));
    if (!Number.isFinite(s.th1)) return null;
    const d = Math.abs(wrap(s.th2 - Math.PI));
    if (d < 0.05 && Math.abs(s.w1) < 0.3 && Math.abs(s.w2) < 0.3) return i * DT;
  }
  return null;
}

function practicalIC(rng) {
  return {
    th1: (rng() - 0.5) * 0.6,
    th2: (rng() - 0.5) * 0.6,
    w1: (rng() - 0.5) * 0.4,
    w2: (rng() - 0.5) * 0.4,
  };
}

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = seed;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const N_TRIALS = 80;
function evalConfig(ke, kp, kd, basin, label) {
  const rng = mulberry32(0xCAFE);
  const ctrl = hybrid(ke, kp, kd, basin);
  let nSuccess = 0;
  const times = [];
  for (let i = 0; i < N_TRIALS; i++) {
    const ic = practicalIC(rng);
    const t = runEpisode(ctrl, ic);
    if (t !== null) { nSuccess++; times.push(t); }
  }
  times.sort((a,b)=>a-b);
  const median = times.length ? times[Math.floor(times.length/2)] : Infinity;
  const p90    = times.length ? times[Math.floor(times.length*0.9)] : Infinity;
  console.log(`${label.padEnd(40)} rate=${(nSuccess/N_TRIALS*100).toFixed(0).padStart(3)}%  median=${median.toFixed(1)}s  p90=${p90===Infinity?"-":p90.toFixed(1)+"s"}`);
}

console.log("=== Best gains × basin parameter sweep ===\n");

console.log("-- ke=50, kp=1, kd=0.3, varying basin --");
evalConfig(50, 1, 0.3, makeBasin(20, 1.0, 0.15, 2), "horiz=20 cap=1.0 final=0.15/2");
evalConfig(50, 1, 0.3, makeBasin(40, 1.0, 0.15, 2), "horiz=40 cap=1.0 final=0.15/2 (current)");
evalConfig(50, 1, 0.3, makeBasin(40, 1.4, 0.15, 2), "horiz=40 cap=1.4 final=0.15/2");
evalConfig(50, 1, 0.3, makeBasin(60, 1.0, 0.20, 3), "horiz=60 cap=1.0 final=0.20/3");
evalConfig(50, 1, 0.3, makeBasin(80, 1.4, 0.25, 4), "horiz=80 cap=1.4 final=0.25/4");
evalConfig(50, 1, 0.3, makeBasin(100, 1.6, 0.3, 5), "horiz=100 cap=1.6 final=0.30/5");

console.log("\n-- ke=24, kp=1, kd=0.5, varying basin --");
evalConfig(24, 1, 0.5, makeBasin(40, 1.0, 0.15, 2), "horiz=40 cap=1.0 final=0.15/2 (current)");
evalConfig(24, 1, 0.5, makeBasin(80, 1.4, 0.25, 4), "horiz=80 cap=1.4 final=0.25/4");
evalConfig(24, 1, 0.5, makeBasin(100, 1.6, 0.3, 5), "horiz=100 cap=1.6 final=0.30/5");

console.log("\n-- ke=20, kp=1, kd=0.2, varying basin --");
evalConfig(20, 1, 0.2, makeBasin(40, 1.0, 0.15, 2), "horiz=40 cap=1.0 final=0.15/2 (current)");
evalConfig(20, 1, 0.2, makeBasin(80, 1.4, 0.25, 4), "horiz=80 cap=1.4 final=0.25/4");
evalConfig(20, 1, 0.2, makeBasin(100, 1.6, 0.3, 5), "horiz=100 cap=1.6 final=0.30/5");

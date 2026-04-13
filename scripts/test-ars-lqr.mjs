// Augmented Random Search (ARS) for the pendubot linear policy.
// ARS is provably convergent for LQR and dead simple:
//   1. Sample N perturbations δ_i ~ N(0, ν²I)
//   2. Evaluate J(K + δ_i) and J(K - δ_i)
//   3. Pick top-b by max(J+, J-)
//   4. K += lr/(b·σ_R) * Σ (J(K+δ_i) - J(K-δ_i)) * δ_i
// We use ARS-V2-t (with state normalization optional, top-b filtering).

const L1=1, L2=1, M1=1, M2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;
const LQR_K = [-41.64, 101.62, -17.31, 29.14];
const LQR_NORM = Math.sqrt(LQR_K.reduce((a,b)=>a+b*b, 0));

const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };
function deriv(s,u){const{th1,th2,w1,w2}=s;const d=th1-th2,cd=Math.cos(d),sd=Math.sin(d);const M11=(M1+M2)*L1*L1,M12=M2*L1*L2*cd,M22=M2*L2*L2,det=M11*M22-M12*M12;const C1=M2*L1*L2*sd*w2*w2,C2=-M2*L1*L2*sd*w1*w1;const G1=(M1+M2)*G*L1*Math.sin(th1),G2=M2*G*L2*Math.sin(th2);const r1=u-DAMPING*w1-C1-G1,r2=-DAMPING*w2-C2-G2;return [w1,w2,(M22*r1-M12*r2)/det,(-M12*r1+M11*r2)/det];}
function stepEnv(s,u,dt=DT){const k1=deriv(s,u);const s2={th1:s.th1+k1[0]*dt/2,th2:s.th2+k1[1]*dt/2,w1:s.w1+k1[2]*dt/2,w2:s.w2+k1[3]*dt/2};const k2=deriv(s2,u);const s3={th1:s.th1+k2[0]*dt/2,th2:s.th2+k2[1]*dt/2,w1:s.w1+k2[2]*dt/2,w2:s.w2+k2[3]*dt/2};const k3=deriv(s3,u);const s4={th1:s.th1+k3[0]*dt,th2:s.th2+k3[1]*dt,w1:s.w1+k3[2]*dt,w2:s.w2+k3[3]*dt};const k4=deriv(s4,u);return{th1:s.th1+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,th2:s.th2+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6,w1:s.w1+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*dt/6,w2:s.w2+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*dt/6};}

const features = s => [s.th1, wrap(s.th2 - Math.PI), s.w1, s.w2];
const reward   = (f, u) => -(10*f[0]*f[0] + 10*f[1]*f[1] + f[2]*f[2] + f[3]*f[3] + 0.1*u*u);
const clip     = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

function gauss() { const u1 = Math.max(1e-9, Math.random()), u2 = Math.random(); return Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2); }
function randomBasinIC() {
  return { th1: (Math.random()-0.5)*0.2, th2: Math.PI + (Math.random()-0.5)*0.2,
           w1:  (Math.random()-0.5)*1,   w2:  (Math.random()-0.5)*1 };
}

const EP_STEPS = 80;

// Evaluate J(K) — average return over N_ROLLOUTS rollouts (shared seeds across K+/K-).
function rollout(K, ic) {
  let s = { ...ic };
  let total = 0;
  for (let t = 0; t < EP_STEPS; t++) {
    const f = features(s);
    let u = -(K[0]*f[0] + K[1]*f[1] + K[2]*f[2] + K[3]*f[3]);
    u = clip(u, -TORQUE_MAX, TORQUE_MAX);
    const next = stepEnv(s, u);
    const nf = features(next);
    total += reward(nf, u);
    if (Math.abs(nf[1]) > 1.0 || Math.abs(nf[0]) > 1.5 || !Number.isFinite(nf[0])) {
      // Penalize remaining steps with the worst observed reward.
      total += (EP_STEPS - t - 1) * (-200);
      break;
    }
    s = next;
  }
  return total;
}

function evalK(K, ics) {
  let total = 0;
  for (const ic of ics) total += rollout(K, ic);
  return total / ics.length;
}

// ----- ARS hyperparams -----
const N_DIRS  = 8;        // perturbations per update
const N_TOP   = 4;        // top-b directions to use
const NU      = 0.5;      // perturbation magnitude
const LR      = 0.5;      // step size
const N_ROLLOUTS_PER_EVAL = 2;
const MAX_UPDATES = 600;

let K = [0, 0, 0, 0];

console.log("=== ARS linear-policy LQR convergence ===");
console.log(`Target K_LQR: [${LQR_K.map(x=>x.toFixed(2)).join(", ")}]`);
console.log(`||K_LQR|| = ${LQR_NORM.toFixed(2)}`);
console.log(`N_DIRS=${N_DIRS}, N_TOP=${N_TOP}, ν=${NU}, lr=${LR}\n`);

for (let upd = 1; upd <= MAX_UPDATES; upd++) {
  // Shared ICs across all perturbations this update (variance reduction).
  const ics = Array.from({length: N_ROLLOUTS_PER_EVAL}, () => randomBasinIC());

  // Sample directions and evaluate K ± ν·δ.
  const dirs = [];
  for (let i = 0; i < N_DIRS; i++) {
    const d = [gauss(), gauss(), gauss(), gauss()];
    const Kp = [K[0]+NU*d[0], K[1]+NU*d[1], K[2]+NU*d[2], K[3]+NU*d[3]];
    const Kn = [K[0]-NU*d[0], K[1]-NU*d[1], K[2]-NU*d[2], K[3]-NU*d[3]];
    const rp = evalK(Kp, ics);
    const rn = evalK(Kn, ics);
    dirs.push({ d, rp, rn, score: Math.max(rp, rn) });
  }

  // Top-b by max(rp, rn).
  dirs.sort((a, b) => b.score - a.score);
  const top = dirs.slice(0, N_TOP);

  // σ_R: std of returns used (2·N_TOP returns).
  const allR = top.flatMap(t => [t.rp, t.rn]);
  const meanR = allR.reduce((a,b)=>a+b, 0) / allR.length;
  let varR = 0;
  for (const r of allR) varR += (r - meanR) ** 2;
  const sigmaR = Math.sqrt(varR / allR.length) + 1e-8;

  // K += lr/(b·σ_R) * Σ (rp - rn) * d
  for (let j = 0; j < 4; j++) {
    let g = 0;
    for (const t of top) g += (t.rp - t.rn) * t.d[j];
    K[j] += LR / (N_TOP * sigmaR) * g;
  }

  if (upd % 25 === 0 || upd === 1) {
    let normDiff = 0;
    for (let i = 0; i < 4; i++) normDiff += (K[i] - LQR_K[i]) ** 2;
    const conv = Math.sqrt(normDiff) / LQR_NORM;
    const fmt = a => a.map(x => x.toFixed(2).padStart(8)).join(" ");
    const evalIcs = Array.from({length: 8}, () => randomBasinIC());
    const J = evalK(K, evalIcs);
    console.log(
      `upd ${String(upd).padStart(3)}  ` +
      `K=[${fmt(K)}]  ` +
      `err=${(conv*100).toFixed(1).padStart(5)}%  ` +
      `J=${J.toFixed(1)}`
    );
  }
}

console.log(`\nFinal K   : [${K.map(x=>x.toFixed(2).padStart(8)).join(" ")}]`);
console.log(`Target LQR: [${LQR_K.map(x=>x.toFixed(2).padStart(8)).join(" ")}]`);
let normDiff = 0;
for (let i = 0; i < 4; i++) normDiff += (K[i] - LQR_K[i]) ** 2;
console.log(`Relative error: ${(Math.sqrt(normDiff) / LQR_NORM * 100).toFixed(1)}%`);

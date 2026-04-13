// Verify that PPO with a 4-parameter linear-Gaussian policy converges
// to the LQR gains on the pendubot near the inverted equilibrium.
// If this works in pure Node, the in-browser tfjs version will work too.
//
// The reward is the negated LQR cost (Q=diag(10,10,1,1), R=0.1), so the
// optimal linear policy under this reward IS the LQR solution by
// construction. PPO should rediscover the LQR gains.

const L1=1, L2=1, M1=1, M2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;
const LQR_K = [-41.64, 101.62, -17.31, 29.14];
const LQR_NORM = Math.sqrt(LQR_K.reduce((a,b)=>a+b*b, 0));

const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };
function deriv(s,u){const{th1,th2,w1,w2}=s;const d=th1-th2,cd=Math.cos(d),sd=Math.sin(d);const M11=(M1+M2)*L1*L1,M12=M2*L1*L2*cd,M22=M2*L2*L2,det=M11*M22-M12*M12;const C1=M2*L1*L2*sd*w2*w2,C2=-M2*L1*L2*sd*w1*w1;const G1=(M1+M2)*G*L1*Math.sin(th1),G2=M2*G*L2*Math.sin(th2);const r1=u-DAMPING*w1-C1-G1,r2=-DAMPING*w2-C2-G2;return [w1,w2,(M22*r1-M12*r2)/det,(-M12*r1+M11*r2)/det];}
function stepEnv(s,u,dt=DT){const k1=deriv(s,u);const s2={th1:s.th1+k1[0]*dt/2,th2:s.th2+k1[1]*dt/2,w1:s.w1+k1[2]*dt/2,w2:s.w2+k1[3]*dt/2};const k2=deriv(s2,u);const s3={th1:s.th1+k2[0]*dt/2,th2:s.th2+k2[1]*dt/2,w1:s.w1+k2[2]*dt/2,w2:s.w2+k2[3]*dt/2};const k3=deriv(s3,u);const s4={th1:s.th1+k3[0]*dt,th2:s.th2+k3[1]*dt,w1:s.w1+k3[2]*dt,w2:s.w2+k3[3]*dt};const k4=deriv(s4,u);return{th1:s.th1+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,th2:s.th2+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6,w1:s.w1+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*dt/6,w2:s.w2+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*dt/6};}

const features = s => [s.th1, wrap(s.th2 - Math.PI), s.w1, s.w2];
const reward   = (f, u) => -(10*f[0]*f[0] + 10*f[1]*f[1] + f[2]*f[2] + f[3]*f[3] + 0.1*u*u);
const isTerm   = f => Math.abs(f[1]) > 1.0 || Math.abs(f[0]) > 1.5;
const clip     = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
function gauss() { const u1 = Math.max(1e-9, Math.random()), u2 = Math.random(); return Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2); }
function randomBasinIC() {
  return { th1: (Math.random()-0.5)*0.2, th2: Math.PI + (Math.random()-0.5)*0.2,
           w1:  (Math.random()-0.5)*1,   w2:  (Math.random()-0.5)*1 };
}

// ----- PPO hyperparams -----
const PPO_GAMMA = 0.99, PPO_LAM = 0.95, PPO_CLIP = 0.2;
const PPO_LR = 0.1, PPO_EPOCHS = 10, PPO_BATCH = 64, PPO_VF_COEF = 0.5;
const SIGMA_INIT = 3.0, SIGMA_MIN = 0.3, SIGMA_DECAY = 0.99;
const N_ENVS = 16, ROLLOUT_LEN = 32, MAX_EP_STEPS = 80;
const MAX_UPDATES = 500;

// ----- Linear-Gaussian policy & linear value baseline -----
// REAL TEST: start from K=0. PPO must drive K toward LQR_K from scratch.
let K  = [0, 0, 0, 0];
let Vw = [0, 0, 0, 0];        // V(f) = Vw @ f + Vb
let Vb = 0;
let sigma = SIGMA_INIT;

// ----- Adam state -----
const ADAM_B1 = 0.9, ADAM_B2 = 0.999, ADAM_EPS = 1e-8;
const adamM_K  = [0,0,0,0], adamV_K  = [0,0,0,0];
const adamM_Vw = [0,0,0,0], adamV_Vw = [0,0,0,0];
let   adamM_Vb = 0,         adamV_Vb = 0;
let adamT = 0;

function adamStep(gK, gVw, gVb) {
  adamT += 1;
  const bc1 = 1 - Math.pow(ADAM_B1, adamT);
  const bc2 = 1 - Math.pow(ADAM_B2, adamT);
  for (let i = 0; i < 4; i++) {
    adamM_K[i] = ADAM_B1*adamM_K[i] + (1-ADAM_B1)*gK[i];
    adamV_K[i] = ADAM_B2*adamV_K[i] + (1-ADAM_B2)*gK[i]*gK[i];
    K[i]  -= PPO_LR * (adamM_K[i]/bc1) / (Math.sqrt(adamV_K[i]/bc2) + ADAM_EPS);

    adamM_Vw[i] = ADAM_B1*adamM_Vw[i] + (1-ADAM_B1)*gVw[i];
    adamV_Vw[i] = ADAM_B2*adamV_Vw[i] + (1-ADAM_B2)*gVw[i]*gVw[i];
    Vw[i] -= PPO_LR * (adamM_Vw[i]/bc1) / (Math.sqrt(adamV_Vw[i]/bc2) + ADAM_EPS);
  }
  adamM_Vb = ADAM_B1*adamM_Vb + (1-ADAM_B1)*gVb;
  adamV_Vb = ADAM_B2*adamV_Vb + (1-ADAM_B2)*gVb*gVb;
  Vb -= PPO_LR * (adamM_Vb/bc1) / (Math.sqrt(adamV_Vb/bc2) + ADAM_EPS);
}

const policyMean = f => -(K[0]*f[0] + K[1]*f[1] + K[2]*f[2] + K[3]*f[3]);
const valueOf    = f =>  Vw[0]*f[0] + Vw[1]*f[1] + Vw[2]*f[2] + Vw[3]*f[3] + Vb;

function actStochastic(f) {
  const m = policyMean(f);
  const raw = m + sigma * gauss();
  const action = clip(raw, -TORQUE_MAX, TORQUE_MAX);
  // log-prob of the SAMPLED (pre-clip) action under N(m, σ²)
  const lp = -0.5*Math.log(2*Math.PI) - Math.log(sigma) - 0.5*((raw-m)/sigma)**2;
  return { action, raw, lp, value: valueOf(f) };
}

function ppoUpdate(buf) {
  const N = buf.states.length;
  if (N < PPO_BATCH) return;

  // Normalize advantages
  let mean = 0;
  for (const a of buf.advantages) mean += a;
  mean /= N;
  let varSum = 0;
  for (const a of buf.advantages) varSum += (a-mean)*(a-mean);
  const std = Math.sqrt(varSum/N) + 1e-8;
  const normAdv = buf.advantages.map(a => (a-mean)/std);

  for (let epoch = 0; epoch < PPO_EPOCHS; epoch++) {
    const idx = Array.from({length:N}, (_,i)=>i);
    for (let i = N-1; i > 0; i--) {
      const j = Math.floor(Math.random()*(i+1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }

    for (let start = 0; start < N; start += PPO_BATCH) {
      const batch = idx.slice(start, start + PPO_BATCH);
      if (batch.length < PPO_BATCH) continue;
      const B = batch.length;

      const gK  = [0,0,0,0];
      const gVw = [0,0,0,0];
      let   gVb = 0;

      for (const i of batch) {
        const f = buf.states[i];
        const a = buf.actions[i];        // raw (pre-clip) action
        const lpOld = buf.logProbs[i];
        const adv = normAdv[i];
        const ret = buf.returns[i];

        // ----- policy gradient (clipped surrogate) -----
        const m = policyMean(f);
        const newLp = -0.5*Math.log(2*Math.PI) - Math.log(sigma) - 0.5*((a-m)/sigma)**2;
        const ratio = Math.exp(newLp - lpOld);

        // Standard PPO gradient w.r.t. ratio:
        //   zero if we're on the "clipped" side (don't push further)
        //   else -adv (we want to maximize the surrogate)
        let dRatio;
        if (adv > 0 && ratio > 1 + PPO_CLIP) dRatio = 0;
        else if (adv < 0 && ratio < 1 - PPO_CLIP) dRatio = 0;
        else dRatio = -adv;

        // Chain rule: ratio = exp(newLp - lpOld); dratio/dnewLp = ratio
        // newLp = const - 0.5*((a - m)/σ)²;  dnewLp/dm = (a - m)/σ²
        // m = -K @ f;  dm/dK_j = -f_j
        const dlpdm = (a - m) / (sigma * sigma);
        const k = (dRatio * ratio * dlpdm) / B;
        for (let j = 0; j < 4; j++) gK[j] += k * (-f[j]);

        // ----- value gradient (MSE) -----
        const v = valueOf(f);
        const valDiff = (v - ret) / B;
        for (let j = 0; j < 4; j++) gVw[j] += PPO_VF_COEF * valDiff * f[j];
        gVb += PPO_VF_COEF * valDiff;
      }

      adamStep(gK, gVw, gVb);
    }
  }

  sigma = Math.max(SIGMA_MIN, sigma * SIGMA_DECAY);
}

// ----- rollout state -----
const envs = Array.from({length:N_ENVS}, () => randomBasinIC());
const epSteps = new Array(N_ENVS).fill(0);
const trajs = Array.from({length:N_ENVS}, () => []);
let avgReward = 0;

function flushTraj(traj, lastValue, buf) {
  const T = traj.length;
  let gae = 0;
  let nextV = lastValue;
  for (let t = T - 1; t >= 0; t--) {
    const delta = traj[t].reward + PPO_GAMMA * nextV - traj[t].value;
    gae = delta + PPO_GAMMA * PPO_LAM * gae;
    buf.states.push(traj[t].state);
    buf.actions.push(traj[t].rawAction);
    buf.logProbs.push(traj[t].logProb);
    buf.returns.push(gae + traj[t].value);
    buf.advantages.push(gae);
    nextV = traj[t].value;
  }
}

console.log("=== PPO linear-policy LQR convergence test ===");
console.log(`Target K_LQR: [${LQR_K.map(x=>x.toFixed(2)).join(", ")}]`);
console.log(`||K_LQR|| = ${LQR_NORM.toFixed(2)}\n`);

for (let upd = 1; upd <= MAX_UPDATES; upd++) {
  const buf = { states: [], actions: [], logProbs: [], returns: [], advantages: [] };

  // Roll out ROLLOUT_LEN steps per env per update
  for (let step = 0; step < ROLLOUT_LEN; step++) {
    for (let i = 0; i < N_ENVS; i++) {
      const f = features(envs[i]);
      const { action, raw, lp, value } = actStochastic(f);
      const next = stepEnv(envs[i], action);
      const nf = features(next);
      const r = reward(nf, action);
      const term = isTerm(nf) || epSteps[i] >= MAX_EP_STEPS;

      trajs[i].push({ state: f, rawAction: raw, logProb: lp, value, reward: r });
      avgReward = avgReward * 0.999 + r * 0.001;
      envs[i] = next;
      epSteps[i] += 1;

      if (term) {
        flushTraj(trajs[i], 0, buf);  // bootstrap = 0 at terminal
        trajs[i].length = 0;
        envs[i] = randomBasinIC();
        epSteps[i] = 0;
      }
    }
  }
  // Flush ongoing (non-terminal) trajectories with V-bootstrap
  for (let i = 0; i < N_ENVS; i++) {
    if (trajs[i].length > 0) {
      const lastV = valueOf(features(envs[i]));
      flushTraj(trajs[i], lastV, buf);
      trajs[i].length = 0;
    }
  }

  ppoUpdate(buf);

  if (upd % 25 === 0 || upd === 1) {
    let normDiff = 0;
    for (let i = 0; i < 4; i++) normDiff += (K[i] - LQR_K[i]) ** 2;
    const conv = Math.sqrt(normDiff) / LQR_NORM;
    const fmt = a => a.map(x => x.toFixed(2).padStart(7)).join(" ");
    console.log(
      `upd ${String(upd).padStart(3)}  ` +
      `K=[${fmt(K)}]  ` +
      `err=${(conv*100).toFixed(1).padStart(5)}%  ` +
      `σ=${sigma.toFixed(2)}  ` +
      `r̄=${avgReward.toFixed(2)}`
    );
  }
}

console.log(`\nFinal K   : [${K.map(x=>x.toFixed(2).padStart(7)).join(" ")}]`);
console.log(`Target LQR: [${LQR_K.map(x=>x.toFixed(2).padStart(7)).join(" ")}]`);
let normDiff = 0;
for (let i = 0; i < 4; i++) normDiff += (K[i] - LQR_K[i]) ** 2;
console.log(`Relative error: ${(Math.sqrt(normDiff) / LQR_NORM * 100).toFixed(1)}%`);

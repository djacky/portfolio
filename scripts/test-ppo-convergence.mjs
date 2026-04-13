// PPO convergence test with SEPARATE policy/value networks
// Matches CleanRL Pendulum-v1 configuration

const G_PHYS = 10, MASS = 1, LEN = 1, MAX_TORQUE = 2, DT = 0.05, MAX_SPEED = 8, EP_LEN = 200;
const OBS_DIM = 3, HIDDEN = 64, BUFFER_SIZE = 2048;
const GAMMA = 0.99, LAM = 0.95, CLIP_EPS = 0.2;
const PPO_EPOCHS = 10, MINI_BATCH = 64, PPO_LR = 3e-4;
const VALUE_COEFF = 0.5, ENTROPY_COEFF = 0.0;
const MAX_GRAD_NORM = 0.5;
const CONVERGE_FRAC = 0.3, CONVERGE_HOLD = 3;
const LOG2PI_HALF = 0.9189385332;

const wrap = a => { let x = (a + Math.PI) % (2 * Math.PI); if (x < 0) x += 2 * Math.PI; return x - Math.PI; };

function pendDeriv(th, w, u) { return [w, (3 * G_PHYS) / (2 * LEN) * Math.sin(th) + 3 / (MASS * LEN * LEN) * u]; }

function stepPend(s, u) {
  const dt = DT;
  const k1 = pendDeriv(s.th, s.w, u);
  const k2 = pendDeriv(s.th + k1[0]*dt/2, s.w + k1[1]*dt/2, u);
  const k3 = pendDeriv(s.th + k2[0]*dt/2, s.w + k2[1]*dt/2, u);
  const k4 = pendDeriv(s.th + k3[0]*dt, s.w + k3[1]*dt, u);
  return {
    th: s.th + (k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,
    w: Math.max(-MAX_SPEED, Math.min(MAX_SPEED, s.w + (k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6)),
  };
}

function getObs(s) { return Float64Array.of(Math.cos(s.th), Math.sin(s.th), s.w / MAX_SPEED); }
function pendReward(s, u) { const th = wrap(s.th); return -(th*th + 0.1*s.w*s.w + 0.001*u*u); }
function randomState(rng) { return { th: (rng()*2-1)*Math.PI, w: (rng()*2-1) }; }

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = seed;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function seededGauss(rng) { return Math.sqrt(-2*Math.log(rng()+1e-10))*Math.cos(2*Math.PI*rng()); }
function gaussRand() { return Math.sqrt(-2*Math.log(Math.random()+1e-10))*Math.cos(2*Math.PI*Math.random()); }

function orthoInit(rows, cols, gain, rng) {
  const n = Math.max(rows, cols);
  const a = Float64Array.from({length:n*n}, ()=>seededGauss(rng));
  // Simple QR via Gram-Schmidt
  const q = new Float64Array(rows * cols);
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) q[i*cols+j] = a[i*n+j];
    for (let k = 0; k < j; k++) {
      let dot = 0;
      for (let i = 0; i < rows; i++) dot += q[i*cols+j] * q[i*cols+k];
      for (let i = 0; i < rows; i++) q[i*cols+j] -= dot * q[i*cols+k];
    }
    let norm = 0;
    for (let i = 0; i < rows; i++) norm += q[i*cols+j] * q[i*cols+j];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < rows; i++) q[i*cols+j] = q[i*cols+j] / norm * gain;
  }
  return q;
}

class MLP {
  constructor(rng, outGain) {
    this.W1 = orthoInit(HIDDEN, OBS_DIM, Math.sqrt(2), rng);
    this.b1 = new Float64Array(HIDDEN);
    this.W2 = orthoInit(HIDDEN, HIDDEN, Math.sqrt(2), rng);
    this.b2 = new Float64Array(HIDDEN);
    this.Wo = orthoInit(1, HIDDEN, outGain, rng);
    this.bo = new Float64Array(1);
    this.allP = [this.W1, this.b1, this.W2, this.b2, this.Wo, this.bo];
    this.mAd = this.allP.map(a => new Float64Array(a.length));
    this.vAd = this.allP.map(a => new Float64Array(a.length));
    this.grads = this.allP.map(a => new Float64Array(a.length));
    this.adamT = 0;
    this._h1 = new Float64Array(HIDDEN);
    this._h2 = new Float64Array(HIDDEN);
    this._dh2 = new Float64Array(HIDDEN);
    this._dh1 = new Float64Array(HIDDEN);
  }

  forward(x) {
    const h1 = this._h1, h2 = this._h2;
    for (let i = 0; i < HIDDEN; i++) {
      let s = this.b1[i];
      for (let j = 0; j < OBS_DIM; j++) s += this.W1[i*OBS_DIM+j]*x[j];
      h1[i] = Math.tanh(s);
    }
    for (let i = 0; i < HIDDEN; i++) {
      let s = this.b2[i];
      for (let j = 0; j < HIDDEN; j++) s += this.W2[i*HIDDEN+j]*h1[j];
      h2[i] = Math.tanh(s);
    }
    let out = this.bo[0];
    for (let j = 0; j < HIDDEN; j++) out += this.Wo[j]*h2[j];
    return out;
  }

  backward(dout) {
    const h1 = this._h1, h2 = this._h2;
    const G = this.grads;
    for (let j = 0; j < HIDDEN; j++) {
      G[4][j] += dout * h2[j];
    }
    G[5][0] += dout;

    const dh2 = this._dh2;
    for (let j = 0; j < HIDDEN; j++)
      dh2[j] = dout * this.Wo[j] * (1 - h2[j]*h2[j]);
    for (let i = 0; i < HIDDEN; i++) {
      G[3][i] += dh2[i];
      for (let j = 0; j < HIDDEN; j++) G[2][i*HIDDEN+j] += dh2[i]*h1[j];
    }

    const dh1 = this._dh1;
    for (let j = 0; j < HIDDEN; j++) {
      let s = 0;
      for (let i = 0; i < HIDDEN; i++) s += dh2[i]*this.W2[i*HIDDEN+j];
      dh1[j] = s*(1-h1[j]*h1[j]);
    }
    for (let i = 0; i < HIDDEN; i++) {
      G[1][i] += dh1[i];
      for (let j = 0; j < OBS_DIM; j++) G[0][i*OBS_DIM+j] += dh1[i]*x_cache[j];
    }
  }

  zeroGrad() { for (const g of this.grads) g.fill(0); }

  adamStep(bs) {
    const G = this.grads;
    for (const g of G) for (let i = 0; i < g.length; i++) g[i] /= bs;

    let norm = 0;
    for (const g of G) for (let i = 0; i < g.length; i++) norm += g[i]*g[i];
    norm = Math.sqrt(norm);
    if (norm > MAX_GRAD_NORM) {
      const sc = MAX_GRAD_NORM / norm;
      for (const g of G) for (let i = 0; i < g.length; i++) g[i] *= sc;
    }

    this.adamT++;
    const bc1 = 1 - Math.pow(0.9, this.adamT);
    const bc2 = 1 - Math.pow(0.999, this.adamT);
    for (let pi = 0; pi < this.allP.length; pi++) {
      const par = this.allP[pi], g = G[pi], m = this.mAd[pi], v = this.vAd[pi];
      for (let i = 0; i < par.length; i++) {
        m[i] = 0.9*m[i] + 0.1*g[i];
        v[i] = 0.999*v[i] + 0.001*g[i]*g[i];
        par[i] -= PPO_LR*(m[i]/bc1)/(Math.sqrt(v[i]/bc2)+1e-8);
      }
    }
  }
}

let x_cache;

class PPOAgent {
  constructor() {
    const rng = mulberry32(42);
    this.bgRng = mulberry32(123);
    this.bgEnv = randomState(this.bgRng);
    this.bgStep = 0;

    this.policyNet = new MLP(rng, 0.01);
    this.valueNet = new MLP(rng, 1.0);
    this.logStd = Float64Array.of(0);
    this.logStd_m = 0;
    this.logStd_v = 0;
    this.logStd_t = 0;

    this.bufO = new Float64Array(BUFFER_SIZE * OBS_DIM);
    this.bufA = new Float64Array(BUFFER_SIZE);
    this.bufR = new Float64Array(BUFFER_SIZE);
    this.bufV = new Float64Array(BUFFER_SIZE);
    this.bufL = new Float64Array(BUFFER_SIZE);
    this.bufD = new Uint8Array(BUFFER_SIZE);
    this.bIdx = 0;

    this.rewardHistory = [];
    this.uprightHistory = [];
    this.updates = 0;
    this.converged = false;
    this.convCount = 0;
  }

  act(obs, deterministic) {
    const mu = this.policyNet.forward(obs);
    const val = this.valueNet.forward(obs);
    const std = Math.exp(this.logStd[0]);
    const action = deterministic
      ? Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, mu))
      : Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, mu + std * gaussRand()));
    const d = action - mu;
    const logProb = -0.5*(d/std)**2 - this.logStd[0] - LOG2PI_HALF;
    return { action, logProb, value: val };
  }

  collectBuffer() {
    while (this.bIdx < BUFFER_SIZE) {
      const obs = getObs(this.bgEnv);
      const { action, logProb, value } = this.act(obs, false);
      const ns = stepPend(this.bgEnv, action);
      const rew = pendReward(ns, action);
      this.bgStep++;
      const done = this.bgStep >= EP_LEN;
      const idx = this.bIdx;
      this.bufO.set(obs, idx*OBS_DIM);
      this.bufA[idx] = action;
      this.bufR[idx] = rew;
      this.bufV[idx] = value;
      this.bufL[idx] = logProb;
      this.bufD[idx] = done ? 1 : 0;
      this.bIdx++;
      if (done) { this.bgEnv = randomState(this.bgRng); this.bgStep = 0; }
      else this.bgEnv = ns;
    }
    const lastObs = getObs(this.bgEnv);
    const { value: lastV } = this.act(lastObs, false);
    return this.bufD[this.bIdx - 1] ? 0 : lastV;
  }

  ppoUpdate(lastVal) {
    const T = this.bIdx;
    const adv = new Float64Array(T);
    const ret = new Float64Array(T);

    let lastA = 0;
    for (let t = T-1; t >= 0; t--) {
      const mask = 1 - this.bufD[t];
      const nv = t === T-1 ? lastVal : this.bufV[t+1];
      const delta = this.bufR[t] + GAMMA*nv*mask - this.bufV[t];
      lastA = delta + GAMMA*LAM*mask*lastA;
      adv[t] = lastA;
      ret[t] = lastA + this.bufV[t];
    }

    let am = 0;
    for (let t = 0; t < T; t++) am += adv[t];
    am /= T;
    let av = 0;
    for (let t = 0; t < T; t++) av += (adv[t]-am)**2;
    const as2 = Math.sqrt(av/T) + 1e-8;
    for (let t = 0; t < T; t++) adv[t] = (adv[t]-am)/as2;

    let totalR = 0;
    for (let t = 0; t < T; t++) totalR += this.bufR[t];
    this.rewardHistory.push(totalR / T);

    const indices = Array.from({length:T}, (_,i) => i);

    for (let ep = 0; ep < PPO_EPOCHS; ep++) {
      for (let i = T-1; i > 0; i--) {
        const j = Math.floor(this.bgRng()*(i+1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      for (let st = 0; st < T; st += MINI_BATCH) {
        const end = Math.min(st+MINI_BATCH, T);
        const bs = end - st;

        this.policyNet.zeroGrad();
        this.valueNet.zeroGrad();
        let dls_accum = 0;

        for (let bi = st; bi < end; bi++) {
          const si = indices[bi];
          const obs = this.bufO.subarray(si*OBS_DIM, si*OBS_DIM+OBS_DIM);
          x_cache = obs;
          const oldAct = this.bufA[si];
          const oldLp = this.bufL[si];
          const advantage = adv[si];
          const target = ret[si];

          const mu = this.policyNet.forward(obs);
          const val = this.valueNet.forward(obs);
          const std = Math.exp(this.logStd[0]);
          const std2 = std*std;
          const diff = oldAct - mu;
          const newLp = -0.5*diff*diff/std2 - this.logStd[0] - LOG2PI_HALF;
          const ratio = Math.exp(newLp - oldLp);

          const clipped = (ratio > 1+CLIP_EPS && advantage > 0) ||
                          (ratio < 1-CLIP_EPS && advantage < 0);

          let dmu = 0, dls = 0;
          if (!clipped) {
            dmu = -advantage*ratio*diff/std2;
            dls = -advantage*ratio*(diff*diff/std2 - 1);
          }
          dls -= ENTROPY_COEFF;
          dls_accum += dls;

          this.policyNet.backward(dmu);

          const dval = VALUE_COEFF*(val - target);
          x_cache = obs;
          this.valueNet.forward(obs);
          this.valueNet.backward(dval);
        }

        this.policyNet.adamStep(bs);
        this.valueNet.adamStep(bs);

        // logStd Adam
        const g_ls = dls_accum / bs;
        this.logStd_t++;
        this.logStd_m = 0.9*this.logStd_m + 0.1*g_ls;
        this.logStd_v = 0.999*this.logStd_v + 0.001*g_ls*g_ls;
        const bc1 = 1 - Math.pow(0.9, this.logStd_t);
        const bc2 = 1 - Math.pow(0.999, this.logStd_t);
        this.logStd[0] -= PPO_LR*(this.logStd_m/bc1)/(Math.sqrt(this.logStd_v/bc2)+1e-8);
      }
    }

    this.logStd[0] = Math.max(-3, Math.min(2, this.logStd[0]));

    const uf = this.evalPolicy();
    this.uprightHistory.push(uf);
    this.updates++;
    if (uf >= CONVERGE_FRAC) this.convCount++;
    else this.convCount = Math.max(0, this.convCount - 1);
    if (this.convCount >= CONVERGE_HOLD) this.converged = true;
    this.bIdx = 0;
  }

  evalPolicy() {
    const rng = mulberry32(0xCAFE);
    let env = randomState(rng);
    let up = 0;
    for (let t = 0; t < EP_LEN; t++) {
      const obs = getObs(env);
      const { action } = this.act(obs, true);
      env = stepPend(env, action);
      if (Math.abs(wrap(env.th)) < 0.3 && Math.abs(env.w) < 1) up++;
    }
    return up / EP_LEN;
  }
}

const agent = new PPOAgent();

console.log("PPO convergence test — SEPARATE policy/value networks");
console.log(`H=${HIDDEN} epochs=${PPO_EPOCHS} mb=${MINI_BATCH} lr=${PPO_LR}`);
console.log(`gamma=${GAMMA} lam=${LAM} clip=${CLIP_EPS} ent=${ENTROPY_COEFF}`);
console.log(`max_grad_norm=${MAX_GRAD_NORM}\n`);
console.log("  upd |  meanR   | upright% | logStd | time");
console.log("-".repeat(55));

const t0 = Date.now();
for (let u = 0; u < 50; u++) {
  const lastV = agent.collectBuffer();
  agent.ppoUpdate(lastV);
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const rH = agent.rewardHistory;
  const uH = agent.uprightHistory;
  console.log(
    `${String(u+1).padStart(5)} | ${rH[rH.length-1].toFixed(3).padStart(8)} | ` +
    `${(uH[uH.length-1]*100).toFixed(0).padStart(7)}% | ` +
    `${agent.logStd[0].toFixed(3).padStart(6)} | ${elapsed}s`
  );
  if (agent.converged) {
    console.log(`\nConverged at update ${u+1}!`);
    break;
  }
}
if (!agent.converged) console.log("\nDid not converge in 50 updates.");

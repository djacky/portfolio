// ARS (Augmented Random Search) for single pendulum swing-up.
// No backprop, no Adam — pure parameter-space search.
// If this converges, the physics/reward are fine and the PPO backprop has a bug.

const G_PHYS = 10, MASS = 1, LEN = 1, MAX_TORQUE = 2, DT = 0.05, MAX_SPEED = 8, EP_LEN = 200;
const wrap = a => { let x = (a + Math.PI) % (2 * Math.PI); if (x < 0) x += 2 * Math.PI; return x - Math.PI; };

function pendDeriv(th, w, u) { return [w, (3*G_PHYS)/(2*LEN)*Math.sin(th) + 3/(MASS*LEN*LEN)*u]; }
function stepPend(s, u) {
  const dt = DT;
  const k1 = pendDeriv(s.th, s.w, u);
  const k2 = pendDeriv(s.th+k1[0]*dt/2, s.w+k1[1]*dt/2, u);
  const k3 = pendDeriv(s.th+k2[0]*dt/2, s.w+k2[1]*dt/2, u);
  const k4 = pendDeriv(s.th+k3[0]*dt, s.w+k3[1]*dt, u);
  return {
    th: s.th + (k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,
    w: Math.max(-MAX_SPEED, Math.min(MAX_SPEED, s.w+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6)),
  };
}

function getObs(s) { return [Math.cos(s.th), Math.sin(s.th), s.w / MAX_SPEED]; }
function pendReward(s, u) { const th = wrap(s.th); return -(th*th + 0.1*s.w*s.w + 0.001*u*u); }

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

const OBS_DIM = 3, HIDDEN = 32;

class Policy {
  constructor(rng) {
    const xI = Math.sqrt(2/OBS_DIM), xH = Math.sqrt(2/HIDDEN), xO = 0.01;
    this.W1 = Float64Array.from({length:HIDDEN*OBS_DIM}, ()=>seededGauss(rng)*xI);
    this.b1 = new Float64Array(HIDDEN);
    this.W2 = Float64Array.from({length:HIDDEN*HIDDEN}, ()=>seededGauss(rng)*xH);
    this.b2 = new Float64Array(HIDDEN);
    this.Wo = Float64Array.from({length:HIDDEN}, ()=>seededGauss(rng)*xO);
    this.bo = new Float64Array(1);
    this.allP = [this.W1, this.b1, this.W2, this.b2, this.Wo, this.bo];
  }

  act(obs) {
    const h1 = new Float64Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      let s = this.b1[i];
      for (let j = 0; j < OBS_DIM; j++) s += this.W1[i*OBS_DIM+j]*obs[j];
      h1[i] = Math.tanh(s);
    }
    const h2 = new Float64Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      let s = this.b2[i];
      for (let j = 0; j < HIDDEN; j++) s += this.W2[i*HIDDEN+j]*h1[j];
      h2[i] = Math.tanh(s);
    }
    let mu = this.bo[0];
    for (let j = 0; j < HIDDEN; j++) mu += this.Wo[j]*h2[j];
    return Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, mu));
  }

  numParams() {
    let n = 0;
    for (const p of this.allP) n += p.length;
    return n;
  }

  getFlat() {
    const n = this.numParams();
    const flat = new Float64Array(n);
    let idx = 0;
    for (const p of this.allP) { flat.set(p, idx); idx += p.length; }
    return flat;
  }

  setFlat(flat) {
    let idx = 0;
    for (const p of this.allP) {
      for (let i = 0; i < p.length; i++) p[i] = flat[idx++];
    }
  }
}

function evaluate(policy, seeds) {
  let totalR = 0;
  for (const seed of seeds) {
    const rng = mulberry32(seed);
    let s = { th: (rng()*2-1)*Math.PI, w: (rng()*2-1) };
    let epR = 0;
    for (let t = 0; t < EP_LEN; t++) {
      const obs = getObs(s);
      const u = policy.act(obs);
      s = stepPend(s, u);
      epR += pendReward(s, u);
    }
    totalR += epR / EP_LEN;
  }
  return totalR / seeds.length;
}

function evalUpright(policy, seeds) {
  let totalUp = 0;
  for (const seed of seeds) {
    const rng = mulberry32(seed);
    let s = { th: (rng()*2-1)*Math.PI, w: (rng()*2-1) };
    let up = 0;
    for (let t = 0; t < EP_LEN; t++) {
      const obs = getObs(s);
      const u = policy.act(obs);
      s = stepPend(s, u);
      if (Math.abs(wrap(s.th)) < 0.3 && Math.abs(s.w) < 1) up++;
    }
    totalUp += up / EP_LEN;
  }
  return totalUp / seeds.length;
}

// ARS hyperparams
const N_DIRS = 16;       // number of perturbation directions
const NOISE_STD = 0.03;  // perturbation std
const STEP_SIZE = 0.02;  // learning rate
const TOP_B = 8;          // use top-b directions
const N_EVAL = 8;         // episodes for evaluation
const N_ITERS = 300;

const rng = mulberry32(42);
const policy = new Policy(rng);
const nP = policy.numParams();

const evalSeeds = Array.from({length:20}, (_,i) => 0xCAFE + i);

console.log(`ARS: ${nP} params, ${N_DIRS} directions, noise=${NOISE_STD}, lr=${STEP_SIZE}`);
console.log("  iter |  meanR   | upright% | time");
console.log("-".repeat(50));

const t0 = Date.now();
const trainSeeds = Array.from({length:N_EVAL}, () => Math.floor(rng()*0x7FFFFFFF));

for (let iter = 0; iter < N_ITERS; iter++) {
  const baseParams = policy.getFlat();

  // Generate perturbations and evaluate
  const deltas = [];
  const rewards_pos = [];
  const rewards_neg = [];

  for (let d = 0; d < N_DIRS; d++) {
    const delta = Float64Array.from({length:nP}, ()=>seededGauss(rng));
    deltas.push(delta);

    // Positive direction
    const posP = new Float64Array(nP);
    for (let i = 0; i < nP; i++) posP[i] = baseParams[i] + NOISE_STD * delta[i];
    policy.setFlat(posP);
    rewards_pos.push(evaluate(policy, trainSeeds));

    // Negative direction
    const negP = new Float64Array(nP);
    for (let i = 0; i < nP; i++) negP[i] = baseParams[i] - NOISE_STD * delta[i];
    policy.setFlat(negP);
    rewards_neg.push(evaluate(policy, trainSeeds));
  }

  // Select top-b directions by max(r+, r-)
  const scored = deltas.map((d, i) => ({
    d, rp: rewards_pos[i], rn: rewards_neg[i],
    maxR: Math.max(rewards_pos[i], rewards_neg[i])
  }));
  scored.sort((a, b) => b.maxR - a.maxR);
  const topDirs = scored.slice(0, TOP_B);

  // Compute reward std for normalization
  const allR = [...rewards_pos, ...rewards_neg];
  let rMean = 0;
  for (const r of allR) rMean += r;
  rMean /= allR.length;
  let rVar = 0;
  for (const r of allR) rVar += (r - rMean) ** 2;
  const rStd = Math.sqrt(rVar / allR.length) + 1e-8;

  // Update
  const update = new Float64Array(nP);
  for (const { d, rp, rn } of topDirs) {
    const coeff = (rp - rn) / rStd;
    for (let i = 0; i < nP; i++) update[i] += coeff * d[i];
  }
  const newP = new Float64Array(nP);
  for (let i = 0; i < nP; i++) newP[i] = baseParams[i] + (STEP_SIZE / TOP_B) * update[i];
  policy.setFlat(newP);

  if (iter % 10 === 0 || iter === N_ITERS - 1) {
    const meanR = evaluate(policy, evalSeeds);
    const upPct = evalUpright(policy, evalSeeds);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(
      `${String(iter).padStart(5)} | ${meanR.toFixed(3).padStart(8)} | ` +
      `${(upPct*100).toFixed(0).padStart(7)}% | ${elapsed}s`
    );
    if (upPct >= 0.3) {
      console.log(`\nConverged at iter ${iter}!`);
      break;
    }
  }
}

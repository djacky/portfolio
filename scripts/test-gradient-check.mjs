// Gradient checker: verify analytical vs numerical gradients for the PPO MLP

const OBS_DIM = 3, HIDDEN = 32;

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

const rng = mulberry32(42);

const W1 = Float64Array.from({length:HIDDEN*OBS_DIM}, ()=>seededGauss(rng)/Math.sqrt(OBS_DIM));
const b1 = new Float64Array(HIDDEN);
const W2 = Float64Array.from({length:HIDDEN*HIDDEN}, ()=>seededGauss(rng)/Math.sqrt(HIDDEN));
const b2 = new Float64Array(HIDDEN);
const Wmu = Float64Array.from({length:HIDDEN}, ()=>seededGauss(rng)*0.01);
const bmu = new Float64Array(1);
const Wv = Float64Array.from({length:HIDDEN}, ()=>seededGauss(rng)/Math.sqrt(HIDDEN));
const bv = new Float64Array(1);
const p = [W1, b1, W2, b2, Wmu, bmu, Wv, bv];

function forward(x) {
  const h1 = new Float64Array(HIDDEN), h2 = new Float64Array(HIDDEN);
  for (let i = 0; i < HIDDEN; i++) {
    let s = b1[i];
    for (let j = 0; j < OBS_DIM; j++) s += W1[i*OBS_DIM+j]*x[j];
    h1[i] = Math.tanh(s);
  }
  for (let i = 0; i < HIDDEN; i++) {
    let s = b2[i];
    for (let j = 0; j < HIDDEN; j++) s += W2[i*HIDDEN+j]*h1[j];
    h2[i] = Math.tanh(s);
  }
  let mu = bmu[0], val = bv[0];
  for (let j = 0; j < HIDDEN; j++) { mu += Wmu[j]*h2[j]; val += Wv[j]*h2[j]; }
  return { mu, val, h1, h2 };
}

// Analytical gradient for dmu=1, dval=0
function analyticGrad_mu(x) {
  const { h1, h2 } = forward(x);
  const G = p.map(a => new Float64Array(a.length));
  const dmu = 1;

  // Layer 3
  for (let j = 0; j < HIDDEN; j++) G[4][j] = dmu * h2[j];
  G[5][0] = dmu;

  // Backprop through h2 — ONLY policy gradient (no value)
  const dz2 = new Float64Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j++)
    dz2[j] = dmu * Wmu[j] * (1 - h2[j]*h2[j]);

  for (let i = 0; i < HIDDEN; i++) {
    G[3][i] = dz2[i];
    for (let j = 0; j < HIDDEN; j++) G[2][i*HIDDEN+j] = dz2[i]*h1[j];
  }

  const dz1 = new Float64Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let s = 0;
    for (let i = 0; i < HIDDEN; i++) s += dz2[i]*W2[i*HIDDEN+j];
    dz1[j] = s*(1-h1[j]*h1[j]);
  }
  for (let i = 0; i < HIDDEN; i++) {
    G[1][i] = dz1[i];
    for (let j = 0; j < OBS_DIM; j++) G[0][i*OBS_DIM+j] = dz1[i]*x[j];
  }

  return G;
}

// Analytical gradient WITH value gradient through shared layers
function analyticGrad_mu_withValue(x) {
  const { h1, h2 } = forward(x);
  const G = p.map(a => new Float64Array(a.length));
  const dmu = 1, dval = 1;

  for (let j = 0; j < HIDDEN; j++) {
    G[4][j] = dmu * h2[j];
    G[6][j] = dval * h2[j];
  }
  G[5][0] = dmu;
  G[7][0] = dval;

  const dz2 = new Float64Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j++)
    dz2[j] = (dmu * Wmu[j] + dval * Wv[j]) * (1 - h2[j]*h2[j]);

  for (let i = 0; i < HIDDEN; i++) {
    G[3][i] = dz2[i];
    for (let j = 0; j < HIDDEN; j++) G[2][i*HIDDEN+j] = dz2[i]*h1[j];
  }

  const dz1 = new Float64Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let s = 0;
    for (let i = 0; i < HIDDEN; i++) s += dz2[i]*W2[i*HIDDEN+j];
    dz1[j] = s*(1-h1[j]*h1[j]);
  }
  for (let i = 0; i < HIDDEN; i++) {
    G[1][i] = dz1[i];
    for (let j = 0; j < OBS_DIM; j++) G[0][i*OBS_DIM+j] = dz1[i]*x[j];
  }

  return G;
}

// Numerical gradient
function numGrad(x, outputFn) {
  const eps = 1e-5;
  const G = p.map(a => new Float64Array(a.length));
  for (let pi = 0; pi < p.length; pi++) {
    for (let i = 0; i < p[pi].length; i++) {
      const orig = p[pi][i];
      p[pi][i] = orig + eps;
      const fPlus = outputFn(x);
      p[pi][i] = orig - eps;
      const fMinus = outputFn(x);
      p[pi][i] = orig;
      G[pi][i] = (fPlus - fMinus) / (2 * eps);
    }
  }
  return G;
}

const x = Float64Array.of(0.5, -0.3, 0.7);

console.log("=== Gradient check: d(mu)/d(params) ===");
const aG = analyticGrad_mu(x);
const nG = numGrad(x, (obs) => forward(obs).mu);

const layerNames = ['W1', 'b1', 'W2', 'b2', 'Wmu', 'bmu', 'Wv', 'bv'];
for (let pi = 0; pi < p.length; pi++) {
  let maxErr = 0, maxRel = 0;
  for (let i = 0; i < p[pi].length; i++) {
    const err = Math.abs(aG[pi][i] - nG[pi][i]);
    const rel = err / (Math.abs(nG[pi][i]) + 1e-10);
    maxErr = Math.max(maxErr, err);
    maxRel = Math.max(maxRel, rel);
  }
  const status = maxRel < 1e-4 ? 'OK' : maxRel < 1e-2 ? 'WARN' : 'FAIL';
  console.log(`  ${layerNames[pi].padEnd(4)}: maxErr=${maxErr.toExponential(2)}, maxRel=${maxRel.toExponential(2)} ${status}`);
}

console.log("\n=== Gradient check: d(val)/d(params) ===");
const aGv = p.map(a => new Float64Array(a.length));
// For val gradient: only Wv and bv should have gradient
{
  const { h2 } = forward(x);
  for (let j = 0; j < HIDDEN; j++) aGv[6][j] = h2[j];
  aGv[7][0] = 1;
}
const nGv = numGrad(x, (obs) => forward(obs).val);
for (let pi = 0; pi < p.length; pi++) {
  let maxErr = 0, maxRel = 0;
  for (let i = 0; i < p[pi].length; i++) {
    const err = Math.abs(aGv[pi][i] - nGv[pi][i]);
    const rel = err / (Math.abs(nGv[pi][i]) + 1e-10);
    maxErr = Math.max(maxErr, err);
    maxRel = Math.max(maxRel, rel);
  }
  const status = maxRel < 1e-4 ? 'OK' : maxRel < 1e-2 ? 'WARN' : 'FAIL';
  console.log(`  ${layerNames[pi].padEnd(4)}: maxErr=${maxErr.toExponential(2)}, maxRel=${maxRel.toExponential(2)} ${status}`);
}

console.log("\n=== Gradient check: d(mu+val)/d(params) — with value gradient through shared layers ===");
const aGmv = analyticGrad_mu_withValue(x);
const nGmv = numGrad(x, (obs) => { const r = forward(obs); return r.mu + r.val; });
for (let pi = 0; pi < p.length; pi++) {
  let maxErr = 0, maxRel = 0;
  for (let i = 0; i < p[pi].length; i++) {
    const err = Math.abs(aGmv[pi][i] - nGmv[pi][i]);
    const rel = err / (Math.abs(nGmv[pi][i]) + 1e-10);
    maxErr = Math.max(maxErr, err);
    maxRel = Math.max(maxRel, rel);
  }
  const status = maxRel < 1e-4 ? 'OK' : maxRel < 1e-2 ? 'WARN' : 'FAIL';
  console.log(`  ${layerNames[pi].padEnd(4)}: maxErr=${maxErr.toExponential(2)}, maxRel=${maxRel.toExponential(2)} ${status}`);
}

// Debug: test if the pendulum physics/reward even supports swing-up
// Then test minimal REINFORCE before PPO

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

// Test 1: Energy-pumping controller
console.log("=== Test 1: Energy-pumping controller ===");
function energyController(s) {
  const th = wrap(s.th);
  // Energy-based swing-up
  // E = 0.5*I*w^2 - m*g*L/2*cos(th), I = mL^2/3
  const E = 0.5*(MASS*LEN*LEN/3)*s.w*s.w - MASS*G_PHYS*LEN/2*Math.cos(s.th);
  const E_up = MASS*G_PHYS*LEN/2; // energy at upright
  const Etilde = E - E_up;

  // Near upright: PD controller
  if (Math.abs(th) < 0.4 && Math.abs(s.w) < 2) {
    return Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, -5*th - 0.5*s.w));
  }

  // Swing-up: pump energy
  const u = -0.5 * Etilde * s.w;
  return Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, u));
}

const rng = mulberry32(0xCAFE);
for (let trial = 0; trial < 5; trial++) {
  let s = { th: (rng()*2-1)*Math.PI, w: (rng()*2-1) };
  let totalR = 0, up = 0;
  for (let t = 0; t < EP_LEN; t++) {
    const u = energyController(s);
    s = stepPend(s, u);
    totalR += pendReward(s, u);
    if (Math.abs(wrap(s.th)) < 0.3 && Math.abs(s.w) < 1) up++;
  }
  console.log(`  trial ${trial}: meanR=${(totalR/EP_LEN).toFixed(3)}, upright=${(up/EP_LEN*100).toFixed(0)}%`);
}

// Test 2: Random policy baseline
console.log("\n=== Test 2: Random policy baseline ===");
for (let trial = 0; trial < 5; trial++) {
  let s = { th: (rng()*2-1)*Math.PI, w: (rng()*2-1) };
  let totalR = 0, up = 0;
  for (let t = 0; t < EP_LEN; t++) {
    const u = (Math.random()*2-1)*MAX_TORQUE;
    s = stepPend(s, u);
    totalR += pendReward(s, u);
    if (Math.abs(wrap(s.th)) < 0.3 && Math.abs(s.w) < 1) up++;
  }
  console.log(`  trial ${trial}: meanR=${(totalR/EP_LEN).toFixed(3)}, upright=${(up/EP_LEN*100).toFixed(0)}%`);
}

// Test 3: REINFORCE with linear policy (simplest possible)
console.log("\n=== Test 3: REINFORCE with linear policy ===");
const OBS_DIM = 3;

function getObs(s) { return [Math.cos(s.th), Math.sin(s.th), s.w/MAX_SPEED]; }

// Linear policy: mu = w · obs
let pw = [0, 0, 0];
let logStd = 0;
const LR = 0.01;
const BASELINE_ALPHA = 0.1;
let baseline = -6;

for (let ep = 0; ep < 500; ep++) {
  let s = { th: (rng()*2-1)*Math.PI, w: (rng()*2-1) };
  const trajectory = [];

  for (let t = 0; t < EP_LEN; t++) {
    const obs = getObs(s);
    const mu = pw[0]*obs[0] + pw[1]*obs[1] + pw[2]*obs[2];
    const std = Math.exp(logStd);
    const noise = Math.sqrt(-2*Math.log(Math.random()+1e-10))*Math.cos(2*Math.PI*Math.random());
    const action = Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, mu + std*noise));
    const ns = stepPend(s, action);
    const rew = pendReward(ns, action);
    trajectory.push({ obs, action, mu, rew });
    s = ns;
  }

  // Compute returns
  let totalR = 0;
  for (const step of trajectory) totalR += step.rew;
  const meanR = totalR / EP_LEN;
  baseline = BASELINE_ALPHA * meanR + (1-BASELINE_ALPHA) * baseline;

  // REINFORCE gradient
  const advantage = meanR - baseline;
  for (const step of trajectory) {
    const { obs, action, mu } = step;
    const std = Math.exp(logStd);
    const diff = action - mu;
    // d log pi / d w_i = (a - mu) / std^2 * obs_i
    for (let i = 0; i < 3; i++) {
      pw[i] += LR * advantage * (diff / (std*std)) * obs[i] / EP_LEN;
    }
    // d log pi / d logStd = (diff^2/std^2 - 1)
    logStd += LR * advantage * (diff*diff/(std*std) - 1) / EP_LEN;
  }
  logStd = Math.max(-3, Math.min(2, logStd));

  if (ep % 50 === 0 || ep === 499) {
    // Evaluate deterministically
    let es = { th: (mulberry32(0xBEEF)()* 2 - 1) * Math.PI, w: 0 };
    let evalR = 0, evalUp = 0;
    for (let t = 0; t < EP_LEN; t++) {
      const obs = getObs(es);
      const mu = pw[0]*obs[0] + pw[1]*obs[1] + pw[2]*obs[2];
      const u = Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, mu));
      es = stepPend(es, u);
      evalR += pendReward(es, u);
      if (Math.abs(wrap(es.th)) < 0.3 && Math.abs(es.w) < 1) evalUp++;
    }
    console.log(`  ep ${String(ep).padStart(3)}: trainR=${meanR.toFixed(3)}, evalR=${(evalR/EP_LEN).toFixed(3)}, up=${(evalUp/EP_LEN*100).toFixed(0)}%, w=[${pw.map(x=>x.toFixed(3)).join(',')}], logStd=${logStd.toFixed(3)}`);
  }
}

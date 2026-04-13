// Test: Can ARS (Augmented Random Search) converge from ZERO gains
// to the optimal energy-shaping + LQR controller gains for pendubot?
//
// Controller structure (fixed, not learned):
//   swing-up: u = ke * (-Ẽ) * ω₁ - kp * sin(θ₁) - kd * ω₁
//   balance:  u = LQR(s)   when state ∈ basin of attraction
//
// ARS optimizes θ = [ke, kp, kd] starting from [0, 0, 0].
// Known-optimal from test-tuning4: ke=50, kp=1, kd=0.3

const L1=1, L2=1, M1=1, M2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;
const LQR_K = [-41.64, 101.62, -17.31, 29.14];
const E_TARGET = -(M1+M2)*G*L1 + M2*G*L2;

const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };
const energy = s =>
  0.5*(M1+M2)*L1*L1*s.w1*s.w1
  + 0.5*M2*L2*L2*s.w2*s.w2
  + M2*L1*L2*s.w1*s.w2*Math.cos(s.th1-s.th2)
  - (M1+M2)*G*L1*Math.cos(s.th1)
  - M2*G*L2*Math.cos(s.th2);

function deriv(s, u) {
  const { th1, th2, w1, w2 } = s;
  const d = th1-th2, cd = Math.cos(d), sd = Math.sin(d);
  const M11=(M1+M2)*L1*L1, M12=M2*L1*L2*cd, M22=M2*L2*L2, det=M11*M22-M12*M12;
  const C1=M2*L1*L2*sd*w2*w2, C2=-M2*L1*L2*sd*w1*w1;
  const G1=(M1+M2)*G*L1*Math.sin(th1), G2=M2*G*L2*Math.sin(th2);
  const r1=u-DAMPING*w1-C1-G1, r2=-DAMPING*w2-C2-G2;
  return [w1, w2, (M22*r1-M12*r2)/det, (-M12*r1+M11*r2)/det];
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

function lqrU(s) {
  const dth2 = wrap(s.th2 - Math.PI);
  const u = -(LQR_K[0]*s.th1 + LQR_K[1]*dth2 + LQR_K[2]*s.w1 + LQR_K[3]*s.w2);
  return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
}

function inBasin(s) {
  let t = { ...s };
  for (let i = 0; i < 40; i++) {
    t = stepEnv(t, lqrU(t));
    const d = Math.abs(wrap(t.th2 - Math.PI));
    if (!Number.isFinite(t.th1) || d > 1.0) return false;
  }
  const d = Math.abs(wrap(t.th2 - Math.PI));
  return d < 0.15 && Math.abs(t.w1) < 2 && Math.abs(t.w2) < 2;
}

function controller(s, ke, kp, kd) {
  if (inBasin(s)) return lqrU(s);
  const Etilde = energy(s) - E_TARGET;
  const u = ke * (-Etilde) * s.w1 - kp * Math.sin(s.th1) - kd * s.w1;
  return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
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

function runEpisode(ke, kp, kd, init, maxSteps = 500) {
  let s = { ...init };
  let totalReward = 0;
  let successSteps = 0;

  for (let i = 0; i < maxSteps; i++) {
    const u = controller(s, ke, kp, kd);
    s = stepEnv(s, u);
    if (!Number.isFinite(s.th1) || Math.abs(s.w1) > 50 || Math.abs(s.w2) > 50) {
      return { reward: totalReward - 100, successSteps };
    }

    const dth2 = Math.abs(wrap(s.th2 - Math.PI));
    const dth1 = Math.abs(wrap(s.th1));

    // Dense reward: pendubot proximity
    if (dth2 < 0.15 && dth1 < 0.3 && Math.abs(s.w1) < 1 && Math.abs(s.w2) < 1) {
      totalReward += 2.0;
      successSteps++;
    } else if (dth2 < 0.5) {
      totalReward += 0.5;
    }

    // Energy shaping: reward getting energy close to target
    const Eerr = Math.abs(energy(s) - E_TARGET);
    totalReward += 0.1 * Math.max(0, 1 - Eerr / 20);

    // Small penalty for th1 deviation
    totalReward -= 0.02 * dth1 * dth1;
  }

  return { reward: totalReward, successSteps };
}

function evaluate(gains, nEpisodes, rng) {
  let totalReward = 0;
  let totalSuccess = 0;
  for (let i = 0; i < nEpisodes; i++) {
    const init = {
      th1: (rng() - 0.5) * 0.6,
      th2: (rng() - 0.5) * 0.6,
      w1:  (rng() - 0.5) * 0.4,
      w2:  (rng() - 0.5) * 0.4,
    };
    const { reward, successSteps } = runEpisode(gains[0], gains[1], gains[2], init);
    totalReward += reward;
    totalSuccess += successSteps > 50 ? 1 : 0;
  }
  return { reward: totalReward / nEpisodes, successRate: totalSuccess / nEpisodes };
}

function std(arr) {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
}

function gaussianRandom(rng) {
  return Math.sqrt(-2 * Math.log(rng())) * Math.cos(2 * Math.PI * rng());
}

function arsOptimize(config) {
  const { nIter, nDirections, stepSize, noiseScale, nEvalEpisodes, initGains, label } = config;
  const rng = mulberry32(0xBEEF);

  let theta = [...initGains];
  // Per-parameter noise scale — ke needs much larger exploration than kd
  const paramScale = [20, 2, 0.5];

  console.log(`\n${"=".repeat(60)}`);
  console.log(`${label}`);
  console.log(`${"=".repeat(60)}`);
  console.log(`Init:   [${theta.map(t => t.toFixed(2))}]`);
  console.log(`Target: ~[50, 1, 0.3]  (known-optimal from test-tuning4)\n`);

  let bestReward = -Infinity;
  let bestTheta = [...theta];

  for (let iter = 0; iter < nIter; iter++) {
    const deltas = [];
    const rPlus = [];
    const rMinus = [];

    for (let i = 0; i < nDirections; i++) {
      const delta = paramScale.map(s => gaussianRandom(rng) * s * noiseScale);
      deltas.push(delta);

      const thetaPlus  = theta.map((t, j) => Math.max(0, t + delta[j]));
      const thetaMinus = theta.map((t, j) => Math.max(0, t - delta[j]));

      const evalRng1 = mulberry32(iter * 1000 + i * 2);
      const evalRng2 = mulberry32(iter * 1000 + i * 2 + 1);

      rPlus.push(evaluate(thetaPlus, nEvalEpisodes, evalRng1).reward);
      rMinus.push(evaluate(thetaMinus, nEvalEpisodes, evalRng2).reward);
    }

    // Use top-b directions (top half by max reward)
    const ranked = deltas.map((d, i) => ({
      d, rp: rPlus[i], rm: rMinus[i],
      maxR: Math.max(rPlus[i], rMinus[i]),
    }));
    ranked.sort((a, b) => b.maxR - a.maxR);
    const topB = Math.max(1, Math.floor(nDirections * 0.5));
    const used = ranked.slice(0, topB);

    const allR = used.flatMap(x => [x.rp, x.rm]);
    const sigmaR = Math.max(1e-6, std(allR));

    for (let j = 0; j < 3; j++) {
      let update = 0;
      for (const { d, rp, rm } of used) {
        update += (rp - rm) * d[j];
      }
      theta[j] += stepSize / (topB * sigmaR) * update;
    }

    // Clamp to reasonable range
    theta[0] = Math.max(0, Math.min(100, theta[0]));
    theta[1] = Math.max(0, Math.min(20, theta[1]));
    theta[2] = Math.max(0, Math.min(5, theta[2]));

    if (iter % 5 === 0 || iter === nIter - 1) {
      const evalRng = mulberry32(0xCAFE);
      const { reward, successRate } = evaluate(theta, 20, evalRng);
      if (reward > bestReward) { bestReward = reward; bestTheta = [...theta]; }
      console.log(
        `iter ${String(iter).padStart(3)}: ` +
        `θ=[${theta.map(t => t.toFixed(2).padStart(6))}]  ` +
        `reward=${reward.toFixed(1).padStart(7)}  ` +
        `success=${(successRate * 100).toFixed(0).padStart(3)}%`
      );
    }
  }

  console.log(`\nBest:   [${bestTheta.map(t => t.toFixed(2))}]  reward=${bestReward.toFixed(1)}`);
  return bestTheta;
}

// ---- Baselines ----
console.log("=== BASELINES ===");
{
  const rng0 = mulberry32(0xCAFE);
  const { reward: r0, successRate: sr0 } = evaluate([0, 0, 0], 20, rng0);
  console.log(`[0, 0, 0]       reward=${r0.toFixed(1)}  success=${(sr0*100).toFixed(0)}%`);

  const rng1 = mulberry32(0xCAFE);
  const { reward: r1, successRate: sr1 } = evaluate([50, 1, 0.3], 20, rng1);
  console.log(`[50, 1, 0.3]    reward=${r1.toFixed(1)}  success=${(sr1*100).toFixed(0)}%  (known-optimal)`);

  const rng2 = mulberry32(0xCAFE);
  const { reward: r2, successRate: sr2 } = evaluate([20, 1, 0.2], 20, rng2);
  console.log(`[20, 1, 0.2]    reward=${r2.toFixed(1)}  success=${(sr2*100).toFixed(0)}%`);

  const rng3 = mulberry32(0xCAFE);
  const { reward: r3, successRate: sr3 } = evaluate([5, 0.5, 0.1], 20, rng3);
  console.log(`[5, 0.5, 0.1]   reward=${r3.toFixed(1)}  success=${(sr3*100).toFixed(0)}%`);
}

// ---- Test 1: ARS from zero ----
arsOptimize({
  nIter: 100,
  nDirections: 16,
  stepSize: 0.05,
  noiseScale: 1.0,
  nEvalEpisodes: 8,
  initGains: [0, 0, 0],
  label: "TEST 1: ARS from [0, 0, 0]",
});

// ---- Test 2: ARS from small gains ----
arsOptimize({
  nIter: 100,
  nDirections: 16,
  stepSize: 0.05,
  noiseScale: 1.0,
  nEvalEpisodes: 8,
  initGains: [5, 0.5, 0.1],
  label: "TEST 2: ARS from [5, 0.5, 0.1]",
});

// ---- Test 3: Larger noise / step for escaping zero plateau ----
arsOptimize({
  nIter: 100,
  nDirections: 24,
  stepSize: 0.1,
  noiseScale: 1.5,
  nEvalEpisodes: 8,
  initGains: [0, 0, 0],
  label: "TEST 3: ARS from [0, 0, 0] (larger noise+step)",
});

// ---- Test 4: How many sim steps does convergence cost? ----
// Each ARS iteration: nDirections * 2 * nEvalEpisodes * 500 steps
// = 16 * 2 * 8 * 500 = 128,000 steps per iteration
// At 150 steps/frame (60fps), that's 128000/150 ≈ 853 frames ≈ 14 seconds per iteration
// 100 iterations = ~23 minutes — too slow for browser
//
// With nDirections=8, nEvalEpisodes=4:
// = 8 * 2 * 4 * 500 = 32,000 steps per iteration
// = 32000/150 ≈ 213 frames ≈ 3.6 seconds per iteration
// If converges in 30 iterations: ~108 seconds ≈ 1.8 minutes — feasible!
console.log("\n\n=== TEST 4: Budget-constrained ARS (browser-realistic) ===");
arsOptimize({
  nIter: 60,
  nDirections: 8,
  stepSize: 0.08,
  noiseScale: 1.2,
  nEvalEpisodes: 4,
  initGains: [0, 0, 0],
  label: "TEST 4: Budget ARS from [0, 0, 0] (8 dirs, 4 evals)",
});

// DQN agent test — validates convergence on pendulum swing-up in <10800 steps (3 min at 60fps)
const G_PHYS=10,MASS=1,LEN=1,MAX_TORQUE=2,DT=0.05,MAX_SPEED=8,EP_LEN=200;
const OBS_DIM=3;

// DQN hyperparameters
const N_ACTIONS=11;
const ACTION_TORQUES=Float64Array.from({length:N_ACTIONS},(_,i)=>-MAX_TORQUE+(2*MAX_TORQUE/(N_ACTIONS-1))*i);
const DQN_HIDDEN=64;
const REPLAY_CAP=20000;
const BATCH_SIZE=64;
const UPDATES_PER_STEP=4;
const DQN_LR=0.001;
const GAMMA=0.95;
const EPS_START=1.0;
const EPS_END=0.05;
const EPS_DECAY=3000;
const TARGET_SYNC=200;
const MIN_REPLAY=200;
const REWARD_SCALE=0.1;
const CONVERGE_FRAC=0.50;
const CONVERGE_HOLD=3;

const wrap=a=>{let x=(a+Math.PI)%(2*Math.PI);if(x<0)x+=2*Math.PI;return x-Math.PI};
function pendDeriv(th,w,u){return[w,(3*G_PHYS)/(2*LEN)*Math.sin(th)+3/(MASS*LEN*LEN)*u]}
function stepPend(s,u){const dt=DT;const k1=pendDeriv(s.th,s.w,u);const k2=pendDeriv(s.th+k1[0]*dt/2,s.w+k1[1]*dt/2,u);const k3=pendDeriv(s.th+k2[0]*dt/2,s.w+k2[1]*dt/2,u);const k4=pendDeriv(s.th+k3[0]*dt,s.w+k3[1]*dt,u);return{th:s.th+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,w:Math.max(-MAX_SPEED,Math.min(MAX_SPEED,s.w+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6))}}
function getObs(s){return Float64Array.of(Math.cos(s.th),Math.sin(s.th),s.w/MAX_SPEED)}
function pendReward(s,u){const th=wrap(s.th);return-(th*th+0.1*s.w*s.w+0.001*u*u)}
function mulberry32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function seededGauss(rng){return Math.sqrt(-2*Math.log(rng()+1e-10))*Math.cos(2*Math.PI*rng())}

// ---- Replay Buffer ----
class ReplayBuffer {
  constructor(cap) {
    this.cap = cap;
    this.stride = OBS_DIM + 1 + 1 + OBS_DIM; // s, a, r, s'
    this.buf = new Float64Array(cap * this.stride);
    this.sz = 0;
    this.pos = 0;
  }
  get size() { return this.sz; }
  push(obs, actionIdx, reward, nextObs) {
    const off = this.pos * this.stride;
    this.buf.set(obs, off);
    this.buf[off + OBS_DIM] = actionIdx;
    this.buf[off + OBS_DIM + 1] = reward;
    this.buf.set(nextObs, off + OBS_DIM + 2);
    this.pos = (this.pos + 1) % this.cap;
    if (this.sz < this.cap) this.sz++;
  }
  getObs(idx) { return this.buf.subarray(idx * this.stride, idx * this.stride + OBS_DIM); }
  getAction(idx) { return this.buf[idx * this.stride + OBS_DIM]; }
  getReward(idx) { return this.buf[idx * this.stride + OBS_DIM + 1]; }
  getNext(idx) { return this.buf.subarray(idx * this.stride + OBS_DIM + 2, idx * this.stride + OBS_DIM + 2 + OBS_DIM); }
}

// ---- DQN Network ----
class DQNNet {
  constructor(rng) {
    const S = [DQN_HIDDEN * OBS_DIM, DQN_HIDDEN, DQN_HIDDEN * DQN_HIDDEN, DQN_HIDDEN, N_ACTIONS * DQN_HIDDEN, N_ACTIONS];
    this.params = S.map(n => new Float64Array(n));
    this.grads = S.map(n => new Float64Array(n));
    this.adam_m = S.map(n => new Float64Array(n));
    this.adam_v = S.map(n => new Float64Array(n));

    // He init for hidden layers
    for (let i = 0; i < this.params[0].length; i++) this.params[0][i] = seededGauss(rng) * Math.sqrt(2 / OBS_DIM);
    for (let i = 0; i < this.params[2].length; i++) this.params[2][i] = seededGauss(rng) * Math.sqrt(2 / DQN_HIDDEN);
    // Small init for output layer
    for (let i = 0; i < this.params[4].length; i++) this.params[4][i] = seededGauss(rng) * 0.01;

    // Scratch buffers
    this.h1 = new Float64Array(DQN_HIDDEN);
    this.h2 = new Float64Array(DQN_HIDDEN);
    this.out = new Float64Array(N_ACTIONS);
    this.h1pre = new Float64Array(DQN_HIDDEN);
    this.h2pre = new Float64Array(DQN_HIDDEN);
    this.obsC = new Float64Array(OBS_DIM);
    this.dh2 = new Float64Array(DQN_HIDDEN);
    this.dh1 = new Float64Array(DQN_HIDDEN);
  }

  get W1(){return this.params[0]} get b1(){return this.params[1]}
  get W2(){return this.params[2]} get b2(){return this.params[3]}
  get Wo(){return this.params[4]} get bo(){return this.params[5]}

  forward(obs) {
    for (let i = 0; i < OBS_DIM; i++) this.obsC[i] = obs[i];
    const {W1,b1,W2,b2,Wo,bo,h1,h2,out,h1pre,h2pre} = this;
    for (let i = 0; i < DQN_HIDDEN; i++) {
      let s = b1[i]; for (let j = 0; j < OBS_DIM; j++) s += W1[i*OBS_DIM+j]*obs[j];
      h1pre[i] = s; h1[i] = s > 0 ? s : 0;
    }
    for (let i = 0; i < DQN_HIDDEN; i++) {
      let s = b2[i]; for (let j = 0; j < DQN_HIDDEN; j++) s += W2[i*DQN_HIDDEN+j]*h1[j];
      h2pre[i] = s; h2[i] = s > 0 ? s : 0;
    }
    for (let k = 0; k < N_ACTIONS; k++) {
      let s = bo[k]; for (let j = 0; j < DQN_HIDDEN; j++) s += Wo[k*DQN_HIDDEN+j]*h2[j];
      out[k] = s;
    }
    return out;
  }

  backward(action, target) {
    const {Wo,h1,h2,h1pre,h2pre,obsC,dh2,dh1} = this;
    const [gW1,gb1,gW2,gb2,gWo,gbo] = this.grads;
    const err = this.out[action] - target;
    const g = Math.abs(err) < 1 ? 2*err : 2*Math.sign(err); // Huber-like gradient clip

    dh2.fill(0);
    gbo[action] += g;
    for (let j = 0; j < DQN_HIDDEN; j++) {
      gWo[action*DQN_HIDDEN+j] += g*h2[j];
      dh2[j] = g*Wo[action*DQN_HIDDEN+j];
    }
    for (let j = 0; j < DQN_HIDDEN; j++) if (h2pre[j] <= 0) dh2[j] = 0;

    dh1.fill(0);
    for (let i = 0; i < DQN_HIDDEN; i++) {
      if (dh2[i] === 0) continue;
      gb2[i] += dh2[i];
      for (let j = 0; j < DQN_HIDDEN; j++) {
        gW2[i*DQN_HIDDEN+j] += dh2[i]*h1[j];
        dh1[j] += dh2[i]*this.W2[i*DQN_HIDDEN+j];
      }
    }
    for (let j = 0; j < DQN_HIDDEN; j++) if (h1pre[j] <= 0) dh1[j] = 0;

    for (let i = 0; i < DQN_HIDDEN; i++) {
      if (dh1[i] === 0) continue;
      gb1[i] += dh1[i];
      for (let j = 0; j < OBS_DIM; j++) gW1[i*OBS_DIM+j] += dh1[i]*obsC[j];
    }
  }

  zeroGrad() { for (const g of this.grads) g.fill(0); }

  adamStep(lr, t) {
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - b1**t, bc2 = 1 - b2**t;
    for (let p = 0; p < this.params.length; p++) {
      const param = this.params[p], grad = this.grads[p], m = this.adam_m[p], v = this.adam_v[p];
      for (let i = 0; i < param.length; i++) {
        m[i] = b1*m[i] + (1-b1)*grad[i];
        v[i] = b2*v[i] + (1-b2)*grad[i]*grad[i];
        param[i] -= lr * (m[i]/bc1) / (Math.sqrt(v[i]/bc2) + eps);
      }
    }
  }

  copyFrom(other) {
    for (let p = 0; p < this.params.length; p++) this.params[p].set(other.params[p]);
  }
}

// ---- DQN Agent ----
class DQNAgent {
  constructor(seed) {
    const rng = mulberry32(seed);
    this.rng = mulberry32(seed + 81);
    this.onlineNet = new DQNNet(rng);
    this.targetNet = new DQNNet(() => 0);
    this.targetNet.copyFrom(this.onlineNet);
    this.replay = new ReplayBuffer(REPLAY_CAP);
    this.epsilon = EPS_START;
    this.totalSteps = 0;
    this.adamT = 0;
    this.lastActionIdx = 0;
    this.updates = 0;
    this.converged = false;
    this.bestUpright = 0;
    this.convCount = 0;
    this.visSteps = 0;
    this.visRewardAcc = 0;
    this.visUprightAcc = 0;
    this.rewardHistory = [];
    this.uprightHistory = [];
  }

  act(obs, deterministic) {
    if (!deterministic && this.rng() < this.epsilon) {
      this.lastActionIdx = Math.floor(this.rng() * N_ACTIONS);
    } else {
      const q = this.onlineNet.forward(obs);
      let best = 0;
      for (let i = 1; i < N_ACTIONS; i++) if (q[i] > q[best]) best = i;
      this.lastActionIdx = best;
    }
    return ACTION_TORQUES[this.lastActionIdx];
  }

  step(prevState, torque, nextState) {
    const reward = pendReward(nextState, torque) * REWARD_SCALE;
    const obs = getObs(prevState);
    const nextObs = getObs(nextState);

    this.visRewardAcc += pendReward(nextState, torque);
    if (Math.abs(wrap(nextState.th)) < 0.3 && Math.abs(nextState.w) < 1) this.visUprightAcc++;
    this.visSteps++;

    this.replay.push(obs, this.lastActionIdx, reward, nextObs);
    this.totalSteps++;
    this.epsilon = Math.max(EPS_END, EPS_START - (EPS_START - EPS_END) * this.totalSteps / EPS_DECAY);

    if (this.replay.size >= MIN_REPLAY) {
      for (let u = 0; u < UPDATES_PER_STEP; u++) this.trainStep();
    }
    if (this.totalSteps % TARGET_SYNC === 0) this.targetNet.copyFrom(this.onlineNet);

    if (this.visSteps >= EP_LEN) {
      const epR = this.visRewardAcc / EP_LEN;
      const epUp = this.visUprightAcc / EP_LEN;
      this.rewardHistory.push(epR);
      this.uprightHistory.push(epUp);
      this.bestUpright = Math.max(this.bestUpright, epUp);
      if (!this.converged) {
        if (epUp >= CONVERGE_FRAC) this.convCount++;
        else this.convCount = Math.max(0, this.convCount - 1);
        if (this.convCount >= CONVERGE_HOLD) this.converged = true;
      }
      this.visRewardAcc = 0;
      this.visUprightAcc = 0;
      this.visSteps = 0;
      return { epR, epUp };
    }
    return null;
  }

  trainStep() {
    this.onlineNet.zeroGrad();
    for (let i = 0; i < BATCH_SIZE; i++) {
      const idx = Math.floor(this.rng() * this.replay.size);
      const sObs = this.replay.getObs(idx);
      const a = this.replay.getAction(idx);
      const r = this.replay.getReward(idx);
      const spObs = this.replay.getNext(idx);

      const tq = this.targetNet.forward(spObs);
      let maxQ = tq[0]; for (let k = 1; k < N_ACTIONS; k++) if (tq[k] > maxQ) maxQ = tq[k];
      const target = r + GAMMA * maxQ;

      this.onlineNet.forward(sObs);
      this.onlineNet.backward(a, target);
    }
    this.adamT++;
    this.onlineNet.adamStep(DQN_LR / BATCH_SIZE, this.adamT);
    this.updates++;
  }
}

// ---- Run Test ----
function runTest(seed) {
  const agent = new DQNAgent(seed);
  let env = { th: Math.PI, w: 0 }; // start hanging
  const t0 = Date.now();
  let convStep = -1;

  for (let step = 0; step < 10800; step++) { // 3 min at 60fps
    const prevState = env;
    const obs = getObs(prevState);
    const torque = agent.act(obs, false);
    env = stepPend(prevState, torque);

    const result = agent.step(prevState, torque, env);
    if (result) {
      const epNum = agent.rewardHistory.length;
      if (epNum % 5 === 0 || agent.converged) {
        const el = ((Date.now() - t0) / 1000).toFixed(1);
        console.log(`  ep ${String(epNum).padStart(3)} (step ${step}) | R=${result.epR.toFixed(2).padStart(7)} | up=${(result.epUp*100).toFixed(0).padStart(3)}% | ε=${agent.epsilon.toFixed(3)} | best=${(agent.bestUpright*100).toFixed(0)}% | ${el}s`);
      }
      if (agent.converged && convStep < 0) {
        convStep = step;
        console.log(`  >>> CONVERGED at step ${step} (ep ${epNum}), ${((Date.now()-t0)/1000).toFixed(1)}s <<<`);
      }
    }
  }

  const finalUp = agent.uprightHistory.length > 0 ? agent.uprightHistory[agent.uprightHistory.length - 1] : 0;
  return { seed, convStep, bestUpright: agent.bestUpright, finalUp, updates: agent.updates };
}

console.log("DQN Pendulum Swing-Up Test");
console.log("==========================\n");

const seeds = [42, 123, 777, 1337, 2024];
const results = [];
for (const seed of seeds) {
  console.log(`\nSeed ${seed}:`);
  const r = runTest(seed);
  results.push(r);
  console.log(`  Result: conv=${r.convStep>=0?'YES':'NO'} step=${r.convStep} best=${(r.bestUpright*100).toFixed(0)}% final=${(r.finalUp*100).toFixed(0)}% updates=${r.updates}`);
}

const convCount = results.filter(r => r.convStep >= 0).length;
console.log(`\n${convCount}/${results.length} converged within 10800 steps (3 min at 60fps)`);

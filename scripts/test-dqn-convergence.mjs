// Double DQN convergence test for pendubot swing-up.
// Inspired by Ben Hazem (Springer 2024): DQN learned Furuta pendulum
// swing-up in 1000 episodes on a laptop CPU.
//
// Key design choices:
//   - Discrete actions (bang-bang suits energy pumping)
//   - Experience replay for sample efficiency
//   - Double DQN to avoid Q-value overestimation
//   - Energy-shaped reward for dense gradient

const L1=1, L2=1, M1=1, M2=1, G=9.81, DAMPING=0.0015, DT=0.02, TORQUE_MAX=30;
const E_TARGET = -(M1+M2)*G*L1 + M2*G*L2;

const ACTIONS = [-30, -15, 0, 15, 30];
const N_ACT = ACTIONS.length;
const OBS_DIM = 6;
const HIDDEN = 32;

const wrap = a => { let x=(a+Math.PI)%(2*Math.PI); if(x<0)x+=2*Math.PI; return x-Math.PI; };
const energy = s =>
  0.5*(M1+M2)*L1*L1*s.w1*s.w1 + 0.5*M2*L2*L2*s.w2*s.w2
  + M2*L1*L2*s.w1*s.w2*Math.cos(s.th1-s.th2)
  - (M1+M2)*G*L1*Math.cos(s.th1) - M2*G*L2*Math.cos(s.th2);

function deriv(s, u) {
  const {th1,th2,w1,w2}=s, d=th1-th2, cd=Math.cos(d), sd=Math.sin(d);
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

function obsFromState(s) {
  return [Math.cos(s.th1), Math.sin(s.th1), Math.cos(s.th2), Math.sin(s.th2), s.w1*0.1, s.w2*0.1];
}

function rlReward(s) {
  const dth1 = wrap(s.th1);
  const rPos = -Math.cos(s.th2);
  const Eerr = Math.abs(energy(s) - E_TARGET);
  const rEnergy = Math.max(0, 1 - Eerr / 20);
  const rCenter = -0.5 * Math.abs(dth1);
  return rPos + 0.3 * rEnergy + rCenter;
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

// ===================== QNet =====================
class QNet {
  constructor(rng) {
    const r = rng;
    const xI = Math.sqrt(6/(OBS_DIM+HIDDEN));
    const xH = Math.sqrt(6/(HIDDEN+HIDDEN));
    const xO = Math.sqrt(6/(HIDDEN+N_ACT));
    this.w1 = Float64Array.from({length:HIDDEN*OBS_DIM}, ()=>(r()-0.5)*2*xI);
    this.b1 = new Float64Array(HIDDEN);
    this.w2 = Float64Array.from({length:HIDDEN*HIDDEN}, ()=>(r()-0.5)*2*xH);
    this.b2 = new Float64Array(HIDDEN);
    this.w3 = Float64Array.from({length:N_ACT*HIDDEN}, ()=>(r()-0.5)*2*xO);
    this.b3 = new Float64Array(N_ACT);
    this.allP = [this.w1,this.b1,this.w2,this.b2,this.w3,this.b3];
    this.mArr = this.allP.map(p=>new Float64Array(p.length));
    this.vArr = this.allP.map(p=>new Float64Array(p.length));
    this.t = 0;
  }

  forward(x) {
    const z1=new Float64Array(HIDDEN), h1=new Float64Array(HIDDEN);
    for (let i=0;i<HIDDEN;i++){
      let s=this.b1[i];
      for(let j=0;j<OBS_DIM;j++) s+=this.w1[i*OBS_DIM+j]*x[j];
      z1[i]=s; h1[i]=s>0?s:0;
    }
    const z2=new Float64Array(HIDDEN), h2=new Float64Array(HIDDEN);
    for (let i=0;i<HIDDEN;i++){
      let s=this.b2[i];
      for(let j=0;j<HIDDEN;j++) s+=this.w2[i*HIDDEN+j]*h1[j];
      z2[i]=s; h2[i]=s>0?s:0;
    }
    const q=new Float64Array(N_ACT);
    for (let i=0;i<N_ACT;i++){
      let s=this.b3[i];
      for(let j=0;j<HIDDEN;j++) s+=this.w3[i*HIDDEN+j]*h2[j];
      q[i]=s;
    }
    return {q,h1,h2,z1,z2};
  }

  bestAction(x) {
    const{q}=this.forward(x);
    let b=0; for(let i=1;i<N_ACT;i++) if(q[i]>q[b]) b=i;
    return b;
  }

  train(batch, tgtNet, gamma, lr) {
    this.t++;
    const G = this.allP.map(p=>new Float64Array(p.length));
    const n = batch.length;
    for (const {s,a,r,sp,done} of batch) {
      const {q,h1,h2,z1,z2} = this.forward(s);
      let target = r;
      if (!done) {
        const aB = this.bestAction(sp);
        const {q:qT} = tgtNet.forward(sp);
        target = r + gamma * qT[aB];
      }
      let td = q[a] - target;
      td = Math.max(-10, Math.min(10, td));

      // Layer 3 grads
      for (let j=0;j<HIDDEN;j++) G[4][a*HIDDEN+j] += td*h2[j];
      G[5][a] += td;

      // dh2
      const dh2 = new Float64Array(HIDDEN);
      for (let j=0;j<HIDDEN;j++) dh2[j] = z2[j]>0 ? td*this.w3[a*HIDDEN+j] : 0;

      // Layer 2 grads
      for (let i=0;i<HIDDEN;i++){
        G[3][i] += dh2[i];
        for(let j=0;j<HIDDEN;j++) G[2][i*HIDDEN+j] += dh2[i]*h1[j];
      }

      // dh1
      const dh1 = new Float64Array(HIDDEN);
      for (let j=0;j<HIDDEN;j++){
        let sum=0;
        for(let i=0;i<HIDDEN;i++) sum+=dh2[i]*this.w2[i*HIDDEN+j];
        dh1[j] = z1[j]>0 ? sum : 0;
      }

      // Layer 1 grads
      for (let i=0;i<HIDDEN;i++){
        G[1][i] += dh1[i];
        for(let j=0;j<OBS_DIM;j++) G[0][i*OBS_DIM+j] += dh1[i]*s[j];
      }
    }

    // Adam
    const b1=0.9, b2=0.999, eps=1e-8;
    const bc1=1-Math.pow(b1,this.t), bc2=1-Math.pow(b2,this.t);
    for (let p=0;p<this.allP.length;p++){
      const par=this.allP[p], g=G[p], m=this.mArr[p], v=this.vArr[p];
      for(let i=0;i<par.length;i++){
        const gi=g[i]/n;
        m[i]=b1*m[i]+(1-b1)*gi;
        v[i]=b2*v[i]+(1-b2)*gi*gi;
        par[i]-=lr*(m[i]/bc1)/(Math.sqrt(v[i]/bc2)+eps);
      }
    }
  }

  copyFrom(o) {
    for(let p=0;p<this.allP.length;p++) this.allP[p].set(o.allP[p]);
  }
}

// ===================== Replay Buffer =====================
class ReplayBuffer {
  constructor(max) { this.buf=[]; this.max=max; this.idx=0; }
  push(t) {
    if(this.buf.length<this.max) this.buf.push(t);
    else this.buf[this.idx]=t;
    this.idx=(this.idx+1)%this.max;
  }
  sample(n, rng) {
    const out=[];
    for(let i=0;i<n;i++) out.push(this.buf[Math.floor(rng()*this.buf.length)]);
    return out;
  }
  get size() { return this.buf.length; }
}

// ===================== Training =====================
const GAMMA = 0.99;
const LR = 5e-4;
const BATCH = 32;
const BUF_SIZE = 50000;
const TGT_SYNC = 1000;
const MIN_BUF = 500;
const EP_STEPS = 500;
const N_EP = 2000;
const EPS_START = 1.0, EPS_END = 0.05, EPS_DECAY = 800;

function evaluate(net, rng, nEval=20) {
  let totalR=0, nSucc=0;
  for (let i=0;i<nEval;i++){
    let s={th1:(rng()-0.5)*0.6,th2:(rng()-0.5)*0.6,w1:(rng()-0.5)*0.4,w2:(rng()-0.5)*0.4};
    let epR=0, succSteps=0;
    for(let t=0;t<EP_STEPS;t++){
      const o=obsFromState(s);
      s=stepEnv(s, ACTIONS[net.bestAction(o)]);
      if(!Number.isFinite(s.th1)||Math.abs(s.w1)>50||Math.abs(s.w2)>50) break;
      epR+=rlReward(s);
      const d2=Math.abs(wrap(s.th2-Math.PI)), d1=Math.abs(wrap(s.th1));
      if(d2<0.15&&d1<0.3&&Math.abs(s.w1)<1&&Math.abs(s.w2)<1) succSteps++;
    }
    totalR+=epR;
    if(succSteps>30) nSucc++;
  }
  return {meanR:totalR/nEval, rate:nSucc/nEval};
}

function train(label, actions, hidden, lr, gamma, epSteps, nEp) {
  const ACTS = actions;
  const NA = ACTS.length;
  const H = hidden;

  // Override globals for this run (hacky but fine for test)
  const origNA = N_ACT;
  // We'll just use the global QNet which uses N_ACT and HIDDEN globals
  // So we run with the global defaults for now.

  const rng = mulberry32(0xDEAD);
  const online = new QNet(rng);
  const target = new QNet(rng);
  target.copyFrom(online);
  const buffer = new ReplayBuffer(BUF_SIZE);

  let totalSteps=0;
  const t0=Date.now();

  console.log(`\n${"=".repeat(60)}`);
  console.log(label);
  console.log(`${"=".repeat(60)}`);
  console.log(`actions=${JSON.stringify(ACTIONS)} hidden=${HIDDEN} lr=${LR} γ=${GAMMA}`);
  console.log(`episodes=${nEp} steps/ep=${epSteps} batch=${BATCH}\n`);
  console.log("  ep  |  ε    | eval rate | eval meanR | total steps | time");
  console.log("-".repeat(68));

  for(let ep=0;ep<nEp;ep++){
    const eps=Math.max(EPS_END, EPS_START-(EPS_START-EPS_END)*ep/EPS_DECAY);
    let s={th1:(rng()-0.5)*0.6,th2:(rng()-0.5)*0.6,w1:(rng()-0.5)*0.4,w2:(rng()-0.5)*0.4};

    for(let t=0;t<epSteps;t++){
      const o=obsFromState(s);
      const aIdx = rng()<eps ? Math.floor(rng()*N_ACT) : online.bestAction(o);
      const ns=stepEnv(s, ACTIONS[aIdx]);
      const blowup=!Number.isFinite(ns.th1)||Math.abs(ns.w1)>50||Math.abs(ns.w2)>50;
      const r = blowup ? -10 : rlReward(ns);
      buffer.push({s:o, a:aIdx, r, sp:blowup?o:obsFromState(ns), done:blowup});
      if(blowup) break;
      s=ns;
      if(buffer.size>=MIN_BUF && t%4===0){
        online.train(buffer.sample(BATCH,rng), target, GAMMA, LR);
      }
      totalSteps++;
      if(totalSteps%TGT_SYNC===0) target.copyFrom(online);
    }

    if(ep%100===0 || ep===nEp-1){
      const ev=evaluate(online, mulberry32(0xCAFE));
      const elapsed=((Date.now()-t0)/1000).toFixed(1);
      console.log(
        `${String(ep).padStart(5)} | ${eps.toFixed(3)} | `+
        `${(ev.rate*100).toFixed(0).padStart(7)}%  | `+
        `${ev.meanR.toFixed(1).padStart(10)} | `+
        `${String(totalSteps).padStart(11)} | ${elapsed}s`
      );
    }
  }

  const wallSec = (Date.now()-t0)/1000;
  console.log(`\nTotal steps: ${totalSteps}`);
  console.log(`Wall time: ${wallSec.toFixed(1)}s`);
  console.log(`Browser estimate (150 steps/frame, 60fps): ${(totalSteps/150/60).toFixed(0)}s`);
}

train(
  "Double DQN: 5 actions, H=32, lr=5e-4",
  ACTIONS, HIDDEN, LR, GAMMA, EP_STEPS, N_EP
);

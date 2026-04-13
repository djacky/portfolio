// Double DQN + curriculum learning for pendubot swing-up.
//
// Key insight: raw DQN can't discover swing-up by random exploration.
// Curriculum starts episodes near θ₂=π (easy balance), then gradually
// shifts toward θ₂=0 (full swing-up). This mirrors the browser demo:
// the USER is the curriculum — they challenge the agent with harder
// starting positions as it improves.

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
  const rPos = -Math.cos(s.th2);
  const dth1 = wrap(s.th1);
  const rCenter = -0.5 * dth1 * dth1;
  const dth2 = Math.abs(wrap(s.th2 - Math.PI));
  const inZone = dth2 < 0.15 && Math.abs(dth1) < 0.3 && Math.abs(s.w1) < 1 && Math.abs(s.w2) < 1;
  return rPos + rCenter + (inZone ? 10 : 0);
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

// Curriculum: θ₂ center shifts from π (inverted) → 0 (hanging) over training
function curriculumIC(ep, maxEp, rng) {
  const progress = Math.min(1, ep / (maxEp * 0.7));
  const th2Center = Math.PI * (1 - progress);
  return {
    th1: (rng()-0.5) * 0.6,
    th2: th2Center + (rng()-0.5) * 0.8,
    w1:  (rng()-0.5) * 0.5,
    w2:  (rng()-0.5) * 0.5,
  };
}

// ===================== QNet =====================
class QNet {
  constructor(rng) {
    const r = rng;
    const xI = Math.sqrt(6/(OBS_DIM+HIDDEN)), xH = Math.sqrt(6/(HIDDEN+HIDDEN)), xO = Math.sqrt(6/(HIDDEN+N_ACT));
    this.w1 = Float64Array.from({length:HIDDEN*OBS_DIM}, ()=>(r()-0.5)*2*xI);
    this.b1 = new Float64Array(HIDDEN);
    this.w2 = Float64Array.from({length:HIDDEN*HIDDEN}, ()=>(r()-0.5)*2*xH);
    this.b2 = new Float64Array(HIDDEN);
    this.w3 = Float64Array.from({length:N_ACT*HIDDEN}, ()=>(r()-0.5)*2*xO);
    this.b3 = new Float64Array(N_ACT);
    this.allP = [this.w1,this.b1,this.w2,this.b2,this.w3,this.b3];
    this.mA = this.allP.map(p=>new Float64Array(p.length));
    this.vA = this.allP.map(p=>new Float64Array(p.length));
    this.t = 0;
  }

  forward(x) {
    const z1=new Float64Array(HIDDEN), h1=new Float64Array(HIDDEN);
    for(let i=0;i<HIDDEN;i++){ let s=this.b1[i]; for(let j=0;j<OBS_DIM;j++) s+=this.w1[i*OBS_DIM+j]*x[j]; z1[i]=s; h1[i]=s>0?s:0; }
    const z2=new Float64Array(HIDDEN), h2=new Float64Array(HIDDEN);
    for(let i=0;i<HIDDEN;i++){ let s=this.b2[i]; for(let j=0;j<HIDDEN;j++) s+=this.w2[i*HIDDEN+j]*h1[j]; z2[i]=s; h2[i]=s>0?s:0; }
    const q=new Float64Array(N_ACT);
    for(let i=0;i<N_ACT;i++){ let s=this.b3[i]; for(let j=0;j<HIDDEN;j++) s+=this.w3[i*HIDDEN+j]*h2[j]; q[i]=s; }
    return {q,h1,h2,z1,z2};
  }

  bestAction(x) {
    const{q}=this.forward(x); let b=0; for(let i=1;i<N_ACT;i++) if(q[i]>q[b]) b=i; return b;
  }

  train(batch, tgtNet, gamma, lr) {
    this.t++;
    const G=this.allP.map(p=>new Float64Array(p.length));
    const n=batch.length;
    for(const {s,a,r,sp,done} of batch){
      const{q,h1,h2,z1,z2}=this.forward(s);
      let target=r;
      if(!done){ const aB=this.bestAction(sp); const{q:qT}=tgtNet.forward(sp); target=r+gamma*qT[aB]; }
      let td=q[a]-target;
      td=Math.max(-10,Math.min(10,td));
      for(let j=0;j<HIDDEN;j++) G[4][a*HIDDEN+j]+=td*h2[j];
      G[5][a]+=td;
      const dh2=new Float64Array(HIDDEN);
      for(let j=0;j<HIDDEN;j++) dh2[j]=z2[j]>0?td*this.w3[a*HIDDEN+j]:0;
      for(let i=0;i<HIDDEN;i++){ G[3][i]+=dh2[i]; for(let j=0;j<HIDDEN;j++) G[2][i*HIDDEN+j]+=dh2[i]*h1[j]; }
      const dh1=new Float64Array(HIDDEN);
      for(let j=0;j<HIDDEN;j++){ let sum=0; for(let i=0;i<HIDDEN;i++) sum+=dh2[i]*this.w2[i*HIDDEN+j]; dh1[j]=z1[j]>0?sum:0; }
      for(let i=0;i<HIDDEN;i++){ G[1][i]+=dh1[i]; for(let j=0;j<OBS_DIM;j++) G[0][i*OBS_DIM+j]+=dh1[i]*s[j]; }
    }
    const b1c=0.9,b2c=0.999,eps=1e-8;
    const bc1=1-Math.pow(b1c,this.t),bc2=1-Math.pow(b2c,this.t);
    for(let p=0;p<this.allP.length;p++){
      const par=this.allP[p],g=G[p],m=this.mA[p],v=this.vA[p];
      for(let i=0;i<par.length;i++){
        const gi=g[i]/n; m[i]=b1c*m[i]+(1-b1c)*gi; v[i]=b2c*v[i]+(1-b2c)*gi*gi;
        par[i]-=lr*(m[i]/bc1)/(Math.sqrt(v[i]/bc2)+eps);
      }
    }
  }

  copyFrom(o){ for(let p=0;p<this.allP.length;p++) this.allP[p].set(o.allP[p]); }
}

class ReplayBuffer {
  constructor(max){ this.buf=[]; this.max=max; this.idx=0; }
  push(t){ if(this.buf.length<this.max) this.buf.push(t); else this.buf[this.idx]=t; this.idx=(this.idx+1)%this.max; }
  sample(n,rng){ const out=[]; for(let i=0;i<n;i++) out.push(this.buf[Math.floor(rng()*this.buf.length)]); return out; }
  get size(){ return this.buf.length; }
}

// ===================== Eval =====================
function evalFromIC(net, icFn, rng, nEval=20) {
  let totalR=0, nSucc=0;
  for(let i=0;i<nEval;i++){
    let s=icFn(rng);
    let epR=0, succSteps=0;
    for(let t=0;t<500;t++){
      s=stepEnv(s, ACTIONS[net.bestAction(obsFromState(s))]);
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

const hangingIC = rng => ({th1:(rng()-0.5)*0.6, th2:(rng()-0.5)*0.6, w1:(rng()-0.5)*0.4, w2:(rng()-0.5)*0.4});
const nearUpIC  = rng => ({th1:(rng()-0.5)*0.4, th2:Math.PI+(rng()-0.5)*0.6, w1:(rng()-0.5)*0.3, w2:(rng()-0.5)*0.3});

// ===================== Training =====================
function runTest(label, useCurriculum, nEp, bufSize) {
  const GAMMA=0.99, LR=5e-4, BATCH=32, TGT_SYNC=500, MIN_BUF=500, EP_STEPS=500;
  const EPS_START=1.0, EPS_END=0.05, EPS_DECAY=useCurriculum?500:800;

  const rng = mulberry32(0xDEAD);
  const online = new QNet(rng);
  const target = new QNet(rng);
  target.copyFrom(online);
  const buffer = new ReplayBuffer(bufSize);

  let totalSteps=0;
  const t0=Date.now();

  console.log(`\n${"=".repeat(65)}`);
  console.log(label);
  console.log(`${"=".repeat(65)}`);
  console.log(`curriculum=${useCurriculum} episodes=${nEp} buffer=${bufSize}`);
  console.log(`actions=${JSON.stringify(ACTIONS)} hidden=${HIDDEN} lr=${LR} γ=${GAMMA}\n`);
  console.log("  ep  |  ε    | curriculum θ₂ | hang rate | hang meanR | near rate | time");
  console.log("-".repeat(80));

  for(let ep=0;ep<nEp;ep++){
    const eps=Math.max(EPS_END, EPS_START-(EPS_START-EPS_END)*ep/EPS_DECAY);

    let s;
    if(useCurriculum){
      s = curriculumIC(ep, nEp, rng);
    } else {
      s = hangingIC(rng);
    }

    for(let t=0;t<EP_STEPS;t++){
      const o=obsFromState(s);
      const aIdx = rng()<eps ? Math.floor(rng()*N_ACT) : online.bestAction(o);
      const ns=stepEnv(s, ACTIONS[aIdx]);
      const blowup=!Number.isFinite(ns.th1)||Math.abs(ns.w1)>50||Math.abs(ns.w2)>50;
      const r=blowup?-10:rlReward(ns);
      buffer.push({s:o, a:aIdx, r, sp:blowup?o:obsFromState(ns), done:blowup});
      if(blowup) break;
      s=ns;
      if(buffer.size>=MIN_BUF && t%4===0) online.train(buffer.sample(BATCH,rng), target, GAMMA, LR);
      totalSteps++;
      if(totalSteps%TGT_SYNC===0) target.copyFrom(online);
    }

    if(ep%100===0 || ep===nEp-1){
      const evHang=evalFromIC(online, hangingIC, mulberry32(0xCAFE));
      const evNear=evalFromIC(online, nearUpIC, mulberry32(0xBEEF));
      const elapsed=((Date.now()-t0)/1000).toFixed(1);
      const th2c = useCurriculum ? (Math.PI*(1-Math.min(1,ep/(nEp*0.7)))).toFixed(2) : "n/a";
      console.log(
        `${String(ep).padStart(5)} | ${eps.toFixed(3)} | `+
        `${th2c.toString().padStart(13)} | `+
        `${(evHang.rate*100).toFixed(0).padStart(7)}%  | `+
        `${evHang.meanR.toFixed(1).padStart(10)} | `+
        `${(evNear.rate*100).toFixed(0).padStart(7)}%  | ${elapsed}s`
      );
    }
  }

  const wallSec=(Date.now()-t0)/1000;
  console.log(`\nTotal steps: ${totalSteps}  Wall: ${wallSec.toFixed(1)}s`);
  console.log(`Browser estimate (150 steps/frame, 60fps): ${(totalSteps/150/60).toFixed(0)}s`);
}

// Test 1: No curriculum (baseline)
runTest("TEST 1: No curriculum (baseline)", false, 1500, 50000);

// Test 2: With curriculum
runTest("TEST 2: With curriculum (θ₂: π→0 over 70% of training)", true, 1500, 20000);

// Test 3: Curriculum + smaller buffer (faster forgetting of old data)
runTest("TEST 3: Curriculum + small buffer (10k)", true, 1500, 10000);

// DQN v2: test multiple configurations to find one that converges in <10800 steps
const G=10,M=1,L=1,MT=2,DT=0.05,MS=8,EL=200,OD=3;
const wrap=a=>{let x=(a+Math.PI)%(2*Math.PI);if(x<0)x+=2*Math.PI;return x-Math.PI};
function pDeriv(th,w,u){return[w,(3*G)/(2*L)*Math.sin(th)+3/(M*L*L)*u]}
function step(s,u){const dt=DT,k1=pDeriv(s.th,s.w,u),k2=pDeriv(s.th+k1[0]*dt/2,s.w+k1[1]*dt/2,u),k3=pDeriv(s.th+k2[0]*dt/2,s.w+k2[1]*dt/2,u),k4=pDeriv(s.th+k3[0]*dt,s.w+k3[1]*dt,u);return{th:s.th+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,w:Math.max(-MS,Math.min(MS,s.w+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6))}}
function obs(s){return[Math.cos(s.th),Math.sin(s.th),s.w/MS]}
function rew(s,u){const th=wrap(s.th);return-(th*th+0.1*s.w*s.w+0.001*u*u)}
function mb32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function sGauss(rng){return Math.sqrt(-2*Math.log(rng()+1e-10))*Math.cos(2*Math.PI*rng())}

// ---- Config A: Tabular Q-learning (most sample efficient) ----
function runTabular(seed) {
  const rng = mb32(seed);
  const N_TH=36, N_W=21, NA=7;
  const torques = Float64Array.from({length:NA},(_,i)=>-MT+(2*MT/(NA-1))*i);
  const Q = new Float64Array(N_TH * N_W * NA); // init 0 = optimistic
  const alpha = 0.3, gamma = 0.99;
  let eps = 1.0;

  function stateIdx(s) {
    let th = wrap(s.th);
    let ti = Math.floor((th + Math.PI) / (2*Math.PI) * N_TH);
    if (ti >= N_TH) ti = N_TH - 1; if (ti < 0) ti = 0;
    let wi = Math.floor((s.w + MS) / (2*MS) * N_W);
    if (wi >= N_W) wi = N_W - 1; if (wi < 0) wi = 0;
    return ti * N_W + wi;
  }

  function act(s) {
    const si = stateIdx(s);
    if (rng() < eps) return Math.floor(rng() * NA);
    let best = 0, bestQ = Q[si * NA];
    for (let a = 1; a < NA; a++) if (Q[si*NA+a] > bestQ) { bestQ = Q[si*NA+a]; best = a; }
    return best;
  }

  let env = {th: Math.PI, w: 0};
  let visSteps=0, visR=0, visUp=0, convCount=0, converged=false, convStep=-1;
  let bestUp=0;
  const epRewards=[], epUprights=[];

  for (let t = 0; t < 10800; t++) {
    const si = stateIdx(env);
    const ai = act(env);
    const torque = torques[ai];
    const next = step(env, torque);
    const r = rew(next, torque);
    const si2 = stateIdx(next);

    // Q-learning update
    let maxQ2 = Q[si2*NA];
    for (let a = 1; a < NA; a++) if (Q[si2*NA+a] > maxQ2) maxQ2 = Q[si2*NA+a];
    Q[si*NA+ai] += alpha * (r + gamma * maxQ2 - Q[si*NA+ai]);

    env = next;
    visR += r;
    if (Math.abs(wrap(next.th)) < 0.3 && Math.abs(next.w) < 1) visUp++;
    visSteps++;
    eps = Math.max(0.01, 1.0 - t / 2000);

    if (visSteps >= EL) {
      const epR = visR / EL, epUp = visUp / EL;
      epRewards.push(epR); epUprights.push(epUp);
      bestUp = Math.max(bestUp, epUp);
      if (!converged) {
        if (epUp >= 0.5) convCount++; else convCount = Math.max(0, convCount-1);
        if (convCount >= 3) { converged = true; convStep = t; }
      }
      visR = 0; visUp = 0; visSteps = 0;
    }
  }
  const finalUp = epUprights.length > 0 ? epUprights[epUprights.length-1] : 0;
  return { convStep, bestUp, finalUp, epRewards, epUprights };
}

// ---- Config B: DQN no clipping, aggressive replay ----
class Buf{constructor(cap){this.cap=cap;this.s=OD+1+1+OD;this.b=new Float64Array(cap*this.s);this.sz=0;this.p=0}
get size(){return this.sz}
push(o,a,r,n){const off=this.p*this.s;this.b.set(o,off);this.b[off+OD]=a;this.b[off+OD+1]=r;this.b.set(n,off+OD+2);this.p=(this.p+1)%this.cap;if(this.sz<this.cap)this.sz++}
gO(i){return this.b.subarray(i*this.s,i*this.s+OD)}
gA(i){return this.b[i*this.s+OD]}
gR(i){return this.b[i*this.s+OD+1]}
gN(i){return this.b.subarray(i*this.s+OD+2,i*this.s+OD+2+OD)}}

class Net{
constructor(rng,H,NA){
this.H=H;this.NA=NA;
const S=[H*OD,H,H*H,H,NA*H,NA];
this.P=S.map(n=>new Float64Array(n));this.G=S.map(n=>new Float64Array(n));
this.am=S.map(n=>new Float64Array(n));this.av=S.map(n=>new Float64Array(n));
for(let i=0;i<this.P[0].length;i++)this.P[0][i]=sGauss(rng)*Math.sqrt(2/OD);
for(let i=0;i<this.P[2].length;i++)this.P[2][i]=sGauss(rng)*Math.sqrt(2/H);
for(let i=0;i<this.P[4].length;i++)this.P[4][i]=sGauss(rng)*0.01;
this.h1=new Float64Array(H);this.h2=new Float64Array(H);this.out=new Float64Array(NA);
this.h1p=new Float64Array(H);this.h2p=new Float64Array(H);this.oC=new Float64Array(OD);
this.dh2=new Float64Array(H);this.dh1=new Float64Array(H);
}
fwd(o){const{H,NA}=this;const[W1,b1,W2,b2,Wo,bo]=this.P;
for(let i=0;i<OD;i++)this.oC[i]=o[i];
for(let i=0;i<H;i++){let s=b1[i];for(let j=0;j<OD;j++)s+=W1[i*OD+j]*o[j];this.h1p[i]=s;this.h1[i]=s>0?s:0}
for(let i=0;i<H;i++){let s=b2[i];for(let j=0;j<H;j++)s+=W2[i*H+j]*this.h1[j];this.h2p[i]=s;this.h2[i]=s>0?s:0}
for(let k=0;k<NA;k++){let s=bo[k];for(let j=0;j<H;j++)s+=Wo[k*H+j]*this.h2[j];this.out[k]=s}
return this.out}
bwd(a,tgt){const{H}=this;const[gW1,gb1,gW2,gb2,gWo,gbo]=this.G;const Wo=this.P[4],W2=this.P[2];
const err=this.out[a]-tgt;const g=2*err; // pure MSE, no clipping
this.dh2.fill(0);gbo[a]+=g;
for(let j=0;j<H;j++){gWo[a*H+j]+=g*this.h2[j];this.dh2[j]=g*Wo[a*H+j]}
for(let j=0;j<H;j++)if(this.h2p[j]<=0)this.dh2[j]=0;
this.dh1.fill(0);
for(let i=0;i<H;i++){if(this.dh2[i]===0)continue;gb2[i]+=this.dh2[i];
for(let j=0;j<H;j++){gW2[i*H+j]+=this.dh2[i]*this.h1[j];this.dh1[j]+=this.dh2[i]*W2[i*H+j]}}
for(let j=0;j<H;j++)if(this.h1p[j]<=0)this.dh1[j]=0;
for(let i=0;i<H;i++){if(this.dh1[i]===0)continue;gb1[i]+=this.dh1[i];
for(let j=0;j<OD;j++)gW1[i*OD+j]+=this.dh1[i]*this.oC[j]}}
zg(){for(const g of this.G)g.fill(0)}
adam(lr,t){const b1=0.9,b2=0.999,e=1e-8,bc1=1-b1**t,bc2=1-b2**t;
for(let p=0;p<this.P.length;p++){const P=this.P[p],G=this.G[p],m=this.am[p],v=this.av[p];
for(let i=0;i<P.length;i++){m[i]=b1*m[i]+(1-b1)*G[i];v[i]=b2*v[i]+(1-b2)*G[i]*G[i];
P[i]-=lr*(m[i]/bc1)/(Math.sqrt(v[i]/bc2)+e)}}}
copy(o){for(let p=0;p<this.P.length;p++)this.P[p].set(o.P[p])}
}

function runDQN(seed, cfg) {
  const {H,NA,UPS,LR,GAM,ED,BS,RS,TAU} = cfg;
  const torques = Float64Array.from({length:NA},(_,i)=>-MT+(2*MT/(NA-1))*i);
  const rng = mb32(seed), rng2 = mb32(seed+77);
  const online = new Net(rng, H, NA);
  const target = new Net(()=>0, H, NA); target.copy(online);
  const replay = new Buf(20000);
  let eps=1.0, adamT=0, updates=0, lastA=0;
  let env={th:Math.PI,w:0};
  let visSteps=0,visR=0,visUp=0,convCount=0,converged=false,convStep=-1,bestUp=0;
  const epR=[],epU=[];

  for (let t = 0; t < 10800; t++) {
    const o = obs(env);
    if (rng2() < eps) { lastA = Math.floor(rng2()*NA); }
    else { const q=online.fwd(o);let b=0;for(let i=1;i<NA;i++)if(q[i]>q[b])b=i;lastA=b; }
    const torque = torques[lastA];
    const next = step(env, torque);
    const r = rew(next, torque) * RS;

    replay.push(Float64Array.from(o), lastA, r, Float64Array.from(obs(next)));
    env = next;
    visR += rew(next, torque);
    if (Math.abs(wrap(next.th))<0.3 && Math.abs(next.w)<1) visUp++;
    visSteps++;
    eps = Math.max(0.01, 1.0 - t/ED);

    if (replay.size >= 200) {
      for (let u = 0; u < UPS; u++) {
        online.zg();
        for (let i = 0; i < BS; i++) {
          const idx = Math.floor(rng2()*replay.size);
          const tq = target.fwd(replay.gN(idx));
          let mQ=tq[0];for(let k=1;k<NA;k++)if(tq[k]>mQ)mQ=tq[k];
          const tgt = replay.gR(idx) + GAM*mQ;
          online.fwd(replay.gO(idx));
          online.bwd(replay.gA(idx), tgt);
        }
        adamT++;
        online.adam(LR/BS, adamT);
        updates++;
      }
    }
    // Soft target update
    if (TAU > 0) {
      for (let p=0;p<online.P.length;p++) {
        const oP=online.P[p],tP=target.P[p];
        for(let i=0;i<oP.length;i++) tP[i]=TAU*oP[i]+(1-TAU)*tP[i];
      }
    } else if (t % 200 === 0) { target.copy(online); }

    if (visSteps >= EL) {
      const eR=visR/EL, eU=visUp/EL;
      epR.push(eR); epU.push(eU);
      bestUp = Math.max(bestUp, eU);
      if (!converged) { if(eU>=0.5)convCount++;else convCount=Math.max(0,convCount-1);if(convCount>=3){converged=true;convStep=t;}}
      visR=0;visUp=0;visSteps=0;
    }
  }
  return { convStep, bestUp, finalUp: epU.length?epU[epU.length-1]:0, updates };
}

// ---- Run all configs ----
console.log("=== CONFIG A: Tabular Q-learning ===");
for (const seed of [42,123,777,1337,2024]) {
  const t0=Date.now();
  const r=runTabular(seed);
  const el=((Date.now()-t0)/1000).toFixed(1);
  console.log(`  seed=${seed}: conv=${r.convStep>=0?`@${r.convStep}`:'NO'} best=${(r.bestUp*100).toFixed(0)}% final=${(r.finalUp*100).toFixed(0)}% (${el}s)`);
}

console.log("\n=== CONFIG B: DQN 2x32, no clip, 8ups, LR=0.003, γ=0.99 ===");
const cfgB = {H:32,NA:7,UPS:8,LR:0.003,GAM:0.99,ED:2000,BS:32,RS:0.1,TAU:0.005};
for (const seed of [42,123,777,1337,2024]) {
  const t0=Date.now();
  const r=runDQN(seed,cfgB);
  const el=((Date.now()-t0)/1000).toFixed(1);
  console.log(`  seed=${seed}: conv=${r.convStep>=0?`@${r.convStep}`:'NO'} best=${(r.bestUp*100).toFixed(0)}% final=${(r.finalUp*100).toFixed(0)}% ups=${r.updates} (${el}s)`);
}

console.log("\n=== CONFIG C: DQN 2x64, no clip, 16ups, LR=0.001, γ=0.99, 11act ===");
const cfgC = {H:64,NA:11,UPS:16,LR:0.001,GAM:0.99,ED:2000,BS:32,RS:0.1,TAU:0.005};
for (const seed of [42,123,777,1337,2024]) {
  const t0=Date.now();
  const r=runDQN(seed,cfgC);
  const el=((Date.now()-t0)/1000).toFixed(1);
  console.log(`  seed=${seed}: conv=${r.convStep>=0?`@${r.convStep}`:'NO'} best=${(r.bestUp*100).toFixed(0)}% final=${(r.finalUp*100).toFixed(0)}% ups=${r.updates} (${el}s)`);
}

console.log("\n=== CONFIG D: DQN 2x32, no clip, 8ups, LR=0.001, γ=0.95, 7act ===");
const cfgD = {H:32,NA:7,UPS:8,LR:0.001,GAM:0.95,ED:1500,BS:32,RS:0.1,TAU:0.005};
for (const seed of [42,123,777,1337,2024]) {
  const t0=Date.now();
  const r=runDQN(seed,cfgD);
  const el=((Date.now()-t0)/1000).toFixed(1);
  console.log(`  seed=${seed}: conv=${r.convStep>=0?`@${r.convStep}`:'NO'} best=${(r.bestUp*100).toFixed(0)}% final=${(r.finalUp*100).toFixed(0)}% ups=${r.updates} (${el}s)`);
}

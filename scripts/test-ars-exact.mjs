// Exact match of ARSAgent with updated settings: 10 eval seeds, 25% threshold
const G_PHYS=10,MASS=1,LEN=1,MAX_TORQUE=2,DT=0.05,MAX_SPEED=8,EP_LEN=200;
const OBS_DIM=3,HIDDEN=32;
const ARS_DIRS=16,ARS_NOISE=0.05,ARS_LR=0.03,ARS_TOP_B=8,ARS_N_EVAL=4;
const CONVERGE_FRAC=0.25,CONVERGE_HOLD=3;

const wrap=a=>{let x=(a+Math.PI)%(2*Math.PI);if(x<0)x+=2*Math.PI;return x-Math.PI};
function pendDeriv(th,w,u){return[w,(3*G_PHYS)/(2*LEN)*Math.sin(th)+3/(MASS*LEN*LEN)*u]}
function stepPend(s,u){const dt=DT;const k1=pendDeriv(s.th,s.w,u);const k2=pendDeriv(s.th+k1[0]*dt/2,s.w+k1[1]*dt/2,u);const k3=pendDeriv(s.th+k2[0]*dt/2,s.w+k2[1]*dt/2,u);const k4=pendDeriv(s.th+k3[0]*dt,s.w+k3[1]*dt,u);return{th:s.th+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,w:Math.max(-MAX_SPEED,Math.min(MAX_SPEED,s.w+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6))}}
function getObs(s){return[Math.cos(s.th),Math.sin(s.th),s.w/MAX_SPEED]}
function pendReward(s,u){const th=wrap(s.th);return-(th*th+0.1*s.w*s.w+0.001*u*u)}
function randomState(rng){return{th:(rng()*2-1)*Math.PI,w:(rng()*2-1)}}
function mulberry32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function seededGauss(rng){return Math.sqrt(-2*Math.log(rng()+1e-10))*Math.cos(2*Math.PI*rng())}

const initRng=mulberry32(42);
const arsRng=mulberry32(123);

const W1=Float64Array.from({length:HIDDEN*OBS_DIM},()=>seededGauss(initRng)*Math.sqrt(2/OBS_DIM));
const b1=new Float64Array(HIDDEN);
const W2=Float64Array.from({length:HIDDEN*HIDDEN},()=>seededGauss(initRng)*Math.sqrt(2/HIDDEN));
const b2=new Float64Array(HIDDEN);
const Wo=Float64Array.from({length:HIDDEN},()=>seededGauss(initRng)*0.01);
const bo=new Float64Array(1);
const allP=[W1,b1,W2,b2,Wo,bo];
const nP=allP.reduce((s,a)=>s+a.length,0);
const trainSeeds=Array.from({length:ARS_N_EVAL},()=>Math.floor(arsRng()*0x7FFFFFFF));

function forwardFlat(obs,flat){
  const h1=new Float64Array(HIDDEN),h2=new Float64Array(HIDDEN);
  let idx=0;
  for(let i=0;i<HIDDEN;i++){let s=flat[idx+HIDDEN*OBS_DIM+i];for(let j=0;j<OBS_DIM;j++)s+=flat[idx+i*OBS_DIM+j]*obs[j];h1[i]=Math.tanh(s)}
  idx+=HIDDEN*OBS_DIM+HIDDEN;
  for(let i=0;i<HIDDEN;i++){let s=flat[idx+HIDDEN*HIDDEN+i];for(let j=0;j<HIDDEN;j++)s+=flat[idx+i*HIDDEN+j]*h1[j];h2[i]=Math.tanh(s)}
  idx+=HIDDEN*HIDDEN+HIDDEN;
  let mu=flat[idx+HIDDEN];
  for(let j=0;j<HIDDEN;j++)mu+=flat[idx+j]*h2[j];
  return Math.max(-MAX_TORQUE,Math.min(MAX_TORQUE,mu));
}

function forward(obs){
  const h1=new Float64Array(HIDDEN),h2=new Float64Array(HIDDEN);
  for(let i=0;i<HIDDEN;i++){let s=b1[i];for(let j=0;j<OBS_DIM;j++)s+=W1[i*OBS_DIM+j]*obs[j];h1[i]=Math.tanh(s)}
  for(let i=0;i<HIDDEN;i++){let s=b2[i];for(let j=0;j<HIDDEN;j++)s+=W2[i*HIDDEN+j]*h1[j];h2[i]=Math.tanh(s)}
  let mu=bo[0];for(let j=0;j<HIDDEN;j++)mu+=Wo[j]*h2[j];
  return Math.max(-MAX_TORQUE,Math.min(MAX_TORQUE,mu));
}

function getFlat(){const f=new Float64Array(nP);let idx=0;for(const p of allP){f.set(p,idx);idx+=p.length}return f}
function setFlat(f){let idx=0;for(const p of allP){for(let i=0;i<p.length;i++)p[i]=f[idx++]}}

function evaluate(flat,seeds){
  let totalR=0;
  for(const seed of seeds){
    const r=mulberry32(seed);let s=randomState(r);let epR=0;
    for(let t=0;t<EP_LEN;t++){const obs=getObs(s);const u=forwardFlat(obs,flat);s=stepPend(s,u);epR+=pendReward(s,u)}
    totalR+=epR/EP_LEN;
  }
  return totalR/seeds.length;
}

function evalUpright(){
  const seeds=[0xCAFE,0xBEEF,0xDEAD,0xF00D,0xBABE,0xFACE,0xFEED,0xACE1,0xC0DE,0xD00D];
  let totalUp=0;
  for(const seed of seeds){
    const r=mulberry32(seed);let env=randomState(r);let up=0;
    for(let t=0;t<EP_LEN;t++){const obs=getObs(env);const u=forward(obs);env=stepPend(env,u);if(Math.abs(wrap(env.th))<0.3&&Math.abs(env.w)<1)up++}
    totalUp+=up/EP_LEN;
  }
  return totalUp/seeds.length;
}

console.log(`ARS exact test: ${nP} params, 10 eval seeds, 25% threshold`);
console.log("  iter |  meanR   | upright% | convCnt | time");
console.log("-".repeat(55));
const t0=Date.now();
let convCount=0;

for(let iter=0;iter<600;iter++){
  const base=getFlat();
  const deltas=[];
  for(let d=0;d<ARS_DIRS;d++) deltas.push(Float64Array.from({length:nP},()=>seededGauss(arsRng)));

  const rewards=new Float64Array(2*ARS_DIRS);
  for(let ei=0;ei<2*ARS_DIRS;ei++){
    const dirIdx=Math.floor(ei/2);
    const sign=ei%2===0?1:-1;
    const pp=new Float64Array(nP);
    for(let i=0;i<nP;i++) pp[i]=base[i]+sign*ARS_NOISE*deltas[dirIdx][i];
    rewards[ei]=evaluate(pp,trainSeeds);
  }

  const scored=Array.from({length:ARS_DIRS},(_,i)=>({
    idx:i,rp:rewards[i*2],rn:rewards[i*2+1],mx:Math.max(rewards[i*2],rewards[i*2+1])
  }));
  scored.sort((a,b)=>b.mx-a.mx);
  let rM=0;for(let i=0;i<rewards.length;i++)rM+=rewards[i];rM/=rewards.length;
  let rV=0;for(let i=0;i<rewards.length;i++)rV+=(rewards[i]-rM)**2;
  const rS=Math.sqrt(rV/rewards.length)+1e-8;
  const update=new Float64Array(nP);
  for(let k=0;k<ARS_TOP_B;k++){const{idx,rp,rn}=scored[k];const c=(rp-rn)/rS;const d=deltas[idx];for(let i=0;i<nP;i++)update[i]+=c*d[i]}
  const newP=new Float64Array(nP);
  for(let i=0;i<nP;i++)newP[i]=base[i]+(ARS_LR/ARS_TOP_B)*update[i];
  setFlat(newP);

  const uf=evalUpright();
  if(uf>=CONVERGE_FRAC)convCount++;else convCount=Math.max(0,convCount-1);

  if(iter%20===0||iter===599||convCount>=CONVERGE_HOLD){
    const el=((Date.now()-t0)/1000).toFixed(1);
    console.log(`${String(iter).padStart(5)} | ${rM.toFixed(3).padStart(8)} | ${(uf*100).toFixed(0).padStart(7)}% | ${String(convCount).padStart(7)} | ${el}s`);
    if(convCount>=CONVERGE_HOLD){console.log(`\nConverged at iter ${iter}!`);break}
  }
}

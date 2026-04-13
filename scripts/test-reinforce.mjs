// Vanilla REINFORCE with MLP — simplest possible policy gradient.
// If this learns but PPO doesn't, the issue is in PPO machinery.
// If this also doesn't learn, gradient-based methods struggle on this task.

const G_PHYS=10, MASS=1, LEN=1, MAX_TORQUE=2, DT=0.05, MAX_SPEED=8, EP_LEN=200;
const OBS_DIM=3, HIDDEN=64;
const GAMMA=0.99, LR=1e-3;

const wrap=a=>{let x=(a+Math.PI)%(2*Math.PI);if(x<0)x+=2*Math.PI;return x-Math.PI};
function pendDeriv(th,w,u){return[w,(3*G_PHYS)/(2*LEN)*Math.sin(th)+3/(MASS*LEN*LEN)*u]}
function stepPend(s,u){const dt=DT;const k1=pendDeriv(s.th,s.w,u);const k2=pendDeriv(s.th+k1[0]*dt/2,s.w+k1[1]*dt/2,u);const k3=pendDeriv(s.th+k2[0]*dt/2,s.w+k2[1]*dt/2,u);const k4=pendDeriv(s.th+k3[0]*dt,s.w+k3[1]*dt,u);return{th:s.th+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,w:Math.max(-MAX_SPEED,Math.min(MAX_SPEED,s.w+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6))}}
function getObs(s){return[Math.cos(s.th),Math.sin(s.th),s.w/MAX_SPEED]}
function pendReward(s,u){const th=wrap(s.th);return-(th*th+0.1*s.w*s.w+0.001*u*u)}
function mulberry32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function seededGauss(rng){return Math.sqrt(-2*Math.log(rng()+1e-10))*Math.cos(2*Math.PI*rng())}

const rng=mulberry32(42);

// 2-hidden-layer MLP policy
const W1=Float64Array.from({length:HIDDEN*OBS_DIM},()=>seededGauss(rng)*Math.sqrt(2/OBS_DIM));
const b1=new Float64Array(HIDDEN);
const W2=Float64Array.from({length:HIDDEN*HIDDEN},()=>seededGauss(rng)*Math.sqrt(2/HIDDEN));
const b2=new Float64Array(HIDDEN);
const Wo=Float64Array.from({length:HIDDEN},()=>seededGauss(rng)*0.1);
const bo=new Float64Array(1);
const params=[W1,b1,W2,b2,Wo,bo];
let logStd=0.5;

function forward(x){
  const h1=new Float64Array(HIDDEN),h2=new Float64Array(HIDDEN);
  for(let i=0;i<HIDDEN;i++){let s=b1[i];for(let j=0;j<OBS_DIM;j++)s+=W1[i*OBS_DIM+j]*x[j];h1[i]=Math.tanh(s)}
  for(let i=0;i<HIDDEN;i++){let s=b2[i];for(let j=0;j<HIDDEN;j++)s+=W2[i*HIDDEN+j]*h1[j];h2[i]=Math.tanh(s)}
  let mu=bo[0];for(let j=0;j<HIDDEN;j++)mu+=Wo[j]*h2[j];
  return{mu,h1,h2};
}

function accGrad(G,dout,x,h1,h2){
  for(let j=0;j<HIDDEN;j++)G[4][j]+=dout*h2[j];
  G[5][0]+=dout;
  const dz2=new Float64Array(HIDDEN);
  for(let j=0;j<HIDDEN;j++)dz2[j]=dout*Wo[j]*(1-h2[j]*h2[j]);
  for(let i=0;i<HIDDEN;i++){G[3][i]+=dz2[i];for(let j=0;j<HIDDEN;j++)G[2][i*HIDDEN+j]+=dz2[i]*h1[j]}
  const dz1=new Float64Array(HIDDEN);
  for(let j=0;j<HIDDEN;j++){let s=0;for(let i=0;i<HIDDEN;i++)s+=dz2[i]*W2[i*HIDDEN+j];dz1[j]=s*(1-h1[j]*h1[j])}
  for(let i=0;i<HIDDEN;i++){G[1][i]+=dz1[i];for(let j=0;j<OBS_DIM;j++)G[0][i*OBS_DIM+j]+=dz1[i]*x[j]}
}

// Adam state
const m=params.map(a=>new Float64Array(a.length));
const v=params.map(a=>new Float64Array(a.length));
let adamT=0,ls_m=0,ls_v2=0;

function adamUpdate(G,nSamples){
  adamT++;
  const bc1=1-Math.pow(0.9,adamT),bc2=1-Math.pow(0.999,adamT);
  for(let pi=0;pi<params.length;pi++){
    const par=params[pi],g=G[pi],mm=m[pi],vv=v[pi];
    for(let i=0;i<par.length;i++){
      const gi=g[i]/nSamples;
      mm[i]=0.9*mm[i]+0.1*gi;vv[i]=0.999*vv[i]+0.001*gi*gi;
      par[i]-=LR*(mm[i]/bc1)/(Math.sqrt(vv[i]/bc2)+1e-8);
    }
  }
}

const evalRng=mulberry32(0xCAFE);
const evalSeeds=Array.from({length:10},()=>Math.floor(evalRng()*0x7FFFFFFF));

function evaluate(){
  let totalUp=0,totalR=0;
  for(const seed of evalSeeds){
    const r=mulberry32(seed);
    let s={th:(r()*2-1)*Math.PI,w:(r()*2-1)};
    let epR=0,up=0;
    for(let t=0;t<EP_LEN;t++){
      const obs=getObs(s);
      const{mu}=forward(obs);
      const a=Math.max(-MAX_TORQUE,Math.min(MAX_TORQUE,mu));
      s=stepPend(s,a);
      epR+=pendReward(s,a);
      if(Math.abs(wrap(s.th))<0.3&&Math.abs(s.w)<1)up++;
    }
    totalR+=epR/EP_LEN;totalUp+=up/EP_LEN;
  }
  return{meanR:totalR/evalSeeds.length,uprightFrac:totalUp/evalSeeds.length};
}

const N_EPS=10; // episodes per batch
const N_BATCHES=500;
let baseline=-6;

console.log("REINFORCE with 2-hidden-layer MLP");
console.log(`H=${HIDDEN}, lr=${LR}, episodes/batch=${N_EPS}, gamma=${GAMMA}\n`);
console.log("  batch |  trainR  |  evalR   | upright% | logStd | time");
console.log("-".repeat(65));
const t0=Date.now();

for(let batch=0;batch<N_BATCHES;batch++){
  const G=params.map(a=>new Float64Array(a.length));
  let batchR=0;
  let dls_total=0;

  for(let ep=0;ep<N_EPS;ep++){
    let s={th:(rng()*2-1)*Math.PI,w:(rng()*2-1)};
    const traj=[];
    for(let t=0;t<EP_LEN;t++){
      const obs=getObs(s);
      const{mu,h1,h2}=forward(obs);
      const std=Math.exp(logStd);
      const noise=Math.sqrt(-2*Math.log(Math.random()+1e-10))*Math.cos(2*Math.PI*Math.random());
      const a=Math.max(-MAX_TORQUE,Math.min(MAX_TORQUE,mu+std*noise));
      const ns=stepPend(s,a);
      const rew=pendReward(ns,a);
      traj.push({obs,mu,h1:h1.slice(),h2:h2.slice(),a,rew});
      s=ns;
    }

    // Compute discounted returns
    const returns=new Float64Array(EP_LEN);
    let G_ret=0;
    for(let t=EP_LEN-1;t>=0;t--){G_ret=traj[t].rew+GAMMA*G_ret;returns[t]=G_ret}

    let epR=0;for(let t=0;t<EP_LEN;t++)epR+=traj[t].rew;
    batchR+=epR/EP_LEN;

    // Accumulate REINFORCE gradients
    const std=Math.exp(logStd),std2=std*std;
    for(let t=0;t<EP_LEN;t++){
      const{obs,mu,h1,h2,a}=traj[t];
      const advantage=returns[t]-baseline;
      const diff=a-mu;
      const dmu=advantage*diff/std2;
      accGrad(G,dmu,obs,h1,h2);
      dls_total+=advantage*(diff*diff/std2-1);
    }
  }

  batchR/=N_EPS;
  baseline=0.1*batchR+0.9*baseline;
  const totalSamples=N_EPS*EP_LEN;

  // Negate for maximization → minimization
  for(const g of G)for(let i=0;i<g.length;i++)g[i]=-g[i];
  dls_total=-dls_total;

  adamUpdate(G,totalSamples);

  // logStd update
  const gls=dls_total/totalSamples;
  ls_m=0.9*ls_m+0.1*gls;ls_v2=0.999*ls_v2+0.001*gls*gls;
  const bbc1=1-Math.pow(0.9,adamT),bbc2=1-Math.pow(0.999,adamT);
  logStd-=LR*(ls_m/bbc1)/(Math.sqrt(ls_v2/bbc2)+1e-8);
  logStd=Math.max(-2,Math.min(2,logStd));

  if(batch%20===0||batch===N_BATCHES-1){
    const{meanR,uprightFrac}=evaluate();
    const el=((Date.now()-t0)/1000).toFixed(1);
    console.log(`${String(batch+1).padStart(7)} | ${batchR.toFixed(3).padStart(8)} | ${meanR.toFixed(3).padStart(8)} | ${(uprightFrac*100).toFixed(0).padStart(7)}% | ${logStd.toFixed(3).padStart(6)} | ${el}s`);
    if(uprightFrac>=0.3){console.log(`\nConverged at batch ${batch+1}!`);break}
  }
}

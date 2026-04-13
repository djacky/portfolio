// PPO with layer-wise learning rates: 10x higher for hidden layers
const G_PHYS=10,MASS=1,LEN=1,MAX_TORQUE=2,DT=0.05,MAX_SPEED=8,EP_LEN=200;
const OBS_DIM=3,HIDDEN=64,BUFFER_SIZE=2048;
const GAMMA=0.99,LAM=0.95,CLIP_EPS=0.2;
const PPO_EPOCHS=10,MINI_BATCH=64;
const BASE_LR=3e-4;
const LOG2PI_HALF=0.9189385332;

const wrap=a=>{let x=(a+Math.PI)%(2*Math.PI);if(x<0)x+=2*Math.PI;return x-Math.PI};
function pendDeriv(th,w,u){return[w,(3*G_PHYS)/(2*LEN)*Math.sin(th)+3/(MASS*LEN*LEN)*u]}
function stepPend(s,u){const dt=DT;const k1=pendDeriv(s.th,s.w,u);const k2=pendDeriv(s.th+k1[0]*dt/2,s.w+k1[1]*dt/2,u);const k3=pendDeriv(s.th+k2[0]*dt/2,s.w+k2[1]*dt/2,u);const k4=pendDeriv(s.th+k3[0]*dt,s.w+k3[1]*dt,u);return{th:s.th+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,w:Math.max(-MAX_SPEED,Math.min(MAX_SPEED,s.w+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6))}}
function getObs(s){return Float64Array.of(Math.cos(s.th),Math.sin(s.th),s.w/MAX_SPEED)}
function pendReward(s,u){const th=wrap(s.th);return-(th*th+0.1*s.w*s.w+0.001*u*u)}
function randomState(rng){return{th:(rng()*2-1)*Math.PI,w:(rng()*2-1)}}
function mulberry32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function seededGauss(rng){return Math.sqrt(-2*Math.log(rng()+1e-10))*Math.cos(2*Math.PI*rng())}
function gaussRand(){return Math.sqrt(-2*Math.log(Math.random()+1e-10))*Math.cos(2*Math.PI*Math.random())}

class Net {
  constructor(rng,outScale,lrs) {
    this.W1=Float64Array.from({length:HIDDEN*OBS_DIM},()=>seededGauss(rng)*Math.sqrt(2/OBS_DIM));
    this.b1=new Float64Array(HIDDEN);
    this.W2=Float64Array.from({length:HIDDEN*HIDDEN},()=>seededGauss(rng)*Math.sqrt(2/HIDDEN));
    this.b2=new Float64Array(HIDDEN);
    this.Wo=Float64Array.from({length:HIDDEN},()=>seededGauss(rng)*outScale);
    this.bo=new Float64Array(1);
    this.allP=[this.W1,this.b1,this.W2,this.b2,this.Wo,this.bo];
    this.lrs=lrs; // per-layer learning rates
    this.mAd=this.allP.map(a=>new Float64Array(a.length));
    this.vAd=this.allP.map(a=>new Float64Array(a.length));
    this.G=this.allP.map(a=>new Float64Array(a.length));
    this.adamT=0;
    this.h1=new Float64Array(HIDDEN);this.h2=new Float64Array(HIDDEN);
  }
  forward(x){
    const{h1,h2,W1,b1,W2,b2,Wo,bo}=this;
    for(let i=0;i<HIDDEN;i++){let s=b1[i];for(let j=0;j<OBS_DIM;j++)s+=W1[i*OBS_DIM+j]*x[j];h1[i]=Math.tanh(s)}
    for(let i=0;i<HIDDEN;i++){let s=b2[i];for(let j=0;j<HIDDEN;j++)s+=W2[i*HIDDEN+j]*h1[j];h2[i]=Math.tanh(s)}
    let out=bo[0];for(let j=0;j<HIDDEN;j++)out+=Wo[j]*h2[j];
    return out;
  }
  accGrad(dout,x){
    const{h1,h2,W2,Wo,G}=this;
    for(let j=0;j<HIDDEN;j++)G[4][j]+=dout*h2[j];
    G[5][0]+=dout;
    const dz2=new Float64Array(HIDDEN);
    for(let j=0;j<HIDDEN;j++)dz2[j]=dout*Wo[j]*(1-h2[j]*h2[j]);
    for(let i=0;i<HIDDEN;i++){G[3][i]+=dz2[i];for(let j=0;j<HIDDEN;j++)G[2][i*HIDDEN+j]+=dz2[i]*h1[j]}
    const dz1=new Float64Array(HIDDEN);
    for(let j=0;j<HIDDEN;j++){let s=0;for(let i=0;i<HIDDEN;i++)s+=dz2[i]*W2[i*HIDDEN+j];dz1[j]=s*(1-h1[j]*h1[j])}
    for(let i=0;i<HIDDEN;i++){G[1][i]+=dz1[i];for(let j=0;j<OBS_DIM;j++)G[0][i*OBS_DIM+j]+=dz1[i]*x[j]}
  }
  zeroGrad(){for(const g of this.G)g.fill(0)}
  adamStep(bs){
    this.adamT++;
    const bc1=1-Math.pow(0.9,this.adamT),bc2=1-Math.pow(0.999,this.adamT);
    for(let pi=0;pi<this.allP.length;pi++){
      const par=this.allP[pi],g=this.G[pi],m=this.mAd[pi],v=this.vAd[pi];
      const lr=this.lrs[pi];
      for(let i=0;i<par.length;i++){
        const gi=g[i]/bs;
        m[i]=0.9*m[i]+0.1*gi;v[i]=0.999*v[i]+0.001*gi*gi;
        par[i]-=lr*(m[i]/bc1)/(Math.sqrt(v[i]/bc2)+1e-8);
      }
    }
  }
}

// Layer-wise LRs: hidden layers get 10x base, output gets 1x
const piLRs = [BASE_LR*10,BASE_LR*10,BASE_LR*10,BASE_LR*10,BASE_LR,BASE_LR];
const vfLRs = [BASE_LR*10,BASE_LR*10,BASE_LR*10,BASE_LR*10,BASE_LR,BASE_LR];

const rng0=mulberry32(42);
const bgRng=mulberry32(123);
const pi=new Net(rng0,0.1,piLRs);
const vf=new Net(rng0,1.0,vfLRs);
let logStd=0.5,ls_m=0,ls_v=0,ls_t=0;

let bgEnv=randomState(bgRng),bgStep=0;
const bufO=new Float64Array(BUFFER_SIZE*OBS_DIM),bufA=new Float64Array(BUFFER_SIZE);
const bufR=new Float64Array(BUFFER_SIZE),bufV=new Float64Array(BUFFER_SIZE);
const bufL=new Float64Array(BUFFER_SIZE),bufD=new Uint8Array(BUFFER_SIZE);

function act(obs,det){
  const mu=pi.forward(obs);const val=vf.forward(obs);
  const std=Math.exp(logStd);
  const a=det?Math.max(-MAX_TORQUE,Math.min(MAX_TORQUE,mu)):Math.max(-MAX_TORQUE,Math.min(MAX_TORQUE,mu+std*gaussRand()));
  const d=a-mu;
  return{action:a,logProb:-0.5*(d/std)**2-logStd-LOG2PI_HALF,value:val};
}

function collectBuffer(){
  let bIdx=0;
  while(bIdx<BUFFER_SIZE){
    const obs=getObs(bgEnv);
    const{action,logProb,value}=act(obs,false);
    const ns=stepPend(bgEnv,action);
    const rew=pendReward(ns,action);
    bgStep++;
    const done=bgStep>=EP_LEN;
    bufO.set(obs,bIdx*OBS_DIM);bufA[bIdx]=action;bufR[bIdx]=rew;
    bufV[bIdx]=value;bufL[bIdx]=logProb;bufD[bIdx]=done?1:0;
    bIdx++;
    if(done){bgEnv=randomState(bgRng);bgStep=0}else bgEnv=ns;
  }
  const lo=getObs(bgEnv);const{value:lv}=act(lo,false);
  return{T:bIdx,lastVal:bufD[bIdx-1]?0:lv};
}

function update(T,lastVal){
  const adv=new Float64Array(T),ret=new Float64Array(T);
  let lastA=0;
  for(let t=T-1;t>=0;t--){
    const mask=1-bufD[t];const nv=t===T-1?lastVal:bufV[t+1];
    const delta=bufR[t]+GAMMA*nv*mask-bufV[t];
    lastA=delta+GAMMA*LAM*mask*lastA;adv[t]=lastA;ret[t]=lastA+bufV[t];
  }
  let am=0;for(let t=0;t<T;t++)am+=adv[t];am/=T;
  let av=0;for(let t=0;t<T;t++)av+=(adv[t]-am)**2;
  const as2=Math.sqrt(av/T)+1e-8;
  for(let t=0;t<T;t++)adv[t]=(adv[t]-am)/as2;

  const indices=Array.from({length:T},(_,i)=>i);
  for(let ep=0;ep<PPO_EPOCHS;ep++){
    for(let i=T-1;i>0;i--){const j=Math.floor(bgRng()*(i+1));[indices[i],indices[j]]=[indices[j],indices[i]]}
    for(let st=0;st<T;st+=MINI_BATCH){
      const end=Math.min(st+MINI_BATCH,T),bs=end-st;
      pi.zeroGrad();vf.zeroGrad();
      let dls_sum=0;
      for(let bi=st;bi<end;bi++){
        const si=indices[bi];
        const obs=bufO.subarray(si*OBS_DIM,si*OBS_DIM+OBS_DIM);
        const oldAct=bufA[si],oldLp=bufL[si];
        const advantage=adv[si],target=ret[si];
        const mu=pi.forward(obs);const val=vf.forward(obs);
        const std=Math.exp(logStd),std2=std*std;
        const diff=oldAct-mu;
        const newLp=-0.5*diff*diff/std2-logStd-LOG2PI_HALF;
        const ratio=Math.exp(newLp-oldLp);
        const clipped=(ratio>1+CLIP_EPS&&advantage>0)||(ratio<1-CLIP_EPS&&advantage<0);
        let dmu=0,dls=0;
        if(!clipped){dmu=-advantage*ratio*diff/std2;dls=-advantage*ratio*(diff*diff/std2-1)}
        dls-=0.01; // entropy bonus
        dls_sum+=dls;
        pi.accGrad(dmu,obs);
        const dval=0.5*(val-target);
        vf.accGrad(dval,obs);
      }
      pi.adamStep(bs);vf.adamStep(bs);
      const gls=dls_sum/bs;
      ls_t++;ls_m=0.9*ls_m+0.1*gls;ls_v=0.999*ls_v+0.001*gls*gls;
      const bc1=1-Math.pow(0.9,ls_t),bc2=1-Math.pow(0.999,ls_t);
      logStd-=BASE_LR*(ls_m/bc1)/(Math.sqrt(ls_v/bc2)+1e-8);
    }
  }
  logStd=Math.max(-2,Math.min(2,logStd));
}

function evalPolicy(){
  const seeds=[0xCAFE,0xBEEF,0xDEAD,0xF00D,0xBABE];
  let totalUp=0;
  for(const seed of seeds){
    const r=mulberry32(seed);let env=randomState(r);let up=0;
    for(let t=0;t<EP_LEN;t++){const obs=getObs(env);const{action}=act(obs,true);env=stepPend(env,action);if(Math.abs(wrap(env.th))<0.3&&Math.abs(env.w)<1)up++}
    totalUp+=up/EP_LEN;
  }
  return totalUp/seeds.length;
}

console.log("PPO with layer-wise LR (hidden=10x, out=1x)");
console.log(`H=${HIDDEN}, baseLR=${BASE_LR}, hiddenLR=${BASE_LR*10}`);
console.log("  upd |  meanR   | upright% | logStd | time");
console.log("-".repeat(55));
const t0=Date.now();
const rewardHist=[];
for(let u=0;u<200;u++){
  const{T,lastVal}=collectBuffer();
  let totalR=0;for(let t=0;t<T;t++)totalR+=bufR[t];
  rewardHist.push(totalR/T);
  update(T,lastVal);
  if(u%10===0||u===199){
    const uf=evalPolicy();
    const el=((Date.now()-t0)/1000).toFixed(1);
    console.log(`${String(u+1).padStart(5)} | ${rewardHist[rewardHist.length-1].toFixed(3).padStart(8)} | ${(uf*100).toFixed(0).padStart(7)}% | ${logStd.toFixed(3).padStart(6)} | ${el}s`);
    if(uf>=0.3){console.log(`\nConverged at update ${u+1}!`);break}
  }
}

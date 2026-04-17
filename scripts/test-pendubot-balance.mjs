// Balance-only pendubot DQN test
// Agent starts near inverted each episode (simulating user placement)
// Tests if DQN can learn balance in <10K steps

const G=10, M1=1, M2=0.5, L1=0.5, L2=0.5, LC1=L1/2, LC2=L2/2;
const I1=M1*L1*L1/12, I2=M2*L2*L2/12, DT=0.05, MS=10;
const B1=0.1, B2=0.3;
const _D1=M1*LC1*LC1+I1+M2*L1*L1, _D2=M2*LC2*LC2+I2, _H=M2*L1*LC2;
const _G1=(M1*LC1+M2*L1)*G, _G2=M2*LC2*G;
const wrap=a=>{let x=(a+Math.PI)%(2*Math.PI);if(x<0)x+=2*Math.PI;return x-Math.PI};

function deriv(t1,w1,t2,w2,u) {
  const d=t1-t2, cd=Math.cos(d), sd=Math.sin(d);
  const relW=w2-w1;
  const f1=u-B1*w1+B2*relW-_H*sd*w2*w2+_G1*Math.sin(t1);
  const f2=-B2*relW+_H*sd*w1*w1+_G2*Math.sin(t2);
  const det=_D1*_D2-_H*_H*cd*cd;
  return [w1,(_D2*f1-_H*cd*f2)/det, w2,(_D1*f2-_H*cd*f1)/det];
}

function step(s,u) {
  const {t1,w1,t2,w2}=s;
  const k1=deriv(t1,w1,t2,w2,u);
  const k2=deriv(t1+k1[0]*DT/2,w1+k1[1]*DT/2,t2+k1[2]*DT/2,w2+k1[3]*DT/2,u);
  const k3=deriv(t1+k2[0]*DT/2,w1+k2[1]*DT/2,t2+k2[2]*DT/2,w2+k2[3]*DT/2,u);
  const k4=deriv(t1+k3[0]*DT,w1+k3[1]*DT,t2+k3[2]*DT,w2+k3[3]*DT,u);
  return {
    t1:t1+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*DT/6,
    w1:Math.max(-MS,Math.min(MS,w1+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*DT/6)),
    t2:t2+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*DT/6,
    w2:Math.max(-MS,Math.min(MS,w2+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*DT/6)),
  };
}

const OBS=6;
function obs(s){return Float64Array.of(Math.cos(s.t1),Math.sin(s.t1),s.w1/MS,Math.cos(s.t2),Math.sin(s.t2),s.w2/MS)}
function rew(s,u){const up=(Math.cos(s.t1)+1)/2*(Math.cos(s.t2)+1)/2;return 2*up-1-0.005*(s.w1*s.w1+s.w2*s.w2)-0.001*u*u}
function mb32(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function gauss(r){return Math.sqrt(-2*Math.log(r()+1e-10))*Math.cos(2*Math.PI*r())}

class Net{
  constructor(iD,hD,oD,r){this.iD=iD;this.hD=hD;this.oD=oD;const S=[hD*iD,hD,hD*hD,hD,oD*hD,oD];this.P=S.map(n=>new Float64Array(n));this.G=S.map(n=>new Float64Array(n));this.am=S.map(n=>new Float64Array(n));this.av=S.map(n=>new Float64Array(n));this.h1=new Float64Array(hD);this.h2=new Float64Array(hD);this.out=new Float64Array(oD);this.h1p=new Float64Array(hD);this.h2p=new Float64Array(hD);this.oC=new Float64Array(iD);this.dh2=new Float64Array(hD);this.dh1=new Float64Array(hD);for(let i=0;i<this.P[0].length;i++)this.P[0][i]=gauss(r)*Math.sqrt(2/iD);for(let i=0;i<this.P[2].length;i++)this.P[2][i]=gauss(r)*Math.sqrt(2/hD);for(let i=0;i<this.P[4].length;i++)this.P[4][i]=gauss(r)*0.01}
  fwd(o){const[W1,b1,W2,b2,Wo,bo]=this.P;const{iD,hD,oD}=this;for(let i=0;i<iD;i++)this.oC[i]=o[i];for(let i=0;i<hD;i++){let s=b1[i];for(let j=0;j<iD;j++)s+=W1[i*iD+j]*o[j];this.h1p[i]=s;this.h1[i]=s>0?s:0}for(let i=0;i<hD;i++){let s=b2[i];for(let j=0;j<hD;j++)s+=W2[i*hD+j]*this.h1[j];this.h2p[i]=s;this.h2[i]=s>0?s:0}for(let k=0;k<oD;k++){let s=bo[k];for(let j=0;j<hD;j++)s+=Wo[k*hD+j]*this.h2[j];this.out[k]=s}return this.out}
  bwd(a,tgt){const Wo=this.P[4],W2=this.P[2];const[gW1,gb1,gW2,gb2,gWo,gbo]=this.G;const{iD,hD}=this;const err=this.out[a]-tgt;const g=Math.max(-5,Math.min(5,err));this.dh2.fill(0);gbo[a]+=g;for(let j=0;j<hD;j++){gWo[a*hD+j]+=g*this.h2[j];this.dh2[j]=g*Wo[a*hD+j]}for(let j=0;j<hD;j++)if(this.h2p[j]<=0)this.dh2[j]=0;this.dh1.fill(0);for(let i=0;i<hD;i++){if(this.dh2[i]===0)continue;gb2[i]+=this.dh2[i];for(let j=0;j<hD;j++){gW2[i*hD+j]+=this.dh2[i]*this.h1[j];this.dh1[j]+=this.dh2[i]*W2[i*hD+j]}}for(let j=0;j<hD;j++)if(this.h1p[j]<=0)this.dh1[j]=0;for(let i=0;i<hD;i++){if(this.dh1[i]===0)continue;gb1[i]+=this.dh1[i];for(let j=0;j<iD;j++)gW1[i*iD+j]+=this.dh1[i]*this.oC[j]}}
  zero(){for(const g of this.G)g.fill(0)}
  adam(lr,t){const B1=0.9,B2=0.999,e=1e-8,bc1=1-B1**t,bc2=1-B2**t;for(let p=0;p<this.P.length;p++){const P=this.P[p],Gr=this.G[p],m=this.am[p],v=this.av[p];for(let i=0;i<P.length;i++){const cg=Math.max(-10,Math.min(10,Gr[i]));m[i]=B1*m[i]+(1-B1)*cg;v[i]=B2*v[i]+(1-B2)*cg*cg;P[i]-=lr*(m[i]/bc1)/(Math.sqrt(v[i]/bc2)+e)}}}
  copy(o){for(let p=0;p<this.P.length;p++)this.P[p].set(o.P[p])}
  soft(src,tau){for(let p=0;p<this.P.length;p++){const t=this.P[p],s=src.P[p];for(let i=0;i<t.length;i++)t[i]=tau*s[i]+(1-tau)*t[i]}}
}

function run(cfg, seed) {
  const rng=mb32(seed), irng=mb32(seed*7+13);
  const {NA,hD,lr,gamma,ups,maxSteps,epsDecay,maxT,rCap,bs,tau,rScale,perturbRange}=cfg;
  const torques=Float64Array.from({length:NA},(_,i)=>-maxT+(2*maxT/(NA-1))*i);
  const STR=OBS+1+1+OBS;
  const on=new Net(OBS,hD,NA,irng), tg=new Net(OBS,hD,NA,()=>0); tg.copy(on);
  const buf=new Float64Array(rCap*STR); let bSz=0, bPos=0;
  function push(o,a,r,n){const off=bPos*STR;buf.set(o,off);buf[off+OBS]=a;buf[off+OBS+1]=r;buf.set(n,off+OBS+2);bPos=(bPos+1)%rCap;if(bSz<rCap)bSz++}

  let adamT=0;
  function train(){on.zero();for(let i=0;i<bs;i++){const idx=Math.floor(rng()*bSz),off=idx*STR;const sO=buf.subarray(off,off+OBS),a=buf[off+OBS],r=buf[off+OBS+1],nO=buf.subarray(off+OBS+2,off+OBS+2+OBS);const oq=on.fwd(nO);let bA=0;for(let k=1;k<NA;k++)if(oq[k]>oq[bA])bA=k;const tq=tg.fwd(nO);const mQ=tq[bA];on.fwd(sO);on.bwd(a,r+gamma*mQ)}adamT++;on.adam(lr/bs,adamT)}

  // Balance-only: reset near inverted at start of each episode
  function resetNearInverted() {
    const pr = perturbRange;
    return {
      t1: (rng() * 2 - 1) * pr,
      w1: (rng() * 2 - 1) * 1,
      t2: (rng() * 2 - 1) * pr,
      w2: (rng() * 2 - 1) * 1,
    };
  }

  let env = resetNearInverted();
  let eps=1, vS=0, vUp=0, convC=0, conv=false, convStep=-1, bestUp=0;
  const EP = 200;

  for (let t=0; t<maxSteps; t++) {
    let ai;
    if (rng()<eps) ai=Math.floor(rng()*NA);
    else { const q=on.fwd(obs(env)); ai=0; for(let i=1;i<NA;i++) if(q[i]>q[ai]) ai=i; }

    const u=torques[ai], prev=env, nx=step(env,u), r=rew(nx,u);
    push(obs(prev), ai, r*rScale, obs(nx));
    if (bSz>=200) { for(let i=0;i<ups;i++) train(); tg.soft(on,tau); }
    env=nx;
    eps=Math.max(0.01, 1-t/epsDecay);

    const up = Math.abs(wrap(nx.t1))<0.3 && Math.abs(wrap(nx.t2))<0.3 && Math.abs(nx.w1)<2 && Math.abs(nx.w2)<2;
    if(up) vUp++;
    vS++;

    // Reset if fallen too far (simulating user re-grab)
    if (Math.abs(wrap(env.t1)) > 1.5 || Math.abs(wrap(env.t2)) > 1.5) {
      env = resetNearInverted();
    }

    if (vS>=EP) {
      const epUp=vUp/EP;
      bestUp=Math.max(bestUp,epUp);
      if(!conv){if(epUp>=0.5)convC++;else convC=Math.max(0,convC-1);if(convC>=3){conv=true;convStep=t}}
      vUp=0;vS=0;
    }
  }
  return {convStep, bestUp};
}

const seeds = [42, 123, 777, 1337, 2024];

const configs = [
  // MATLAB-inspired: 2u, larger net+batch+buffer, Huber(5)+gradclip
  {label:"MATLAB: 128h 2u LR=3e-3 bs=64 buf=50K",NA:11,hD:128,lr:0.003,gamma:0.99,ups:2,maxSteps:10000,epsDecay:5000,maxT:10,rCap:50000,bs:64,tau:0.005,rScale:1.0,perturbRange:0.3},
  // Smaller net for speed comparison
  {label:"SMALL: 64h 2u LR=3e-3 bs=64 buf=50K", NA:11,hD:64, lr:0.003,gamma:0.99,ups:2,maxSteps:10000,epsDecay:5000,maxT:10,rCap:50000,bs:64,tau:0.005,rScale:1.0,perturbRange:0.3},
  // 4 updates variant
  {label:"4U: 128h 4u LR=3e-3 bs=64 buf=50K",   NA:11,hD:128,lr:0.003,gamma:0.99,ups:4,maxSteps:10000,epsDecay:5000,maxT:10,rCap:50000,bs:64,tau:0.005,rScale:1.0,perturbRange:0.3},
];

console.log(`Pendubot BALANCE-ONLY DQN test — ${seeds.length} seeds, 10K steps`);
console.log(`Start near inverted (perturbRange), auto-reset if |θ|>1.5 rad\n`);

for (const cfg of configs) {
  const t0=Date.now(), details=[];
  let convT=0;
  for (const seed of seeds) {
    const r=run(cfg,seed);
    if(r.convStep>=0)convT++;
    details.push(`${seed}:${r.convStep>=0?r.convStep:'--'}/${(r.bestUp*100).toFixed(0)}%`);
  }
  console.log(`[${cfg.label}]  ${convT}/${seeds.length} conv  (${((Date.now()-t0)/1000).toFixed(1)}s)`);
  console.log(`  ${details.join(' | ')}`);
}

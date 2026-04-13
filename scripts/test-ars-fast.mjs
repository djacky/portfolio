// ARS fast: HIDDEN=32, tuned for fast convergence in browser
const G_PHYS=10,MASS=1,LEN=1,MAX_TORQUE=2,DT=0.05,MAX_SPEED=8,EP_LEN=200;
const OBS_DIM=3,wrap=a=>{let x=(a+Math.PI)%(2*Math.PI);if(x<0)x+=2*Math.PI;return x-Math.PI};
function pendDeriv(th,w,u){return[w,(3*G_PHYS)/(2*LEN)*Math.sin(th)+3/(MASS*LEN*LEN)*u]}
function stepPend(s,u){const dt=DT;const k1=pendDeriv(s.th,s.w,u);const k2=pendDeriv(s.th+k1[0]*dt/2,s.w+k1[1]*dt/2,u);const k3=pendDeriv(s.th+k2[0]*dt/2,s.w+k2[1]*dt/2,u);const k4=pendDeriv(s.th+k3[0]*dt,s.w+k3[1]*dt,u);return{th:s.th+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,w:Math.max(-MAX_SPEED,Math.min(MAX_SPEED,s.w+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6))}}
function getObs(s){return[Math.cos(s.th),Math.sin(s.th),s.w/MAX_SPEED]}
function pendReward(s,u){const th=wrap(s.th);return-(th*th+0.1*s.w*s.w+0.001*u*u)}
function mulberry32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
function seededGauss(rng){return Math.sqrt(-2*Math.log(rng()+1e-10))*Math.cos(2*Math.PI*rng())}

function runARS(config, seed) {
  const { HIDDEN, N_DIRS, NOISE_STD, STEP_SIZE, TOP_B, N_EVAL, N_ITERS } = config;
  const rng = mulberry32(seed);
  const xI=Math.sqrt(2/OBS_DIM), xH=Math.sqrt(2/HIDDEN), xO=0.01;
  const W1=Float64Array.from({length:HIDDEN*OBS_DIM},()=>seededGauss(rng)*xI);
  const b1=new Float64Array(HIDDEN);
  const W2=Float64Array.from({length:HIDDEN*HIDDEN},()=>seededGauss(rng)*xH);
  const b2=new Float64Array(HIDDEN);
  const Wo=Float64Array.from({length:HIDDEN},()=>seededGauss(rng)*xO);
  const bo=new Float64Array(1);
  const allP=[W1,b1,W2,b2,Wo,bo];
  const nP=allP.reduce((s,a)=>s+a.length,0);

  function act(obs,flat){
    let idx=0;
    const h1=new Float64Array(HIDDEN);
    for(let i=0;i<HIDDEN;i++){let s=flat[idx+HIDDEN*OBS_DIM+i];for(let j=0;j<OBS_DIM;j++)s+=flat[idx+i*OBS_DIM+j]*obs[j];h1[i]=Math.tanh(s)}
    idx+=HIDDEN*OBS_DIM+HIDDEN;
    const h2=new Float64Array(HIDDEN);
    for(let i=0;i<HIDDEN;i++){let s=flat[idx+HIDDEN*HIDDEN+i];for(let j=0;j<HIDDEN;j++)s+=flat[idx+i*HIDDEN+j]*h1[j];h2[i]=Math.tanh(s)}
    idx+=HIDDEN*HIDDEN+HIDDEN;
    let mu=flat[idx+HIDDEN];
    for(let j=0;j<HIDDEN;j++)mu+=flat[idx+j]*h2[j];
    return Math.max(-MAX_TORQUE,Math.min(MAX_TORQUE,mu));
  }

  function getFlat(){const f=new Float64Array(nP);let idx=0;for(const p of allP){f.set(p,idx);idx+=p.length}return f}
  function setFlat(f){let idx=0;for(const p of allP){for(let i=0;i<p.length;i++)p[i]=f[idx++]}}

  function evaluate(flat,seeds){
    let totalR=0;
    for(const seed of seeds){
      const r=mulberry32(seed);let s={th:(r()*2-1)*Math.PI,w:(r()*2-1)};let epR=0;
      for(let t=0;t<EP_LEN;t++){const obs=getObs(s);const u=act(obs,flat);s=stepPend(s,u);epR+=pendReward(s,u)}
      totalR+=epR/EP_LEN;
    }
    return totalR/seeds.length;
  }

  function evalUpright(flat,seeds){
    let totalUp=0;
    for(const seed of seeds){
      const r=mulberry32(seed);let s={th:(r()*2-1)*Math.PI,w:(r()*2-1)};let up=0;
      for(let t=0;t<EP_LEN;t++){const obs=getObs(s);const u=act(obs,flat);s=stepPend(s,u);if(Math.abs(wrap(s.th))<0.3&&Math.abs(s.w)<1)up++}
      totalUp+=up/EP_LEN;
    }
    return totalUp/seeds.length;
  }

  const trainSeeds=Array.from({length:N_EVAL},()=>Math.floor(rng()*0x7FFFFFFF));
  const evalSeeds=Array.from({length:20},(_,i)=>0xCAFE+i);
  const t0=Date.now();
  let bestUpright=0;

  for(let iter=0;iter<N_ITERS;iter++){
    const baseP=getFlat();
    const deltas=[],rp=[],rn=[];
    for(let d=0;d<N_DIRS;d++){
      const delta=Float64Array.from({length:nP},()=>seededGauss(rng));
      deltas.push(delta);
      const posP=new Float64Array(nP),negP=new Float64Array(nP);
      for(let i=0;i<nP;i++){posP[i]=baseP[i]+NOISE_STD*delta[i];negP[i]=baseP[i]-NOISE_STD*delta[i]}
      rp.push(evaluate(posP,trainSeeds));rn.push(evaluate(negP,trainSeeds));
    }
    const scored=deltas.map((d,i)=>({d,rp:rp[i],rn:rn[i],mx:Math.max(rp[i],rn[i])}));
    scored.sort((a,b)=>b.mx-a.mx);
    const allR=[...rp,...rn];let rM=0;for(const r of allR)rM+=r;rM/=allR.length;
    let rV=0;for(const r of allR)rV+=(r-rM)**2;const rS=Math.sqrt(rV/allR.length)+1e-8;
    const update=new Float64Array(nP);
    for(let k=0;k<TOP_B;k++){const{d,rp:rr,rn:nr}=scored[k];const c=(rr-nr)/rS;for(let i=0;i<nP;i++)update[i]+=c*d[i]}
    const newP=new Float64Array(nP);
    for(let i=0;i<nP;i++)newP[i]=baseP[i]+(STEP_SIZE/TOP_B)*update[i];
    setFlat(newP);

    if(iter%25===0||iter===N_ITERS-1){
      const flat=getFlat();
      const meanR=evaluate(flat,evalSeeds);
      const upPct=evalUpright(flat,evalSeeds);
      bestUpright=Math.max(bestUpright,upPct);
      const el=((Date.now()-t0)/1000).toFixed(1);
      console.log(`${String(iter).padStart(5)} | ${meanR.toFixed(3).padStart(8)} | ${(upPct*100).toFixed(0).padStart(7)}% | ${el}s`);
      if(upPct>=0.3){return{converged:true,iter,time:(Date.now()-t0)/1000}}
    }
  }
  return{converged:false,bestUpright,time:(Date.now()-t0)/1000};
}

// Test configs with multiple seeds
const configs = [
  { name: "H32,16d,n0.05,lr0.03", HIDDEN:32, N_DIRS:16, NOISE_STD:0.05, STEP_SIZE:0.03, TOP_B:8, N_EVAL:8, N_ITERS:500 },
  { name: "H32,32d,n0.05,lr0.03", HIDDEN:32, N_DIRS:32, NOISE_STD:0.05, STEP_SIZE:0.03, TOP_B:16, N_EVAL:8, N_ITERS:500 },
  { name: "H32,16d,n0.05,lr0.05", HIDDEN:32, N_DIRS:16, NOISE_STD:0.05, STEP_SIZE:0.05, TOP_B:8, N_EVAL:4, N_ITERS:500 },
  { name: "H32,32d,n0.08,lr0.03", HIDDEN:32, N_DIRS:32, NOISE_STD:0.08, STEP_SIZE:0.03, TOP_B:16, N_EVAL:4, N_ITERS:500 },
];

for (const cfg of configs) {
  console.log(`\n=== ${cfg.name} ===`);
  console.log("  iter |  meanR   | upright% | time");
  console.log("-".repeat(45));
  const result = runARS(cfg, 42);
  if(result.converged) console.log(`  Converged at iter ${result.iter} in ${result.time.toFixed(1)}s!`);
  else console.log(`  Did not converge. Best upright: ${(result.bestUpright*100).toFixed(0)}%`);
}

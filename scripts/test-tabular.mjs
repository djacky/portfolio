// Tabular Q-learning with finer discretization + Dyna-Q imagination
const G=10,M=1,L=1,MT=2,DT=0.05,MS=8,EL=200;
const wrap=a=>{let x=(a+Math.PI)%(2*Math.PI);if(x<0)x+=2*Math.PI;return x-Math.PI};
function pD(th,w,u){return[w,(3*G)/(2*L)*Math.sin(th)+3/(M*L*L)*u]}
function step(s,u){const dt=DT,k1=pD(s.th,s.w,u),k2=pD(s.th+k1[0]*dt/2,s.w+k1[1]*dt/2,u),k3=pD(s.th+k2[0]*dt/2,s.w+k2[1]*dt/2,u),k4=pD(s.th+k3[0]*dt,s.w+k3[1]*dt,u);return{th:s.th+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dt/6,w:Math.max(-MS,Math.min(MS,s.w+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dt/6))}}
function rew(s,u){const th=wrap(s.th);return-(th*th+0.1*s.w*s.w+0.001*u*u)}
function mb32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}

function run(seed, NTH, NW, NA, alpha, gamma, epsDecay, label) {
  const rng = mb32(seed);
  const torques = Float64Array.from({length:NA},(_,i)=>-MT+(2*MT/(NA-1))*i);
  const Q = new Float64Array(NTH * NW * NA); // init 0 = optimistic

  function si(s) {
    let th = wrap(s.th);
    let ti = Math.floor((th + Math.PI) / (2*Math.PI) * NTH);
    ti = Math.max(0, Math.min(NTH-1, ti));
    let wi = Math.floor((s.w + MS) / (2*MS) * NW);
    wi = Math.max(0, Math.min(NW-1, wi));
    return ti * NW + wi;
  }

  let env = {th: Math.PI, w: 0};
  let visSteps=0, visR=0, visUp=0, convCount=0, converged=false, convStep=-1, bestUp=0;
  let eps = 1.0;

  for (let t = 0; t < 10800; t++) {
    const s = si(env);
    let ai;
    if (rng() < eps) ai = Math.floor(rng() * NA);
    else { ai=0; let bq=Q[s*NA]; for(let a=1;a<NA;a++) if(Q[s*NA+a]>bq){bq=Q[s*NA+a];ai=a;} }

    const torque = torques[ai];
    const next = step(env, torque);
    const r = rew(next, torque);
    const s2 = si(next);

    let mQ=Q[s2*NA]; for(let a=1;a<NA;a++) if(Q[s2*NA+a]>mQ) mQ=Q[s2*NA+a];
    Q[s*NA+ai] += alpha * (r + gamma * mQ - Q[s*NA+ai]);

    env = next;
    visR += r; visSteps++;
    if (Math.abs(wrap(next.th)) < 0.3 && Math.abs(next.w) < 1) visUp++;
    eps = Math.max(0.01, 1.0 - t / epsDecay);

    if (visSteps >= EL) {
      const eR = visR/EL, eU = visUp/EL;
      bestUp = Math.max(bestUp, eU);
      if (!converged) { if(eU>=0.5) convCount++; else convCount=Math.max(0,convCount-1); if(convCount>=3){converged=true;convStep=t;}}
      visR=0;visUp=0;visSteps=0;
    }
  }
  return { convStep, bestUp };
}

const seeds = [42,123,777,1337,2024,7,99,256,444,888];

const configs = [
  {NTH:36,NW:21,NA:7,alpha:0.3,gamma:0.99,epsDecay:2000,label:"36x21x7 α=0.3 γ=0.99"},
  {NTH:36,NW:21,NA:7,alpha:0.5,gamma:0.99,epsDecay:1500,label:"36x21x7 α=0.5 γ=0.99 fastε"},
  {NTH:36,NW:21,NA:11,alpha:0.3,gamma:0.99,epsDecay:2000,label:"36x21x11 α=0.3"},
  {NTH:72,NW:41,NA:7,alpha:0.2,gamma:0.99,epsDecay:2000,label:"72x41x7 α=0.2 fine"},
  {NTH:24,NW:15,NA:5,alpha:0.5,gamma:0.99,epsDecay:1000,label:"24x15x5 α=0.5 coarse"},
  {NTH:36,NW:21,NA:7,alpha:0.3,gamma:0.95,epsDecay:2000,label:"36x21x7 α=0.3 γ=0.95"},
];

for (const cfg of configs) {
  let convTotal = 0, bestTotal = 0;
  const t0 = Date.now();
  const details = [];
  for (const seed of seeds) {
    const r = run(seed, cfg.NTH, cfg.NW, cfg.NA, cfg.alpha, cfg.gamma, cfg.epsDecay, cfg.label);
    if (r.convStep >= 0) convTotal++;
    bestTotal += r.bestUp;
    details.push(`${seed}:${r.convStep>=0?Math.round(r.convStep/200)+'ep':'--'}/${(r.bestUp*100).toFixed(0)}%`);
  }
  const el = ((Date.now()-t0)/1000).toFixed(1);
  console.log(`[${cfg.label}]  ${convTotal}/${seeds.length} conv  avgBest=${(bestTotal/seeds.length*100).toFixed(0)}%  (${el}s)`);
  console.log(`  ${details.join(' | ')}`);
}

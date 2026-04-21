/* ------------------------------------------------------------------
   Time-budget diagnostic for the torsional synthesis.  Prints per
   bisection step: SCS iter count, SCS solveTime, wall-clock, status.
   Lets us see whether the 80 s torsional run is spent in:
     - many bisection iters  (loosen BIS_TOL)
     - many SCS iters per solve (cap maxIters / raise eps)
     - SCS hitting the 10 s per-SOCP cap (ill-conditioned — real cost)
   Usage: npx tsx scripts/hinf-timing-diag.ts
   ------------------------------------------------------------------ */

import { plantById } from "../lib/hinf-plants";
import {
  getSCS,
  synthesizeController,
  synthesizeAtGamma,
  setScsSettingsOverride,
} from "../lib/hinf-synthesis";

void getSCS;

async function main() {
  const plant = plantById("torsional-3mass");
  const Ts = 5e-4;
  const specs = {
    desMm: 0.3,
    desBw: 150,
    desZeta: 0.8,
    order: 8,
    Ts,
  };
  const grid = plant.buildGrid(Ts, specs.desBw);

  console.log(`plant=${plant.id} mm=${specs.desMm} bw=${specs.desBw}Hz order=${specs.order}`);
  console.log(`grid n=${grid.w.length}`);

  // First: full bisection trace with wall-clock per step (wraps
  // synthesizeAtGamma since synthesizeController doesn't expose per-step
  // wall clock by itself).
  console.log("\n── γ sweep (one-shot per γ, no warm start) ──");
  console.log("  γ         feas   status            wall       ");
  for (const γ of [0.1, 0.2, 0.5, 0.7, 0.9, 0.95, 0.98, 1.0]) {
    const t0 = Date.now();
    const r = await synthesizeAtGamma(grid, specs, γ);
    const ms = Date.now() - t0;
    const ok = r.residual < 1e-3;
    console.log(
      `  ${γ.toFixed(3).padStart(6)}   ${ok ? "FEAS " : "infeas"}  ${r.status.padEnd(18)} ${ms.toString().padStart(6)}ms`,
    );
  }

  console.log("\n── full bisection (with warm start, BIS_TOL=current) ──");
  let lastT = Date.now();
  const totalT0 = Date.now();
  const res = await synthesizeController(grid, specs, (p) => {
    const now = Date.now();
    const dt = now - lastT;
    lastT = now;
    console.log(
      `  bis=${String(p.iter).padStart(2)} γ=${p.gamma.toFixed(4)} feas=${String(p.feasible).padEnd(5)} dt=${String(dt).padStart(5)}ms`,
    );
  });
  const total = Date.now() - totalT0;
  console.log(
    `\nresult: feas=${res.feasible} γ_pyf=${res.gammaPyfresco.toFixed(3)} H∞=${res.gammaOpt.toFixed(3)} bw=${res.achievedBw.toFixed(1)}Hz bis=${res.iterations.length} total=${total}ms`,
  );

  console.log("\n── lowering per-SOCP budget: maxIters=20000, timeLimitSecs=3 ──");
  setScsSettingsOverride({ maxIters: 20000, timeLimitSecs: 3 });
  lastT = Date.now();
  const totalT1 = Date.now();
  const res2 = await synthesizeController(grid, specs, (p) => {
    const now = Date.now();
    const dt = now - lastT;
    lastT = now;
    console.log(
      `  bis=${String(p.iter).padStart(2)} γ=${p.gamma.toFixed(4)} feas=${String(p.feasible).padEnd(5)} dt=${String(dt).padStart(5)}ms`,
    );
  });
  const total2 = Date.now() - totalT1;
  console.log(
    `\nresult: feas=${res2.feasible} γ_pyf=${res2.gammaPyfresco.toFixed(3)} H∞=${res2.gammaOpt.toFixed(3)} bw=${res2.achievedBw.toFixed(1)}Hz bis=${res2.iterations.length} total=${total2}ms`,
  );

  console.log("\n── loosened eps: 1e-5 / 1e-5, timeLimitSecs=5 ──");
  setScsSettingsOverride({ epsAbs: 1e-5, epsRel: 1e-5, maxIters: 50000, timeLimitSecs: 5 });
  lastT = Date.now();
  const totalT2 = Date.now();
  const res3 = await synthesizeController(grid, specs, (p) => {
    const now = Date.now();
    const dt = now - lastT;
    lastT = now;
    console.log(
      `  bis=${String(p.iter).padStart(2)} γ=${p.gamma.toFixed(4)} feas=${String(p.feasible).padEnd(5)} dt=${String(dt).padStart(5)}ms`,
    );
  });
  const total3 = Date.now() - totalT2;
  console.log(
    `\nresult: feas=${res3.feasible} γ_pyf=${res3.gammaPyfresco.toFixed(3)} H∞=${res3.gammaOpt.toFixed(3)} bw=${res3.achievedBw.toFixed(1)}Hz bis=${res3.iterations.length} total=${total3}ms`,
  );
}
main().catch((e) => {
  console.error(e);
  process.exit(1);
});

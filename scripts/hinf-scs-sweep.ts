/* ------------------------------------------------------------------
   Sweep SCS settings on the torsional-chain hard case and see which
   configuration closes the γ gap vs. pyfresco (γ ≈ 1.08 with ECOS).

   Torsional plant, Ts=5e-4, desBw=150 Hz, desMm=0.3, desZ=0.8, order 8.
   Baseline browser run: γ = 7.98, f_c ≈ 24 Hz.

   Usage:  npx tsx scripts/hinf-scs-sweep.ts
   ------------------------------------------------------------------ */

import { plantById } from "../lib/hinf-plants";
import {
  synthesizeController,
  setScsSettingsOverride,
  ScsSettingsOverride,
} from "../lib/hinf-synthesis";

const plant = plantById("torsional-3mass");
const Ts = 5e-4;
const specs = {
  desMm: 0.3,
  desBw: 150,
  desZeta: 0.8,
  order: 8,
  Ts,
};

async function runOnce(label: string, override: ScsSettingsOverride | null) {
  setScsSettingsOverride(override);
  const grid = plant.buildGrid(Ts, specs.desBw);
  const t0 = Date.now();
  const res = await synthesizeController(grid, specs);
  const elapsed = Date.now() - t0;
  const g = isFinite(res.gammaOpt) ? res.gammaOpt.toFixed(3) : "inf";
  console.log(
    `${label.padEnd(42)}  γ=${g.padStart(7)}  γ_pyf=${res.gammaPyfresco.toFixed(3).padStart(6)}  bw=${res.achievedBw.toFixed(1).padStart(5)}Hz  feas=${String(res.feasible).padEnd(5)}  bis=${String(res.iterations.length).padStart(2)}  ${elapsed}ms`,
  );
}

async function main() {
  console.log(
    `Target: torsional Ts=${Ts}s, desBw=${specs.desBw}Hz, desMm=${specs.desMm}, desZ=${specs.desZeta}, order=${specs.order}`,
  );
  console.log(`pyfresco (ECOS) reference: γ ≈ 1.08  f_c ≈ target bw\n`);

  // 1. baseline = whatever innerSolveSCS ships with.
  await runOnce("baseline (current)", null);

  // 2. loosen time limit (probably the biggest one — 5 s is suspiciously tight)
  await runOnce("no time limit, 100k iters",
    { timeLimitSecs: 0, maxIters: 100000 });

  // 3. tighter eps with more room to iterate.
  await runOnce("eps 1e-8 / 100k iters / no timeout",
    { epsAbs: 1e-8, epsRel: 1e-8, maxIters: 100000, timeLimitSecs: 0 });

  // 4. very tight eps + big iter budget.
  await runOnce("eps 1e-9 / 200k iters / no timeout",
    { epsAbs: 1e-9, epsRel: 1e-9, maxIters: 200000, timeLimitSecs: 0 });

  // 5. disable adaptive scaling (sometimes helps on ill-conditioned SOCPs).
  await runOnce("adaptiveScale=false",
    { adaptiveScale: false, timeLimitSecs: 0, maxIters: 100000 });

  // 6. disable normalization.
  await runOnce("normalize=false",
    { normalize: false, timeLimitSecs: 0, maxIters: 100000 });

  // 7. manual scale — pyfresco/ECOS preprocessing uses equilibration;
  //    larger initial `scale` sometimes helps SCS on SOCPs with a wide
  //    coefficient dynamic range (this plant has |G| swing >10⁴).
  await runOnce("scale=10, adaptiveScale=false",
    { scale: 10, adaptiveScale: false, timeLimitSecs: 0, maxIters: 100000 });
  await runOnce("scale=100, adaptiveScale=false",
    { scale: 100, adaptiveScale: false, timeLimitSecs: 0, maxIters: 100000 });
  await runOnce("scale=0.01, adaptiveScale=false",
    { scale: 0.01, adaptiveScale: false, timeLimitSecs: 0, maxIters: 100000 });

  // 8. alpha (Douglas-Rachford relaxation).
  await runOnce("alpha=1.0",
    { alpha: 1.0, timeLimitSecs: 0, maxIters: 100000 });
  await runOnce("alpha=1.8",
    { alpha: 1.8, timeLimitSecs: 0, maxIters: 100000 });

  // 9. rhoX (primal scaling) — tiny by default (1e-6); try larger.
  await runOnce("rhoX=1e-4",
    { rhoX: 1e-4, timeLimitSecs: 0, maxIters: 100000 });
  await runOnce("rhoX=1.0",
    { rhoX: 1.0, timeLimitSecs: 0, maxIters: 100000 });

  // 10. everything and the kitchen sink.
  await runOnce("tight eps + big budget + no normalize",
    {
      epsAbs: 1e-9, epsRel: 1e-9, maxIters: 200000, timeLimitSecs: 0,
      normalize: false, adaptiveScale: false,
    });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

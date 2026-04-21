/* ------------------------------------------------------------------
   Feasibility harness — runs the H∞ RST synthesis against every
   preset plant at its default specs and prints a pass/fail report.

   Usage:
     npx tsx scripts/hinf-feasibility-harness.ts
     PLANT=eddy-fractional npx tsx scripts/hinf-feasibility-harness.ts
     STRESS=1 npx tsx scripts/hinf-feasibility-harness.ts   (sweep mm)
   ------------------------------------------------------------------ */

import { PLANTS } from "../lib/hinf-plants";
import { synthesizeController } from "../lib/hinf-synthesis";

async function runOne(plant: (typeof PLANTS)[number], mm?: number, bw?: number, zeta?: number) {
  const specs = {
    desMm: mm ?? plant.defaults.desMm,
    desBw: bw ?? plant.defaults.desBw,
    desZeta: zeta ?? plant.defaults.desZeta,
    Ts: plant.Ts,
  };
  const grid = plant.buildGrid(plant.Ts, specs.desBw);
  const start = Date.now();
  const res = await synthesizeController(grid, specs);
  const elapsed = Date.now() - start;

  const tag = res.feasible ? "PASS" : "FAIL";
  const hinfNorm = res.feasible ? res.gammaOpt.toFixed(3) : "∞";
  const gammaPyf = res.feasible ? res.gammaPyfresco.toFixed(3) : "-";
  const bwStr = res.feasible ? `${res.achievedBw.toFixed(1)} Hz` : "-";
  console.log(
    `[${tag}] ${plant.id.padEnd(22)} mm=${specs.desMm.toFixed(2)} ` +
      `bw=${specs.desBw.toFixed(1)}Hz zeta=${specs.desZeta.toFixed(2)} ` +
      `→ H∞=${hinfNorm} γ_pyf=${gammaPyf} achievedBw=${bwStr} ` +
      `bis=${res.iterations.length} ${elapsed}ms`,
  );
  return res;
}

async function main() {
  const only = process.env.PLANT;
  const stress = process.env.STRESS === "1";
  const plants = only ? PLANTS.filter((p) => p.id === only) : PLANTS;

  console.log("\n── default-spec feasibility ─────────────────────────");
  let passes = 0;
  for (const p of plants) {
    const r = await runOne(p);
    if (r.feasible) passes++;
  }
  console.log(`\n${passes}/${plants.length} plants feasible at default specs.`);

  if (!stress) return;

  console.log("\n── modulus-margin sweep (bw=default, ζ=default) ────");
  for (const p of plants) {
    for (const mm of [0.3, 0.5, 0.7]) {
      await runOne(p, mm);
    }
  }

  console.log("\n── bandwidth sweep at fixed mm=0.5 ──────────────────");
  for (const p of plants) {
    const Fs = 1 / p.Ts;
    for (const fc of [Fs / 25, Fs / 15, Fs / 8]) {
      await runOne(p, 0.5, fc);
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

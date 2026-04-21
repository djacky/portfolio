/* ------------------------------------------------------------------
   Run the browser H∞ synthesis on Plant 1 (dipole correction coil)
   at the user-specified test case — Ts=5e-4, desBw=200, desMm=0.5,
   desZeta=0.8, n_r=n_s=n_t=6 — and print the RST coefficients so the
   python comparison script can pick them up.

   Usage:  npx tsx scripts/hinf-dipole-usercase.ts
   ------------------------------------------------------------------ */

import { plantById } from "../lib/hinf-plants";
import { synthesizeController } from "../lib/hinf-synthesis";

async function main() {
  const plant = plantById("magnet-rl");
  const Ts = 5e-4;
  const specs = {
    desMm: 0.5,
    desBw: 200,
    desZeta: 0.8,
    Ts,
  };
  const grid = plant.buildGrid(Ts, specs.desBw);
  const t0 = Date.now();
  const res = await synthesizeController(grid, specs);
  const elapsed = Date.now() - t0;

  console.log(`elapsed      = ${elapsed} ms`);
  console.log(`feasible     = ${res.feasible}`);
  console.log(`H∞ norm      = ${res.gammaOpt.toFixed(6)}`);
  console.log(`γ_pyfresco   = ${res.gammaPyfresco.toFixed(6)}`);
  console.log(`achieved f_c = ${res.achievedBw.toFixed(3)} Hz`);
  console.log(`bisection    = ${res.iterations.length} iters`);
  console.log(`Gain (ΣT/ΣR) = ${res.gain.toFixed(6)}`);

  const fmt = (arr: number[]) =>
    "[" + arr.map((x) => x.toFixed(6)).join(", ") + "]";
  console.log(`R            = ${fmt(res.RFull)}`);
  console.log(`S            = ${fmt(res.SFull)}`);
  console.log(`T            = ${fmt(res.TFull)}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

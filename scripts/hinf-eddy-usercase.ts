/* ------------------------------------------------------------------
   Run the browser H∞ synthesis on Plant 3 (eddy-current magnet,
   fractional α=0.75 · −15 dB/dec) at its default specs and print
   the RST coefficients for the python comparison.

   Usage:  npx tsx scripts/hinf-eddy-usercase.ts
   ------------------------------------------------------------------ */

import { plantById } from "../lib/hinf-plants";
import { synthesizeController } from "../lib/hinf-synthesis";

async function main() {
  const plant = plantById("eddy-fractional");
  const Ts = plant.Ts;
  const specs = {
    desMm: plant.defaults.desMm,
    desBw: plant.defaults.desBw,
    desZeta: plant.defaults.desZeta,
    Ts,
  };
  const grid = plant.buildGrid(Ts, specs.desBw);
  const t0 = Date.now();
  const res = await synthesizeController(grid, specs);
  const elapsed = Date.now() - t0;

  console.log(`Ts           = ${Ts}`);
  console.log(`desBw        = ${specs.desBw.toFixed(3)} Hz`);
  console.log(`desMm/desZ   = ${specs.desMm} / ${specs.desZeta}`);
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

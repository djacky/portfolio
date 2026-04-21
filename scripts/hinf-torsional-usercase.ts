/* ------------------------------------------------------------------
   Run the browser H∞ synthesis on Plant 4 (torsional 3-mass chain,
   stable / trivial-coprime) at the user-specified test case —
   Ts=5e-4, desBw=150 Hz, desMm=0.3, desZ=0.8, 8th-order controller
   (n_r=n_s=n_t=9) — and print the RST coefficients so the pyfresco
   comparison script can pick them up.

   Usage:  npx tsx scripts/hinf-torsional-usercase.ts
   ------------------------------------------------------------------ */

import { plantById } from "../lib/hinf-plants";
import { synthesizeController } from "../lib/hinf-synthesis";

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
  const t0 = Date.now();
  const res = await synthesizeController(grid, specs);
  const elapsed = Date.now() - t0;

  console.log(`plant        = ${plant.id}`);
  console.log(`Ts           = ${Ts}`);
  console.log(`desBw        = ${specs.desBw} Hz`);
  console.log(`desMm/desZ   = ${specs.desMm} / ${specs.desZeta}`);
  console.log(`order        = ${specs.order}  (${specs.order + 1} coefficients)`);
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

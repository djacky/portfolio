/* Diagnostic — traces one plant's bisection step by step. */
import { PLANTS } from "../lib/hinf-plants";
import { synthesizeController } from "../lib/hinf-synthesis";

const plant = PLANTS[0]; // magnet-rl
const specs = {
  desMm: plant.defaults.desMm,
  desBw: plant.defaults.desBw,
  desZeta: plant.defaults.desZeta,
  order: plant.defaults.order,
  Ts: plant.Ts,
};
const grid = plant.buildGrid(plant.Ts, specs.desBw);

console.log(`plant=${plant.id} mm=${specs.desMm} bw=${specs.desBw}Hz ζ=${specs.desZeta} Ts=${specs.Ts}`);
console.log(`grid: n=${grid.w.length} wMin=${grid.w[0].toFixed(3)} wMax=${grid.w[grid.w.length - 1].toFixed(3)} rad/s`);

// Inspect |G| and |Wd| at a few points.
import { cAbs, cExp, cSub, Complex, cMul } from "../lib/hinf-synthesis";
// Recompute Wd in-script to trace it
const zeta = specs.desZeta;
const wd = (2 * Math.PI * specs.desBw) / Math.sqrt(1 - 2 * zeta * zeta + Math.sqrt(2 - 4 * zeta * zeta + 4 * Math.pow(zeta, 4)));
console.log(`wd=${wd.toFixed(2)} rad/s (${(wd / (2 * Math.PI)).toFixed(2)} Hz)`);

console.log("\n  k     ω           f(Hz)       |G|        |Wd|");
for (const k of [0, 20, 40, 60, 80, 99]) {
  const w = grid.w[k];
  const denomRe = wd * wd - w * w;
  const denomIm = 2 * zeta * wd * w;
  const mag2 = denomRe * denomRe + denomIm * denomIm;
  const Td: Complex = { re: (wd * wd * denomRe) / mag2, im: -(wd * wd * denomIm) / mag2 };
  const omt = { re: 1 - Td.re, im: -Td.im };
  const m = omt.re * omt.re + omt.im * omt.im;
  const Wd = { re: omt.re / m, im: -omt.im / m };
  console.log(`  ${String(k).padStart(3)}  ${w.toFixed(3).padStart(10)}  ${(w / (2 * Math.PI)).toFixed(3).padStart(10)}  ${cAbs(grid.G[k]).toExponential(2)}   ${cAbs(Wd).toExponential(2)}`);
}

import { synthesizeAtGamma } from "../lib/hinf-synthesis";

async function main() {
  console.log("\n── bisection trace ──");
  const result = await synthesizeController(grid, specs, (p) => {
    console.log(`  bis=${p.iter} γ=${p.gamma.toFixed(4)} feas=${p.feasible} bw=${p.bw?.toFixed(1) ?? "-"}`);
  });
  console.log(`\nfinal: feasible=${result.feasible} H∞=${result.gammaOpt} bw=${result.achievedBw}`);

  console.log("\n── γ sweep ──");
  for (const γ of [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]) {
    const r = await synthesizeAtGamma(grid, specs, γ);
    const ok = r.residual < 1e-3;
    console.log(`  γ=${γ.toFixed(2).padStart(5)} ${ok ? "FEAS" : "infeas"} status=${r.status} residual=${r.residual.toExponential(2)}`);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });

/* Trace the inner AL solver at fixed γ. */
import { PLANTS } from "../lib/hinf-plants";
import { synthesizeAtGamma } from "../lib/hinf-synthesis";

const plant = PLANTS[0];
const specs = {
  desMm: 0.3,
  desBw: 200,
  desZeta: 0.8,
  Ts: plant.Ts,
};
const grid = plant.buildGrid(plant.Ts, specs.desBw);

async function main() {
  console.log(`plant=${plant.id} mm=${specs.desMm} bw=${specs.desBw}Hz`);
  for (const γ of [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]) {
    const r = await synthesizeAtGamma(grid, specs, γ);
    console.log(`  γ=${γ} residual=${r.residual.toExponential(2)} status=${r.status} ${r.residual < 1e-3 ? "FEAS" : "infeas"}`);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });

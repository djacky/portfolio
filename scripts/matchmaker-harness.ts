/* ------------------------------------------------------------------
   Headless matchmaker simulation harness.

   Drives the MatchmakerEngine tick loop in a tight loop (no setInterval
   delays) and captures metrics at each checkpoint.  Purpose: diagnose
   σ-calibration / μ̂-oscillation behaviour without waiting for wall-
   clock training in the browser.

   Run:
     npx tsx scripts/matchmaker-harness.ts
------------------------------------------------------------------ */

import { getEngine } from "../lib/matchmaker-engine";

const SEEDS = parseInt(process.env.SEEDS ?? "1", 10);
const TARGET_MATCHES = parseInt(process.env.MATCHES ?? "150", 10);
const CHECKPOINTS = [20, 50, 100, 150, 200, 250, 300].filter((m) => m <= TARGET_MATCHES);
const SMURF_INJECTIONS_AT = [60, 120]; // matches at which to inject a smurf

interface Sample {
  seed: number;
  match: number;
  rho: number;
  cal1: number;
  cal2: number;
  sigMSE: number;
  trainSkillLoss: number;
  heldOutNll: number;
  delta: number;
  smurfP: number;
  smurfR: number;
  // per-player distribution diagnostics
  medErr: number;
  p25Err: number;
  p75Err: number;
  medSig: number;
  p25Sig: number;
  p75Sig: number;
  ratio: number;       // medSig / medErr
  errStd: number;      // std of |err| across players — oscillation indicator
  muMean: number;      // mean of muHatEma across pool
  muStd: number;       // std of muHatEma across pool (compare to trueSkill std ≈ 0.29)
  muMin: number;
  muMax: number;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * sorted.length)));
  return sorted[idx];
}

function runSeed(seedIdx: number, onCheckpoint: (s: Sample) => void) {
  const engine = getEngine() as any;
  engine.resetModel();
  engine.reseed(160);

  const hit = new Set<number>();
  let smurfIdx = 0;

  while (engine.matchesPlayed < TARGET_MATCHES) {
    // Engine normally: trainTimer @ 5 Hz, matchTimer @ 2.5 Hz (2:1 ratio).
    // Mimic that ratio in tight loop.
    engine.trainTick();
    engine.trainTick();
    engine.matchTick();

    const m: number = engine.matchesPlayed;

    // Smurf injection at designated match counts
    while (smurfIdx < SMURF_INJECTIONS_AT.length && m >= SMURF_INJECTIONS_AT[smurfIdx]) {
      engine.injectSmurf();
      smurfIdx++;
    }

    // Checkpoint metrics
    if (CHECKPOINTS.includes(m) && !hit.has(m)) {
      hit.add(m);
      const snap = engine.snapshot();
      const pool = engine.players.filter((p: any) => p.history.length >= 2);
      const errs = pool.map((p: any) => Math.abs(p.trueSkill - p.muHatEma)).sort((a: number, b: number) => a - b);
      const sigmas = pool.map((p: any) => p.sigmaHat).sort((a: number, b: number) => a - b);
      const medErr = percentile(errs, 0.5);
      const medSig = percentile(sigmas, 0.5);
      const errMean = errs.reduce((x: number, y: number) => x + y, 0) / (errs.length || 1);
      const errVar = errs.reduce((x: number, y: number) => x + (y - errMean) ** 2, 0) / (errs.length || 1);
      const errStd = Math.sqrt(errVar);
      const mus = pool.map((p: any) => p.muHatEma);
      const muMean = mus.reduce((x: number, y: number) => x + y, 0) / (mus.length || 1);
      const muVar = mus.reduce((x: number, y: number) => x + (y - muMean) ** 2, 0) / (mus.length || 1);
      const muStd = Math.sqrt(muVar);
      const muMin = Math.min(...mus);
      const muMax = Math.max(...mus);

      onCheckpoint({
        seed: seedIdx,
        match: m,
        rho: snap.rankCorrelation,
        cal1: snap.calibration1 ?? 0,
        cal2: snap.calibration2 ?? 0,
        sigMSE: snap.sigmaLoss ?? 0,
        trainSkillLoss: snap.skillLoss,
        heldOutNll: snap.heldOutNll ?? 0,
        delta: (snap.heldOutNll ?? 0) - snap.skillLoss,
        smurfP: snap.smurfPrecision,
        smurfR: snap.smurfRecall,
        medErr, p25Err: percentile(errs, 0.25), p75Err: percentile(errs, 0.75),
        medSig, p25Sig: percentile(sigmas, 0.25), p75Sig: percentile(sigmas, 0.75),
        ratio: medErr > 1e-9 ? medSig / medErr : Infinity,
        errStd,
        muMean, muStd, muMin, muMax,
      });
    }
  }
}

function pad(s: string | number, w: number): string {
  const str = typeof s === "number" ? s.toFixed(s > 10 ? 1 : 3) : s;
  return str.padStart(w, " ");
}

async function main() {
  console.log(`Matchmaker headless harness — ${SEEDS} seeds × ${TARGET_MATCHES} matches each\n`);
  const allSamples: Sample[] = [];
  const t0 = Date.now();

  for (let s = 0; s < SEEDS; s++) {
    const seedStart = Date.now();
    console.log(`\n--- seed ${s} ---`);
    console.log(
      pad("match", 6) + pad("ρ", 6) + pad("cal1", 6) + pad("cal2", 6)
      + pad("σMSE", 7) + pad("Δ", 7) + pad("smP", 6) + pad("smR", 6)
      + pad("medErr", 8) + pad("medσ̂", 8) + pad("σ̂/err", 7) + pad("errStd", 8)
      + pad("μ̄", 7) + pad("μσ", 7) + pad("μrng", 12)
    );
    runSeed(s, (sample) => {
      allSamples.push(sample);
      console.log(
        pad(sample.match, 6)
        + pad(sample.rho.toFixed(2), 6)
        + pad((sample.cal1 * 100).toFixed(0) + "%", 6)
        + pad((sample.cal2 * 100).toFixed(0) + "%", 6)
        + pad(sample.sigMSE.toFixed(4), 7)
        + pad((sample.delta >= 0 ? "+" : "") + sample.delta.toFixed(3), 7)
        + pad((sample.smurfP * 100).toFixed(0) + "%", 6)
        + pad((sample.smurfR * 100).toFixed(0) + "%", 6)
        + pad(sample.medErr.toFixed(3), 8)
        + pad(sample.medSig.toFixed(3), 8)
        + pad(sample.ratio.toFixed(2), 7)
        + pad(sample.errStd.toFixed(3), 8)
        + pad(sample.muMean.toFixed(3), 7)
        + pad(sample.muStd.toFixed(3), 7)
        + pad(`${sample.muMin.toFixed(2)}..${sample.muMax.toFixed(2)}`, 12)
      );
    });
    const seedElapsed = ((Date.now() - seedStart) / 1000).toFixed(1);
    console.log(`(seed ${s} took ${seedElapsed}s)`);
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\ntotal runtime: ${elapsed}s`);

  // Aggregate across seeds per checkpoint
  console.log(`\n--- cross-seed aggregates ---`);
  console.log(
    pad("match", 6) + pad("ρ̄", 8) + pad("cal1̄", 10) + pad("cal2̄", 10)
    + pad("σ̂/err̄", 10) + pad("cal1 σ", 8)
  );
  for (const cp of CHECKPOINTS) {
    const rows = allSamples.filter((s) => s.match === cp);
    if (rows.length === 0) continue;
    const rhoMean = rows.reduce((a, r) => a + r.rho, 0) / rows.length;
    const cal1Mean = rows.reduce((a, r) => a + r.cal1, 0) / rows.length;
    const cal2Mean = rows.reduce((a, r) => a + r.cal2, 0) / rows.length;
    const ratioMean = rows.reduce((a, r) => a + r.ratio, 0) / rows.length;
    const cal1Var = rows.reduce((a, r) => a + (r.cal1 - cal1Mean) ** 2, 0) / rows.length;
    const cal1Std = Math.sqrt(cal1Var);
    console.log(
      pad(cp, 6)
      + pad(rhoMean.toFixed(2), 8)
      + pad((cal1Mean * 100).toFixed(1) + "%", 10)
      + pad((cal2Mean * 100).toFixed(1) + "%", 10)
      + pad(ratioMean.toFixed(2), 10)
      + pad((cal1Std * 100).toFixed(1) + "pp", 8)
    );
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

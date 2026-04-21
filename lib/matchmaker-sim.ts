/* ------------------------------------------------------------------
   Matchmaker Simulation — the environment that feeds the transformer.

   Each player carries hidden latent state that the transformer has to
   infer from the sequence of matches alone:

     • trueSkill        ∈ [0,1]   — baseline strength
     • playstyle        ∈ R⁴      — hidden style vector (sampled on spawn)
         The four dimensions map to recognizable battle-royale archetypes:
           0 — aggression   (hunter ↔ camper)   drives +kills +deaths +dmg
           1 — precision    (marksman ↔ spray)  drives +accuracy, +K/D,
                                                slightly lower damage volume
           2 — survivor     (positioner ↔ brawler) drives −kills −deaths,
                                                a small placement bonus
           3 — objective    (looter ↔ fighter)  drives +damage (better gear),
                                                −kills (avoids fights while
                                                rotating), small placement
                                                bonus from good zone play
     • fatigue          ∈ [0,1]   — decays performance after many matches
                                    in short succession; recovers over idle
     • tilt             ∈ [0,1]   — kicks in after 2+ bottom-5 finishes;
                                    temporary performance hit
     • isSmurf          : bool    — first ~8 matches: visible stats are
                                    suppressed vs trueSkill; anomaly head
                                    should learn to flag these
     • matchesPlayed    : int

   Matches are 30-player battle royales. We use a Plackett–Luce sampler
   on the EFFECTIVE skill (trueSkill modulated by fatigue, tilt, playstyle
   synergy with the lobby composition) so the ordering has the right
   stochastic structure (upsets happen, but strong players usually win).

   Each match produces a 12-D token per participant that is appended to
   that player's history ring-buffer — this is the input sequence the
   transformer consumes.
------------------------------------------------------------------ */

import { INPUT_DIM, SEQ_LEN } from "./transformer-matchmaker";

export const LOBBY_SIZE = 30;
export const DEFAULT_QUEUE_SIZE = 160;
export const PLAYSTYLE_HIDDEN_DIM = 4;

/* ---------------- Glicko-2-style σ target ----------------
   The transformer's σ head is supervised against a closed-form Bayesian
   target (Glicko-2-style: precision shrinks per match, variance inflates
   between matches).  Learnable PL temperatures collapse — see Bruch 2020
   — so we pin σ to an external teacher instead.  The transformer still
   does the work of *predicting* the target from the token sequence (so
   a fresh smurf's low-match σ gets flagged BEFORE the analytic update
   has had time to shrink), just not of inventing uncertainty from whole
   cloth under an unsupervised objective.

   Scale lives in [0,1] skill-space (same units as trueSkill / muHat):
     σ≈0.35 ≈ "±1 tier" uncertainty on a new player,
     σ≈0.04 ≈ floor after many matches (reflects irreducible PL noise).
------------------------------------------------------- */
export const SIGMA_INIT = 0.35;
export const SIGMA_MIN = 0.04;
export const SIGMA_MAX = 0.6;
// Precision added per observed match.  Empirical sweep at steady-state ρ≈0.98:
// 3.0 → cal 1σ 43%, 5.0 → cal 1σ 77-87%, 4.0 → cal 1σ 82-96%.  Monotonic the
// OTHER direction from what you'd expect naively — more precision shrinks σ̂
// but also shrinks σ_target faster, which the student tracks; net effect is
// σ̂ becomes a TIGHTER but more honestly-sized predictor of |err|.  7.0 is
// the next point in the sweep, targeting cal 1σ ≈ 68%.
export const SIGMA_PRECISION_PER_MATCH = 7.0;
// Variance inflation per idle tick — mirrors Glicko-2's between-period
// drift term.  Small so players that idle briefly don't reset to σ_init.
export const SIGMA_INFLATION_PER_TICK = 2e-5;

export type Tier = "Bronze" | "Silver" | "Gold" | "Diamond";

export interface MatchToken {
  placement: number;     // 1..30
  kills: number;         // 0..1
  deaths: number;        // 0..1
  damage: number;        // 0..1
  accuracy: number;      // 0..1
  lobbyMuMean: number;   // avg learned μ of the lobby
  lobbyMuSpread: number; // std of learned μ
  timeSinceLast: number; // ticks since this player's last match (normalized)
  patch: number;         // 0..3 — patch epoch index at match time
  // MMR-climb-velocity proxy: current win-mag minus trailing mean of past
  // win-mags, in [-1, 1].  Real matchmaking systems (Riot "Smurf Queue",
  // Blizzard's SR velocity flag) use rapid positive deflection as the
  // strongest early smurf signal — it shows up in the sequence long
  // before the raw stat profile normalizes.
  climbVelocity: number;
}

export interface Player {
  id: number;
  name: string;
  // hidden truth — the transformer doesn't see these directly
  trueSkill: number;
  playstyle: Float32Array; // length PLAYSTYLE_HIDDEN_DIM
  fatigue: number;
  tilt: number;
  isSmurf: boolean;
  recentBottomFinishes: number; // for tilt onset
  // visible / derived
  matchesPlayed: number;
  lastMatchAt: number; // global tick index
  history: MatchToken[]; // ring-buffered to SEQ_LEN
  // predictions from the transformer (updated after each re-encode)
  muHat: number;
  muHatEma: number; // EMA-shadow forward — used for stable ρ, not lobby picking
  sigmaHat: number;
  // Glicko-2-style teacher σ.  Updated in closed form after every match; the
  // transformer's σ head is trained to predict this value (amortized Bayesian
  // inference — Kingma/Welling 2014, Garnelo NP 2018, Radev BayesFlow 2020).
  sigmaTarget: number;
  playstyleEmb: Float32Array; // length PLAYSTYLE_HIDDEN_DIM
  anomalyP: number;
  formMod: number;
  // attention cache — last-query-position weights per layer, shape [heads × T]
  // populated by the engine during encodeAll, read by the HUD on hover
  attentionLayers: Float32Array[] | null;
  attentionT: number;
  attentionHeads: number;
  // bookkeeping
  cooldown: number;
  waitTicks: number;
  color: string;
  tier: Tier;
}

const NAMES = [
  "aurora", "neo", "kairo", "vex", "luma", "rift", "hex", "onyx",
  "nova", "echo", "zephyr", "sable", "pyra", "drift", "glitch", "flux",
  "ember", "cipher", "mira", "halo", "quasar", "orbit", "vector", "lumen",
  "axon", "blitz", "cinder", "dusk", "frost", "ghost", "helix", "iris",
  "jade", "karma", "lyra", "myst", "nyx", "omen", "prism", "quartz",
];

const TIER_COLORS: Record<Tier, string> = {
  Bronze:  "#c2843f",
  Silver:  "#cbd5e1",
  Gold:    "#fbbf24",
  Diamond: "#22d3ee",
};

export function tierOf(skill: number): Tier {
  if (skill < 0.25) return "Bronze";
  if (skill < 0.5)  return "Silver";
  if (skill < 0.8)  return "Gold";
  return "Diamond";
}

function randn(std = 1): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function makePlayer(id: number, smurfRate = 0.12): Player {
  const trueSkill = Math.random();
  const playstyle = new Float32Array(PLAYSTYLE_HIDDEN_DIM);
  for (let i = 0; i < PLAYSTYLE_HIDDEN_DIM; i++) playstyle[i] = randn(0.6);
  const tier = tierOf(trueSkill);
  // Pre-match visual jitter: spread orbs before the first forward pass so
  // the scene isn't a single clump at the origin on cold start.
  const embInit = new Float32Array(PLAYSTYLE_HIDDEN_DIM);
  for (let i = 0; i < PLAYSTYLE_HIDDEN_DIM; i++) embInit[i] = randn(0.3);
  return {
    id,
    name: NAMES[id % NAMES.length] + (id >= NAMES.length ? Math.floor(id / NAMES.length) : ""),
    trueSkill,
    playstyle,
    fatigue: 0,
    tilt: 0,
    isSmurf: Math.random() < smurfRate,
    recentBottomFinishes: 0,
    matchesPlayed: 0,
    lastMatchAt: 0,
    history: [],
    muHat: 0.5 + randn(0.05),
    muHatEma: 0.5 + randn(0.05),
    sigmaHat: SIGMA_INIT,
    sigmaTarget: SIGMA_INIT,
    playstyleEmb: embInit,
    attentionLayers: null,
    attentionT: 0,
    attentionHeads: 2,
    anomalyP: 0,
    formMod: 0,
    cooldown: 0,
    waitTicks: 0,
    color: TIER_COLORS[tier],
    tier,
  };
}

/**
 * Effective skill used by the PL sampler.
 *   baseline = trueSkill
 *   − fatigue penalty  (up to −0.25)
 *   − tilt penalty     (up to −0.20)
 *   + playstyle synergy: small reward for being in a lobby whose mean
 *       playstyle is orthogonal to yours (rewards diversity/niche)
 *   + small survivor bonus & aggression penalty on placement probability
 *       (positioners place higher; hunters die early more often).  Kept
 *       small so trueSkill remains the dominant driver of outcome.
 *   For smurfs in early matches, trueSkill is NOT suppressed — only the
 *   VISIBLE stats are.  This creates the "smurf signature": a first-match
 *   placement way higher than the visible-stat profile would predict.
 */
function effectiveSkill(p: Player, lobby: Player[]): number {
  const fatiguePenalty = p.fatigue * 0.25;
  const tiltPenalty = p.tilt * 0.20;
  // playstyle synergy — cosine distance from lobby mean playstyle
  let meanSim = 0;
  if (lobby.length > 1) {
    const mean = new Float32Array(PLAYSTYLE_HIDDEN_DIM);
    for (const q of lobby) {
      if (q.id === p.id) continue;
      for (let i = 0; i < PLAYSTYLE_HIDDEN_DIM; i++) mean[i] += q.playstyle[i];
    }
    for (let i = 0; i < PLAYSTYLE_HIDDEN_DIM; i++) mean[i] /= (lobby.length - 1);
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < PLAYSTYLE_HIDDEN_DIM; i++) {
      dot += mean[i] * p.playstyle[i];
      na += mean[i] * mean[i];
      nb += p.playstyle[i] * p.playstyle[i];
    }
    const denom = Math.sqrt(na * nb) + 1e-6;
    meanSim = dot / denom;
  }
  const synergy = (1 - meanSim) * 0.05;
  const survBonus = p.playstyle[2] * 0.04;   // survivors place higher
  const aggrPenalty = p.playstyle[0] * 0.02; // hunters die earlier more often
  const objBonus = p.playstyle[3] * 0.018;   // objective/looter — good rotations
  return Math.max(
    0,
    p.trueSkill - fatiguePenalty - tiltPenalty + synergy + survBonus - aggrPenalty + objBonus,
  );
}

/**
 * Plackett–Luce sampling over the lobby.
 *   probability of each next finisher ∝ exp(β · effectiveSkill)
 *   β controls how often upsets happen. β=6 ~ "usually the strong win"
 */
export function plackettLuce(lobby: Player[], beta = 6): Player[] {
  const remaining = [...lobby];
  const out: Player[] = [];
  while (remaining.length > 0) {
    const weights = remaining.map((p) => Math.exp(beta * effectiveSkill(p, lobby)));
    const total = weights.reduce((a, b) => a + b, 0);
    let r = Math.random() * total;
    let picked = 0;
    for (let i = 0; i < remaining.length; i++) {
      r -= weights[i];
      if (r <= 0) { picked = i; break; }
    }
    out.push(remaining[picked]);
    remaining.splice(picked, 1);
  }
  return out;
}

/**
 * Produce a MatchToken for a player given their placement and the lobby.
 * Visible stats are built from three overlapping drivers:
 *
 *   1. skillSignal   — the dominant term (~60-80% of stat variance).
 *                      Derived from effectiveSkill + win magnitude.
 *   2. playstyle     — four hidden archetype axes that bias stat *shape*
 *                      (e.g. high-aggression = +kills +deaths, high-
 *                      precision = +accuracy, +K/D, survivor = fewer
 *                      engagements but higher placement).  This is the
 *                      signal the contrastive head latches onto.
 *   3. noise         — per-match variance, scaled by the consistency axis
 *                      so "steady" players have narrower distributions
 *                      than "volatile" ones.
 *
 * Coefficients are tuned so playstyle contributes ~20–25% of stat variance:
 * strong enough that 4-D SimCLR can separate archetypes in < 60 matches
 * per player, small enough that a single player's matches still vary
 * match-to-match (log-normal shaping on kills/damage adds realism).
 *
 * Smurf suppression stays strong and separate: for the first ~8 matches,
 * visible skillSignal is halved so the stat profile LOOKS like a low-skill
 * player despite high true placement — the anomaly head's signal.
 */
export function buildToken(
  p: Player,
  placement: number,
  lobby: Player[],
  tickNow: number,
  patchEpoch: number,
  avgLobbyMu: number,
  spreadLobbyMu: number,
): MatchToken {
  const placementNorm = placement / LOBBY_SIZE;
  const winMagnitude = 1 - placementNorm; // 1 at 1st, 0 at 30th

  let skillSignal = effectiveSkill(p, lobby) * 0.6 + winMagnitude * 0.4;
  if (p.isSmurf && p.matchesPlayed < 8) {
    skillSignal *= 0.45; // visible skill looks ~halved
  }

  // Archetype biases.  Playstyle values are Gaussian(0, 0.6), so typical
  // |value| ≈ 0.5–1.0 and extreme players (top/bottom ~2.5%) have |value|
  // ≳ 1.2.  Coefficients here are ~1.6× the previous pass — the old values
  // produced ~10% playstyle variance, which SimCLR couldn't reliably
  // separate in 4-D; at ~20–25% the archetype clusters open up enough
  // for the embedding head to latch onto them.
  const aggr = p.playstyle[0]; // hunter ↔ camper
  const prec = p.playstyle[1]; // marksman ↔ spray
  const surv = p.playstyle[2]; // positioner ↔ brawler
  const obj  = p.playstyle[3]; // looter ↔ fighter

  const noise = (amp: number) => (Math.random() - 0.5) * amp;
  // Multiplicative log-normal shaper: kills/damage in real BR data follow
  // heavy-tailed distributions (Apex, PUBG public leaderboards both show
  // log-normal K/D and DPG curves).  exp(N(0,σ)) gives that shape cheaply;
  // after clamping to [0,1] the right tail compresses but the LEFT tail
  // (good players getting unlucky early exits) comes through correctly.
  const logMult = (sigma: number) => Math.exp(randn(sigma));
  const clamp = (v: number) => Math.max(0, Math.min(1, v));

  // Kills: mostly skill, boosted by aggression, dampened by survivors
  // and objective (looters avoid fights), slight boost for precision.
  const kills = clamp(
    (0.15 + skillSignal * 0.75
      + aggr * 0.18 - surv * 0.11 + prec * 0.06 - obj * 0.09
      + noise(0.12)) * logMult(0.18),
  );
  // Deaths: anti-correlated with skill, boosted by aggression,
  // reduced by survivors and precision; objective plays it safer.
  const deaths = clamp(
    (0.90 - skillSignal * 0.65
      + aggr * 0.14 - surv * 0.18 - prec * 0.08 - obj * 0.05
      + noise(0.15)) * logMult(0.12),
  );
  // Damage: scales with skill; aggression ↑, precision ↓ (efficient kills
  // need less ammo), survivor ↓, objective ↑ (better gear from loot runs).
  const damage = clamp(
    (0.20 + skillSignal * 0.70
      + aggr * 0.16 - prec * 0.10 - surv * 0.06 + obj * 0.12
      + noise(0.12)) * logMult(0.18),
  );
  // Accuracy: precision is the primary driver (coef 0.35);
  // skill contributes moderately; looters get a small bump from better
  // gear.  This is the axis that most cleanly separates marksmen from
  // spray-and-pray in real data.
  const accuracy = clamp(
    0.30 + skillSignal * 0.35 + prec * 0.35 + obj * 0.06 + noise(0.10),
  );

  const tSinceLast = Math.min(1, (tickNow - p.lastMatchAt) / 200);

  // Climb velocity = winMag - mean(recent history winMags).  Looks at the
  // last 6 matches (or fewer if history is short); positive = improving,
  // negative = sliding.  Smurfs produce strongly positive values from
  // match 1 because their history is effectively bootstrapped from 0.
  let climbVelocity = 0;
  if (p.history.length > 0) {
    const LOOKBACK = 6;
    const start = Math.max(0, p.history.length - LOOKBACK);
    let sum = 0, n = 0;
    for (let i = start; i < p.history.length; i++) {
      sum += 1 - p.history[i].placement / LOBBY_SIZE;
      n++;
    }
    climbVelocity = winMagnitude - (n > 0 ? sum / n : 0);
  }

  return {
    placement,
    kills,
    deaths,
    damage,
    accuracy,
    lobbyMuMean: avgLobbyMu,
    lobbyMuSpread: spreadLobbyMu,
    timeSinceLast: tSinceLast,
    patch: patchEpoch,
    climbVelocity,
  };
}

/**
 * Serialize a player's match-history ring buffer into a contiguous
 * Float32Array of shape [T × INPUT_DIM] for the transformer.
 * Returns T = actual number of matches (up to SEQ_LEN).
 */
export function tokensForPlayer(p: Player): { tokens: Float32Array; T: number } {
  const T = Math.min(p.history.length, SEQ_LEN);
  const out = new Float32Array(T * INPUT_DIM);
  // Patch embedding uses 4 slots (one-hot of patch ∈ {0,1,2,3})
  for (let t = 0; t < T; t++) {
    const tok = p.history[p.history.length - T + t];
    const b = t * INPUT_DIM;
    out[b + 0] = tok.placement / LOBBY_SIZE;
    out[b + 1] = tok.kills;
    out[b + 2] = tok.deaths;
    out[b + 3] = tok.damage;
    out[b + 4] = tok.accuracy;
    out[b + 5] = tok.lobbyMuMean;
    out[b + 6] = tok.lobbyMuSpread;
    out[b + 7] = tok.timeSinceLast;
    out[b + 8] = tok.patch === 0 ? 1 : 0;
    out[b + 9] = tok.patch === 1 ? 1 : 0;
    out[b + 10] = tok.patch === 2 ? 1 : 0;
    out[b + 11] = tok.patch === 3 ? 1 : 0;
    out[b + 12] = tok.climbVelocity;
  }
  return { tokens: out, T };
}

/**
 * Closed-form Glicko-2-style σ update: variance inflates while the player
 * is idle, then precision 1/σ² gains a fixed increment for the new match.
 * Floored at SIGMA_MIN² — irreducible uncertainty from PL sampling noise
 * means an infinitely-confident model would be wrong, so σ̂ should plateau.
 */
export function updateSigmaTarget(p: Player, ticksSinceLast: number): void {
  let sig2 = p.sigmaTarget * p.sigmaTarget;
  sig2 += SIGMA_INFLATION_PER_TICK * Math.max(0, ticksSinceLast);
  const prec = 1 / sig2 + SIGMA_PRECISION_PER_MATCH;
  sig2 = 1 / prec;
  sig2 = Math.max(SIGMA_MIN * SIGMA_MIN, sig2);
  p.sigmaTarget = Math.sqrt(sig2);
}

/**
 * Apply post-match effects to a player: update fatigue, tilt, smurf clock,
 * the Glicko-2 σ target, and push the new token onto the history.
 */
export function postMatchUpdate(
  p: Player,
  tok: MatchToken,
  placement: number,
  tickNow: number,
) {
  updateSigmaTarget(p, tickNow - p.lastMatchAt);
  p.history.push(tok);
  if (p.history.length > SEQ_LEN) p.history.shift();
  p.matchesPlayed += 1;
  p.lastMatchAt = tickNow;

  // Fatigue: +0.10 per match, recovers only when idle (handled elsewhere).
  p.fatigue = Math.min(1, p.fatigue + 0.10);

  // Tilt: climbs after 2+ bottom-5 finishes in a row; bleeds off on recovery.
  const wasBottom = placement >= LOBBY_SIZE - 4;
  p.recentBottomFinishes = wasBottom ? p.recentBottomFinishes + 1 : 0;
  if (p.recentBottomFinishes >= 2) {
    p.tilt = Math.min(1, p.tilt + 0.4);
  } else {
    p.tilt = Math.max(0, p.tilt - 0.15);
  }
}

/**
 * Recovery tick: called every sim step for players NOT selected for a lobby.
 * Fatigue bleeds off; tilt decays slowly.
 */
export function idleRecover(p: Player) {
  p.fatigue = Math.max(0, p.fatigue - 0.02);
  p.tilt = Math.max(0, p.tilt - 0.01);
  p.waitTicks += 1;
}

/* ---------------- lobby selection ----------------
   Production-style: anchor = longest-waiting eligible, then gather
   the LOBBY_SIZE-1 players closest in (skill × playstyle) joint space.
   Uses the TRANSFORMER'S learned μ AND playstyle embedding.  As the
   model learns, lobby quality improves.
------------------------------------------------------- */

export function selectLobby(queue: Player[]): Player[] {
  const eligible = queue.filter((p) => p.cooldown === 0);
  if (eligible.length < LOBBY_SIZE) return eligible;

  // anchor = longest-waiting eligible (small random tiebreak)
  let anchor = eligible[0];
  let bestWait = -1;
  for (const p of eligible) {
    const w = p.waitTicks + Math.random() * 0.5;
    if (w > bestWait) { bestWait = w; anchor = p; }
  }

  // Distance in joint (μ, playstyle) space.  Skill is heavily weighted
  // (fair match is first priority); playstyle is a secondary tiebreaker.
  //
  // Playstyle uses COSINE distance (1 − cos θ), bounded in [0, 2].  The
  // earlier L2 version became dominant after ~100 matches once the
  // contrastive head had grown embedding norms to ~10 — the raw L2
  // distance grew with norm while dSkill stayed bounded in [0, 1], so
  // the matchmaker drifted from "same skill" to "same playstyle" and
  // started mixing Gold with Diamond.  Cosine is norm-invariant, so the
  // weighting stays valid as training progresses.
  const ALPHA_SKILL = 1.0;
  const BETA_STYLE = 0.15;
  const cosStyle = (a: Player, b: Player) => {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < PLAYSTYLE_HIDDEN_DIM; i++) {
      dot += a.playstyleEmb[i] * b.playstyleEmb[i];
      na += a.playstyleEmb[i] * a.playstyleEmb[i];
      nb += b.playstyleEmb[i] * b.playstyleEmb[i];
    }
    const denom = Math.sqrt(na * nb);
    if (denom < 1e-6) return 0;
    return 1 - dot / denom;
  };
  const distTo = (q: Player) => {
    const dSkill = Math.abs(q.muHat - anchor.muHat);
    return ALPHA_SKILL * dSkill + BETA_STYLE * cosStyle(q, anchor);
  };
  const rest = eligible.filter((p) => p.id !== anchor.id);
  rest.sort((a, b) => distTo(a) - distTo(b));
  return [anchor, ...rest.slice(0, LOBBY_SIZE - 1)];
}

/**
 * Inject a fresh smurf — brand-new account with hidden-high trueSkill.
 * For the "inject smurf" demo button.
 */
export function makeSmurfInjection(id: number): Player {
  const p = makePlayer(id, /*smurfRate=*/0);
  // force smurf flag + high hidden skill
  p.isSmurf = true;
  p.trueSkill = 0.85 + Math.random() * 0.15; // top tier
  p.matchesPlayed = 0;
  p.history = [];
  p.sigmaTarget = SIGMA_INIT;
  p.tier = tierOf(p.trueSkill);
  p.color = TIER_COLORS[p.tier];
  p.name = `smurf_${p.name}`;
  p.attentionLayers = null;
  p.attentionT = 0;
  return p;
}

/* ---------------- training pair sampler ----------------
   SimCLR-style for playstyle head.  Positive pair = the same player at
   two different times (stats noise is the "augmentation").  Negatives
   = other random players at random times.
---------------------------------------------------------- */

export function sampleContrastivePair(queue: Player[], N = 8): {
  anchorId: number;
  positiveTokens: { tokens: Float32Array; T: number };
  anchorTokens:   { tokens: Float32Array; T: number };
  negativeTokens: { tokens: Float32Array; T: number }[];
  negativeIds: number[];
} | null {
  const candidates = queue.filter((p) => p.history.length >= 4);
  if (candidates.length < N + 1) return null;
  const anchor = candidates[Math.floor(Math.random() * candidates.length)];
  // build a "view" by dropping a random early token — augmentation
  const aTokens = tokensForPlayer(anchor);
  let pTokens = aTokens;
  if (aTokens.T > 4) {
    const dropT = Math.max(4, aTokens.T - 1);
    pTokens = { tokens: aTokens.tokens.subarray(0, dropT * INPUT_DIM), T: dropT };
  }
  const negs: { tokens: Float32Array; T: number }[] = [];
  const negIds: number[] = [];
  const pool = candidates.filter((p) => p.id !== anchor.id);
  for (let i = 0; i < N && pool.length > 0; i++) {
    const idx = Math.floor(Math.random() * pool.length);
    negs.push(tokensForPlayer(pool[idx]));
    negIds.push(pool[idx].id);
    pool.splice(idx, 1);
  }
  return {
    anchorId: anchor.id,
    positiveTokens: pTokens,
    anchorTokens: aTokens,
    negativeTokens: negs,
    negativeIds: negIds,
  };
}

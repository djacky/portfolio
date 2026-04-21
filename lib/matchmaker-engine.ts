/* ------------------------------------------------------------------
   MatchmakerEngine — module-level singleton that owns the transformer,
   the player queue, and the training loop.  Persists across React
   mount/unmount (like the PendulumEngine pattern) so that training
   continues even when the demo tab is briefly hidden.

   The scene and HUD components subscribe to this engine via
   subscribe() + snapshot().  No React state is held inside the engine;
   consumers mutate a version counter to trigger re-renders.
------------------------------------------------------------------ */

import {
  TransformerMatchmaker,
  makeGrads,
  anomalyBceLoss,
  contrastivePlaystyleLoss,
  plackettLuceLoss,
  PLAYSTYLE_DIM,
  ANOMALY_DIM,
  FORM_DIM,
  N_HEADS,
  SKILL_DIM,
} from "./transformer-matchmaker";
import {
  Player,
  LOBBY_SIZE,
  makePlayer,
  plackettLuce,
  buildToken,
  tokensForPlayer,
  postMatchUpdate,
  idleRecover,
  selectLobby,
  sampleContrastivePair,
  makeSmurfInjection,
  DEFAULT_QUEUE_SIZE,
  SIGMA_MIN,
  SIGMA_MAX,
} from "./matchmaker-sim";

// Supervised σ head weight.  Small enough that the shared transformer trunk
// is still dominated by the PL / SimCLR / anomaly objectives (which need to
// learn the representation from scratch), but large enough to keep σ̂
// tracking the Glicko-2 target with ±0.02 error at equilibrium.  Matches the
// ~0.3 recommendation from Balan et al. 2015 "Bayesian Dark Knowledge".
const SIGMA_LOSS_WEIGHT = 0.3;

// Scale-anchor weight on sigmoid(μ) against per-match placement percentile.
// PL loss is shift-invariant on raw θ (Bradley-Terry pathology), so without
// an anchor, mean(μ̂) drifts and sigmoid saturates — seed 0 / seed 2 hit
// μ̂ ≈ 0.9 for every player at match 20 in the headless sweep, and seed 2
// drifted to mean μ̂ = 0.27 by match 150 (vs trueSkill mean 0.5), tanking
// cal1 to 22% despite ρ = 0.96.  λ = 0.2 is strong enough to pin the scale
// but light relative to PL so the ordinal signal still dominates the rank.
const SKILL_ANCHOR_WEIGHT = 0.3;

export type Phase = "idle" | "forming" | "playing" | "finishing";

// Dev-only diagnostics gate.  Next.js / SWC substitutes process.env.NODE_ENV
// with a string literal at build time, so `DEV` folds to a constant and the
// guarded branches are dead-code-eliminated from the production bundle.
const DEV = process.env.NODE_ENV !== "production";

export interface EngineSnapshot {
  version: number;
  players: Player[];
  phase: Phase;
  lobbyIds: Set<number>;
  lastOrdering: Player[] | null;
  matchesPlayed: number;
  // KPIs
  loss: number;
  skillLoss: number;
  anomalyLoss: number;
  contrastLoss: number;
  sigmaLoss: number;
  lobbySpread: number;
  randomBaseline: number;
  smurfPrecision: number;
  smurfRecall: number;
  rankCorrelation: number;
  // event toasts (ephemeral)
  lastEvent: string | null;
  lastEventAt: number;
  // dev-only diagnostics (undefined in production builds)
  calibration1?: number;
  calibration2?: number;
  baselineRho?: number;
  heldOutNll?: number;
  dev?: boolean;
}

type Listener = () => void;

class _MatchmakerEngine {
  model: TransformerMatchmaker;
  players: Player[] = [];
  phase: Phase = "idle";
  lobbyIds = new Set<number>();
  lobby: Player[] = [];
  lastOrdering: Player[] | null = null;
  matchesPlayed = 0;
  tickCount = 0;
  trainingPaused = false;
  // Halved from 0.002 — at 0.002 the multi-task gradients (skill NLL +
  // SimCLR + BCE) regularly push past the clip threshold, which caused
  // ρ to oscillate between 0.5 and 0.93 instead of settling.
  lr = 0.001;
  patchEpoch = 0;

  // Hysteretic LR decay: once the skill ordering has settled (ρ > 0.90),
  // halve the effective LR so a single unlucky batch can't tip the trunk
  // out of the basin.  Release the decay if ρ slips back below 0.85 so
  // the model can re-converge with the full step size.  Window (0.85,
  // 0.90) is intentionally wide to prevent chattering at the threshold.
  private lrDecayActive = false;
  private effectiveLr(): number {
    if (this.lrDecayActive) {
      if (this.rankCorrelation < 0.85) this.lrDecayActive = false;
    } else {
      if (this.rankCorrelation > 0.90) this.lrDecayActive = true;
    }
    return this.lrDecayActive ? this.lr * 0.5 : this.lr;
  }
  // Global L2-norm gradient clip threshold.  Caps the per-step parameter
  // update regardless of which head spiked — kills the occasional
  // pathological batch (fresh smurf with huge σ_target contrast, etc.)
  // that used to yank the shared trunk out of the converged basin.
  private readonly GRAD_CLIP_NORM = 1.0;

  // running KPIs
  lossEma = 0;
  skillLossEma = 0;
  anomalyLossEma = 0;
  contrastLossEma = 0;
  sigmaLossEma = 0;
  lobbySpread = 0;
  randomBaseline = 0;
  smurfTP = 0; smurfFP = 0; smurfFN = 0; smurfTN = 0;
  rankCorrelation = 0;
  // Rolling-average Spearman (last ~30 instantaneous samples).  Without
  // this, a single unlucky training step could drop the displayed ρ from
  // 0.93 to 0.5, then it would rebound — making the HUD read like the
  // model was unstable when the underlying ordering was actually fine.
  private rankHistory: number[] = [];
  private readonly RANK_WINDOW = 30;

  // Dev-only convergence diagnostics.  All stripped from prod by DEV guard.
  calibration1 = 0; // fraction of players with |trueSkill - μ̂| ≤ σ̂ (target 0.68)
  calibration2 = 0; // ≤ 2σ̂ (target 0.95)
  baselineRho = 0;  // Spearman of dumb EMA-winMag heuristic vs truth
  heldOutNll = 0;   // PL loss on each fresh lobby BEFORE any training touches it

  // Kendall et al. 2018 — learnable log-variances (s = log σ²) per task.
  // Each task's gradient is scaled by 0.5·exp(-s), and s itself is
  // optimized by SGD.  Replaces the old hand-tuned 1.0·skill + 0.2·anom
  // + 1.0·contrast mixture, which under-weighted SimCLR in early training
  // and over-weighted it after skill had converged.
  logVarSkill = 0;
  logVarAnom = 0;
  logVarContr = 0;

  // Ring buffer of completed-lobby snapshots — each entry holds the
  // pre-match tokens for all 30 players plus their actual placements.
  // Used by the Plackett–Luce training step as the ordinal label.
  private completedLobbies: Array<{
    tokens: { tokens: Float32Array; T: number }[];
    placements: number[];
    playerIds: number[];
  }> = [];
  // Ring-buffer of completed lobbies that the PL trainer replays from.  At 8
  // the trunk memorizes specific orderings within ~50 training visits (empirically:
  // peaked ρ=0.95 / cal1σ=68% at match 113, then drifted to 0.92 / 49% by 191 —
  // classic overfit signature with train PL flat while held-out PL climbed).
  // 32 quadruples sample diversity; combined with the 1-in-2 train cadence
  // and wd=1e-3, each lobby now sees ~3 training visits before eviction.
  private readonly LOBBY_BUFFER = 32;
  // PL subsample size per training step.  10 gives a tractable O(K²) = 100-op
  // gradient pass per lobby while still providing ~45 pairwise ordinal
  // comparisons — enough to train a 2-layer transformer's skill head.
  private readonly PL_SAMPLE_K = 10;

  lastEvent: string | null = null;
  lastEventAt = 0;

  // rendering subscription
  version = 0;
  private listeners = new Set<Listener>();

  // wall-clock intervals
  private trainTimer: ReturnType<typeof setInterval> | null = null;
  private matchTimer: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this.model = new TransformerMatchmaker();
    this.reseed(DEFAULT_QUEUE_SIZE);
  }

  reseed(queueSize: number) {
    this.players = [];
    for (let i = 0; i < queueSize; i++) this.players.push(makePlayer(i));
    this.tickCount = 0;
    this.matchesPlayed = 0;
    this.phase = "idle";
    this.lobby = [];
    this.lobbyIds.clear();
    this.lastOrdering = null;
    this.lossEma = this.skillLossEma = this.anomalyLossEma = this.contrastLossEma = 0;
    this.sigmaLossEma = 0;
    this.lobbySpread = 0;
    this.randomBaseline = 0;
    this.smurfTP = this.smurfFP = this.smurfFN = this.smurfTN = 0;
    this.rankCorrelation = 0;
    this.calibration1 = this.calibration2 = this.baselineRho = this.heldOutNll = 0;
    this.lastEvent = null;
    this.seedWarmup(8);
    this.bumpVersion();
  }

  // Synthesize N completed lobbies before the live loop starts so the PL
  // trainer has labels from tick 0.  Uses random lobby selection (the
  // untrained model's muHat is effectively random anyway) + the real sim
  // outcome sampler, so player histories and the ring buffer end up in
  // the exact shape they'd reach after N organic matches.
  private seedWarmup(count: number) {
    if (this.players.length < LOBBY_SIZE) return;
    for (let k = 0; k < count; k++) {
      const shuffled = [...this.players];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      const lobby = shuffled.slice(0, LOBBY_SIZE);
      const ordering = plackettLuce(lobby, 6);

      let muSum = 0;
      for (const p of lobby) muSum += p.muHat;
      const muMean = muSum / lobby.length;
      let muVar = 0;
      for (const p of lobby) muVar += (p.muHat - muMean) ** 2;
      const muSpread = Math.sqrt(muVar / lobby.length);

      const preMatchTokens = ordering.map((p) => tokensForPlayer(p));
      const placements = ordering.map((_, i) => i + 1);
      const playerIds = ordering.map((p) => p.id);

      for (let i = 0; i < ordering.length; i++) {
        const p = ordering[i];
        const placement = i + 1;
        const tok = buildToken(p, placement, lobby, this.tickCount, this.patchEpoch, muMean, muSpread);
        postMatchUpdate(p, tok, placement, this.tickCount);
      }
      // Approximate ~12 ticks between real matches (forming+playing+finishing).
      this.tickCount += 12;

      this.completedLobbies.push({ tokens: preMatchTokens, placements, playerIds });
      if (this.completedLobbies.length > this.LOBBY_BUFFER) this.completedLobbies.shift();
      this.matchesPlayed += 1;
    }
  }

  resetModel() {
    this.model.reset();
    this.lossEma = this.skillLossEma = this.anomalyLossEma = this.contrastLossEma = 0;
    this.sigmaLossEma = 0;
    this.bumpVersion();
  }

  // Dev-only speed multiplier (1 / 3 / 5).  Shortens both timer periods by
  // the same factor so training + match-generation scale together — each
  // lobby still sees a proportional number of trainStep visits before
  // eviction, so regularization behaviour is unchanged.
  speedMultiplier: 1 | 3 | 5 = 3;

  start() {
    if (this.trainTimer) return;
    // Re-encode + train: 5 Hz (heavy) at 1×; scales with speedMultiplier.
    this.trainTimer = setInterval(() => this.trainTick(), 200 / this.speedMultiplier);
    // Match lifecycle: drives phase transitions.
    this.matchTimer = setInterval(() => this.matchTick(), 400 / this.speedMultiplier);
  }
  stop() {
    if (this.trainTimer) { clearInterval(this.trainTimer); this.trainTimer = null; }
    if (this.matchTimer) { clearInterval(this.matchTimer); this.matchTimer = null; }
  }
  setLr(lr: number) { this.lr = lr; }
  setPaused(p: boolean) { this.trainingPaused = p; }
  setSpeed(mult: 1 | 3 | 5) {
    if (mult === this.speedMultiplier) return;
    this.speedMultiplier = mult;
    const wasRunning = this.trainTimer !== null;
    if (wasRunning) {
      this.stop();
      this.start();
    }
    this.bumpVersion();
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
  private bumpVersion() {
    this.version += 1;
    for (const fn of this.listeners) fn();
  }

  snapshot(): EngineSnapshot {
    const s: EngineSnapshot = {
      version: this.version,
      players: this.players,
      phase: this.phase,
      lobbyIds: this.lobbyIds,
      lastOrdering: this.lastOrdering,
      matchesPlayed: this.matchesPlayed,
      loss: this.lossEma,
      skillLoss: this.skillLossEma,
      anomalyLoss: this.anomalyLossEma,
      contrastLoss: this.contrastLossEma,
      sigmaLoss: this.sigmaLossEma,
      lobbySpread: this.lobbySpread,
      randomBaseline: this.randomBaseline,
      smurfPrecision: this.smurfTP + this.smurfFP > 0
        ? this.smurfTP / (this.smurfTP + this.smurfFP) : 0,
      smurfRecall: this.smurfTP + this.smurfFN > 0
        ? this.smurfTP / (this.smurfTP + this.smurfFN) : 0,
      rankCorrelation: this.rankCorrelation,
      lastEvent: this.lastEvent,
      lastEventAt: this.lastEventAt,
    };
    if (DEV) {
      s.dev = true;
      s.calibration1 = this.calibration1;
      s.calibration2 = this.calibration2;
      s.baselineRho = this.baselineRho;
      s.heldOutNll = this.heldOutNll;
    }
    return s;
  }

  injectSmurf() {
    const id = this.players.length;
    const smurf = makeSmurfInjection(id);
    this.players.push(smurf);
    this.emitEvent(`smurf injected: ${smurf.name} (hidden skill ${smurf.trueSkill.toFixed(2)})`);
    this.bumpVersion();
  }

  emitEvent(msg: string) {
    this.lastEvent = msg;
    this.lastEventAt = Date.now();
  }

  /* ---------------- training tick ----------------
     Every 200ms: re-encode a random subset of players (so activations
     stay fresh for the scene), plus run one gradient step if we have
     completed match data to learn from.
  ---------------------------------------------------- */
  // Tick counter used to gate trainStep() to every-other call.  Encode still
  // runs every tick (scene/HUD need fresh activations), but the gradient update
  // skips every 2nd tick — halves replay pressure on the lobby ring without
  // slowing the visible μ̂ updates.
  private trainTickCounter = 0;

  private trainTick() {
    // Budget: keep each tick under ~80ms so RAF stays at 60fps.  Encode a
    // modest batch per tick; over several ticks the whole queue gets refreshed.
    // Priority = current lobby (so scene/HUD show their μ̂ up to date).
    const priority = this.players.filter((p) => this.lobbyIds.has(p.id));
    const pool = this.players.filter((p) => !this.lobbyIds.has(p.id));
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const batch = [...priority, ...pool.slice(0, Math.max(4, 12 - priority.length))];
    this.encodeAll(batch);

    this.trainTickCounter++;
    if (!this.trainingPaused && (this.trainTickCounter & 1) === 0) {
      this.trainStep();
    }
    this.updateRankCorrelation();
    this.bumpVersion();
  }

  private encodeAll(batch: Player[]) {
    for (const p of batch) {
      if (p.history.length < 1) continue;
      const { tokens, T } = tokensForPlayer(p);
      const fc = this.model.forward(tokens, T);
      p.muHat = clamp01(sigmoid(fc.skill[0]));
      // skill[1] is log σ̂ under the amortized-Bayesian supervision scheme.
      // The supervised NLL target is log σ_target (Glicko-2 teacher), so σ̂
      // = exp(skill[1]).  Clamp to [SIGMA_MIN, SIGMA_MAX] — protects the
      // sparklines and lobby UI from a transient init where the head's
      // log-σ output blows up before the first training step touches it.
      p.sigmaHat = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, Math.exp(fc.skill[1])));
      for (let i = 0; i < PLAYSTYLE_DIM; i++) p.playstyleEmb[i] = fc.playstyle[i];
      p.anomalyP = sigmoid(fc.anomaly[0]);
      p.formMod = Math.tanh(fc.form[0]) * 0.3;
      // EMA-shadow forward, same batch — cost ~2× but only runs on the
      // batched subset (~12 players/tick), not all 160.  Used for ρ.
      const fcEma = this.model.forwardEma(tokens, T);
      p.muHatEma = clamp01(sigmoid(fcEma.skill[0]));
      // Stash last-query attention per layer so the HUD can read it without
      // re-running a forward pass.  Shape per layer: [heads × T].
      const perLayer: Float32Array[] = [];
      for (const bc of fc.blocks) {
        const w = new Float32Array(N_HEADS * T);
        for (let h = 0; h < N_HEADS; h++) {
          for (let s = 0; s < T; s++) {
            w[h * T + s] = bc.attn[(h * T + (T - 1)) * T + s];
          }
        }
        perLayer.push(w);
      }
      p.attentionLayers = perLayer;
      p.attentionT = T;
      p.attentionHeads = N_HEADS;
      // update smurf confusion matrix (labels visible from sim ground truth)
      const detected = p.anomalyP > 0.5;
      if (p.isSmurf && p.matchesPlayed < 12) {
        if (detected) this.smurfTP++; else this.smurfFN++;
      } else {
        if (detected) this.smurfFP++; else this.smurfTN++;
      }
    }
    // decay confusion matrix — keeps metric responsive
    const decay = 0.98;
    this.smurfTP *= decay; this.smurfFP *= decay;
    this.smurfFN *= decay; this.smurfTN *= decay;
  }

  /**
   * One training step.  Plackett–Luce loss on a subsampled completed
   * lobby drives the skill head (ordinal objective — matches the
   * generative model).  BCE on the anomaly head and SimCLR on the
   * playstyle head ride along on the same forward passes.
   */
  private trainStep() {
    if (this.players.filter((p) => p.history.length >= 3).length < 16) return;

    const grads = makeGrads();
    let batchLoss = 0, skillL = 0, anomL = 0, contrL = 0, sigmaL = 0, nb = 0;
    // Kendall 2018: precompute task-specific gradient weights w_i = 0.5·exp(-s_i).
    // Clamp the inverse precision so the early-training huge-loss regime
    // can't push s to −∞ (which would blow up w) or divergent s to +∞
    // (which would zero out gradients permanently).
    const sSkill = Math.max(-2, Math.min(3, this.logVarSkill));
    const sAnom  = Math.max(-2, Math.min(3, this.logVarAnom));
    const sContr = Math.max(-2, Math.min(3, this.logVarContr));
    const wSkill = 0.5 * Math.exp(-sSkill);
    const wAnom  = 0.5 * Math.exp(-sAnom);
    const wContr = 0.5 * Math.exp(-sContr);

    // PL step — only if we have a completed lobby in the buffer.
    if (this.completedLobbies.length > 0) {
      const lobby = this.completedLobbies[
        Math.floor(Math.random() * this.completedLobbies.length)
      ];
      // Subsample K players from the lobby without replacement, keeping
      // their relative finishing order (PL is invariant to subset because
      // ranks of subset elements induce a valid PL sample on the subset).
      // Only sample slots with actual history — brand-new players playing
      // their first match have T=0, which would make the forward pass read
      // past the empty array and poison the skill head with NaN.
      const usable = [...lobby.tokens.keys()].filter((i) => lobby.tokens[i].T >= 1);
      if (usable.length < 2) return;
      const K = Math.min(this.PL_SAMPLE_K, usable.length);
      const idxs = usable.sort(() => Math.random() - 0.5).slice(0, K);
      idxs.sort((a, b) => lobby.placements[a] - lobby.placements[b]);
      const thetas = new Float32Array(K);
      const caches: ReturnType<TransformerMatchmaker["forward"]>[] = [];
      for (let k = 0; k < K; k++) {
        const idx = idxs[k];
        const { tokens, T } = lobby.tokens[idx];
        const fc = this.model.forward(tokens, T);
        caches.push(fc);
        thetas[k] = fc.skill[0]; // μ — raw, pre-sigmoid
      }
      const { loss: lPL, grad: gT } = plackettLuceLoss(thetas);
      skillL = lPL / K;
      // Backprop: PL gradient flows to skill[0] (μ); supervised MSE on
      // log σ_target drives skill[1] (log σ̂) independently.  Anomaly BCE
      // rides along on the same forward.  Learnable PL temperatures were
      // the prior attempt — Bruch et al. 2020 show they collapse under
      // SGD without a prior, which is what we observed empirically.
      for (let k = 0; k < K; k++) {
        const idx = idxs[k];
        const p = this.players.find((q) => q.id === lobby.playerIds[idx]);
        const gSkill = new Float32Array(SKILL_DIM);
        // Scale anchor: soft-supervise sigmoid(θ) against placement percentile.
        //   target = 1 − (placement − 0.5) / LOBBY_SIZE  ∈ (0.017, 0.983)
        //   L = (σ(θ) − target)²,  dL/dθ = 2·(σ(θ) − target)·σ(θ)·(1−σ(θ))
        // Weak per-match signal (placement is noisy) but over many matches
        // anchors mean(μ̂) ≈ 0.5 and std(μ̂) to match trueSkill's [0,1] scale.
        // Additive to the PL gradient — ordinal signal still dominates ranks.
        const placement = lobby.placements[idx];
        const target = 1 - (placement - 0.5) / LOBBY_SIZE;
        const mu = 1 / (1 + Math.exp(-thetas[k]));
        const gAnchor = 2 * (mu - target) * mu * (1 - mu);
        gSkill[0] = gT[k] * wSkill + SKILL_ANCHOR_WEIGHT * gAnchor;
        // Amortized Bayesian σ: supervise log σ̂ against log σ_target (Glicko-2).
        //   L_σ = (skill[1] − log σ_target)²   →   dL/dskill[1] = 2·(skill[1] − log σ_target)
        // Uses log space so σ̂ is guaranteed positive and errors scale
        // multiplicatively (relative, not absolute).  Gaussian NLL form per
        // Kendall/Gal 2017 and BayesFlow (Radev 2020).
        if (p) {
          const logTarget = Math.log(p.sigmaTarget);
          const dlog = caches[k].skill[1] - logTarget;
          gSkill[1] = SIGMA_LOSS_WEIGHT * 2 * dlog;
          sigmaL += dlog * dlog;
        }
        const gPlay = new Float32Array(PLAYSTYLE_DIM);
        const gAnom = new Float32Array(ANOMALY_DIM);
        const gForm = new Float32Array(FORM_DIM);
        if (p) {
          const smurfLabel = p.isSmurf && p.matchesPlayed < 12 ? 1 : 0;
          const { loss: lAnom, grad: gA } = anomalyBceLoss(caches[k].anomaly, smurfLabel);
          for (let j = 0; j < gA.length; j++) gAnom[j] = gA[j] * wAnom;
          anomL += lAnom;
        }
        this.model.backward(caches[k], gSkill, gPlay, gAnom, gForm, grads);
        nb++;
      }
      anomL /= Math.max(1, nb);
      sigmaL /= Math.max(1, nb);
      batchLoss = wSkill * skillL + wAnom * anomL + SIGMA_LOSS_WEIGHT * sigmaL;
    }

    // Contrastive playstyle step — separate forward/backward on anchor only
    const contra = sampleContrastivePair(this.players, 3);
    if (contra) {
      const fcA = this.model.forward(contra.anchorTokens.tokens, contra.anchorTokens.T);
      const fcP = this.model.forward(contra.positiveTokens.tokens, contra.positiveTokens.T);
      const candEmbs: Float32Array[] = [fcP.playstyle];
      for (const neg of contra.negativeTokens) {
        const fcN = this.model.forward(neg.tokens, neg.T);
        candEmbs.push(fcN.playstyle);
      }
      const { loss: lC, grad: gC } = contrastivePlaystyleLoss(fcA.playstyle, candEmbs, 0.2);
      for (let i = 0; i < gC.length; i++) gC[i] *= wContr;
      const zGrad = new Float32Array(2);
      const zAnom = new Float32Array(1);
      const zForm = new Float32Array(1);
      this.model.backward(fcA, zGrad, gC, zAnom, zForm, grads);
      contrL = lC;
    }

    // Kendall: update s_i via plain SGD.  dL/ds_i = -0.5·exp(-s_i)·L_i + 0.5.
    // Use a smaller step for the log-variances than for the model weights
    // (Liebel & Körner 2018 observation — otherwise s can race away from
    // the optimum before the backbone has stabilized).  skillL / anomL are
    // already per-player-averaged above.
    const sLr = this.effectiveLr() * 0.1;
    if (nb > 0) {
      this.logVarSkill -= sLr * (-wSkill * skillL + 0.5);
      this.logVarAnom  -= sLr * (-wAnom  * anomL  + 0.5);
    }
    this.logVarContr -= sLr * (-wContr * contrL + 0.5);

    // Apply Adam step — scale grads by batch size first so the Adam step
    // is on a per-player-averaged gradient magnitude regardless of K.
    if (nb > 0) {
      scaleGrads(grads, 1 / nb);
    }
    // Global L2-norm gradient clipping (Pascanu 2013).  Caps total update
    // magnitude so one outlier batch can't knock the trunk out of its
    // converged basin — empirically the source of the "diagonal →
    // scramble → diagonal" oscillation cycle we were seeing at ρ≈0.97.
    const gNorm = gradNorm(grads);
    if (gNorm > this.GRAD_CLIP_NORM) {
      scaleGrads(grads, this.GRAD_CLIP_NORM / gNorm);
    }
    // wd bumped 1e-4 → 1e-3: extra regularization to counter ring-buffer replay
    // overfitting.  The overfit sparkline gap (held-out PL − train PL) grew from
    // ~0 to ~+0.45 between match 113 and match 191; heavier weight decay pulls
    // the trunk back toward simpler solutions that generalize across lobbies.
    this.model.adamStep(grads, this.effectiveLr(), 0.9, 0.999, 1e-3);
    // Slowly bleed online weights into the EMA shadow.  Decay 0.995 →
    // effective averaging window ~200 training steps, matching roughly
    // ~40 matches of queue activity — long enough to dampen a single
    // noisy step but short enough to still track real convergence.
    this.model.stepEma();

    // Display EMAs — batchLoss / skillL / anomL are already per-player means.
    // Guard each blend: if a past step poisoned the EMA with NaN, reset to
    // the fresh value instead of carrying the NaN forward forever.
    const alpha = 0.05;
    const blendEma = (prev: number, next: number) => {
      if (!Number.isFinite(next)) return prev; // skip bad sample
      if (!Number.isFinite(prev)) return next; // reset from NaN
      return (1 - alpha) * prev + alpha * next;
    };
    if (nb > 0) {
      this.lossEma = blendEma(this.lossEma, batchLoss);
      this.skillLossEma = blendEma(this.skillLossEma, skillL);
      this.anomalyLossEma = blendEma(this.anomalyLossEma, anomL);
      this.sigmaLossEma = blendEma(this.sigmaLossEma, sigmaL);
    }
    this.contrastLossEma = blendEma(this.contrastLossEma, contrL);
  }

  /* ---------------- match lifecycle tick ----------------
     This drives the phase state machine at 400ms cadence:
       idle     → selectLobby → forming
       forming  → (2 ticks) → playing
       playing  → (10 ticks) → finishing (sample ordering, emit tokens)
       finishing→ (1 tick) → idle
  ---------------------------------------------------- */
  private phaseTicks = 0;
  private matchTick() {
    this.tickCount += 1;

    // Recover idle players
    for (const p of this.players) {
      if (!this.lobbyIds.has(p.id) && p.cooldown === 0) idleRecover(p);
      if (p.cooldown > 0) p.cooldown -= 1;
    }

    if (this.phase === "idle") {
      if (this.phaseTicks-- > 0) return;
      const lobby = selectLobby(this.players);
      if (lobby.length < LOBBY_SIZE) {
        // not enough eligible — clear cooldowns
        for (const p of this.players) p.cooldown = 0;
        return;
      }
      this.lobby = lobby;
      this.lobbyIds = new Set(lobby.map((p) => p.id));
      // Compute lobby KPIs immediately
      this.lobbySpread = stdSkill(lobby);
      const randSample = [...this.players].sort(() => Math.random() - 0.5).slice(0, LOBBY_SIZE);
      this.randomBaseline = stdSkill(randSample);
      this.phase = "forming";
      this.phaseTicks = 3;
      return;
    }

    if (this.phase === "forming") {
      if (this.phaseTicks-- > 0) return;
      this.phase = "playing";
      this.phaseTicks = 8;
      return;
    }

    if (this.phase === "playing") {
      if (this.phaseTicks-- > 0) return;
      this.phase = "finishing";
      this.phaseTicks = 1;
      // Sample ordering
      const ordering = plackettLuce(this.lobby, 6);
      this.lastOrdering = ordering;
      // Compute lobby μ stats (for the tokens)
      let muSum = 0;
      for (const p of this.lobby) muSum += p.muHat;
      const muMean = muSum / this.lobby.length;
      let muVar = 0;
      for (const p of this.lobby) muVar += (p.muHat - muMean) ** 2;
      const muSpread = Math.sqrt(muVar / this.lobby.length);
      // Snapshot pre-match token sequences for every player in ordering —
      // this is what the PL training step will forward through.  Grabbing
      // it HERE (before buildToken + history.push) is critical: once the
      // match-outcome token is appended, forward() would read it and the
      // PL objective would become trivially memorizable.
      const preMatchTokens = ordering.map((p) => tokensForPlayer(p));
      const placements = ordering.map((_, i) => i + 1);
      const playerIds = ordering.map((p) => p.id);
      // Build + append tokens
      for (let i = 0; i < ordering.length; i++) {
        const p = ordering[i];
        const placement = i + 1;
        const tok = buildToken(p, placement, this.lobby, this.tickCount, this.patchEpoch, muMean, muSpread);
        postMatchUpdate(p, tok, placement, this.tickCount);
        p.cooldown = 1;
        p.waitTicks = 0;
        // Detect tilt events for toasts
        if (p.tilt > 0.5 && Math.random() < 0.1) {
          this.emitEvent(`tilt onset: ${p.name}`);
        }
      }
      // Push snapshot into ring buffer; cap at LOBBY_BUFFER.
      this.completedLobbies.push({ tokens: preMatchTokens, placements, playerIds });
      if (this.completedLobbies.length > this.LOBBY_BUFFER) this.completedLobbies.shift();
      this.matchesPlayed += 1;
      // Evaluate held-out PL NLL on this fresh lobby before any trainStep
      // has touched it.  Dev-only — stripped from prod by DEV guard.
      if (DEV) this.evalHeldOutNll();
      return;
    }

    if (this.phase === "finishing") {
      if (this.phaseTicks-- > 0) return;
      this.lobbyIds.clear();
      this.lobby = [];
      this.phase = "idle";
      this.phaseTicks = 1;
    }
  }

  private updateRankCorrelation() {
    // Spearman on (EMA-forward μ, trueSkill) — visible-queue-wide.  Using
    // the EMA shadow weights here (not the live online params that
    // encodeAll writes into player.muHat) gives a ~3× more stable ρ;
    // encodeAll is computed against online weights for lobby matching
    // purposes, where freshness matters.
    const pool = this.players.filter((p) => p.history.length >= 2);
    if (pool.length < 10) return;

    // Read cached muHatEma populated by encodeAll — zero extra forward cost.
    const byMu = [...pool].sort((a, b) => a.muHatEma - b.muHatEma);
    const byTrue = [...pool].sort((a, b) => a.trueSkill - b.trueSkill);
    const rMu = new Map<number, number>();
    const rTrue = new Map<number, number>();
    byMu.forEach((p, i) => rMu.set(p.id, i));
    byTrue.forEach((p, i) => rTrue.set(p.id, i));
    let sumD2 = 0;
    for (const p of pool) {
      const d = (rMu.get(p.id)! - rTrue.get(p.id)!);
      sumD2 += d * d;
    }
    const n = pool.length;
    const inst = 1 - (6 * sumD2) / (n * (n * n - 1));

    this.rankHistory.push(inst);
    if (this.rankHistory.length > this.RANK_WINDOW) this.rankHistory.shift();
    let s = 0;
    for (const r of this.rankHistory) s += r;
    this.rankCorrelation = s / this.rankHistory.length;

    if (DEV) this.updateDevDiagnostics(pool, rTrue);
  }

  /**
   * Calibration + baseline-beat diagnostics.  Dev-only — the entire method
   * is dead-code-eliminated in production via the DEV guard at callsite.
   *
   *  calibration: σ̂ is trained as an amortized predictor of the Glicko-2
   *  teacher (which already lives in [0,1]-skill-space units), so σ̂ is
   *  directly comparable to |trueSkill − μ̂|.  A well-calibrated Gaussian
   *  target has 68% within ±1σ, 95% within ±2σ.
   *
   *  baseline: rank players by a trailing-exponential mean of their past
   *  winMag (a naive "averaged placement" rating).  Spearman vs trueSkill
   *  tells you what a dumb non-transformer heuristic achieves — if the
   *  transformer's ρ barely beats it, the attention / playstyle / form
   *  machinery isn't earning its keep.
   */
  private updateDevDiagnostics(pool: Player[], rTrue: Map<number, number>) {
    // ---- 1σ / 2σ coverage ----
    let in1 = 0, in2 = 0, denom = 0;
    for (const p of pool) {
      if (!Number.isFinite(p.sigmaHat) || !Number.isFinite(p.muHatEma)) continue;
      // σ̂ is now amortized-Bayesian, already in skill-space units.
      const sigMu = p.sigmaHat;
      if (sigMu <= 1e-6) continue;
      const err = Math.abs(p.trueSkill - p.muHatEma);
      if (err <= sigMu) in1++;
      if (err <= 2 * sigMu) in2++;
      denom++;
    }
    if (denom > 0) {
      this.calibration1 = in1 / denom;
      this.calibration2 = in2 / denom;
    }

    // ---- baseline ρ: exponentially-weighted mean of past winMag ----
    // α = 0.3 puts ~70% of weight on the most recent 4 matches — enough
    // history to average out lobby variance but short enough to still
    // respond to improving players.
    const alpha = 0.3;
    const baseline = new Map<number, number>();
    for (const p of pool) {
      let num = 0, wsum = 0, w = 1;
      for (let i = p.history.length - 1; i >= 0; i--) {
        const winMag = 1 - (p.history[i].placement - 1) / 29;
        num += w * winMag;
        wsum += w;
        w *= (1 - alpha);
      }
      baseline.set(p.id, wsum > 0 ? num / wsum : 0);
    }
    const byBase = [...pool].sort((a, b) => baseline.get(a.id)! - baseline.get(b.id)!);
    const rBase = new Map<number, number>();
    byBase.forEach((p, i) => rBase.set(p.id, i));
    let sumD2 = 0;
    for (const p of pool) {
      const d = rBase.get(p.id)! - rTrue.get(p.id)!;
      sumD2 += d * d;
    }
    const n = pool.length;
    if (n >= 2) {
      this.baselineRho = 1 - (6 * sumD2) / (n * (n * n - 1));
    }
  }

  /**
   * Held-out PL NLL — pure forward pass on a freshly-completed lobby, BEFORE
   * any trainStep has seen it.  Same K=10 subsample as training so losses
   * are directly comparable.  Dev-only; called from matchTick.
   */
  private evalHeldOutNll() {
    const lobby = this.completedLobbies[this.completedLobbies.length - 1];
    if (!lobby) return;
    const usable = [...lobby.tokens.keys()].filter((i) => lobby.tokens[i].T >= 1);
    if (usable.length < 2) return;
    const K = Math.min(this.PL_SAMPLE_K, usable.length);
    const idxs = [...usable].sort(() => Math.random() - 0.5).slice(0, K);
    idxs.sort((a, b) => lobby.placements[a] - lobby.placements[b]);
    const thetas = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      const { tokens, T } = lobby.tokens[idxs[k]];
      const fc = this.model.forward(tokens, T);
      thetas[k] = fc.skill[0];
    }
    const { loss } = plackettLuceLoss(thetas);
    const perPlayerNll = loss / K;
    if (!Number.isFinite(perPlayerNll)) return;
    const a = 0.1;
    this.heldOutNll = this.heldOutNll === 0
      ? perPlayerNll
      : (1 - a) * this.heldOutNll + a * perPlayerNll;
  }

  /**
   * Pull one player's attention for the HUD inspector.  Returns the
   * last-token's attention weights per layer, per head.
   */
  /**
   * Return the cached attention for a player.  encodeAll() populates the
   * cache every tick that includes this player; reading here is zero-cost.
   * If the cache is cold (fresh player not yet encoded), run one forward
   * pass so the HUD has something to show.
   */
  attentionFor(playerId: number): {
    tokens: number; layers: number; heads: number;
    weights: Float32Array[];
    historyTokens: Player["history"];
  } | null {
    const p = this.players.find((x) => x.id === playerId);
    if (!p || p.history.length < 1) return null;
    if (!p.attentionLayers) {
      this.encodeAll([p]);
    }
    if (!p.attentionLayers) return null;
    return {
      tokens: p.attentionT,
      layers: p.attentionLayers.length,
      heads: p.attentionHeads,
      weights: p.attentionLayers,
      historyTokens: p.history.slice(-p.attentionT),
    };
  }
}

function gradArrays(g: ReturnType<typeof makeGrads>): Float32Array[] {
  const arrs: Float32Array[] = [
    g.Wemb, g.bemb, g.lnFg, g.Wsk, g.bsk, g.Wps, g.bps, g.Wan, g.ban, g.Wfm, g.bfm,
  ];
  for (const bl of g.blocks) {
    arrs.push(bl.ln1g, bl.Wqkv, bl.bqkv, bl.Wo, bl.bo, bl.ln2g, bl.Wff1, bl.bff1, bl.Wff2, bl.bff2);
  }
  return arrs;
}

function scaleGrads(g: ReturnType<typeof makeGrads>, s: number) {
  const arrs = gradArrays(g);
  for (const a of arrs) for (let i = 0; i < a.length; i++) a[i] *= s;
}

function gradNorm(g: ReturnType<typeof makeGrads>): number {
  const arrs = gradArrays(g);
  let sum = 0;
  for (const a of arrs) for (let i = 0; i < a.length; i++) sum += a[i] * a[i];
  return Math.sqrt(sum);
}

function sigmoid(x: number): number {
  if (x > 20) return 1;
  if (x < -20) return 0;
  return 1 / (1 + Math.exp(-x));
}
function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}
function stdSkill(lobby: Player[]): number {
  if (lobby.length === 0) return 0;
  const mean = lobby.reduce((a, p) => a + p.trueSkill, 0) / lobby.length;
  const v = lobby.reduce((a, p) => a + (p.trueSkill - mean) ** 2, 0) / lobby.length;
  return Math.sqrt(v);
}

/* ---------------- module-level singleton ---------------- */

let _instance: _MatchmakerEngine | null = null;
export function getEngine(): _MatchmakerEngine {
  if (typeof window === "undefined") {
    // SSR guard — return a stub that does nothing.  Scene is client-only anyway.
    if (!_instance) _instance = new _MatchmakerEngine();
  }
  if (!_instance) _instance = new _MatchmakerEngine();
  return _instance;
}

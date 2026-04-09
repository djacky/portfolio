"use client";
import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Users, Pause, Play, RefreshCw, Swords, Brain } from "lucide-react";

/* ------------------------------------------------------------------
   Disruptive Labs — Siamese matchmaker, 30-player battle-royale edition.

   Architecture (unchanged from the pairwise version):
     Encoder:    8 → 24 → 16 → 2   shared ReLU MLP (Siamese legs)
     Score head: score(p) = Ws · encoder(x_p) + bs
     Loss:       L = max(0, margin − (score_win − score_lose))

   What changed for the BR setting:
     - The matchmaker assembles a LOBBY of 30 players by sliding a
       window of length 30 along the queue sorted by learned score,
       and keeping the window with the smallest score spread.
     - Each match is simulated with a Plackett–Luce model on the
       hidden true skills: you get a full finishing order 1..30,
       with upsets baked in via the softmax temperature.
     - Training signal: every higher-finisher beat every lower-finisher,
       so one 30-player match yields C(30,2) = 435 ordered pairs, which
       we stream into the margin ranking loss across the match's
       "playing" phase so the scatter moves smoothly.
     - Headline metric is LOBBY SKILL SPREAD — std-dev of TRUE skills
       inside the formed lobby. A learned matchmaker should beat the
       "random 30 from the queue" baseline by a wide margin.

   Everything (encoder, forward, backward, SGD, simulation) is
   hand-written in pure TypeScript. No fake weights.
------------------------------------------------------------------ */

const INPUT_DIM = 8;
const H1 = 24;
const H2 = 16;
const EMB_DIM = 2;
const MARGIN = 0.5;

const LOBBY_SIZE = 30;
const DEFAULT_QUEUE_SIZE = 160;

type Tier = "Bronze" | "Silver" | "Gold" | "Diamond";
type Phase = "idle" | "forming" | "playing";

interface Player {
  id: number;
  name: string;
  trueSkill: number;        // hidden ground truth ∈ [0, 1]
  stats: Float32Array;      // input features (noisy reflection of trueSkill)
  emb: [number, number];    // current 2-D embedding
  score: number;            // learned scalar score = Ws·emb + bs
  wins: number;
  losses: number;
  tier: Tier;
  color: string;
  cooldown: number;         // lobby formations to skip (rotation)
  waitTicks: number;        // ticks spent eligible and not selected
  lastPlacement: number | null;
}

const TIER_COLORS: Record<Tier, string> = {
  Bronze:  "#c2843f",
  Silver:  "#cbd5e1",
  Gold:    "#fbbf24",
  Diamond: "#22d3ee",
};

function tierOf(skill: number): Tier {
  if (skill < 0.25) return "Bronze";
  if (skill < 0.5)  return "Silver";
  if (skill < 0.8)  return "Gold";
  return "Diamond";
}

const NAMES = [
  "aurora", "neo", "kairo", "vex", "luma", "rift", "hex", "onyx",
  "nova", "echo", "zephyr", "sable", "pyra", "drift", "glitch", "flux",
  "ember", "cipher", "mira", "halo", "quasar", "orbit", "vector", "lumen",
  "axon", "blitz", "cinder", "dusk", "frost", "ghost", "helix", "iris",
  "jade", "karma", "lyra", "myst", "nyx", "omen", "prism", "quartz",
];

/* ---------------- tiny linear algebra ---------------- */

function randn(std = 0.1) {
  const u = 1 - Math.random();
  const v = Math.random();
  return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function matInit(rows: number, cols: number) {
  const m = new Float32Array(rows * cols);
  const s = Math.sqrt(2 / cols); // He init
  for (let i = 0; i < m.length; i++) m[i] = randn(s);
  return m;
}
function vec(n: number) {
  return new Float32Array(n);
}

/* ---------------- siamese encoder + score head ---------------- */

class SiameseEncoder {
  W1: Float32Array; b1: Float32Array;
  W2: Float32Array; b2: Float32Array;
  W3: Float32Array; b3: Float32Array;
  // Learned skill-direction score head.  score(p) = Ws · emb(p) + bs
  Ws: Float32Array; bs: Float32Array;

  constructor() {
    this.W1 = matInit(H1, INPUT_DIM); this.b1 = vec(H1);
    this.W2 = matInit(H2, H1);        this.b2 = vec(H2);
    this.W3 = matInit(EMB_DIM, H2);   this.b3 = vec(EMB_DIM);
    this.Ws = matInit(1, EMB_DIM);    this.bs = vec(1);
  }

  reset() {
    const p = new SiameseEncoder();
    this.W1 = p.W1; this.b1 = p.b1;
    this.W2 = p.W2; this.b2 = p.b2;
    this.W3 = p.W3; this.b3 = p.b3;
    this.Ws = p.Ws; this.bs = p.bs;
  }

  forward(x: Float32Array) {
    const z1 = vec(H1);
    for (let k = 0; k < H1; k++) {
      let s = this.b1[k];
      const row = k * INPUT_DIM;
      for (let j = 0; j < INPUT_DIM; j++) s += this.W1[row + j] * x[j];
      z1[k] = s;
    }
    const a1 = vec(H1);
    for (let k = 0; k < H1; k++) a1[k] = z1[k] > 0 ? z1[k] : 0;

    const z2 = vec(H2);
    for (let k = 0; k < H2; k++) {
      let s = this.b2[k];
      const row = k * H1;
      for (let j = 0; j < H1; j++) s += this.W2[row + j] * a1[j];
      z2[k] = s;
    }
    const a2 = vec(H2);
    for (let k = 0; k < H2; k++) a2[k] = z2[k] > 0 ? z2[k] : 0;

    const emb = vec(EMB_DIM);
    for (let k = 0; k < EMB_DIM; k++) {
      let s = this.b3[k];
      const row = k * H2;
      for (let j = 0; j < H2; j++) s += this.W3[row + j] * a2[j];
      emb[k] = s;
    }

    let score = this.bs[0];
    for (let k = 0; k < EMB_DIM; k++) score += this.Ws[k] * emb[k];

    return { x, a1, a2, emb, score };
  }

  /** Chain from ∂L/∂score through score head → emb → encoder MLP. */
  backwardFromScore(
    fwd: { x: Float32Array; a1: Float32Array; a2: Float32Array; emb: Float32Array; score: number },
    dScore: number,
    g: Grads
  ) {
    g.bs[0] += dScore;
    const dEmb = vec(EMB_DIM);
    for (let k = 0; k < EMB_DIM; k++) {
      g.Ws[k] += dScore * fwd.emb[k];
      dEmb[k]  = dScore * this.Ws[k];
    }

    // layer 3 (linear to emb)
    const dA2 = vec(H2);
    for (let k = 0; k < EMB_DIM; k++) {
      g.b3[k] += dEmb[k];
      const row = k * H2;
      for (let j = 0; j < H2; j++) {
        g.W3[row + j] += dEmb[k] * fwd.a2[j];
        dA2[j]        += dEmb[k] * this.W3[row + j];
      }
    }
    for (let j = 0; j < H2; j++) if (fwd.a2[j] <= 0) dA2[j] = 0;

    // layer 2
    const dA1 = vec(H1);
    for (let k = 0; k < H2; k++) {
      g.b2[k] += dA2[k];
      const row = k * H1;
      for (let j = 0; j < H1; j++) {
        g.W2[row + j] += dA2[k] * fwd.a1[j];
        dA1[j]        += dA2[k] * this.W2[row + j];
      }
    }
    for (let j = 0; j < H1; j++) if (fwd.a1[j] <= 0) dA1[j] = 0;

    // layer 1
    for (let k = 0; k < H1; k++) {
      g.b1[k] += dA1[k];
      const row = k * INPUT_DIM;
      for (let j = 0; j < INPUT_DIM; j++) {
        g.W1[row + j] += dA1[k] * fwd.x[j];
      }
    }
  }

  applyGrads(g: Grads, lr: number) {
    for (let i = 0; i < this.W1.length; i++) this.W1[i] -= lr * g.W1[i];
    for (let i = 0; i < this.b1.length; i++) this.b1[i] -= lr * g.b1[i];
    for (let i = 0; i < this.W2.length; i++) this.W2[i] -= lr * g.W2[i];
    for (let i = 0; i < this.b2.length; i++) this.b2[i] -= lr * g.b2[i];
    for (let i = 0; i < this.W3.length; i++) this.W3[i] -= lr * g.W3[i];
    for (let i = 0; i < this.b3.length; i++) this.b3[i] -= lr * g.b3[i];
    for (let i = 0; i < this.Ws.length; i++) this.Ws[i] -= lr * g.Ws[i];
    this.bs[0] -= lr * g.bs[0];
  }
}

type Grads = {
  W1: Float32Array; b1: Float32Array;
  W2: Float32Array; b2: Float32Array;
  W3: Float32Array; b3: Float32Array;
  Ws: Float32Array; bs: Float32Array;
};
function zeroGrads(): Grads {
  return {
    W1: vec(H1 * INPUT_DIM), b1: vec(H1),
    W2: vec(H2 * H1),        b2: vec(H2),
    W3: vec(EMB_DIM * H2),   b3: vec(EMB_DIM),
    Ws: vec(EMB_DIM),        bs: vec(1),
  };
}

/* ---------------- players & simulation ---------------- */

/** Observed 8-D stats from hidden true skill.
 *  Noise amplitude shrinks with matches played: a brand-new account has very
 *  uncertain stats, and they converge toward the true skill after ~100 BRs.
 *  std_mult(N) = 1 / sqrt(1 + N/10)  →  N=0: 1.00,  N=10: 0.71,  N=100: 0.30 */
function statsFromSkill(skill: number, matchesPlayed = 0): Float32Array {
  const stdMult = 1 / Math.sqrt(1 + matchesPlayed / 10);
  const noise = () => (Math.random() - 0.5) * 0.5 * stdMult;
  const s = vec(INPUT_DIM);
  s[0] = skill + noise();                // win rate
  s[1] = 0.3 + skill * 0.7 + noise();    // kills
  s[2] = 0.9 - skill * 0.8 + noise();    // deaths (negative correlation)
  s[3] = 0.2 + skill * 0.8 + noise();    // damage
  s[4] = 0.3 + skill * 0.6 + noise();    // accuracy
  s[5] = 0.2 + skill * 0.7 + noise();    // objectives
  s[6] = 0.3 + skill * 0.6 + noise();    // recent perf
  s[7] = Math.random();                  // playstyle (pure noise — encoder should ignore)
  for (let i = 0; i < INPUT_DIM; i++) s[i] = Math.max(0, Math.min(1, s[i]));
  return s;
}

function makePlayer(id: number): Player {
  const trueSkill = Math.random();
  return {
    id,
    name: NAMES[id % NAMES.length] + (id >= NAMES.length ? Math.floor(id / NAMES.length) : ""),
    trueSkill,
    stats: statsFromSkill(trueSkill),
    emb: [0, 0],
    score: 0,
    wins: 0,
    losses: 0,
    tier: tierOf(trueSkill),
    color: TIER_COLORS[tierOf(trueSkill)],
    cooldown: 0,
    waitTicks: 0,
    lastPlacement: null,
  };
}

/** Plackett–Luce sampling. Probability of each next finisher is proportional
 *  to exp(strength * trueSkill). Higher strength = more deterministic (skill
 *  almost always wins); lower = more upsets. */
function plackettLuce(lobby: Player[], strength = 5): Player[] {
  const remaining = [...lobby];
  const out: Player[] = [];
  while (remaining.length > 0) {
    const weights = remaining.map((p) => Math.exp(strength * p.trueSkill));
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

/** Std-dev of true skill within a group. Lower = fairer match. */
function skillSpread(group: Player[]): number {
  if (group.length === 0) return 0;
  const mean = group.reduce((a, p) => a + p.trueSkill, 0) / group.length;
  const v = group.reduce((a, p) => a + (p.trueSkill - mean) ** 2, 0) / group.length;
  return Math.sqrt(v);
}

/** Production-style lobby selection: anchor on the longest-waiting eligible
 *  player, then gather the 29 other eligible players closest to them in
 *  learned score space. This avoids the order-statistic bias of "tightest
 *  contiguous window" (which concentrates on the dense middle of the score
 *  distribution and starves the tails), and it serves every tier fairly —
 *  the longer a Bronze waits, the higher their chance of being the anchor. */
function selectLobby(queue: Player[]): Player[] {
  const eligible = queue.filter((p) => p.cooldown === 0);
  if (eligible.length < LOBBY_SIZE) return eligible;
  // Anchor = eligible player with the most ticks waited (ties → random).
  let anchor = eligible[0];
  let bestWait = -1;
  for (const p of eligible) {
    const w = p.waitTicks + Math.random() * 0.5; // small jitter for ties
    if (w > bestWait) { bestWait = w; anchor = p; }
  }
  // Gather the LOBBY_SIZE-1 closest neighbors by score.
  const rest = eligible.filter((p) => p.id !== anchor.id);
  rest.sort((a, b) => Math.abs(a.score - anchor.score) - Math.abs(b.score - anchor.score));
  return [anchor, ...rest.slice(0, LOBBY_SIZE - 1)];
}

/* ---------------- component ---------------- */

const LOT_W = 640;
const LOT_H = 440;
const PAD = 40;

export default function MatchmakerDemo() {
  const [running, setRunning]   = useState(true);
  const [training, setTraining] = useState(true);
  const [lr, setLr]             = useState(0.02);
  const [queueSize, setQueueSize] = useState(DEFAULT_QUEUE_SIZE);
  const [matchesPlayed, setMatchesPlayed] = useState(0);
  const [rankCorrelation, setRankCorrelation] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [spreadHistory, setSpreadHistory] = useState<number[]>([]);
  const [lastLobbySpread, setLastLobbySpread] = useState<number | null>(null);
  const [randomBaseline, setRandomBaseline] = useState<number>(0);
  const [frame, setFrame] = useState<[number, number, number, number]>([1, 0, 0, 1]);

  const [players, setPlayers] = useState<Player[]>([]);
  const [phase, setPhase] = useState<Phase>("idle");
  const [lobbyIds, setLobbyIds] = useState<Set<number>>(new Set());
  const [lastFinish, setLastFinish] = useState<Player[]>([]);

  const encoderRef = useRef<SiameseEncoder | null>(null);
  const lossBufRef = useRef<number[]>([]);
  const playersRef = useRef<Player[]>([]);
  const phaseRef = useRef<Phase>("idle");
  const phaseTicksRef = useRef<number>(0);
  const lobbyRef = useRef<Player[]>([]);
  const orderingRef = useRef<Player[]>([]);
  const trainPairsRef = useRef<{ w: Player; l: Player }[]>([]);

  useEffect(() => { playersRef.current = players; }, [players]);

  // init encoder + queue (client-only)
  useEffect(() => {
    encoderRef.current = new SiameseEncoder();
    const init: Player[] = [];
    for (let i = 0; i < queueSize; i++) init.push(makePlayer(i));
    setPlayers(init);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // queue resize
  useEffect(() => {
    setPlayers((prev) => {
      if (prev.length === queueSize) return prev;
      if (prev.length < queueSize) {
        const extra: Player[] = [];
        for (let i = prev.length; i < queueSize; i++) extra.push(makePlayer(i));
        return [...prev, ...extra];
      }
      return prev.slice(0, queueSize);
    });
  }, [queueSize]);

  /* ------------- main sim + training loop (phase state machine) -------------
     Every 80 ms:
       idle    → pick the best-spread lobby, pre-sample the BR outcome, build
                 training pairs, transition to "forming".
       forming → pulse the lobby bubble for a few ticks, then "playing".
       playing → each tick, train on a slice of the pair buffer (spreads 435
                 pairs across ~10 ticks of the match) so the scatter moves
                 smoothly. At the last tick, record spread + release players.
  ----------------------------------------------------------------------- */
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      const enc = encoderRef.current;
      if (!enc) return;
      const queue = playersRef.current;
      if (queue.length < LOBBY_SIZE) return;

      // ---------- IDLE ----------
      if (phaseRef.current === "idle") {
        phaseTicksRef.current -= 1;
        if (phaseTicksRef.current > 0) return;

        const lobby = selectLobby(queue);
        if (lobby.length < LOBBY_SIZE) {
          // not enough eligible — clear cooldowns and try next tick
          setPlayers((prev) => prev.map((p) => ({ ...p, cooldown: 0 })));
          return;
        }
        lobbyRef.current = lobby;

        // Pre-sample the BR outcome on the true skills
        const ordering = plackettLuce(lobby, 6);
        orderingRef.current = ordering;

        // 435 ordered training pairs from finishing positions
        const pairs: { w: Player; l: Player }[] = [];
        for (let i = 0; i < ordering.length; i++) {
          for (let j = i + 1; j < ordering.length; j++) {
            pairs.push({ w: ordering[i], l: ordering[j] });
          }
        }
        // Fisher–Yates shuffle so we stream them in random order
        for (let i = pairs.length - 1; i > 0; i--) {
          const k = Math.floor(Math.random() * (i + 1));
          [pairs[i], pairs[k]] = [pairs[k], pairs[i]];
        }
        trainPairsRef.current = pairs;

        setLobbyIds(new Set(lobby.map((p) => p.id)));
        phaseRef.current = "forming";
        phaseTicksRef.current = 5;   // ~400 ms
        setPhase("forming");
        return;
      }

      // ---------- FORMING ----------
      if (phaseRef.current === "forming") {
        phaseTicksRef.current -= 1;
        if (phaseTicksRef.current <= 0) {
          phaseRef.current = "playing";
          phaseTicksRef.current = 10; // ~800 ms of in-match training
          setPhase("playing");
        }
        return;
      }

      // ---------- PLAYING ----------
      if (phaseRef.current === "playing") {
        const pairs = trainPairsRef.current;
        const batchSize = Math.max(1, Math.ceil(pairs.length / Math.max(1, phaseTicksRef.current)));
        const batch = pairs.splice(0, batchSize);

        let batchLoss = 0;
        if (training && batch.length > 0) {
          const g = zeroGrads();
          let nViolating = 0;
          for (const { w, l } of batch) {
            const fw = enc.forward(w.stats);
            const fl = enc.forward(l.stats);
            const gap = fw.score - fl.score;
            const violation = MARGIN - gap;
            if (violation <= 0) continue;
            batchLoss += violation;
            nViolating++;
            enc.backwardFromScore(fw, -1, g);
            enc.backwardFromScore(fl, +1, g);
          }
          // Average gradients over the batch so effective LR is not
          // silently multiplied by batch.length (~40).
          const denom = Math.max(1, nViolating);
          const scale = 1 / denom;
          const arrs: Float32Array[] = [g.W1, g.b1, g.W2, g.b2, g.W3, g.b3, g.Ws, g.bs];
          for (const a of arrs) for (let i = 0; i < a.length; i++) a[i] *= scale;

          // L2 weight decay
          for (let i = 0; i < enc.W1.length; i++) g.W1[i] += 1e-4 * enc.W1[i];
          for (let i = 0; i < enc.W2.length; i++) g.W2[i] += 1e-4 * enc.W2[i];
          for (let i = 0; i < enc.W3.length; i++) g.W3[i] += 1e-4 * enc.W3[i];
          for (let i = 0; i < enc.Ws.length; i++) g.Ws[i] += 1e-4 * enc.Ws[i];

          // Global-norm gradient clipping (guards against rare spikes).
          let gnorm2 = 0;
          for (const a of arrs) for (let i = 0; i < a.length; i++) gnorm2 += a[i] * a[i];
          const gnorm = Math.sqrt(gnorm2);
          const clip = 5.0;
          if (gnorm > clip) {
            const k = clip / gnorm;
            for (const a of arrs) for (let i = 0; i < a.length; i++) a[i] *= k;
          }

          enc.applyGrads(g, lr);

          // NaN / Inf guard: if the encoder ever blows up, re-initialize
          // rather than leave the scatter dead.
          let bad = false;
          const ws: Float32Array[] = [enc.W1, enc.b1, enc.W2, enc.b2, enc.W3, enc.b3, enc.Ws, enc.bs];
          for (const a of ws) { for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) { bad = true; break; } if (bad) break; }
          if (bad) {
            enc.reset();
            lossBufRef.current = [];
          }
        }
        lossBufRef.current.push(batch.length > 0 ? batchLoss / batch.length : 0);
        if (lossBufRef.current.length > 80) lossBufRef.current.shift();

        // Re-encode everyone so the scatter moves smoothly with training
        setPlayers((prev) => prev.map((p) => {
          const f = enc.forward(p.stats);
          return { ...p, emb: [f.emb[0], f.emb[1]] as [number, number], score: f.score };
        }));

        // Update the rotation frame (skill direction ŵ)
        const wx = enc.Ws[0], wy = enc.Ws[1];
        const wlen = Math.sqrt(wx * wx + wy * wy) || 1;
        const ux = wx / wlen, uy = wy / wlen;
        setFrame([ux, uy, -uy, ux]);

        setLossHistory([...lossBufRef.current]);

        phaseTicksRef.current -= 1;
        if (phaseTicksRef.current <= 0) {
          // Match done — record metrics and release players
          const lobby = lobbyRef.current;
          const ordering = orderingRef.current;
          const spread = skillSpread(lobby);
          setLastLobbySpread(spread);
          setSpreadHistory((h) => {
            const next = [...h, spread];
            return next.length > 60 ? next.slice(-60) : next;
          });
          setLastFinish(ordering);
          setMatchesPlayed((c) => c + 1);

          // Random-baseline for the KPI: 30 random players from the full queue
          const shuffled = [...queue].sort(() => Math.random() - 0.5).slice(0, LOBBY_SIZE);
          setRandomBaseline(skillSpread(shuffled));

          // Apply cooldowns so the next few formations rotate through the queue
          const lobbyIdSet = new Set(lobby.map((p) => p.id));
          const placementById = new Map(ordering.map((p, i) => [p.id, i + 1]));
          setPlayers((prev) => prev.map((p) => {
            if (lobbyIdSet.has(p.id)) {
              const place = placementById.get(p.id) ?? LOBBY_SIZE;
              const isWin = place <= LOBBY_SIZE / 2;
              const newWins   = isWin   ? p.wins + 1   : p.wins;
              const newLosses = !isWin  ? p.losses + 1 : p.losses;
              const newMatches = newWins + newLosses;
              // Observed stats converge toward true skill as matches accumulate.
              // We resample from statsFromSkill() with a noise amplitude that
              // decays ∝ 1/√(1+N/10), so after ~100 BRs the stats are a tight
              // estimate of trueSkill (std ~0.3× the initial spread).
              const fresh = statsFromSkill(p.trueSkill, newMatches);
              return {
                ...p,
                stats: fresh,
                cooldown: 1,
                waitTicks: 0,
                lastPlacement: place,
                wins: newWins,
                losses: newLosses,
              };
            }
            // Eligible non-selected players accumulate wait; cooldown ones decay.
            const nextCooldown = Math.max(0, p.cooldown - 1);
            const nextWait = nextCooldown === 0 ? p.waitTicks + 1 : 0;
            return { ...p, cooldown: nextCooldown, waitTicks: nextWait };
          }));

          setLobbyIds(new Set());
          phaseRef.current = "idle";
          phaseTicksRef.current = 3; // short pause before next formation
          setPhase("idle");
        }
      }
    }, 80);
    return () => clearInterval(id);
  }, [running, training, lr]);

  // Spearman rank correlation: learned ranking vs hidden ground truth.
  useEffect(() => {
    if (players.length < 2) return;
    const n = players.length;
    const byScore = [...players].sort((a, b) => a.score - b.score);
    const byTruth = [...players].sort((a, b) => a.trueSkill - b.trueSkill);
    const rScore = new Map<number, number>();
    const rTruth = new Map<number, number>();
    byScore.forEach((p, i) => rScore.set(p.id, i));
    byTruth.forEach((p, i) => rTruth.set(p.id, i));
    let sumD2 = 0;
    for (const p of players) {
      const d = (rScore.get(p.id) ?? 0) - (rTruth.get(p.id) ?? 0);
      sumD2 += d * d;
    }
    setRankCorrelation(1 - (6 * sumD2) / (n * (n * n - 1)));
  }, [players]);

  // Rotate embeddings into the learned (skill, residual) frame.
  function toFrame(e: [number, number]): [number, number] {
    const [ux, uy, vx, vy] = frame;
    return [e[0] * ux + e[1] * uy, e[0] * vx + e[1] * vy];
  }
  const bounds = (() => {
    if (players.length === 0) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of players) {
      const [x, y] = toFrame(p.emb);
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const padX = Math.max(0.3, (maxX - minX) * 0.1);
    const padY = Math.max(0.3, (maxY - minY) * 0.1);
    return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
  })();
  function toPx(e: [number, number]): [number, number] {
    const [fx, fy] = toFrame(e);
    const spanX = Math.max(0.01, bounds.maxX - bounds.minX);
    const spanY = Math.max(0.01, bounds.maxY - bounds.minY);
    const x = PAD + ((fx - bounds.minX) / spanX) * (LOT_W - 2 * PAD);
    const y = LOT_H - PAD - ((fy - bounds.minY) / spanY) * (LOT_H - 2 * PAD);
    return [x, y];
  }

  // Lobby bubble ellipse in screen space (wraps the 30 selected dots).
  const lobbyBubble = (() => {
    if (lobbyIds.size === 0) return null;
    const lobbyPlayers = players.filter((p) => lobbyIds.has(p.id));
    if (lobbyPlayers.length === 0) return null;
    const pts = lobbyPlayers.map((p) => toPx(p.emb));
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of pts) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const rx = Math.max(40, (maxX - minX) / 2 + 22);
    const ry = Math.max(34, (maxY - minY) / 2 + 18);
    return { cx, cy, rx, ry };
  })();

  const resetAll = () => {
    encoderRef.current?.reset();
    lossBufRef.current = [];
    lobbyRef.current = [];
    trainPairsRef.current = [];
    orderingRef.current = [];
    phaseRef.current = "idle";
    phaseTicksRef.current = 0;
    setLossHistory([]);
    setSpreadHistory([]);
    setLastLobbySpread(null);
    setRandomBaseline(0);
    setMatchesPlayed(0);
    setRankCorrelation(0);
    setFrame([1, 0, 0, 1]);
    setLobbyIds(new Set());
    setLastFinish([]);
    setPhase("idle");
    setPlayers((prev) =>
      prev.map((p) => ({
        ...p,
        wins: 0, losses: 0,
        emb: [0, 0], score: 0,
        cooldown: 0, waitTicks: 0, lastPlacement: null,
      }))
    );
  };

  const maxLoss = Math.max(0.001, ...lossHistory);
  const maxSpread = Math.max(0.05, randomBaseline * 1.1, ...spreadHistory);
  const improvement =
    randomBaseline > 0 && lastLobbySpread != null
      ? Math.max(0, 1 - lastLobbySpread / randomBaseline)
      : 0;
  // Theoretical floor: tightest 30-of-N contiguous window in uniform skills
  // has std ≈ (LOBBY_SIZE / queueSize) / √12. Random baseline ≈ 1/√12.
  // So the best ANY matchmaker can achieve is 1 − LOBBY_SIZE/queueSize.
  const ceiling = players.length > 0 ? Math.max(0, 1 - LOBBY_SIZE / players.length) : 0;

  return (
    <div className="shimmer-border rounded-3xl">
      <div className="glass rounded-3xl p-6 md:p-8">
        {/* header */}
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-accent2">
              <Users className="w-3 h-3" /> Disruptive Labs · Siamese matchmaker
            </div>
            <h3 className="mt-2 text-2xl font-semibold text-white flex items-center gap-2 flex-wrap">
              30-player battle-royale matchmaker
              <span className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-accent2 border border-accent2/30 rounded-full px-2 py-0.5">
                <Brain className="w-3 h-3" /> siamese network · live training
              </span>
            </h3>
            <p className="mt-2 text-sm text-gray-400 max-w-2xl">
              A shared-weight encoder maps 8-D player stats into a 2-D
              embedding, then a learned linear head projects it onto a
              scalar skill score. Every few seconds the matchmaker assembles
              a <span className="text-gray-200">30-player lobby</span> — the
              cyan bubble you see on the scatter — by sliding a window along
              the learned skill axis and keeping the tightest cluster. That
              lobby then plays out a battle royale (Plackett–Luce on hidden
              true skills, so upsets still happen) and every higher-finisher
              beat every lower-finisher:{" "}
              <span className="font-mono text-gray-300">C(30,2) = 435</span>
              {" "}training pairs per match feed the margin ranking loss.
              After each match, the participants' observed stats are refreshed
              to simulate a rolling recent-performance window. Watch the
              <span className="text-gray-200"> lobby skill spread</span> drop
              below the random-matchmaking baseline as the network learns.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setRunning((r) => !r)}
              className="inline-flex items-center gap-1.5 rounded-full bg-white/5 border border-white/10 px-4 py-2 text-xs hover:bg-white/10"
            >
              {running ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {running ? "Pause" : "Resume"}
            </button>
            <button
              onClick={resetAll}
              className="inline-flex items-center gap-1.5 rounded-full bg-white/5 border border-white/10 px-4 py-2 text-xs hover:bg-white/10"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Reset
            </button>
          </div>
        </div>

        {/* metric strip */}
        <div className="mt-6 grid grid-cols-2 md:grid-cols-5 gap-3">
          <Metric
            label="Rank correlation"
            value={rankCorrelation.toFixed(2)}
            sub="Spearman ρ vs truth"
            pct={Math.max(0, rankCorrelation)}
          />
          <Metric
            label="Lobby spread"
            value={lastLobbySpread != null ? lastLobbySpread.toFixed(3) : "—"}
            sub={randomBaseline > 0 ? `${(improvement * 100).toFixed(0)}% better than random · ceiling ${(ceiling * 100).toFixed(0)}%` : "warming up"}
            pct={improvement}
          />
          <Metric label="BRs played" value={`${matchesPlayed}`} sub={`${LOBBY_SIZE} players each`} />
          <Metric label="Queue size" value={`${players.length}`} sub={`lobby = ${LOBBY_SIZE}`} />
          <Metric
            label="Loss"
            value={lossHistory.length ? lossHistory[lossHistory.length - 1].toFixed(3) : "—"}
            sub="margin ranking"
          />
        </div>

        {/* scatter + sidebar */}
        <div className="mt-6 grid lg:grid-cols-5 gap-6">
          <div className="lg:col-span-3 rounded-2xl bg-bg/60 border border-white/5 p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm text-gray-300">
                Queue — embedding space (rotated into skill frame)
              </div>
              <div className="text-[10px] font-mono text-gray-500">
                phase:{" "}
                <span
                  className={
                    phase === "forming"
                      ? "text-warn"
                      : phase === "playing"
                      ? "text-accent2"
                      : "text-gray-500"
                  }
                >
                  {phase}
                </span>
              </div>
            </div>
            <div
              className="relative mx-auto rounded-xl overflow-hidden"
              style={{
                width: LOT_W,
                height: LOT_H,
                background:
                  "radial-gradient(ellipse at center, rgba(124,92,255,0.08) 0%, rgba(5,7,13,0) 70%)",
                border: "1px solid rgba(255,255,255,0.05)",
                maxWidth: "100%",
              }}
            >
              <div className="absolute inset-x-0 top-1/2 h-px bg-white/5" />
              <div className="absolute inset-y-0 left-1/2 w-px bg-white/5" />
              {/* Bronze → Diamond gradient strip under the X axis */}
              <div
                className="absolute left-[40px] right-[40px] bottom-4 h-1 rounded-full"
                style={{
                  background:
                    "linear-gradient(90deg, #c2843f 0%, #cbd5e1 33%, #fbbf24 66%, #22d3ee 100%)",
                  opacity: 0.6,
                }}
              />
              <div className="absolute left-3 bottom-1.5 text-[9px] font-mono uppercase tracking-wider text-gray-500">
                ← low skill
              </div>
              <div className="absolute right-3 bottom-1.5 text-[9px] font-mono uppercase tracking-wider text-gray-500">
                high skill →
              </div>
              <div className="absolute left-2 top-1.5 text-[9px] font-mono text-gray-600">residual ↑</div>

              {/* lobby bubble overlay */}
              <AnimatePresence>
                {lobbyBubble && (
                  <motion.svg
                    key="bubble"
                    className="absolute inset-0 pointer-events-none"
                    width={LOT_W}
                    height={LOT_H}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: phase === "playing" ? 1 : 0.55 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <defs>
                      <radialGradient id="lobbyGrad" cx="50%" cy="50%" r="50%">
                        <stop offset="0%"   stopColor="#22d3ee" stopOpacity="0.22" />
                        <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
                      </radialGradient>
                    </defs>
                    <motion.ellipse
                      cx={lobbyBubble.cx}
                      cy={lobbyBubble.cy}
                      rx={lobbyBubble.rx}
                      ry={lobbyBubble.ry}
                      fill="url(#lobbyGrad)"
                      stroke="#22d3ee"
                      strokeWidth={1.5}
                      strokeDasharray="5 5"
                      animate={{
                        strokeDashoffset: phase === "playing" ? [0, -20] : 0,
                      }}
                      transition={{
                        duration: 1.2,
                        repeat: phase === "playing" ? Infinity : 0,
                        ease: "linear",
                      }}
                    />
                    <motion.text
                      x={lobbyBubble.cx}
                      y={lobbyBubble.cy - lobbyBubble.ry - 8}
                      textAnchor="middle"
                      fill="#22d3ee"
                      fontSize="10"
                      fontFamily="monospace"
                      style={{ textTransform: "uppercase", letterSpacing: 1 }}
                      animate={{ opacity: phase === "playing" ? [0.6, 1, 0.6] : 0.9 }}
                      transition={{ duration: 1.2, repeat: phase === "playing" ? Infinity : 0 }}
                    >
                      {phase === "forming"
                        ? `forming lobby · 30 closest-skill players`
                        : phase === "playing"
                        ? `battle royale in progress · 30 players`
                        : `selected lobby · 30 players`}
                    </motion.text>
                  </motion.svg>
                )}
              </AnimatePresence>

              {/* player dots */}
              {players.map((p) => {
                const [x, y] = toPx(p.emb);
                const inLobby = lobbyIds.has(p.id);
                const onCooldown = p.cooldown > 0 && !inLobby;
                return (
                  <motion.div
                    key={p.id}
                    className="absolute group"
                    animate={{ left: x - 6, top: y - 6 }}
                    transition={{ type: "spring", stiffness: 160, damping: 22 }}
                    style={{ width: 12, height: 12 }}
                  >
                    <motion.div
                      className="w-3 h-3 rounded-full border"
                      animate={inLobby ? { scale: [1, 1.45, 1] } : { scale: 1 }}
                      transition={inLobby ? { duration: 0.9, repeat: Infinity } : { duration: 0.2 }}
                      style={{
                        background: p.color,
                        borderColor: inLobby ? "#fff" : "rgba(0,0,0,0.5)",
                        boxShadow: inLobby
                          ? `0 0 14px ${p.color}, 0 0 6px #fff`
                          : `0 0 6px ${p.color}66`,
                        opacity: onCooldown ? 0.35 : 1,
                      }}
                    />
                    <div className="absolute left-1/2 -translate-x-1/2 top-4 text-[9px] font-mono text-gray-300 opacity-0 group-hover:opacity-100 whitespace-nowrap pointer-events-none bg-bg/90 border border-white/10 px-1.5 py-0.5 rounded z-10">
                      {p.name} · {p.tier}
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* legend */}
            <div className="mt-3 flex flex-wrap items-center justify-center gap-4 text-[10px] text-gray-400">
              {(Object.keys(TIER_COLORS) as Tier[]).map((t) => (
                <span key={t} className="inline-flex items-center gap-1.5">
                  <span
                    className="inline-block w-2.5 h-2.5 rounded-full"
                    style={{ background: TIER_COLORS[t] }}
                  />
                  {t}
                </span>
              ))}
              <span className="text-gray-600">
                · colors = hidden truth · dim dots = on cooldown
              </span>
            </div>
          </div>

          {/* right column: controls + spread chart + loss + last match */}
          <div className="lg:col-span-2 space-y-4">
            {/* controls */}
            <div className="rounded-2xl bg-bg/60 border border-white/5 p-5 space-y-3">
              <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={training}
                  onChange={(e) => setTraining(e.target.checked)}
                  className="accent-accent"
                />
                Train encoder
              </label>
              <Slider label="Learning rate" value={lr} min={0.001} max={0.08} step={0.001} unit="" onChange={setLr} fmt={(v) => v.toFixed(3)} />
              <Slider label="Queue size" value={queueSize} min={60} max={240} step={10} unit="" onChange={setQueueSize} />
            </div>

            {/* lobby spread — the money chart */}
            <div className="rounded-2xl bg-bg/60 border border-white/5 p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-300">Lobby skill spread</div>
                <div className="text-[10px] font-mono text-gray-500">std-dev · lower = fairer</div>
              </div>
              <svg viewBox="0 0 240 70" width="100%" height="70" preserveAspectRatio="none">
                {/* random-baseline reference */}
                {randomBaseline > 0 && (
                  <>
                    <line
                      x1={0} y1={64 - (randomBaseline / maxSpread) * 56}
                      x2={240} y2={64 - (randomBaseline / maxSpread) * 56}
                      stroke="#6b7280" strokeWidth={1} strokeDasharray="3 3"
                    />
                    <text
                      x={238} y={64 - (randomBaseline / maxSpread) * 56 - 2}
                      textAnchor="end" fill="#6b7280" fontSize="8" fontFamily="monospace"
                    >
                      random
                    </text>
                  </>
                )}
                {spreadHistory.length > 1 && (
                  <>
                    <defs>
                      <linearGradient id="spreadFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%"   stopColor="#34d399" stopOpacity="0.35" />
                        <stop offset="100%" stopColor="#34d399" stopOpacity="0" />
                      </linearGradient>
                    </defs>
                    <path
                      d={
                        "M 0 64 L " +
                        spreadHistory
                          .map((s, i) => {
                            const x = (i / Math.max(1, spreadHistory.length - 1)) * 240;
                            const y = 64 - (s / maxSpread) * 56;
                            return `${x.toFixed(1)} ${y.toFixed(1)}`;
                          })
                          .join(" L ") +
                        ` L 240 64 Z`
                      }
                      fill="url(#spreadFill)"
                    />
                    <path
                      d={
                        "M " +
                        spreadHistory
                          .map((s, i) => {
                            const x = (i / Math.max(1, spreadHistory.length - 1)) * 240;
                            const y = 64 - (s / maxSpread) * 56;
                            return `${x.toFixed(1)} ${y.toFixed(1)}`;
                          })
                          .join(" L ")
                      }
                      fill="none"
                      stroke="#34d399"
                      strokeWidth={2}
                    />
                  </>
                )}
              </svg>
              <div className="flex items-center justify-between text-[10px] text-gray-500 mt-1">
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block w-3 h-0.5 bg-good" /> learned matchmaker
                </span>
                <span className="font-mono">
                  random = {randomBaseline.toFixed(3)}
                </span>
              </div>
            </div>

            {/* loss sparkline */}
            <div className="rounded-2xl bg-bg/60 border border-white/5 p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-gray-300">Ranking loss</div>
                <div className="text-[10px] font-mono text-gray-500">margin = {MARGIN}</div>
              </div>
              <svg viewBox="0 0 240 50" width="100%" height="50" preserveAspectRatio="none">
                <path
                  d={
                    lossHistory.length
                      ? "M " +
                        lossHistory
                          .map((l, i) => {
                            const x = (i / Math.max(1, lossHistory.length - 1)) * 240;
                            const y = 46 - (l / maxLoss) * 42;
                            return `${x.toFixed(1)} ${y.toFixed(1)}`;
                          })
                          .join(" L ")
                      : ""
                  }
                  fill="none"
                  stroke="#22d3ee"
                  strokeWidth={2}
                />
              </svg>
            </div>

            {/* last match — finishing positions (stable mount, no re-key) */}
            {lastFinish.length > 0 && (
                <div
                  className="rounded-2xl bg-bg/60 border border-white/5 p-4"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="text-sm text-gray-300 inline-flex items-center gap-1.5">
                      <Swords className="w-3.5 h-3.5 text-accent2" /> Last battle royale
                    </div>
                    <div className="text-[10px] font-mono text-gray-500">
                      spread {lastLobbySpread?.toFixed(3) ?? "—"}
                    </div>
                  </div>
                  {/* vertical finishing-order bar: one cell per finisher,
                      colored by hidden true tier — the cleaner the Diamond →
                      Bronze gradient, the better the matchmaker. */}
                  <div className="flex h-6 rounded-md overflow-hidden border border-white/10">
                    {lastFinish.map((p, i) => (
                      <div
                        key={p.id}
                        className="flex-1 relative group"
                        style={{ background: p.color, opacity: 0.85 }}
                        title={`#${i + 1} — ${p.name} (${p.tier})`}
                      >
                        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 flex items-center justify-center text-[8px] font-mono text-black/80">
                          {i + 1}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="flex justify-between text-[9px] text-gray-500 mt-1 font-mono uppercase tracking-wider">
                    <span>1st</span>
                    <span>#30</span>
                  </div>
                </div>
              )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Metric({
  label, value, sub, pct,
}: { label: string; value: string; sub?: string; pct?: number }) {
  return (
    <div className="rounded-2xl bg-bg/60 border border-white/5 p-4">
      <div className="text-[10px] uppercase tracking-wider text-gray-500">{label}</div>
      <div className="mt-1 text-xl font-semibold text-white tabular-nums">{value}</div>
      {sub && <div className="text-[10px] text-gray-500 mt-0.5">{sub}</div>}
      {pct != null && (
        <div className="mt-2 h-1 rounded-full bg-white/5 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-accent to-accent2"
            animate={{ width: `${Math.min(100, Math.max(0, pct * 100))}%` }}
            transition={{ duration: 0.4 }}
          />
        </div>
      )}
    </div>
  );
}

function Slider({
  label, value, min, max, step, unit, onChange, fmt,
}: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (v: number) => void; fmt?: (v: number) => string;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span className="tabular-nums text-white">
          {fmt ? fmt(value) : step < 1 ? value.toFixed(2) : value.toFixed(0)}
          {unit}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent"
      />
    </div>
  );
}

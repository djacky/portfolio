"use client";
import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Legend,
} from "recharts";
import { Pause, Play, RefreshCw, Zap, Brain } from "lucide-react";

/* ------------------------------------------------------------------
   EV Fleet Load Balancer — real reinforcement learning, in-browser.

   Policy:   a 2-layer MLP (30 -> 32 -> 6) with a critic head (32 -> 1),
             written in pure TypeScript with hand-rolled forward +
             backprop. No fake weights, no hand-tuned heuristic.

   Training: online actor-critic / REINFORCE with value baseline.
             Every simulation step we observe the fleet, sample a slot
             to prioritize ~ Categorical(softmax(logits)), compute a
             shaped reward, and do one SGD step on the policy + critic.

   Reward:   exactly the shaping from the Python spec — efficiency,
             readiness, fairness, grid overflow, priority weighting.

   Environment:
     - 6 charger slots in a parking lot.
     - Cars arrive on empty spots, charge, then leave. Lifecycle is
       driven by the sim tick, animations by Framer Motion.
     - Each tick = 60 simulated seconds.
     - The trained policy's softmax output is used DIRECTLY as the
       continuous allocation vector (scaled by MAX_GRID_CURRENT),
       exactly like `action = softmax(policy_net(state)) * MAX_GRID`
       in the Python reference. The gradient signal uses a sampled
       categorical action (which slot to prioritize) for REINFORCE.
   ------------------------------------------------------------------ */

type Priority = "VIP" | "Standard" | "Low";
type LifeState = "arriving" | "charging" | "leaving";

const N_SPOTS = 6;
const FEAT_PER_SLOT = 6;

// Charger-type physical limits
const AC_MIN = 6;   // A — below this, AC standard requires the contactor to open (0 A)
const AC_MAX = 32;  // A — single-phase 32 A wallbox ceiling
const DC_MAX = 55;  // A — small DC fast-charger bus limit
const SLOT_HIDDEN = 16;       // per-slot encoder hidden size (shared across slots)
const INPUT_DIM = N_SPOTS * FEAT_PER_SLOT;
const ENTROPY_BETA = 0.02;    // exploration bonus — fights REINFORCE mode collapse

const VOLTAGE = 400;
const TICK_MS = 200;
const SIM_SECONDS_PER_TICK = 60;
const ARRIVE_TICKS = 4; // 1.6s arrive animation
const LEAVE_TICKS = 5;  // 2.0s leave animation

const PRIORITY_W: Record<Priority, number> = { VIP: 1.6, Standard: 1.0, Low: 0.55 };
const PRIORITY_SCORE: Record<Priority, number> = { VIP: 1.0, Standard: 0.5, Low: 0.2 };

const CAR_COLORS = [
  "#7c5cff",
  "#22d3ee",
  "#34d399",
  "#fbbf24",
  "#f472b6",
  "#60a5fa",
  "#fb923c",
  "#a78bfa",
];

// Parking spot pixel coordinates (lot canvas is 900 x 520)
const LOT_W = 900;
const LOT_H = 520;
const SPOTS: { x: number; y: number }[] = [
  { x: 180, y: 150 }, { x: 420, y: 150 }, { x: 660, y: 150 },
  { x: 180, y: 360 }, { x: 420, y: 360 }, { x: 660, y: 360 },
];
const ENTRANCE = { x: -160, y: 260 };
const EXIT = { x: 1060, y: 260 };
const CAR_W = 110;
const CAR_H = 60;

interface Vehicle {
  id: number;
  spotIdx: number;
  state: LifeState;
  stateUntil: number;
  color: string;
  soc: number;
  capacityKwh: number;
  deadline: number;
  timeConnected: number;
  priority: Priority;
  currentDraw: number;   // ACTUAL amps drawn after CC-CV taper
  maxAccept: number;     // battery's max charge-accept current (A) during CC phase
  done: boolean;
  label: string;
  chargerType: "AC" | "DC";
  // For DC cars: once charging starts at a given current, it's latched for the
  // rest of the session. null until the first charging tick.
  latchedCurrent: number | null;
}

/**
 * Li-ion CC-CV charge profile.
 *   SoC ≤ 80%: constant-current phase — battery accepts full maxAccept.
 *   SoC > 80%: constant-voltage phase — accepted current tapers roughly
 *              linearly with (1 - SoC), floored so the last few percent
 *              still trickle in. Real cells are closer to exponential;
 *              linear-with-floor is the standard simplified model.
 */
function batteryAcceptance(soc: number, maxAccept: number): number {
  if (soc >= 1) return 0;
  if (soc <= 0.8) return maxAccept;
  const taper = (1 - soc) / 0.2;            // 1.0 at 0.8 SoC, 0.0 at 1.0 SoC
  return maxAccept * Math.max(0.04, taper); // 4% floor so charging still completes
}

/* ---------------- tiny linear algebra ---------------- */

function randn(std = 0.1) {
  const u = 1 - Math.random();
  const v = Math.random();
  return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function mat(rows: number, cols: number, std: number) {
  const m = new Float32Array(rows * cols);
  for (let i = 0; i < m.length; i++) m[i] = randn(std);
  return m;
}
function vec(n: number) {
  return new Float32Array(n);
}

/* ---------------- policy network ----------------

   Permutation-equivariant actor + critic.

   Actor: a SHARED per-slot MLP (5 -> 16 -> 1). The SAME weights
   are applied to every slot's feature vector, so the decision
   depends only on what the slot looks like (SoC, urgency,
   priority, ...), not on which cabinet the charger sits in.
   This kills the "slot 0 always wins" mode-collapse bug you saw
   with the earlier independent-head architecture.

   Critic: a small MLP (5 -> 16 -> 1) over the mean-pooled feature
   vector of active slots. Provides the REINFORCE baseline.

   Training: actor-critic / REINFORCE with value baseline AND an
   entropy bonus (β) on the softmax, which keeps the policy
   exploring so that every slot still gets visited during training.
---------------------------------------------------- */

class Policy {
  // shared per-slot actor
  W1: Float32Array;   // [SLOT_HIDDEN x FEAT_PER_SLOT]
  b1: Float32Array;   // [SLOT_HIDDEN]
  w2: Float32Array;   // [SLOT_HIDDEN]  (one logit per slot, shared vector)
  b2: Float32Array;   // [1]            (shared scalar bias)
  // critic
  Wv1: Float32Array;  // [SLOT_HIDDEN x FEAT_PER_SLOT]
  bv1: Float32Array;  // [SLOT_HIDDEN]
  wv2: Float32Array;  // [SLOT_HIDDEN]
  bv2: Float32Array;  // [1]

  constructor() {
    const s1 = Math.sqrt(2 / FEAT_PER_SLOT);
    const s2 = Math.sqrt(2 / SLOT_HIDDEN);
    this.W1  = mat(SLOT_HIDDEN, FEAT_PER_SLOT, s1);
    this.b1  = vec(SLOT_HIDDEN);
    this.w2  = mat(1, SLOT_HIDDEN, s2);
    this.b2  = vec(1);
    this.Wv1 = mat(SLOT_HIDDEN, FEAT_PER_SLOT, s1);
    this.bv1 = vec(SLOT_HIDDEN);
    this.wv2 = mat(1, SLOT_HIDDEN, s2);
    this.bv2 = vec(1);
  }

  reset() {
    const p = new Policy();
    this.W1  = p.W1;  this.b1  = p.b1;
    this.w2  = p.w2;  this.b2  = p.b2;
    this.Wv1 = p.Wv1; this.bv1 = p.bv1;
    this.wv2 = p.wv2; this.bv2 = p.bv2;
  }

  /**
   * Forward pass.
   * slotFeats: Float32Array of length N_SPOTS * FEAT_PER_SLOT
   * mask:      boolean[] length N_SPOTS — which slots are currently charging
   */
  forward(slotFeats: Float32Array, mask: boolean[]) {
    const hPerSlot = new Float32Array(N_SPOTS * SLOT_HIDDEN);
    const logits = vec(N_SPOTS);
    const pool = vec(FEAT_PER_SLOT);
    let nActive = 0;

    // per-slot encoder — SHARED weights applied independently to each slot
    for (let i = 0; i < N_SPOTS; i++) {
      const fBase = i * FEAT_PER_SLOT;
      for (let k = 0; k < SLOT_HIDDEN; k++) {
        let s = this.b1[k];
        const row = k * FEAT_PER_SLOT;
        for (let j = 0; j < FEAT_PER_SLOT; j++) {
          s += this.W1[row + j] * slotFeats[fBase + j];
        }
        hPerSlot[i * SLOT_HIDDEN + k] = s > 0 ? s : 0; // ReLU
      }
      // scalar logit = w2 · h_i + b2
      let lg = this.b2[0];
      for (let k = 0; k < SLOT_HIDDEN; k++) {
        lg += this.w2[k] * hPerSlot[i * SLOT_HIDDEN + k];
      }
      logits[i] = lg;
      if (mask[i]) {
        for (let j = 0; j < FEAT_PER_SLOT; j++) pool[j] += slotFeats[fBase + j];
        nActive += 1;
      }
    }
    if (nActive > 0) {
      for (let j = 0; j < FEAT_PER_SLOT; j++) pool[j] /= nActive;
    }

    // masked softmax (zero probability for inactive slots)
    let m = -Infinity;
    for (let i = 0; i < N_SPOTS; i++) if (mask[i] && logits[i] > m) m = logits[i];
    if (!isFinite(m)) m = 0;
    const probs = vec(N_SPOTS);
    let sumE = 0;
    for (let i = 0; i < N_SPOTS; i++) {
      if (!mask[i]) { probs[i] = 0; continue; }
      probs[i] = Math.exp(logits[i] - m);
      sumE += probs[i];
    }
    if (sumE > 0) {
      for (let i = 0; i < N_SPOTS; i++) probs[i] /= sumE;
    }

    // critic head on mean-pooled features
    const hv = vec(SLOT_HIDDEN);
    for (let k = 0; k < SLOT_HIDDEN; k++) {
      let s = this.bv1[k];
      const row = k * FEAT_PER_SLOT;
      for (let j = 0; j < FEAT_PER_SLOT; j++) s += this.Wv1[row + j] * pool[j];
      hv[k] = s > 0 ? s : 0;
    }
    let v = this.bv2[0];
    for (let k = 0; k < SLOT_HIDDEN; k++) v += this.wv2[k] * hv[k];

    return { hPerSlot, logits, probs, pool, hv, v, nActive };
  }

  /**
   * Actor-critic / REINFORCE update with value baseline + entropy bonus.
   * All gradients accumulate across slots into the SHARED actor params.
   */
  update(
    slotFeats: Float32Array,
    mask: boolean[],
    hPerSlot: Float32Array,
    probs: Float32Array,
    pool: Float32Array,
    hv: Float32Array,
    value: number,
    action: number,
    reward: number,
    lr: number,
    beta: number
  ) {
    // advantage, clipped for stability
    let adv = reward - value;
    if (adv > 5) adv = 5;
    if (adv < -5) adv = -5;

    // entropy of current (masked) distribution
    let H = 0;
    for (let i = 0; i < N_SPOTS; i++) {
      if (mask[i] && probs[i] > 1e-9) H -= probs[i] * Math.log(probs[i]);
    }

    // policy loss grad: (probs - onehot(action)) * adv
    //                 + entropy bonus grad: β · p_i · (log p_i + H)
    // (entropy bonus MAXIMIZES H — encourages exploration, fights collapse)
    const dLogits = vec(N_SPOTS);
    for (let i = 0; i < N_SPOTS; i++) {
      if (!mask[i]) { dLogits[i] = 0; continue; }
      dLogits[i] = probs[i] * adv;
      if (probs[i] > 1e-9) {
        dLogits[i] += beta * probs[i] * (Math.log(probs[i]) + H);
      }
    }
    dLogits[action] -= adv;

    // backprop through shared actor: accumulate grads across slots
    const dW1 = vec(SLOT_HIDDEN * FEAT_PER_SLOT);
    const db1 = vec(SLOT_HIDDEN);
    const dw2 = vec(SLOT_HIDDEN);
    let db2 = 0;

    for (let i = 0; i < N_SPOTS; i++) {
      if (!mask[i]) continue;
      const g = dLogits[i];
      db2 += g;
      for (let k = 0; k < SLOT_HIDDEN; k++) {
        const hik = hPerSlot[i * SLOT_HIDDEN + k];
        dw2[k] += g * hik;
      }
      // dH_i = w2 * g, then ReLU backward on h_i
      const fBase = i * FEAT_PER_SLOT;
      for (let k = 0; k < SLOT_HIDDEN; k++) {
        const hik = hPerSlot[i * SLOT_HIDDEN + k];
        const dh = hik > 0 ? this.w2[k] * g : 0;
        db1[k] += dh;
        const row = k * FEAT_PER_SLOT;
        for (let j = 0; j < FEAT_PER_SLOT; j++) {
          dW1[row + j] += dh * slotFeats[fBase + j];
        }
      }
    }

    // SGD update on actor
    for (let i = 0; i < this.W1.length; i++) this.W1[i] -= lr * dW1[i];
    for (let k = 0; k < SLOT_HIDDEN; k++) this.b1[k] -= lr * db1[k];
    for (let k = 0; k < SLOT_HIDDEN; k++) this.w2[k] -= lr * dw2[k];
    this.b2[0] -= lr * db2;

    // critic update: dL_critic/dv = v - r
    const dV = value - reward;
    this.bv2[0] -= lr * dV;
    for (let k = 0; k < SLOT_HIDDEN; k++) {
      const dwv = dV * hv[k];
      const dhv = hv[k] > 0 ? this.wv2[k] * dV : 0;
      this.wv2[k] -= lr * dwv;
      this.bv1[k] -= lr * dhv;
      const row = k * FEAT_PER_SLOT;
      for (let j = 0; j < FEAT_PER_SLOT; j++) {
        this.Wv1[row + j] -= lr * dhv * pool[j];
      }
    }
  }
}

/* ---------------- env helpers ---------------- */

function buildObs(fleet: Vehicle[]): Float32Array {
  const x = vec(INPUT_DIM);
  for (const v of fleet) {
    if (v.state !== "charging" || v.done) continue;
    const b = v.spotIdx * FEAT_PER_SLOT;
    const energyNeeded = Math.max(0, (1 - v.soc) * v.capacityKwh);
    x[b + 0] = v.soc;
    x[b + 1] = Math.min(1, energyNeeded / 100);
    x[b + 2] = Math.min(1, v.deadline / 3600);
    x[b + 3] = PRIORITY_SCORE[v.priority];
    x[b + 4] = 1;
    // Charger-type feature: 1 for DC (latched), 0 for AC (flexible).
    // Lets the policy condition on the constraint class — critical since a
    // DC car's first-tick allocation locks in its current for the whole session.
    x[b + 5] = v.chargerType === "DC" ? 1 : 0;
  }
  return x;
}

function activeMask(fleet: Vehicle[]): boolean[] {
  const m: boolean[] = Array(N_SPOTS).fill(false);
  for (const v of fleet) {
    if (v.state === "charging" && !v.done) m[v.spotIdx] = true;
  }
  return m;
}

function sampleMasked(probs: Float32Array, mask: boolean[]): number {
  let s = 0;
  for (let i = 0; i < N_SPOTS; i++) if (mask[i]) s += probs[i];
  if (s <= 1e-9) {
    const actives = mask.map((m, i) => (m ? i : -1)).filter((i) => i >= 0);
    return actives.length ? actives[Math.floor(Math.random() * actives.length)] : 0;
  }
  let r = Math.random() * s;
  for (let i = 0; i < N_SPOTS; i++) {
    if (!mask[i]) continue;
    r -= probs[i];
    if (r <= 0) return i;
  }
  return 0;
}

function computeReward(
  fleet: Vehicle[],
  gridUsed: number,
  maxGrid: number,
  w: { eff: number; read: number; fair: number; grid: number; prio: number; util: number }
) {
  const active = fleet.filter((v) => v.state === "charging" && !v.done);
  if (active.length === 0) return 0;

  const n = active.length;
  // efficiency: penalize overcharging (SoC > 95%)
  const r_efficiency = -active.reduce(
    (a, v) => a + Math.max(0, v.soc - 0.95) * 10,
    0
  ) / n;
  // readiness: soc / remaining-hours-until-deadline, averaged
  const r_readiness = active.reduce(
    (a, v) => a + v.soc / Math.max(0.15, v.deadline / 3600),
    0
  ) / n;
  // fairness: penalize high variance in allocation
  const draws = active.map((v) => v.currentDraw);
  const mean = draws.reduce((a, b) => a + b, 0) / n;
  const varD = draws.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(varD);
  const r_fairness = -std / Math.max(1, maxGrid);
  // grid: hard penalty if softmax scaling somehow exceeds cap
  const r_grid = -Math.max(0, gridUsed - maxGrid) / Math.max(1, maxGrid);
  // utilization: reward filling the grid up to (but not past) the cap.
  // Scaled to roughly match the magnitude of r_readiness so the weight
  // slider has real pull. The term peaks at ~5.0 when gridUsed == cap
  // and goes to 0 at 0% utilization. Overshooting the cap is still
  // punished by r_grid, so capping at 1.0 before scaling keeps the
  // incentive unambiguous: fill up to cap, not past it.
  const r_utilization = 5 * Math.min(1, gridUsed / Math.max(1, maxGrid));
  // priority: weighted-avg SoC (VIP worth more)
  const r_priority = active.reduce(
    (a, v) => a + PRIORITY_W[v.priority] * v.soc,
    0
  ) / n;
  return (
    w.eff * r_efficiency +
    w.read * r_readiness +
    w.fair * r_fairness +
    w.grid * r_grid +
    w.util * r_utilization +
    w.prio * r_priority
  );
}

/* ---------------- component ---------------- */

export default function EVFleetDemo() {
  const [maxGrid, setMaxGrid] = useState(200);
  const [running, setRunning] = useState(true);
  const [training, setTraining] = useState(true);
  const [lr, setLr] = useState(0.003);
  const [weights, setWeights] = useState({
    eff: 0.5,
    read: 1.0,
    fair: 0.5,
    grid: 2.0,
    prio: 0.8,
    util: 2.0,
  });

  const [fleet, setFleet] = useState<Vehicle[]>([]);
  const [gridHistory, setGridHistory] = useState<{ t: number; grid: number }[]>([]);
  const [allocHistory, setAllocHistory] = useState<
    { t: number; s0: number; s1: number; s2: number; s3: number; s4: number; s5: number }[]
  >([]);
  const [missHistory, setMissHistory] = useState<
    { t: number; vip: number; std: number; low: number }[]
  >([]);
  const [socPrioHistory, setSocPrioHistory] = useState<
    { t: number; vip: number | null; std: number | null; low: number | null }[]
  >([]);
  const [episodes, setEpisodes] = useState(0);
  const [totalCompleted, setTotalCompleted] = useState(0);
  const [meanReward, setMeanReward] = useState(0);

  const policyRef = useRef<Policy | null>(null);
  const nextIdRef = useRef(1);
  const tickRef = useRef(0);
  const rewardBufRef = useRef<number[]>([]);
  const spotCooldownRef = useRef<number[]>(Array(N_SPOTS).fill(0));
  // rolling window of recent departures for the miss-rate chart
  const departureBufRef = useRef<
    { missedVIP: number; missedStd: number; missedLow: number;
      onTimeVIP: number; onTimeStd: number; onTimeLow: number }[]
  >([]);

  // init policy (client-only — avoids SSR hydration drift from random init)
  useEffect(() => {
    policyRef.current = new Policy();
  }, []);

  // seed the lot
  useEffect(() => {
    const init: Vehicle[] = [];
    for (let i = 0; i < N_SPOTS; i++) {
      if (Math.random() < 0.8) init.push(spawn(i, nextIdRef));
    }
    setFleet(init);
  }, []);

  // main simulation loop
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setFleet((prev) => {
        tickRef.current += 1;
        const tick = tickRef.current;

        // 1. advance lifecycle timers + tick deadlines for all non-leaving cars
        //    (deadline is wall-clock — it should decrement whether the car is
        //     already charging or still pulling into the spot).
        let next: Vehicle[] = prev.map((v) => {
          const newDeadline =
            v.state === "leaving" ? v.deadline : Math.max(0, v.deadline - SIM_SECONDS_PER_TICK);
          if (v.state === "arriving") {
            const u = v.stateUntil - 1;
            if (u <= 0) {
              return { ...v, state: "charging", stateUntil: 0, timeConnected: 0, deadline: newDeadline };
            }
            return { ...v, stateUntil: u, deadline: newDeadline };
          }
          if (v.state === "leaving") {
            return { ...v, stateUntil: v.stateUntil - 1 };
          }
          return { ...v, deadline: newDeadline };
        });

        // 2. remove vehicles that finished leaving, set spot cooldown
        next = next.filter((v) => {
          if (v.state === "leaving" && v.stateUntil <= 0) {
            spotCooldownRef.current[v.spotIdx] = 3; // small gap before next arrival
            return false;
          }
          return true;
        });

        // 3. decrement cooldowns + spawn arrivals on empty spots
        for (let s = 0; s < N_SPOTS; s++) {
          if (spotCooldownRef.current[s] > 0) spotCooldownRef.current[s] -= 1;
        }
        const occupied = new Set(next.map((v) => v.spotIdx));
        for (let s = 0; s < N_SPOTS; s++) {
          if (occupied.has(s)) continue;
          if (spotCooldownRef.current[s] > 0) continue;
          if (Math.random() < 0.12) next.push(spawn(s, nextIdRef));
        }

        // 4. policy step
        const mask = activeMask(next);
        const hasActive = mask.some(Boolean);
        let gridUsed = 0;
        let reward = 0;

        if (hasActive && policyRef.current) {
          const obs = buildObs(next);
          const fwd = policyRef.current.forward(obs, mask);
          // probs are already masked to active slots inside forward()
          const action = sampleMasked(fwd.probs, mask);

          // Allocation respecting latched-DC commitments.
          //   1. Latched DC draws are fixed — they take priority on the bus.
          //   2. Remaining budget = maxGrid − sum(latched).
          //   3. Non-latched slots share that budget via the renormalized softmax.
          // This guarantees sum(allocated) ≤ maxGrid, which the naive
          // `probs[i] * maxGrid` scheme violated whenever the policy's
          // probability drifted away from a still-drawing DC slot.
          const alloc = vec(N_SPOTS);
          const latchedIdx = new Set<number>();
          let latchedSum = 0;
          for (const v of next) {
            if (
              v.state === "charging" &&
              !v.done &&
              v.chargerType === "DC" &&
              v.latchedCurrent != null
            ) {
              // Reserve only the EFFECTIVE draw, not the raw latched value.
              // Once a DC car enters CV taper, batteryAcceptance(soc) falls
              // below its latched current and the physical grid draw follows
              // suit — so the headroom the car is no longer using should flow
              // to the rest of the fleet via the waterfall below.
              const effective = Math.min(
                v.latchedCurrent,
                batteryAcceptance(v.soc, v.maxAccept),
              );
              alloc[v.spotIdx] = effective;
              latchedSum += effective;
              latchedIdx.add(v.spotIdx);
            }
          }
          const remainingCap = Math.max(0, maxGrid - latchedSum);

          // Waterfall allocation: give each free slot its softmax share of the
          // remaining grid cap, BUT cap each slot by what the car can physically
          // accept this tick (charger type limit ∧ battery acceptance curve).
          // Any unused headroom is redistributed to still-uncapped slots,
          // again proportionally to the policy's probs, until no slot wants
          // more. Without this, the policy wastes current on AC slots that
          // are already at 32 A while a DC slot sits at 35 A with 20 A of
          // free grid just... gone.
          {
            // Per-slot ceiling for THIS tick. For DC cars that haven't latched
            // yet, we don't cap here — the latch happens in the physics step
            // and they should get as much as the policy wants to hand them.
            const slotCap = new Array<number>(N_SPOTS).fill(0);
            for (let i = 0; i < N_SPOTS; i++) {
              if (!mask[i] || latchedIdx.has(i)) continue;
              const v = next.find((x) => x.spotIdx === i && x.state === "charging" && !x.done);
              if (!v) continue;
              const accept = batteryAcceptance(v.soc, v.maxAccept);
              const typeCap = v.chargerType === "AC" ? AC_MAX : DC_MAX;
              slotCap[i] = Math.min(typeCap, accept);
            }

            const locked = new Array<boolean>(N_SPOTS).fill(false);
            let budget = remainingCap;

            // Up to 6 waterfall passes — converges quickly with only 6 slots.
            for (let pass = 0; pass < N_SPOTS && budget > 1e-6; pass++) {
              let pFreeSum = 0;
              for (let i = 0; i < N_SPOTS; i++) {
                if (mask[i] && !latchedIdx.has(i) && !locked[i]) {
                  pFreeSum += fwd.probs[i];
                }
              }
              if (pFreeSum < 1e-9) break;

              let newlyLocked = false;
              let distributed = 0;
              for (let i = 0; i < N_SPOTS; i++) {
                if (!mask[i] || latchedIdx.has(i) || locked[i]) continue;
                const share = (fwd.probs[i] / pFreeSum) * budget;
                const headroom = Math.max(0, slotCap[i] - alloc[i]);
                if (share >= headroom - 1e-6) {
                  alloc[i] += headroom;
                  distributed += headroom;
                  locked[i] = true;
                  newlyLocked = true;
                } else {
                  alloc[i] += share;
                  distributed += share;
                }
              }
              budget -= distributed;
              if (!newlyLocked) break; // nothing to redistribute
            }
          }

          // 5. step physics on charging vehicles — with CC-CV battery taper.
          //    The policy *allocates* amps via the softmax, but the battery
          //    only *accepts* min(allocated, batteryAcceptance(soc)). Any
          //    unaccepted current is simply not drawn from the grid.
          let justFinished = 0;
          let onTimeVIP = 0, onTimeStd = 0, onTimeLow = 0;
          next = next.map((v) => {
            if (v.state !== "charging" || v.done) return v;
            const rawAlloc = alloc[v.spotIdx];

            // Apply charger-type constraint to the policy's allocation.
            //   AC: either 0 A, or anywhere in [AC_MIN, AC_MAX]; can vary every tick.
            //   DC: up to DC_MAX, but LATCHED — the first tick's current is locked
            //       in for the rest of the session.
            let latched = v.latchedCurrent;
            let constrained: number;
            if (v.chargerType === "AC") {
              if (rawAlloc < AC_MIN) constrained = 0;
              else constrained = Math.min(rawAlloc, AC_MAX);
            } else {
              if (latched != null) {
                constrained = latched;
              } else {
                // First charging tick for this DC car — pick & latch.
                // Tiny allocations aren't worth starting a DC session for.
                constrained = rawAlloc < 8 ? 0 : Math.min(rawAlloc, DC_MAX);
                if (constrained > 0) latched = constrained;
              }
            }

            const accept = batteryAcceptance(v.soc, v.maxAccept);
            const actualAmps = Math.min(constrained, accept);
            const kWh = (actualAmps * VOLTAGE * SIM_SECONDS_PER_TICK) / 3600 / 1000;
            const newSoc = Math.min(1, v.soc + kWh / v.capacityKwh);
            // Done at 95% — matches real-world operator behavior (billing
            // typically stops in the low-90s because CV tail is uneconomic)
            // and avoids the AC_MIN dead zone where a policy-deprioritized
            // near-full car can stall at 0 A until its deadline expires.
            const done = newSoc >= 0.95;
            if (done) {
              justFinished += 1;
              if (v.priority === "VIP") onTimeVIP++;
              else if (v.priority === "Standard") onTimeStd++;
              else onTimeLow++;
            }
            return {
              ...v,
              soc: newSoc,
              currentDraw: done ? 0 : actualAmps,
              timeConnected: v.timeConnected + SIM_SECONDS_PER_TICK,
              done,
              state: done ? ("leaving" as LifeState) : ("charging" as LifeState),
              stateUntil: done ? LEAVE_TICKS : 0,
              latchedCurrent: done ? null : latched,
            };
          });
          if (justFinished > 0) {
            setTotalCompleted((c) => c + justFinished);
          }

          // Deadline expiry: cars that ran out of time while still charging
          // leave as "missed". We count them per priority for the miss-rate chart.
          let missedVIP = 0, missedStd = 0, missedLow = 0;
          next = next.map((v) => {
            if (v.state === "charging" && !v.done && v.deadline <= 0) {
              if (v.priority === "VIP") missedVIP++;
              else if (v.priority === "Standard") missedStd++;
              else missedLow++;
              return {
                ...v,
                currentDraw: 0,
                state: "leaving" as LifeState,
                stateUntil: LEAVE_TICKS,
                latchedCurrent: null,
              };
            }
            return v;
          });
          if (missedVIP || missedStd || missedLow || onTimeVIP || onTimeStd || onTimeLow) {
            departureBufRef.current.push({
              missedVIP, missedStd, missedLow,
              onTimeVIP, onTimeStd, onTimeLow,
            });
            // keep a rolling buffer of the last 50 windows worth
            if (departureBufRef.current.length > 200) {
              departureBufRef.current = departureBufRef.current.slice(-200);
            }
          }

          // gridUsed = sum of ACTUAL draws (not allocations), so the chart
          // visibly dips when cars enter the CV taper phase.
          gridUsed = next.reduce((a, v) => a + v.currentDraw, 0);

          // Reward: shaped terms + fixed bonus for each car that just finished
          // (on-time completion signal — keeps the gradient meaningful even
          // when a vehicle leaves the active set the same tick it finishes).
          reward =
            computeReward(next, gridUsed, maxGrid, weights) +
            justFinished * (weights.prio + 1.0);

          // 6. actor-critic / REINFORCE update (shared weights + entropy bonus)
          if (training) {
            policyRef.current.update(
              obs, mask, fwd.hPerSlot, fwd.probs, fwd.pool, fwd.hv, fwd.v,
              action, reward, lr, ENTROPY_BETA
            );
          }

          rewardBufRef.current.push(reward);
        }

        // 7. logging + diagnostics for the charts
        setGridHistory((h) => {
          const n = [...h, { t: tick, grid: Math.round(gridUsed) }];
          return n.length > 80 ? n.slice(-80) : n;
        });
        // Live per-slot allocation stream (actual amps drawn per charger).
        {
          const drawBySlot = [0, 0, 0, 0, 0, 0];
          for (const v of next) {
            if (v.state === "charging" && !v.done) {
              drawBySlot[v.spotIdx] = v.currentDraw;
            }
          }
          setAllocHistory((h) => {
            const n = [
              ...h,
              {
                t: tick,
                s0: +drawBySlot[0].toFixed(1),
                s1: +drawBySlot[1].toFixed(1),
                s2: +drawBySlot[2].toFixed(1),
                s3: +drawBySlot[3].toFixed(1),
                s4: +drawBySlot[4].toFixed(1),
                s5: +drawBySlot[5].toFixed(1),
              },
            ];
            return n.length > 80 ? n.slice(-80) : n;
          });
        }
        if (tick % 10 === 0) {
          // mean SoC broken down by priority class
          let vs = 0, vn = 0, ss = 0, sn = 0, ls = 0, ln = 0;
          for (const v of next) {
            if (v.state !== "charging") continue;
            if (v.priority === "VIP") { vs += v.soc; vn++; }
            else if (v.priority === "Standard") { ss += v.soc; sn++; }
            else { ls += v.soc; ln++; }
          }

          // Rolling deadline-miss rate by priority over the last 60 departure events.
          const dbuf = departureBufRef.current.slice(-60);
          let mv = 0, nv = 0, ms = 0, ns = 0, ml = 0, nl = 0;
          for (const d of dbuf) {
            mv += d.missedVIP; nv += d.missedVIP + d.onTimeVIP;
            ms += d.missedStd; ns += d.missedStd + d.onTimeStd;
            ml += d.missedLow; nl += d.missedLow + d.onTimeLow;
          }
          setMissHistory((h) => {
            const n = [
              ...h,
              {
                t: tick,
                vip: nv ? +((mv / nv) * 100).toFixed(1) : 0,
                std: ns ? +((ms / ns) * 100).toFixed(1) : 0,
                low: nl ? +((ml / nl) * 100).toFixed(1) : 0,
              },
            ];
            return n.length > 80 ? n.slice(-80) : n;
          });
          setSocPrioHistory((h) => {
            const n = [
              ...h,
              {
                t: tick,
                vip: vn ? +((vs / vn) * 100).toFixed(1) : null,
                std: sn ? +((ss / sn) * 100).toFixed(1) : null,
                low: ln ? +((ls / ln) * 100).toFixed(1) : null,
              },
            ];
            return n.length > 80 ? n.slice(-80) : n;
          });
        }
        if (tick % 25 === 0) {
          const buf = rewardBufRef.current;
          if (buf.length) {
            const m = buf.reduce((a, b) => a + b, 0) / buf.length;
            setMeanReward(m);
            setEpisodes((e) => e + 1);
            rewardBufRef.current = [];
          }
        }

        return next;
      });
    }, TICK_MS);
    return () => clearInterval(id);
  }, [running, maxGrid, weights, training, lr]);

  function spawn(spotIdx: number, idRef: React.MutableRefObject<number>): Vehicle {
    const id = idRef.current++;
    const r = Math.random();
    const prio: Priority = r < 0.2 ? "VIP" : r < 0.75 ? "Standard" : "Low";
    const cap = 40 + Math.round(Math.random() * 60); // 40–100 kWh
    const chargerType: "AC" | "DC" = Math.random() < 0.55 ? "AC" : "DC";
    // Max the battery can accept during CC phase — bounded by the charger class.
    const batteryMax = 80 + (cap - 40) * 0.8 + Math.random() * 15;
    const maxAccept =
      chargerType === "AC"
        ? Math.min(batteryMax, AC_MAX)
        : Math.min(batteryMax, DC_MAX);
    return {
      id,
      spotIdx,
      state: "arriving",
      stateUntil: ARRIVE_TICKS,
      color: CAR_COLORS[id % CAR_COLORS.length],
      soc: 0.08 + Math.random() * 0.4,
      capacityKwh: cap,
      // Deadlines tuned to the sim time-scale (SIM_SECONDS_PER_TICK=60, 400ms/tick).
      // Range ≈ 2–5 hours of sim time so cars don't all expire before charging completes.
      deadline: 7200 + Math.random() * 10800,
      timeConnected: 0,
      priority: prio,
      currentDraw: 0,
      // Max accept roughly scales with pack size. Tuned to 80–140 A so the
      // CC-CV taper is visibly observable when allocations approach ~30–60 A.
      maxAccept,
      done: false,
      label: `EV-${String(id).padStart(3, "0")}`,
      chargerType,
      latchedCurrent: null,
    };
  }

  const resetPolicy = () => {
    policyRef.current?.reset();
    setEpisodes(0);
    setMeanReward(0);
    setMissHistory([]);
    setAllocHistory([]);
    rewardBufRef.current = [];
    departureBufRef.current = [];
  };
  const resetAll = () => {
    resetPolicy();
    tickRef.current = 0;
    setFleet([]);
    setGridHistory([]);
    setSocPrioHistory([]);
    setTotalCompleted(0);
    spotCooldownRef.current = Array(N_SPOTS).fill(0);
    setTimeout(() => {
      const init: Vehicle[] = [];
      for (let i = 0; i < N_SPOTS; i++) {
        if (Math.random() < 0.8) init.push(spawn(i, nextIdRef));
      }
      setFleet(init);
    }, 50);
  };

  const activeCount = fleet.filter((v) => v.state === "charging" && !v.done).length;
  const gridUsed = fleet.reduce((a, v) => a + v.currentDraw, 0);
  const power = (gridUsed * VOLTAGE) / 1000;

  return (
    <div className="shimmer-border rounded-3xl">
      <div className="glass rounded-3xl p-6 md:p-8">
        {/* header */}
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-5">
          <div>
            <div className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.22em] text-accent2">
              <Zap className="w-3.5 h-3.5" />
              Eaton · EV fleet load balancing
            </div>
            <h3 className="mt-2 text-2xl md:text-3xl font-semibold text-gradient">
              Reinforcement-learning current allocator
            </h3>
            <p className="mt-2 text-sm text-gray-400 max-w-2xl leading-relaxed">
              A small policy network is training live in your browser to split
              current across the fleet against the Eaton spec. Watch the mean
              reward climb as it learns to juggle priority, urgency, fairness,
              and the grid cap.
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
        <div className="mt-6 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <Metric label="Grid draw" value={`${gridUsed.toFixed(0)} A`} sub={`cap ${maxGrid} A`} pct={gridUsed / maxGrid} />
          <Metric label="Power" value={`${power.toFixed(1)} kW`} sub={`${VOLTAGE} V bus`} />
          <Metric label="Active" value={`${activeCount}`} sub={`${N_SPOTS} chargers`} />
          <Metric label="Completed" value={`${totalCompleted}`} sub="since reset" />
          <Metric label="Episodes" value={`${episodes}`} sub="25-step windows" />
          <Metric label="Mean reward" value={meanReward.toFixed(2)} sub={training ? "learning" : "frozen"} />
        </div>

        {/* parking lot */}
        <div className="mt-6 rounded-2xl bg-bg/60 border border-white/5 p-3 overflow-hidden">
          <div
            className="relative mx-auto"
            style={{ width: LOT_W, height: LOT_H }}
          >
            <LotBackground />
            {SPOTS.map((s, i) => (
              <ChargerSpot key={i} x={s.x} y={s.y} idx={i} />
            ))}
            <AnimatePresence>
              {fleet.map((v) => (
                <VehicleSprite key={v.id} v={v} maxGrid={maxGrid} />
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* charts + controls */}
        <div className="mt-6 grid lg:grid-cols-5 gap-6">
          <div className="lg:col-span-3 grid md:grid-cols-2 gap-4">
            <ChartCard title="Grid current vs cap" subtitle={`cap ${maxGrid} A`}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={gridHistory}>
                  <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={10}
                    domain={[0, Math.ceil(maxGrid * 1.15)]}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0b0f1a",
                      border: "1px solid #1f2937",
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                  />
                  <ReferenceLine
                    y={maxGrid}
                    stroke="#f87171"
                    strokeDasharray="4 4"
                    label={{ value: "cap", fill: "#f87171", fontSize: 10 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="grid"
                    stroke="#7c5cff"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Deadline-miss rate by priority" subtitle="rolling 60 departures">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={missHistory}>
                  <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={10}
                    domain={[0, 100]}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0b0f1a",
                      border: "1px solid #1f2937",
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                    formatter={(v: number) => `${v}%`}
                  />
                  <Legend
                    verticalAlign="top"
                    height={24}
                    iconType="line"
                    wrapperStyle={{ fontSize: 11, color: "#9ca3af", paddingBottom: 4 }}
                  />
                  <Line type="monotone" dataKey="vip" stroke="#7c5cff" strokeWidth={2} dot={false} isAnimationActive={false} name="VIP" />
                  <Line type="monotone" dataKey="std" stroke="#22d3ee" strokeWidth={2} dot={false} isAnimationActive={false} name="Standard" />
                  <Line type="monotone" dataKey="low" stroke="#6b7280" strokeWidth={2} dot={false} isAnimationActive={false} name="Low" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Mean SoC by priority class" subtitle="VIP · Standard · Low">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={socPrioHistory}>
                  <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={10}
                    domain={[0, 100]}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0b0f1a",
                      border: "1px solid #1f2937",
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                    formatter={(v: unknown) => (v == null ? "—" : `${v}%`)}
                  />
                  <Legend
                    verticalAlign="top"
                    height={24}
                    iconType="line"
                    wrapperStyle={{ fontSize: 11, color: "#9ca3af", paddingBottom: 4 }}
                  />
                  <Line type="monotone" dataKey="vip" stroke="#7c5cff" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls name="VIP" />
                  <Line type="monotone" dataKey="std" stroke="#22d3ee" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls name="Standard" />
                  <Line type="monotone" dataKey="low" stroke="#6b7280" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls name="Low" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Per-charger current" subtitle="amps drawn · one line per charger">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={allocHistory}>
                  <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                  <XAxis dataKey="t" hide />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={10}
                    domain={[0, Math.ceil(Math.max(60, maxGrid * 0.6))]}
                    tickFormatter={(v) => `${v}A`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0b0f1a",
                      border: "1px solid #1f2937",
                      borderRadius: 8,
                      fontSize: 11,
                    }}
                    formatter={(v: number) => `${v} A`}
                  />
                  <Line type="monotone" dataKey="s0" stroke="#7c5cff" strokeWidth={2} dot={false} isAnimationActive={false} name="Ch-01" />
                  <Line type="monotone" dataKey="s1" stroke="#22d3ee" strokeWidth={2} dot={false} isAnimationActive={false} name="Ch-02" />
                  <Line type="monotone" dataKey="s2" stroke="#34d399" strokeWidth={2} dot={false} isAnimationActive={false} name="Ch-03" />
                  <Line type="monotone" dataKey="s3" stroke="#fbbf24" strokeWidth={2} dot={false} isAnimationActive={false} name="Ch-04" />
                  <Line type="monotone" dataKey="s4" stroke="#f472b6" strokeWidth={2} dot={false} isAnimationActive={false} name="Ch-05" />
                  <Line type="monotone" dataKey="s5" stroke="#60a5fa" strokeWidth={2} dot={false} isAnimationActive={false} name="Ch-06" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          <div className="lg:col-span-2 rounded-2xl bg-bg/60 border border-white/5 p-5 space-y-4">
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={training}
                  onChange={(e) => setTraining(e.target.checked)}
                  className="accent-accent"
                />
                Train policy
              </label>
              <button
                onClick={resetPolicy}
                className="ml-auto text-[10px] uppercase tracking-wider text-accent hover:underline"
              >
                Reset weights
              </button>
            </div>
            <Slider label="Grid cap" value={maxGrid} min={80} max={400} step={10} unit=" A" onChange={setMaxGrid} />
            <Slider label="Learning rate" value={lr} min={0.0005} max={0.01} step={0.0005} unit="" onChange={setLr} fmt={(v) => v.toFixed(4)} />
            <div className="h-px bg-white/5" />
            <div className="text-xs text-gray-400">Reward weights</div>
            <Slider label="Readiness" value={weights.read} min={0} max={2} step={0.1} unit="" onChange={(v) => setWeights({ ...weights, read: v })} />
            <Slider label="Priority" value={weights.prio} min={0} max={2} step={0.1} unit="" onChange={(v) => setWeights({ ...weights, prio: v })} />
            <Slider label="Fairness" value={weights.fair} min={0} max={2} step={0.1} unit="" onChange={(v) => setWeights({ ...weights, fair: v })} />
            <Slider label="Efficiency" value={weights.eff} min={0} max={2} step={0.1} unit="" onChange={(v) => setWeights({ ...weights, eff: v })} />
            <Slider label="Grid penalty" value={weights.grid} min={0} max={4} step={0.1} unit="" onChange={(v) => setWeights({ ...weights, grid: v })} />
            <Slider label="Utilization" value={weights.util} min={0} max={5} step={0.1} unit="" onChange={(v) => setWeights({ ...weights, util: v })} />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------- presentational sub-components ---------------- */

function LotBackground() {
  return (
    <>
      <div
        className="absolute inset-0 rounded-xl"
        style={{
          background:
            "radial-gradient(ellipse at center, rgba(124,92,255,0.06) 0%, rgba(5,7,13,0) 70%), linear-gradient(180deg, #0a0f1a 0%, #070a12 100%)",
        }}
      />
      {/* driveway */}
      <div
        className="absolute left-0 right-0"
        style={{
          top: 232,
          height: 56,
          background:
            "repeating-linear-gradient(90deg, rgba(255,255,255,0.04) 0 40px, transparent 40px 80px)",
          borderTop: "1px dashed rgba(255,255,255,0.08)",
          borderBottom: "1px dashed rgba(255,255,255,0.08)",
        }}
      />
      <div className="absolute left-3 top-[252px] text-[10px] font-mono uppercase tracking-widest text-accent2/70">
        entrance →
      </div>
      <div className="absolute right-3 top-[252px] text-[10px] font-mono uppercase tracking-widest text-accent2/70">
        → exit
      </div>
      <div className="absolute left-3 top-2 text-[10px] font-mono uppercase tracking-widest text-gray-500">
        Eaton Smart Lot · L1
      </div>
    </>
  );
}

function ChargerSpot({ x, y, idx }: { x: number; y: number; idx: number }) {
  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: x - CAR_W / 2 - 10,
        top: y - CAR_H / 2 - 8,
        width: CAR_W + 20,
        height: CAR_H + 16,
      }}
    >
      {/* parking stall — dashed border + faint inner glow */}
      <div
        className="w-full h-full rounded-xl border border-dashed border-white/10"
        style={{ boxShadow: "inset 0 0 24px rgba(34,211,238,0.025)" }}
      />
      {/* stall corner ticks for that "marked stall" feel */}
      <div className="absolute top-1 left-1 w-2 h-2 border-l border-t border-white/20" />
      <div className="absolute top-1 right-1 w-2 h-2 border-r border-t border-white/20" />
      <div className="absolute bottom-1 left-1 w-2 h-2 border-l border-b border-white/20" />
      <div className="absolute bottom-1 right-1 w-2 h-2 border-r border-b border-white/20" />

      <div className="absolute -top-4 left-1 text-[9px] font-mono uppercase tracking-widest text-white/25">
        charger-{String(idx + 1).padStart(2, "0")}
      </div>

      {/* pedestal — proper top-down SVG with base + body + idle LED */}
      <svg
        className="absolute"
        width={20}
        height={32}
        viewBox="0 0 20 32"
        style={{ left: -22, top: "50%", transform: "translateY(-50%)" }}
      >
        {/* base shadow */}
        <ellipse cx="10" cy="29" rx="8" ry="2" fill="rgba(0,0,0,0.55)" />
        {/* base plate */}
        <rect x="3" y="22" width="14" height="6" rx="1.5" fill="#0b0f1a" stroke="#1f2937" strokeWidth="0.6" />
        {/* body */}
        <rect x="6" y="6" width="8" height="18" rx="1.5" fill="url(#pedGrad)" stroke="#1f2937" strokeWidth="0.6" />
        {/* face panel */}
        <rect x="7.2" y="9" width="5.6" height="3" rx="0.6" fill="#05070d" />
        <rect x="7.2" y="13" width="5.6" height="1" fill="#0e1526" />
        <rect x="7.2" y="15" width="5.6" height="1" fill="#0e1526" />
        {/* idle LED ring on top */}
        <circle cx="10" cy="5" r="2.6" fill="#0b0f1a" stroke="rgba(34,211,238,0.35)" strokeWidth="0.7" />
        <circle cx="10" cy="5" r="1.2" fill="#22d3ee" opacity="0.5">
          <animate attributeName="opacity" values="0.35;0.7;0.35" dur="3.6s" repeatCount="indefinite" />
        </circle>
        <defs>
          <linearGradient id="pedGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#1a2236" />
            <stop offset="100%" stopColor="#0b0f1a" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
}

function VehicleSprite({ v, maxGrid }: { v: Vehicle; maxGrid: number }) {
  const spot = SPOTS[v.spotIdx];
  const target =
    v.state === "leaving" ? EXIT : spot;
  const duration =
    v.state === "arriving" ? ARRIVE_TICKS * (TICK_MS / 1000)
    : v.state === "leaving" ? LEAVE_TICKS * (TICK_MS / 1000)
    : 0;
  // Bottom-row cars show info card BELOW (to avoid the driveway strip)
  const isBottomRow = spot.y > 250;
  const infoStyle = isBottomRow
    ? { left: (CAR_W - 170) / 2, top: CAR_H + 14, width: 170 }
    : { left: (CAR_W - 170) / 2, top: -96, width: 170 };

  return (
    <motion.div
      className="absolute"
      style={{ width: CAR_W, height: CAR_H }}
      initial={{ x: ENTRANCE.x - CAR_W / 2, y: ENTRANCE.y - CAR_H / 2, opacity: 0 }}
      animate={{
        x: target.x - CAR_W / 2,
        y: target.y - CAR_H / 2,
        opacity: 1,
      }}
      exit={{ opacity: 0 }}
      transition={{
        duration,
        ease: v.state === "arriving" ? "easeOut" : "easeIn",
      }}
    >
      <CarBody color={v.color} charging={v.state === "charging" && !v.done && v.currentDraw > 0} done={v.done} />
      {/* Animated charging cable from pedestal → car charging port.
          Width and energy-flow speed scale with the actual current
          allocated to this car by the policy. Visible only while the
          car is parked at the spot (state === "charging"). */}
      {v.state === "charging" && !v.done && (
        <ChargingCable
          chargerType={v.chargerType}
          currentDraw={v.currentDraw}
          color={v.color}
        />
      )}
      <AnimatePresence>
        {v.state === "charging" && !v.done && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="absolute"
            style={infoStyle}
          >
            <InfoCard v={v} maxGrid={maxGrid} />
          </motion.div>
        )}
        {v.done && v.state === "leaving" && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] font-semibold text-good whitespace-nowrap"
          >
            ✓ {v.label} ready
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function CarBody({ color, charging, done }: { color: string; charging: boolean; done: boolean }) {
  return (
    <div className="relative" style={{ width: CAR_W, height: CAR_H }}>
      <svg viewBox="0 0 110 60" width={CAR_W} height={CAR_H} style={{ overflow: "visible" }}>
        {/* shadow */}
        <ellipse cx="55" cy="56" rx="46" ry="4.5" fill="rgba(0,0,0,0.55)" />
        {/* body */}
        <rect x="10" y="13" width="90" height="36" rx="11" fill={color} />
        <rect x="10" y="13" width="90" height="36" rx="11" fill="url(#carGrad)" opacity="0.4" />
        {/* hood / trunk seams */}
        <line x1="22" y1="13" x2="22" y2="49" stroke="rgba(0,0,0,0.18)" strokeWidth="0.8" />
        <line x1="88" y1="13" x2="88" y2="49" stroke="rgba(0,0,0,0.18)" strokeWidth="0.8" />
        {/* roof / cabin */}
        <rect x="28" y="17" width="54" height="28" rx="6" fill="#0b0f1a" opacity="0.78" />
        {/* windshield + rear window */}
        <rect x="30" y="19" width="50" height="9" rx="2.5" fill="#22d3ee" opacity="0.28" />
        <rect x="30" y="34" width="50" height="9" rx="2.5" fill="#22d3ee" opacity="0.18" />
        {/* center pillar */}
        <line x1="55" y1="18" x2="55" y2="44" stroke="#0b0f1a" strokeWidth="0.6" opacity="0.6" />
        {/* wheels */}
        <rect x="20" y="44" width="12" height="6" rx="1.5" fill="#0b0f1a" stroke="#374151" strokeWidth="0.8" />
        <rect x="78" y="44" width="12" height="6" rx="1.5" fill="#0b0f1a" stroke="#374151" strokeWidth="0.8" />
        <rect x="20" y="12" width="12" height="6" rx="1.5" fill="#0b0f1a" stroke="#374151" strokeWidth="0.8" />
        <rect x="78" y="12" width="12" height="6" rx="1.5" fill="#0b0f1a" stroke="#374151" strokeWidth="0.8" />
        {/* headlights (front = right side, since car drives right out of the lot) */}
        <rect x="99.5" y="22" width="3.5" height="4" fill="#fef3c7" rx="0.8" />
        <rect x="99.5" y="36" width="3.5" height="4" fill="#fef3c7" rx="0.8" />
        {/* tail lights (left side) */}
        <rect x="7" y="22" width="3" height="4" fill="#7f1d1d" rx="0.8" />
        <rect x="7" y="36" width="3" height="4" fill="#7f1d1d" rx="0.8" />
        {/* charging port (left side, mid) */}
        <rect x="6" y="28" width="4" height="6" rx="1" fill="#0b0f1a" stroke={charging ? "#22d3ee" : "#374151"} strokeWidth="0.8" />
        <defs>
          <linearGradient id="carGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#fff" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#000" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>
      {done && (
        <div
          className="absolute"
          style={{
            left: -8,
            top: CAR_H / 2 - 5,
            width: 10,
            height: 10,
            borderRadius: 999,
            background: "#34d399",
            boxShadow: "0 0 10px #34d399",
          }}
        />
      )}
    </div>
  );
}

/* ---------------- Animated charging cable ----------------
   Drawn as an SVG positioned absolutely inside the VehicleSprite,
   stretching from the charger pedestal (just left of the car) to
   the car's charging port. Two layers:
     1. cable casing (thick, semi-transparent, dark)
     2. energy stream (thin, bright, dashed, animated stroke offset)
   Width and energy-flow speed scale with currentDraw / DC_MAX so
   the policy's actual allocation is visible at a glance. Includes
   a soft glow LED ring on the pedestal head reflecting active state.
---------------------------------------------------------- */
function ChargingCable({
  chargerType,
  currentDraw,
  color,
}: {
  chargerType: "AC" | "DC";
  currentDraw: number;
  color: string;
}) {
  // Cable spans from pedestal head (~x = -16) to car port (~x = 4),
  // mid-height vertically (CAR_H / 2 = 30).
  const PORT_X = 4;
  const PORT_Y = CAR_H / 2;
  const PED_X = -16;
  const PED_Y = CAR_H / 2;
  // Curved cable path with a downward droop in the middle.
  const MID_X = (PORT_X + PED_X) / 2;
  const DROOP = 6;
  const path = `M ${PED_X} ${PED_Y} Q ${MID_X} ${PED_Y + DROOP} ${PORT_X} ${PORT_Y}`;

  // Visual scaling.
  const norm = Math.min(1, Math.max(0, currentDraw / DC_MAX));
  const flowing = currentDraw > 0.5;
  const energyColor = chargerType === "DC" ? "#fbbf24" : "#22d3ee";
  // Animation duration in seconds — faster = more energy flowing.
  const flowDuration = flowing ? Math.max(0.45, 1.6 - norm * 1.1) : 0;
  const energyWidth = 1.4 + norm * 1.8;
  const energyOpacity = 0.45 + norm * 0.5;

  // SVG covers the area from x=-26 to x=10, y=20 to y=46.
  // Use a viewBox with negative coords by translating in the SVG transform.
  const SVG_W = 36;
  const SVG_H = 28;
  const OFFSET_X = -26;
  const OFFSET_Y = 16;

  return (
    <svg
      className="absolute pointer-events-none"
      width={SVG_W}
      height={SVG_H}
      style={{ left: OFFSET_X, top: OFFSET_Y, overflow: "visible" }}
      viewBox={`${OFFSET_X} ${OFFSET_Y} ${SVG_W} ${SVG_H}`}
    >
      {/* cable casing */}
      <path
        d={path}
        stroke="#0b0f1a"
        strokeOpacity={0.95}
        strokeWidth={4.2}
        strokeLinecap="round"
        fill="none"
      />
      <path
        d={path}
        stroke="#1f2937"
        strokeOpacity={0.9}
        strokeWidth={2.8}
        strokeLinecap="round"
        fill="none"
      />
      {/* energy glow */}
      {flowing && (
        <path
          d={path}
          stroke={energyColor}
          strokeOpacity={energyOpacity * 0.4}
          strokeWidth={energyWidth + 3}
          strokeLinecap="round"
          fill="none"
          style={{ filter: `blur(2px)` }}
        />
      )}
      {/* energy stream (animated dash flowing toward the car) */}
      {flowing && (
        <path
          d={path}
          stroke={energyColor}
          strokeOpacity={energyOpacity}
          strokeWidth={energyWidth}
          strokeLinecap="round"
          fill="none"
          strokeDasharray="3 5"
        >
          <animate
            attributeName="stroke-dashoffset"
            from="16"
            to="0"
            dur={`${flowDuration}s`}
            repeatCount="indefinite"
          />
        </path>
      )}
      {/* active LED ring on top of the pedestal */}
      <circle
        cx={PED_X - 6}
        cy={PED_Y - 17}
        r={3.4}
        fill="none"
        stroke={energyColor}
        strokeOpacity={0.85}
        strokeWidth={1.1}
      />
      <circle cx={PED_X - 6} cy={PED_Y - 17} r={1.7} fill={energyColor}>
        <animate
          attributeName="opacity"
          values={flowing ? "0.55;1;0.55" : "0.3;0.5;0.3"}
          dur={flowing ? `${Math.max(0.5, flowDuration)}s` : "2.4s"}
          repeatCount="indefinite"
        />
      </circle>
      {/* car port plug glow */}
      {flowing && (
        <circle
          cx={PORT_X - 1}
          cy={PORT_Y}
          r={2.2}
          fill={color}
          opacity={0.6}
          style={{ filter: `drop-shadow(0 0 4px ${energyColor})` }}
        />
      )}
    </svg>
  );
}

function InfoCard({ v, maxGrid }: { v: Vehicle; maxGrid: number }) {
  const prioStyle: Record<Priority, string> = {
    VIP: "bg-accent/20 text-accent border-accent/40",
    Standard: "bg-white/5 text-gray-300 border-white/10",
    Low: "bg-white/5 text-gray-500 border-white/10",
  };
  return (
    <div
      className="glass rounded-lg p-2 text-[10px] leading-tight"
      style={{ borderColor: v.color + "55" }}
    >
      <div className="flex items-center justify-between gap-1">
        <span className="font-mono text-white">{v.label}</span>
        <div className="flex items-center gap-1">
          <span
            className={`text-[8px] uppercase tracking-wider px-1.5 py-0.5 rounded border ${prioStyle[v.priority]}`}
          >
            {v.priority}
          </span>
          <span
            className={`text-[8px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded border ${
              v.chargerType === "DC"
                ? "bg-accent2/15 text-accent2 border-accent2/40"
                : "bg-white/5 text-gray-300 border-white/15"
            }`}
            title={
              v.chargerType === "DC"
                ? `DC fast · latched at ${v.latchedCurrent?.toFixed(0) ?? "—"} A`
                : `AC · ${AC_MIN}–${AC_MAX} A, variable`
            }
          >
            {v.chargerType}
          </span>
        </div>
      </div>
      <div className="mt-1">
        <div className="flex justify-between text-gray-400 tabular-nums">
          <span>
            SoC{v.soc > 0.8 && <span className="text-warn ml-1">· CV</span>}
          </span>
          <span>{(v.soc * 100).toFixed(0)}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
          <motion.div
            className="h-full"
            style={{
              background: `linear-gradient(90deg, ${v.color}44, ${v.color})`,
            }}
            animate={{ width: `${v.soc * 100}%` }}
            transition={{ duration: 0.4 }}
          />
        </div>
      </div>
      <div className="mt-1">
        <div className="flex justify-between text-gray-400 tabular-nums">
          <span>Current</span>
          <span>{v.currentDraw.toFixed(0)} A</span>
        </div>
        <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-accent to-accent2"
            animate={{ width: `${Math.min(100, (v.currentDraw / maxGrid) * 100)}%` }}
            transition={{ duration: 0.4 }}
          />
        </div>
      </div>
      <div className="mt-1 flex justify-between text-[9px] text-gray-500">
        <span>⏱ {formatTime(v.deadline)}</span>
        <span>{v.capacityKwh} kWh</span>
      </div>
    </div>
  );
}

function ChartCard({
  title, subtitle, children,
}: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl bg-bg/60 border border-white/5 p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm text-gray-300">{title}</div>
        {subtitle && <div className="text-[10px] font-mono text-gray-500">{subtitle}</div>}
      </div>
      <div className="h-40">{children}</div>
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
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span className="tabular-nums text-white">
          {fmt ? fmt(value) : step < 1 ? value.toFixed(1) : value.toFixed(0)}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent"
      />
    </div>
  );
}

function formatTime(sec: number) {
  if (sec <= 0) return "0m";
  const m = Math.floor(sec / 60);
  if (m >= 60) return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}`;
  return `${m}m`;
}

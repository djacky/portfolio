"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
} from "react";
import { createPortal } from "react-dom";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

const COLOR_ROD = new THREE.Color("#22d3ee");
const COLOR_BOB = new THREE.Color("#7c5cff");
const COLOR_GOLD = new THREE.Color("#fbbf24");
const COLOR_JOINT = new THREE.Color("#a0aec0");
const HEAT_COLD = new THREE.Color("#3b82f6");
const HEAT_HOT = new THREE.Color("#ef4444");

const PEND_OFFSET_X = 2.3;
const PEND_OFFSET_Y = 0.6;

type PendubotMode = "full" | "partial";
const PENDUBOT_MODE: PendubotMode = "full";

/* ================================================================
   Physics — double pendulum (pendubot)
   Only joint 1 is actuated. θ=0 is upright (unstable), θ=π hanging.
   Lagrangian EOM with mass matrix inversion, RK4 integration.
================================================================ */
const G_PHYS = 10;
const M1 = 1, M2 = 0.3;
const L1 = 0.6, L2 = 0.6;
const LC1 = L1 / 2, LC2 = L2 / 2;
const I1 = M1 * L1 * L1 / 12, I2 = M2 * L2 * L2 / 12;
const MAX_TORQUE = PENDUBOT_MODE == 'full' ? 20 : 10;
const DT = 0.05;
const MAX_SPEED = 15;
const DAMP1 = 0.02;
const DAMP2 = 0.005;
const EP_LEN = 200;

const _D1 = M1 * LC1 * LC1 + I1 + M2 * L1 * L1;
const _D2 = M2 * LC2 * LC2 + I2;
const _H = M2 * L1 * LC2;
const _G1 = (M1 * LC1 + M2 * L1) * G_PHYS;
const _G2 = M2 * LC2 * G_PHYS;

type PendState = { t1: number; w1: number; t2: number; w2: number };

function wrap(a: number): number {
  let x = (a + Math.PI) % (2 * Math.PI);
  if (x < 0) x += 2 * Math.PI;
  return x - Math.PI;
}

function pendDeriv(t1: number, w1: number, t2: number, w2: number, u: number): [number, number, number, number] {
  const d = t1 - t2, cd = Math.cos(d), sd = Math.sin(d);
  const relW = w2 - w1;
  const f1 = u - DAMP1 * w1 + DAMP2 * relW - _H * sd * w2 * w2 + _G1 * Math.sin(t1);
  const f2 = -DAMP2 * relW + _H * sd * w1 * w1 + _G2 * Math.sin(t2);
  const det = _D1 * _D2 - _H * _H * cd * cd;
  return [w1, (_D2 * f1 - _H * cd * f2) / det, w2, (_D1 * f2 - _H * cd * f1) / det];
}

function stepPend(s: PendState, u: number): PendState {
  const { t1, w1, t2, w2 } = s;
  const k1 = pendDeriv(t1, w1, t2, w2, u);
  const k2 = pendDeriv(t1 + k1[0] * DT / 2, w1 + k1[1] * DT / 2, t2 + k1[2] * DT / 2, w2 + k1[3] * DT / 2, u);
  const k3 = pendDeriv(t1 + k2[0] * DT / 2, w1 + k2[1] * DT / 2, t2 + k2[2] * DT / 2, w2 + k2[3] * DT / 2, u);
  const k4 = pendDeriv(t1 + k3[0] * DT, w1 + k3[1] * DT, t2 + k3[2] * DT, w2 + k3[3] * DT, u);
  return {
    t1: t1 + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * DT / 6,
    w1: Math.max(-MAX_SPEED, Math.min(MAX_SPEED, w1 + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * DT / 6)),
    t2: t2 + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * DT / 6,
    w2: Math.max(-MAX_SPEED, Math.min(MAX_SPEED, w2 + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * DT / 6)),
  };
}

function getObs(s: PendState): Float64Array {
  return Float64Array.of(
    Math.cos(s.t1), Math.sin(s.t1), s.w1 / MAX_SPEED,
    Math.cos(s.t2), Math.sin(s.t2), s.w2 / MAX_SPEED,
  );
}

function pendReward(s: PendState, u: number): number {
    const upright = PENDUBOT_MODE === "full"
      ? (Math.cos(s.t1) + 1) / 2 * (Math.cos(s.t2) + 1) / 2
      : (-Math.cos(s.t1) + 1) / 2 * (Math.cos(s.t2) + 1) / 2;

    // u·ω₁ > 0  ⇔ torque pushes the same direction joint 1 is already moving (pumping energy).
    // max(0, …)  ⇒ only pumping is penalized; braking (u·ω₁ < 0) is free.
    // upright²   ⇒ near-zero during swing-up, strong at the top → swing-up corridor preserved.
    const pushWithMotion = Math.max(0, u * s.w1);
    const brakeTerm = 0.01 * upright * upright * pushWithMotion;

    return upright
         - 0.002 * (s.w1 * s.w1 + s.w2 * s.w2)   // ← revert to uniform
         - 0.001 * u * u
         - brakeTerm;
}

function inverseDynJ1(prev: PendState, next: PendState, dt: number): number {
  const alpha1 = (next.w1 - prev.w1) / dt;
  const alpha2 = (next.w2 - prev.w2) / dt;
  const d = prev.t1 - prev.t2, cd = Math.cos(d), sd = Math.sin(d);
  const relW = prev.w2 - prev.w1;
  const rest = -DAMP1 * prev.w1 + DAMP2 * relW - _H * sd * prev.w2 * prev.w2 + _G1 * Math.sin(prev.t1);
  return _D1 * alpha1 + _H * cd * alpha2 - rest;
}

function nearestActionIdx(torque: number): number {
  const c = Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, torque));
  return Math.round((c + MAX_TORQUE) * (N_ACTIONS - 1) / (2 * MAX_TORQUE));
}

function invertedHeat(s: PendState): number {
  return PENDUBOT_MODE === "full"
    ? (Math.cos(s.t1) + 1) / 2 * (Math.cos(s.t2) + 1) / 2
    : (-Math.cos(s.t1) + 1) / 2 * (Math.cos(s.t2) + 1) / 2;
}

/* ================================================================
   PRNG
================================================================ */
function mulberry32(seed: number) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = seed;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function seededGauss(rng: () => number): number {
  return Math.sqrt(-2 * Math.log(rng() + 1e-10)) * Math.cos(2 * Math.PI * rng());
}

/* ================================================================
   DQN — Deep Q-Network for pendubot balance
   Network: [6] → 64 → ReLU → 64 → ReLU → 21 Q-values
   Learns online from actual visible pendulum transitions.
================================================================ */
const OBS_DIM = 6;
const N_ACTIONS = PENDUBOT_MODE == 'full' ? 21 : 11;
const ACTION_TORQUES = Float64Array.from(
  { length: N_ACTIONS }, (_, i) => -MAX_TORQUE + (2 * MAX_TORQUE / (N_ACTIONS - 1)) * i,
);
const DQN_HIDDEN = 64;
const REPLAY_CAP = 50000;
const BATCH_SIZE = 32;
const UPDATES_PER_STEP = 16;
const DQN_LR = PENDUBOT_MODE == 'full' ? 0.003 : 0.03;
const GAMMA = 0.99;
const EPS_START = 1.0;
const EPS_END = 0.01;
const EPS_DECAY = 3000;
const TARGET_TAU = 0.008;
const MIN_REPLAY = 200;
const REWARD_SCALE = 1.0;
const CONVERGE_FRAC = 0.50;
const CONVERGE_HOLD = 3;
const MIN_CONVERGE_STEPS = 1000;
const PLOT_WINDOW = 40;
/** Warm-start Gaussian perturbation applied to loaded weights. Small
 *  enough that the policy still nails balance from equilibrium, large
 *  enough that the HUD shows a visible ~20–30s stabilization before
 *  convergence fires (3 consecutive episodes ≥ CONVERGE_FRAC). */
const WARM_START_SIGMA = 0.02;
/** Master switch for the bestNet snapshot. When false, the agent doesn't
 *  update bestNet on new-best episodes and skips the warm-start seed.
 *  Turn off during a reward-function transition so a stale-reward snapshot
 *  can't be captured; re-enable once the new reward has stabilised. */
const BESTNET_GUARD_ENABLED = true;

type TrainStats = {
  updates: number;
  meanReward: number;
  uprightFrac: number;
  bestUpright: number;
  epsilon: number;
  converged: boolean;
  trainingActive: boolean;
};

const INIT_STATS: TrainStats = {
  updates: 0, meanReward: 0, uprightFrac: 0, bestUpright: 0,
  epsilon: 1, converged: false, trainingActive: false,
};

class DQNNet {
  params: Float64Array[];
  private grads: Float64Array[];
  private adam_m: Float64Array[];
  private adam_v: Float64Array[];
  private h1 = new Float64Array(DQN_HIDDEN);
  private h2 = new Float64Array(DQN_HIDDEN);
  out = new Float64Array(N_ACTIONS);
  private h1p = new Float64Array(DQN_HIDDEN);
  private h2p = new Float64Array(DQN_HIDDEN);
  private oC = new Float64Array(OBS_DIM);
  private dh2 = new Float64Array(DQN_HIDDEN);
  private dh1 = new Float64Array(DQN_HIDDEN);

  constructor(rng: () => number) {
    const S = [DQN_HIDDEN * OBS_DIM, DQN_HIDDEN, DQN_HIDDEN * DQN_HIDDEN, DQN_HIDDEN, N_ACTIONS * DQN_HIDDEN, N_ACTIONS];
    this.params = S.map(n => new Float64Array(n));
    this.grads = S.map(n => new Float64Array(n));
    this.adam_m = S.map(n => new Float64Array(n));
    this.adam_v = S.map(n => new Float64Array(n));
    for (let i = 0; i < this.params[0].length; i++) this.params[0][i] = seededGauss(rng) * Math.sqrt(2 / OBS_DIM);
    for (let i = 0; i < this.params[2].length; i++) this.params[2][i] = seededGauss(rng) * Math.sqrt(2 / DQN_HIDDEN);
    for (let i = 0; i < this.params[4].length; i++) this.params[4][i] = seededGauss(rng) * 0.01;
  }

  forward(o: ArrayLike<number>): Float64Array {
    const [W1, b1, W2, b2, Wo, bo] = this.params;
    for (let i = 0; i < OBS_DIM; i++) this.oC[i] = o[i];
    for (let i = 0; i < DQN_HIDDEN; i++) {
      let s = b1[i]; for (let j = 0; j < OBS_DIM; j++) s += W1[i * OBS_DIM + j] * o[j];
      this.h1p[i] = s; this.h1[i] = s > 0 ? s : 0;
    }
    for (let i = 0; i < DQN_HIDDEN; i++) {
      let s = b2[i]; for (let j = 0; j < DQN_HIDDEN; j++) s += W2[i * DQN_HIDDEN + j] * this.h1[j];
      this.h2p[i] = s; this.h2[i] = s > 0 ? s : 0;
    }
    for (let k = 0; k < N_ACTIONS; k++) {
      let s = bo[k]; for (let j = 0; j < DQN_HIDDEN; j++) s += Wo[k * DQN_HIDDEN + j] * this.h2[j];
      this.out[k] = s;
    }
    return this.out;
  }

  backward(action: number, target: number) {
    const Wo = this.params[4], W2 = this.params[2];
    const [gW1, gb1, gW2, gb2, gWo, gbo] = this.grads;
    const g = 2 * (this.out[action] - target);
    this.dh2.fill(0);
    gbo[action] += g;
    for (let j = 0; j < DQN_HIDDEN; j++) {
      gWo[action * DQN_HIDDEN + j] += g * this.h2[j];
      this.dh2[j] = g * Wo[action * DQN_HIDDEN + j];
    }
    for (let j = 0; j < DQN_HIDDEN; j++) if (this.h2p[j] <= 0) this.dh2[j] = 0;
    this.dh1.fill(0);
    for (let i = 0; i < DQN_HIDDEN; i++) {
      if (this.dh2[i] === 0) continue;
      gb2[i] += this.dh2[i];
      for (let j = 0; j < DQN_HIDDEN; j++) {
        gW2[i * DQN_HIDDEN + j] += this.dh2[i] * this.h1[j];
        this.dh1[j] += this.dh2[i] * W2[i * DQN_HIDDEN + j];
      }
    }
    for (let j = 0; j < DQN_HIDDEN; j++) if (this.h1p[j] <= 0) this.dh1[j] = 0;
    for (let i = 0; i < DQN_HIDDEN; i++) {
      if (this.dh1[i] === 0) continue;
      gb1[i] += this.dh1[i];
      for (let j = 0; j < OBS_DIM; j++) gW1[i * OBS_DIM + j] += this.dh1[i] * this.oC[j];
    }
  }

  zeroGrad() { for (const g of this.grads) g.fill(0); }

  adamStep(lr: number, t: number) {
    const B1 = 0.9, B2 = 0.999, eps = 1e-8;
    const bc1 = 1 - Math.pow(B1, t), bc2 = 1 - Math.pow(B2, t);
    for (let p = 0; p < this.params.length; p++) {
      const P = this.params[p], G = this.grads[p], m = this.adam_m[p], v = this.adam_v[p];
      for (let i = 0; i < P.length; i++) {
        m[i] = B1 * m[i] + (1 - B1) * G[i];
        v[i] = B2 * v[i] + (1 - B2) * G[i] * G[i];
        P[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
      }
    }
  }

  copyFrom(other: DQNNet) {
    for (let p = 0; p < this.params.length; p++) this.params[p].set(other.params[p]);
  }

  exportParams(): number[][] {
    // Full f64 precision — any rounding here lossy-compresses the policy
    // and can flip argmax at tight decision boundaries (especially swing-
    // up states where the margin between adjacent torque bins is small).
    return this.params.map((p) => Array.from(p));
  }

  loadParams(data: number[][]): boolean {
    if (!Array.isArray(data) || data.length !== this.params.length) return false;
    for (let p = 0; p < this.params.length; p++) {
      if (!Array.isArray(data[p]) || data[p].length !== this.params[p].length) return false;
    }
    for (let p = 0; p < this.params.length; p++) {
      const dst = this.params[p], src = data[p];
      for (let i = 0; i < dst.length; i++) dst[i] = src[i];
    }
    return true;
  }

  perturb(sigma: number, rng: () => number) {
    for (const p of this.params) {
      for (let i = 0; i < p.length; i++) p[i] += seededGauss(rng) * sigma;
    }
  }

  softUpdate(src: DQNNet, tau: number) {
    for (let p = 0; p < this.params.length; p++) {
      const t = this.params[p], s = src.params[p];
      for (let i = 0; i < t.length; i++) t[i] = tau * s[i] + (1 - tau) * t[i];
    }
  }
}

class DQNAgent {
  private onlineNet: DQNNet;
  private targetNet: DQNNet;
  private replay: Float64Array;
  private replaySize = 0;
  private replayPos = 0;
  private readonly stride = OBS_DIM + 1 + 1 + OBS_DIM;
  private rng: () => number;
  epsilon = EPS_START;
  private totalSteps = 0;
  private adamT = 0;
  private lastActionIdx = 0;

  rewardHistory: number[] = [];
  uprightHistory: number[] = [];
  updates = 0;
  converged = false;
  bestUpright = 0;
  private convCount = 0;
  trainingActive = false;
  stats: TrainStats = { ...INIT_STATS };
  private bestNet: DQNNet | null = null;
  private bestUprightSnapshot = 0;

  private visSteps = 0;
  private visRewardAcc = 0;
  private visUprightAcc = 0;
  private visRewardWindow: number[] = [];
  private visUprightWindow: number[] = [];

  plotRewardHistory: number[] = [];
  plotUprightHistory: number[] = [];
  plotEpsilonHistory: number[] = [];
  private plotSteps = 0;
  private plotRewardAcc = 0;
  private plotUprightAcc = 0;

  constructor() {
    const initRng = mulberry32(42);
    this.rng = mulberry32(123);
    this.onlineNet = new DQNNet(initRng);
    this.targetNet = new DQNNet(() => 0);
    this.targetNet.copyFrom(this.onlineNet);
    this.replay = new Float64Array(REPLAY_CAP * this.stride);
  }

  injectDemo(o: Float64Array, a: number, r: number, n: Float64Array) {
    this.pushReplay(o, a, r, n);
  }

  exportReplay(maxSamples = 10000): Array<{ o: number[]; a: number; r: number; n: number[] }> {
    const size = this.replaySize;
    if (size === 0) return [];
    const total = Math.min(maxSamples, size);
    const step = size / total;
    const round = (x: number) => Math.round(x * 1e5) / 1e5;
    const out: Array<{ o: number[]; a: number; r: number; n: number[] }> = new Array(total);
    for (let i = 0; i < total; i++) {
      const idx = Math.floor(i * step) % size;
      const off = idx * this.stride;
      const o = new Array<number>(OBS_DIM);
      const n = new Array<number>(OBS_DIM);
      for (let k = 0; k < OBS_DIM; k++) {
        o[k] = round(this.replay[off + k]);
        n[k] = round(this.replay[off + OBS_DIM + 2 + k]);
      }
      out[i] = {
        o,
        a: this.replay[off + OBS_DIM],
        r: round(this.replay[off + OBS_DIM + 1]),
        n,
      };
    }
    return out;
  }

  getReplaySize(): number { return this.replaySize; }

  async injectBatch(
    transitions: ReadonlyArray<{ o: number[]; a: number; r: number; n: number[] }>,
    opts: { chunkSize?: number; onProgress?: (done: number, total: number) => void } = {},
  ): Promise<void> {
    const chunkSize = opts.chunkSize ?? 300;
    const total = transitions.length;
    const oBuf = new Float64Array(OBS_DIM);
    const nBuf = new Float64Array(OBS_DIM);
    for (let i = 0; i < total; i += chunkSize) {
      const end = Math.min(i + chunkSize, total);
      for (let j = i; j < end; j++) {
        const t = transitions[j];
        for (let k = 0; k < OBS_DIM; k++) { oBuf[k] = t.o[k]; nBuf[k] = t.n[k]; }
        this.pushReplay(oBuf, t.a, t.r * REWARD_SCALE, nBuf);
      }
      opts.onProgress?.(end, total);
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
    }
  }

  exportWeights(): { online: number[][]; target: number[][] } {
    return {
      online: this.onlineNet.exportParams(),
      target: this.targetNet.exportParams(),
    };
  }

  /** Load pre-trained weights with a small Gaussian perturbation, so the
   *  policy starts near-converged but has a brief visible stabilization
   *  window (~20–30s) before the HUD flips to converged. Normal online
   *  training runs post-load, pulling the slightly noisy weights back
   *  onto the good optimum. ε is parked at EPS_END (greedy-ish) by
   *  fast-forwarding totalSteps past EPS_DECAY.                         */
  loadWeights(data: { online: number[][]; target: number[][] }): boolean {
    if (!data || !this.onlineNet.loadParams(data.online)) return false;
    if (!this.targetNet.loadParams(data.target)) return false;
    this.onlineNet.perturb(WARM_START_SIGMA, this.rng);
    this.targetNet.copyFrom(this.onlineNet);
    if (BESTNET_GUARD_ENABLED) {
      if (!this.bestNet) this.bestNet = new DQNNet(() => 0);
      this.bestNet.copyFrom(this.onlineNet);
      this.bestUprightSnapshot = CONVERGE_FRAC;
    }
    this.totalSteps = Math.max(this.totalSteps, EPS_DECAY + MIN_CONVERGE_STEPS);
    this.epsilon = EPS_END;
    return true;
  }

  async warmStartTrain(
    totalUpdates: number,
    opts: { updatesPerFrame?: number; onProgress?: (done: number, total: number) => void } = {},
  ): Promise<void> {
    if (this.replaySize < MIN_REPLAY) return;
    const perFrame = opts.updatesPerFrame ?? 8;
    let done = 0;
    while (done < totalUpdates) {
      const n = Math.min(perFrame, totalUpdates - done);
      for (let i = 0; i < n; i++) this.trainStep();
      this.targetNet.softUpdate(this.onlineNet, TARGET_TAU);
      done += n;
      opts.onProgress?.(done, totalUpdates);
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
    }
    this.totalSteps = Math.max(this.totalSteps, Math.round(EPS_DECAY * 0.7));
    this.epsilon = Math.max(EPS_END, EPS_START - (EPS_START - EPS_END) * this.totalSteps / EPS_DECAY);
  }

  private pushReplay(o: Float64Array, a: number, r: number, n: Float64Array) {
    const off = this.replayPos * this.stride;
    this.replay.set(o, off);
    this.replay[off + OBS_DIM] = a;
    this.replay[off + OBS_DIM + 1] = r;
    this.replay.set(n, off + OBS_DIM + 2);
    this.replayPos = (this.replayPos + 1) % REPLAY_CAP;
    if (this.replaySize < REPLAY_CAP) this.replaySize++;
  }

  act(obs: Float64Array, deterministic: boolean): { action: number } {
    if (!deterministic && this.rng() < this.epsilon) {
      this.lastActionIdx = Math.floor(this.rng() * N_ACTIONS);
    } else {
      const q = this.onlineNet.forward(obs);
      let best = 0;
      for (let i = 1; i < N_ACTIONS; i++) if (q[i] > q[best]) best = i;
      this.lastActionIdx = best;
    }
    return { action: ACTION_TORQUES[this.lastActionIdx] };
  }

  startTraining() { this.trainingActive = true; }

  recordVisibleStep(prev: PendState, u: number, next: PendState) {
    const rawR = pendReward(next, u);
    this.pushReplay(getObs(prev), this.lastActionIdx, rawR * REWARD_SCALE, getObs(next));
    this.totalSteps++;
    this.epsilon = Math.max(EPS_END, EPS_START - (EPS_START - EPS_END) * this.totalSteps / EPS_DECAY);

    if (this.replaySize >= MIN_REPLAY) {
      const ups = this.converged ? 2 : UPDATES_PER_STEP;
      for (let i = 0; i < ups; i++) this.trainStep();
      this.targetNet.softUpdate(this.onlineNet, TARGET_TAU);
    }

    const t1Goal = PENDUBOT_MODE === "full" ? 0 : Math.PI;
    const isUp = Math.abs(wrap(next.t1 - t1Goal)) < 0.3 && Math.abs(wrap(next.t2)) < 0.3
              && Math.abs(next.w1) < 2 && Math.abs(next.w2) < 2;

    this.plotRewardAcc += rawR;
    if (isUp) this.plotUprightAcc++;
    this.plotSteps++;
    if (this.plotSteps >= PLOT_WINDOW) {
      this.plotRewardHistory.push(this.plotRewardAcc / PLOT_WINDOW);
      this.plotUprightHistory.push(this.plotUprightAcc / PLOT_WINDOW);
      this.plotEpsilonHistory.push(this.epsilon);
      this.plotRewardAcc = 0;
      this.plotUprightAcc = 0;
      this.plotSteps = 0;
    }

    this.visRewardAcc += rawR;
    if (isUp) this.visUprightAcc++;
    this.visSteps++;
    if (this.visSteps >= EP_LEN) {
      const epR = this.visRewardAcc / EP_LEN;
      const epUp = this.visUprightAcc / EP_LEN;
      this.rewardHistory.push(epR);
      this.uprightHistory.push(epUp);
      this.bestUpright = Math.max(this.bestUpright, epUp);
      if (BESTNET_GUARD_ENABLED && epUp > this.bestUprightSnapshot && epUp >= CONVERGE_FRAC) {
        this.bestUprightSnapshot = epUp;
        if (!this.bestNet) this.bestNet = new DQNNet(() => 0);
        this.bestNet.copyFrom(this.onlineNet);
      }
      // bestNet is only snapshotted, never reloaded: on a post-convergence
      // collapse the snapshot is also OOD (hanging + high-ω is outside its
      // training distribution), so live gradient updates recover faster
      // than a reload would. Snapshot is kept for warm-start export only.
      this.visRewardWindow.push(epR);
      this.visUprightWindow.push(epUp);
      if (this.visRewardWindow.length > 10) this.visRewardWindow.shift();
      if (this.visUprightWindow.length > 10) this.visUprightWindow.shift();
      if (!this.converged && this.totalSteps >= MIN_CONVERGE_STEPS) {
        if (epUp >= CONVERGE_FRAC) this.convCount++;
        else this.convCount = Math.max(0, this.convCount - 1);
        if (this.convCount >= CONVERGE_HOLD) { this.converged = true; this.epsilon = 0; }
      }
      this.visRewardAcc = 0;
      this.visUprightAcc = 0;
      this.visSteps = 0;
    }
  }

  private trainStep() {
    this.onlineNet.zeroGrad();
    for (let i = 0; i < BATCH_SIZE; i++) {
      const idx = Math.floor(this.rng() * this.replaySize);
      const off = idx * this.stride;
      const sO = this.replay.subarray(off, off + OBS_DIM);
      const a = this.replay[off + OBS_DIM];
      const r = this.replay[off + OBS_DIM + 1];
      const nO = this.replay.subarray(off + OBS_DIM + 2, off + OBS_DIM + 2 + OBS_DIM);
      const tq = this.targetNet.forward(nO);
      let mQ = tq[0]; for (let k = 1; k < N_ACTIONS; k++) if (tq[k] > mQ) mQ = tq[k];
      this.onlineNet.forward(sO);
      this.onlineNet.backward(a, r + GAMMA * mQ);
    }
    this.adamT++;
    this.onlineNet.adamStep(DQN_LR / BATCH_SIZE, this.adamT);
    this.updates++;
  }

  resetVisibleEpisode() {
    this.visSteps = 0;
    this.visRewardAcc = 0;
    this.visUprightAcc = 0;
  }

  refreshStats() {
    const meanR = this.visRewardWindow.length > 0
      ? this.visRewardWindow.reduce((a, b) => a + b, 0) / this.visRewardWindow.length : 0;
    const uprFrac = this.visUprightWindow.length > 0
      ? this.visUprightWindow.reduce((a, b) => a + b, 0) / this.visUprightWindow.length : 0;
    this.stats = {
      updates: this.updates, meanReward: meanR, uprightFrac: uprFrac,
      bestUpright: this.bestUpright, epsilon: this.epsilon,
      converged: this.converged, trainingActive: this.trainingActive,
    };
  }
}

/* ================================================================
   Engine singleton — persists across Hero mount/unmount cycles.
   When the Canvas unmounts (user navigates away), a background
   setInterval continues physics + DQN training at ~20 Hz.
================================================================ */
type PendulumEngine = {
  agent: DQNAgent;
  env: PendState;
  trainingStarted: boolean;
  trainingStartTime: number;
  convergenceCounted: boolean;
  bgTimer: ReturnType<typeof setInterval> | null;
};

let _engine: PendulumEngine | null = null;

function getEngine(): PendulumEngine {
  if (!_engine) {
    _engine = {
      agent: new DQNAgent(),
      env: {
        t1: Math.PI + (Math.random() - 0.5) * 0.3, w1: 0,
        t2: Math.PI + (Math.random() - 0.5) * 0.3, w2: 0,
      },
      trainingStarted: false,
      trainingStartTime: 0,
      convergenceCounted: false,
      bgTimer: null,
    };
  }
  return _engine;
}

function startBackground(engine: PendulumEngine) {
  if (engine.bgTimer || !engine.agent.trainingActive) return;
  engine.bgTimer = setInterval(() => {
    const prev = { ...engine.env };
    const { action: torque } = engine.agent.act(getObs(prev), engine.agent.converged);
    engine.env = stepPend(prev, torque);
    engine.agent.recordVisibleStep(prev, torque, engine.env);
    engine.agent.refreshStats();
  }, 50);
}

function stopBackground(engine: PendulumEngine) {
  if (engine.bgTimer) {
    clearInterval(engine.bgTimer);
    engine.bgTimer = null;
  }
}

/* ================================================================
   Rendering helpers
================================================================ */
const _tmpMid = new THREE.Vector3();
const _tmpDir = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _q = new THREE.Quaternion();

function setRod(mesh: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3) {
  _tmpMid.addVectors(a, b).multiplyScalar(0.5);
  _tmpDir.subVectors(b, a);
  const len = _tmpDir.length();
  mesh.position.copy(_tmpMid);
  mesh.scale.set(1, len, 1);
  _tmpDir.normalize();
  _q.setFromUnitVectors(_up, _tmpDir);
  mesh.quaternion.copy(_q);
}

function joint1Pos(s: PendState): THREE.Vector3 {
  return new THREE.Vector3(L1 * Math.sin(s.t1), L1 * Math.cos(s.t1), 0);
}

function tipPos(s: PendState): THREE.Vector3 {
  return new THREE.Vector3(
    L1 * Math.sin(s.t1) + L2 * Math.sin(s.t2),
    L1 * Math.cos(s.t1) + L2 * Math.cos(s.t2),
    0,
  );
}

function solveIK(px: number, py: number, prevT1: number): { t1: number; t2: number } {
  const d2 = px * px + py * py;
  const d = Math.sqrt(d2);
  if (d >= L1 + L2) {
    const angle = Math.atan2(px, py);
    return { t1: angle, t2: angle };
  }
  if (d < 0.01) return { t1: prevT1, t2: prevT1 + Math.PI };
  const cosPhi = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2);
  const phi = Math.acos(Math.max(-1, Math.min(1, cosPhi)));
  const A = L1 + L2 * Math.cos(phi);
  const B = L2 * Math.sin(phi);
  const t1a = Math.atan2(A * px + B * py, A * py - B * px);
  const t2a = t1a - phi;
  const t1b = Math.atan2(A * px - B * py, A * py + B * px);
  const t2b = t1b + phi;
  return Math.abs(wrap(t1a - prevT1)) <= Math.abs(wrap(t1b - prevT1))
    ? { t1: t1a, t2: t2a }
    : { t1: t1b, t2: t2b };
}

function makeGlowTexture(): THREE.Texture {
  const size = 128;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.25, "rgba(255,255,255,0.55)");
  grad.addColorStop(0.55, "rgba(255,255,255,0.12)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

/* ================================================================
   PendulumMesh — double pendulum with heat indicator on tip bob
================================================================ */
function PendulumMesh({
  envRef,
  convergedRef,
  grabActiveRef,
}: {
  envRef: MutableRefObject<PendState>;
  convergedRef: MutableRefObject<boolean>;
  grabActiveRef: MutableRefObject<boolean>;
}) {
  const rod1Ref = useRef<THREE.Mesh>(null);
  const rod1CoreRef = useRef<THREE.Mesh>(null);
  const rod2Ref = useRef<THREE.Mesh>(null);
  const rod2CoreRef = useRef<THREE.Mesh>(null);
  const jointRef = useRef<THREE.Mesh>(null);
  const bobGroupRef = useRef<THREE.Group>(null);
  const bobMeshRef = useRef<THREE.Mesh>(null);
  const bobRingRef = useRef<THREE.Mesh>(null);
  const bobGlowRef = useRef<THREE.Sprite>(null);
  const trailRef = useRef<THREE.Vector3[]>([]);
  const trailGeoRef = useRef<THREE.BufferGeometry>(null);
  const goldRef = useRef(0);
  const smoothHeat = useRef(0);
  const glowTex = useMemo(() => makeGlowTexture(), []);
  const _heatC = useMemo(() => new THREE.Color(), []);
  const TRAIL_LEN = 72;

  useFrame((_, dt) => {
    const s = envRef.current;
    const pivot = new THREE.Vector3(0, 0, 0);
    const j1 = joint1Pos(s);
    const tip = tipPos(s);

    if (rod1Ref.current) setRod(rod1Ref.current, pivot, j1);
    if (rod1CoreRef.current) setRod(rod1CoreRef.current, pivot, j1);
    if (rod2Ref.current) setRod(rod2Ref.current, j1, tip);
    if (rod2CoreRef.current) setRod(rod2CoreRef.current, j1, tip);
    if (jointRef.current) jointRef.current.position.copy(j1);
    if (bobGroupRef.current) bobGroupRef.current.position.copy(tip);
    if (bobRingRef.current) bobRingRef.current.rotation.z -= dt * 0.5;

    const goldTarget = convergedRef.current ? 1 : 0;
    const k = 1 - Math.exp(-dt * 2.2);
    goldRef.current += (goldTarget - goldRef.current) * k;
    const g = goldRef.current;
    const pulse = 0.9 + Math.sin(performance.now() * 0.0042) * 0.18;

    const grabbing = grabActiveRef.current;
    const rawHeat = invertedHeat(s);
    smoothHeat.current += (rawHeat - smoothHeat.current) * Math.min(1, dt * 8);
    const heat = smoothHeat.current;
    const heatPulse = 0.85 + Math.sin(performance.now() * 0.005) * 0.25;

    const lerpMat = (mesh: THREE.Mesh | null, base: THREE.Color, emI: number) => {
      if (!mesh) return;
      const mat = mesh.material as THREE.MeshPhysicalMaterial;
      mat.color.copy(base).lerp(COLOR_GOLD, g);
      mat.emissive.copy(base).lerp(COLOR_GOLD, g);
      mat.emissiveIntensity = emI + g * 0.6 * pulse;
    };
    lerpMat(rod1Ref.current, COLOR_ROD, 0.35);
    lerpMat(rod2Ref.current, COLOR_ROD, 0.35);

    if (rod1CoreRef.current)
      (rod1CoreRef.current.material as THREE.MeshBasicMaterial).color.copy(COLOR_ROD).lerp(COLOR_GOLD, g);
    if (rod2CoreRef.current)
      (rod2CoreRef.current.material as THREE.MeshBasicMaterial).color.copy(COLOR_ROD).lerp(COLOR_GOLD, g);

    if (jointRef.current) {
      const jm = jointRef.current.material as THREE.MeshPhysicalMaterial;
      jm.color.copy(COLOR_JOINT).lerp(COLOR_GOLD, g);
      jm.emissive.copy(COLOR_JOINT).lerp(COLOR_GOLD, g);
      jm.emissiveIntensity = 0.15 + g * 0.4;
    }

    if (bobMeshRef.current) {
      const mat = bobMeshRef.current.material as THREE.MeshPhysicalMaterial;
      if (grabbing && !convergedRef.current) {
        _heatC.copy(HEAT_COLD).lerp(HEAT_HOT, heat);
        mat.color.copy(_heatC);
        mat.emissive.copy(_heatC);
        mat.emissiveIntensity = 1.6 * heatPulse;
      } else {
        mat.color.copy(COLOR_BOB).lerp(COLOR_GOLD, g);
        mat.emissive.copy(COLOR_BOB).lerp(COLOR_GOLD, g);
        mat.emissiveIntensity = 1.4 + g * 0.6 * pulse;
      }
    }

    if (bobGlowRef.current) {
      const gm = bobGlowRef.current.material as THREE.SpriteMaterial;
      if (grabbing && !convergedRef.current) {
        _heatC.copy(HEAT_COLD).lerp(HEAT_HOT, heat);
        gm.color.copy(_heatC);
        bobGlowRef.current.scale.setScalar(1.4 * heatPulse);
      } else {
        gm.color.copy(COLOR_BOB).lerp(COLOR_GOLD, g);
        bobGlowRef.current.scale.setScalar(1.2 * (1 + Math.sin(performance.now() * 0.0032) * 0.06));
      }
    }

    trailRef.current.push(tip.clone());
    if (trailRef.current.length > TRAIL_LEN) trailRef.current.shift();
    const geo = trailGeoRef.current;
    if (geo) {
      const n = trailRef.current.length;
      const pos = new Float32Array(n * 3);
      const col = new Float32Array(n * 3);
      const baseC = new THREE.Color().copy(COLOR_BOB).lerp(COLOR_GOLD, g);
      for (let i = 0; i < n; i++) {
        const v = trailRef.current[i];
        pos[i * 3] = v.x; pos[i * 3 + 1] = v.y; pos[i * 3 + 2] = v.z;
        const t = i / Math.max(1, n - 1);
        col[i * 3] = baseC.r * t; col[i * 3 + 1] = baseC.g * t; col[i * 3 + 2] = baseC.b * t;
      }
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      geo.setDrawRange(0, n);
      geo.computeBoundingSphere();
    }
  });

  return (
    <group>
      {/* stepped mounting base */}
      <mesh position={[0, -0.02, -0.05]}>
        <cylinderGeometry args={[0.28, 0.34, 0.06, 32]} />
        <meshPhysicalMaterial color="#0b0f1a" metalness={0.85} roughness={0.28} />
      </mesh>
      <mesh position={[0, 0.03, -0.05]}>
        <cylinderGeometry args={[0.2, 0.22, 0.08, 32]} />
        <meshPhysicalMaterial color="#1a1f2e" metalness={0.9} roughness={0.2} clearcoat={0.6} clearcoatRoughness={0.2} emissive="#22d3ee" emissiveIntensity={0.12} />
      </mesh>
      <mesh position={[0, 0.07, -0.05]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.215, 0.008, 12, 48]} />
        <meshBasicMaterial color="#22d3ee" />
      </mesh>
      {/* pivot bolt */}
      <mesh>
        <sphereGeometry args={[0.09, 24, 24]} />
        <meshPhysicalMaterial color="#e8ecf5" metalness={1} roughness={0.15} clearcoat={1} clearcoatRoughness={0.1} emissive="#ffffff" emissiveIntensity={0.35} />
      </mesh>
      {/* rod 1 */}
      <mesh ref={rod1Ref}>
        <cylinderGeometry args={[0.045, 0.055, 1, 18]} />
        <meshPhysicalMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={0.35} metalness={0.6} roughness={0.2} clearcoat={0.9} clearcoatRoughness={0.15} />
      </mesh>
      <mesh ref={rod1CoreRef}>
        <cylinderGeometry args={[0.016, 0.016, 1.01, 10]} />
        <meshBasicMaterial color="#22d3ee" toneMapped={false} />
      </mesh>
      {/* joint between links */}
      <mesh ref={jointRef}>
        <sphereGeometry args={[0.065, 20, 20]} />
        <meshPhysicalMaterial color="#a0aec0" metalness={0.95} roughness={0.15} clearcoat={1} clearcoatRoughness={0.1} emissive="#a0aec0" emissiveIntensity={0.15} />
      </mesh>
      {/* rod 2 */}
      <mesh ref={rod2Ref}>
        <cylinderGeometry args={[0.045, 0.055, 1, 18]} />
        <meshPhysicalMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={0.35} metalness={0.6} roughness={0.2} clearcoat={0.9} clearcoatRoughness={0.15} />
      </mesh>
      <mesh ref={rod2CoreRef}>
        <cylinderGeometry args={[0.016, 0.016, 1.01, 10]} />
        <meshBasicMaterial color="#22d3ee" toneMapped={false} />
      </mesh>
      {/* tip bob */}
      <group ref={bobGroupRef}>
        <sprite ref={bobGlowRef}>
          <spriteMaterial map={glowTex} color="#7c5cff" transparent depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
        </sprite>
        <mesh ref={bobRingRef}>
          <torusGeometry args={[0.17, 0.008, 12, 48]} />
          <meshBasicMaterial color="#7c5cff" toneMapped={false} />
        </mesh>
        <mesh ref={bobMeshRef}>
          <sphereGeometry args={[0.12, 28, 28]} />
          <meshPhysicalMaterial color="#7c5cff" emissive="#7c5cff" emissiveIntensity={1.4} metalness={0.7} roughness={0.15} clearcoat={1} clearcoatRoughness={0.08} />
        </mesh>
      </group>
      {/* tip trail */}
      <line>
        <bufferGeometry ref={trailGeoRef} />
        <lineBasicMaterial vertexColors transparent opacity={0.85} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </line>
    </group>
  );
}

/* ================================================================
   GrabPlane — catches pointer events for the tip bob
================================================================ */
type GrabState = { active: boolean; wx: number; wy: number };

function GrabPlane({
  envRef,
  grabRef,
}: {
  envRef: MutableRefObject<PendState>;
  grabRef: MutableRefObject<GrabState>;
}) {
  return (
    <mesh
      onPointerDown={(e) => {
        const lx = e.point.x - PEND_OFFSET_X;
        const ly = e.point.y - PEND_OFFSET_Y;
        const tip = tipPos(envRef.current);
        if (Math.hypot(lx - tip.x, ly - tip.y) > 0.35) return;
        e.stopPropagation();
        grabRef.current = { active: true, wx: lx, wy: ly };
        (e.target as Element)?.setPointerCapture?.(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (!grabRef.current.active) return;
        grabRef.current.wx = e.point.x - PEND_OFFSET_X;
        grabRef.current.wy = e.point.y - PEND_OFFSET_Y;
      }}
      onPointerUp={(e) => {
        if (!grabRef.current.active) return;
        grabRef.current.active = false;
        try { (e.target as Element)?.releasePointerCapture?.(e.pointerId); } catch { /* */ }
      }}
      onPointerOut={() => { grabRef.current.active = false; }}
    >
      <planeGeometry args={[24, 24]} />
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
    </mesh>
  );
}

/* ================================================================
   PivotProjector — projects a point above the base to screen %
================================================================ */
function PivotProjector({ onPosition }: { onPosition: (p: { x: number; y: number }) => void }) {
  const called = useRef(false);
  useFrame((state) => {
    if (called.current) return;
    called.current = true;
    const v = new THREE.Vector3(PEND_OFFSET_X, PEND_OFFSET_Y + 0.3, 0);
    v.project(state.camera);
    onPosition({ x: (v.x * 0.5 + 0.5) * 100, y: (-v.y * 0.5 + 0.5) * 100 });
  });
  return null;
}

function BaseProjector({ onPosition }: { onPosition: (p: { x: number; y: number }) => void }) {
  const called = useRef(false);
  useFrame((state) => {
    if (called.current) return;
    called.current = true;
    const v = new THREE.Vector3(PEND_OFFSET_X, PEND_OFFSET_Y - 1.55, 0);
    v.project(state.camera);
    onPosition({ x: (v.x * 0.5 + 0.5) * 100, y: (-v.y * 0.5 + 0.5) * 100 });
  });
  return null;
}

/* ================================================================
   SceneRunner — game loop with IK grab + double pendulum physics
================================================================ */
function SceneRunner({
  agentRef,
  visibleEnvRef,
  convergedRef,
  grabRef,
  grabActiveRef,
  statsRef,
  onFirstRelease,
}: {
  agentRef: MutableRefObject<DQNAgent>;
  visibleEnvRef: MutableRefObject<PendState>;
  convergedRef: MutableRefObject<boolean>;
  grabRef: MutableRefObject<GrabState>;
  grabActiveRef: MutableRefObject<boolean>;
  statsRef: MutableRefObject<TrainStats>;
  onFirstRelease: () => void;
}) {
  const wasGrabRef = useRef(false);
  const prevT1Ref = useRef(Math.PI);
  const prevT2Ref = useRef(Math.PI);
  const grabVel1Ref = useRef(0);
  const grabVel2Ref = useRef(0);
  const accumRef = useRef(0);
  const physRef = useRef<PendState>({ ...visibleEnvRef.current });
  const physPrevRef = useRef<PendState>({ ...visibleEnvRef.current });

  useFrame((_, delta) => {
    const agent = agentRef.current;
    const nowGrab = grabRef.current.active;
    grabActiveRef.current = nowGrab;

    if (nowGrab) {
      if (!wasGrabRef.current) {
        agent.resetVisibleEpisode();
        prevT1Ref.current = physRef.current.t1;
        prevT2Ref.current = physRef.current.t2;
      }
      wasGrabRef.current = true;
      const { wx, wy } = grabRef.current;
      const ik = solveIK(wx, wy, prevT1Ref.current);
      const frameDt = Math.min(delta, 0.05);
      const prevGrabState: PendState = {
        t1: prevT1Ref.current, w1: grabVel1Ref.current,
        t2: prevT2Ref.current, w2: grabVel2Ref.current,
      };
      if (frameDt > 0) {
        grabVel1Ref.current = wrap(ik.t1 - prevT1Ref.current) / frameDt;
        grabVel2Ref.current = wrap(ik.t2 - prevT2Ref.current) / frameDt;
      }
      prevT1Ref.current = ik.t1;
      prevT2Ref.current = ik.t2;
      const gs: PendState = { t1: ik.t1, w1: grabVel1Ref.current, t2: ik.t2, w2: grabVel2Ref.current };
      if (frameDt > 0 && !agent.converged) {
        const demoU = inverseDynJ1(prevGrabState, gs, frameDt);
        const demoIdx = nearestActionIdx(demoU);
        const demoR = pendReward(gs, ACTION_TORQUES[demoIdx]);
        agent.injectDemo(getObs(prevGrabState), demoIdx, demoR * REWARD_SCALE, getObs(gs));
      }
      physRef.current = gs;
      physPrevRef.current = gs;
      visibleEnvRef.current = gs;
      accumRef.current = 0;
    } else {
      if (wasGrabRef.current) {
        wasGrabRef.current = false;
        const rs: PendState = {
          t1: physRef.current.t1,
          w1: Math.max(-MAX_SPEED, Math.min(MAX_SPEED, grabVel1Ref.current)),
          t2: physRef.current.t2,
          w2: Math.max(-MAX_SPEED, Math.min(MAX_SPEED, grabVel2Ref.current)),
        };
        physRef.current = rs;
        physPrevRef.current = rs;
        visibleEnvRef.current = rs;
        grabVel1Ref.current = 0;
        grabVel2Ref.current = 0;
        accumRef.current = 0;
        agent.resetVisibleEpisode();
        if (!agent.trainingActive) { agent.startTraining(); onFirstRelease(); }
      }

      accumRef.current += Math.min(delta, 0.1);
      while (accumRef.current >= DT) {
        accumRef.current -= DT;
        physPrevRef.current = { ...physRef.current };
        const prev = physPrevRef.current;
        const torque = agent.trainingActive
          ? agent.act(getObs(prev), agent.converged).action
          : 0;
        physRef.current = stepPend(prev, torque);
        getEngine().env = physRef.current;
        if (agent.trainingActive) {
          agent.recordVisibleStep(prev, torque, physRef.current);
        }
      }

      const t = accumRef.current / DT;
      visibleEnvRef.current = {
        t1: physPrevRef.current.t1 + (physRef.current.t1 - physPrevRef.current.t1) * t,
        w1: physPrevRef.current.w1 + (physRef.current.w1 - physPrevRef.current.w1) * t,
        t2: physPrevRef.current.t2 + (physRef.current.t2 - physPrevRef.current.t2) * t,
        w2: physPrevRef.current.w2 + (physRef.current.w2 - physPrevRef.current.w2) * t,
      };
    }

    agent.refreshStats();
    statsRef.current = { ...agent.stats };
    if (agent.converged && !convergedRef.current) convergedRef.current = true;
  });

  return (
    <group position={[PEND_OFFSET_X, PEND_OFFSET_Y, 0]}>
      <PendulumMesh envRef={visibleEnvRef} convergedRef={convergedRef} grabActiveRef={grabActiveRef} />
      <GrabPlane envRef={visibleEnvRef} grabRef={grabRef} />
    </group>
  );
}

/* ================================================================
   HUD overlays
================================================================ */
function RewardPlot({ agentRef, isAbsorbing }: { agentRef: MutableRefObject<DQNAgent>; isAbsorbing?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setTick((x) => x + 1), 100);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const agent = agentRef.current;
    if (!canvas || !agent) return;
    const rH = agent.plotRewardHistory;
    const uH = agent.plotUprightHistory;
    const eH = agent.plotEpsilonHistory;
    if (rH.length < 1) { canvas.width = 0; canvas.height = 0; return; }

    const dpr = window.devicePixelRatio || 1;
    const W = 240, H = 110;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const ml = 28, mr = 26, mt = 16, mb = 4, pw = W - ml - mr, ph = H - mt - mb;

    ctx.font = "7px monospace";
    ctx.textAlign = "left";
    let lx = ml;
    ctx.fillStyle = "#22d3ee"; ctx.fillText("reward", lx, 9); lx += ctx.measureText("reward").width + 6;
    ctx.fillStyle = "rgba(251,191,36,0.7)"; ctx.fillText("upright%", lx, 9); lx += ctx.measureText("upright%").width + 6;
    ctx.fillStyle = "rgba(167,139,250,0.6)"; ctx.fillText("epsilon", lx, 9);

    if (rH.length >= 2) {
      const rMin = Math.min(...rH) - 0.2;
      const rMax = Math.max(...rH) + 0.2;
      const rR = rMax - rMin || 1;

      ctx.fillStyle = "rgba(34,211,238,0.35)";
      ctx.font = "6px monospace";
      ctx.textAlign = "right";
      const ticks = 4;
      for (let t = 0; t <= ticks; t++) {
        const v = rMin + (rR * t) / ticks;
        const y = mt + ph * (1 - t / ticks);
        ctx.fillText(v.toFixed(1), ml - 3, y + 2);
        ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(ml, y); ctx.lineTo(ml + pw, y); ctx.stroke();
      }

      ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 1.5; ctx.beginPath();
      for (let i = 0; i < rH.length; i++) {
        const x = ml + (i / Math.max(1, rH.length - 1)) * pw;
        const y = mt + ph * (1 - (rH[i] - rMin) / rR);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      const zy = mt + ph * (1 - (0 - rMin) / rR);
      if (zy > mt && zy < mt + ph) {
        ctx.strokeStyle = "rgba(52,211,153,0.3)"; ctx.setLineDash([2, 2]); ctx.lineWidth = 0.7;
        ctx.beginPath(); ctx.moveTo(ml, zy); ctx.lineTo(ml + pw, zy); ctx.stroke(); ctx.setLineDash([]);
      }
    }

    if (uH.length >= 2) {
      ctx.strokeStyle = "rgba(251,191,36,0.5)"; ctx.lineWidth = 1; ctx.beginPath();
      for (let i = 0; i < uH.length; i++) {
        const x = ml + (i / Math.max(1, uH.length - 1)) * pw;
        const y = mt + ph * (1 - uH[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    if (eH.length >= 2) {
      ctx.strokeStyle = "rgba(167,139,250,0.45)"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1; ctx.beginPath();
      for (let i = 0; i < eH.length; i++) {
        const x = ml + (i / Math.max(1, eH.length - 1)) * pw;
        const y = mt + ph * (1 - eH[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke(); ctx.setLineDash([]);
    }

    ctx.font = "6px monospace"; ctx.textAlign = "left";
    const rightX = ml + pw + 3;
    ctx.fillStyle = "rgba(251,191,36,0.45)";
    ctx.fillText("1", rightX, mt + 3);
    ctx.fillText("0", rightX, mt + ph + 3);
  }, [tick, agentRef]);

  return (
    <div
      className="rounded-md border bg-black/45 backdrop-blur-sm px-3 py-2 min-w-[220px]"
      style={{
        borderColor: isAbsorbing ? "rgba(34,211,238,0.9)" : "rgba(255,255,255,0.1)",
        animation: isAbsorbing ? "absorbBorder 1s ease-in-out infinite" : undefined,
      }}
    >
      <canvas ref={canvasRef} />
    </div>
  );
}

function Row({ k, v, tint }: { k: string; v: string; tint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 tabular-nums">
      <span className="text-gray-500 uppercase tracking-wider text-[9px]">{k}</span>
      <span style={{ color: tint ?? "#e5e7eb" }}>{v}</span>
    </div>
  );
}

function StatsHud({
  statsRef,
  convergedRef,
}: {
  statsRef: MutableRefObject<TrainStats>;
  convergedRef: MutableRefObject<boolean>;
}) {
  const [, rerender] = useState(0);
  useEffect(() => {
    const id = setInterval(() => rerender((x) => x + 1), 250);
    return () => clearInterval(id);
  }, []);

  const s = statsRef.current;
  const converged = convergedRef.current;
  const explorationDone = Math.min(40, (1 - s.epsilon) / (1 - EPS_END) * 40);
  const progressByUpright = Math.min(95, s.bestUpright / CONVERGE_FRAC * 95);
  const pct = Math.min(100, Math.round(converged ? 100 : s.updates > 0 ? Math.max(explorationDone, progressByUpright) : 0));
  const barColor = converged ? "#34d399" : "#22d3ee";

  return (
    <div className="rounded-md border border-white/10 bg-black/45 backdrop-blur-sm px-3 py-2 font-mono text-[10px] leading-snug text-gray-300 min-w-[200px]">
      <div className="flex items-center gap-2 mb-1">
        <span className={`inline-block w-1.5 h-1.5 rounded-full ${converged ? "bg-good" : s.trainingActive ? "bg-accent2" : "bg-gray-600"} animate-pulse`} />
        <span className="uppercase tracking-[0.2em] text-[9px] text-gray-500">
          {converged ? "RL converged" : s.trainingActive ? "RL training" : "grab & invert"}
        </span>
      </div>
      {s.trainingActive && (
        <div className="mb-1.5">
          <div className="flex items-baseline justify-between mb-0.5">
            <span className="text-gray-500 uppercase tracking-wider text-[9px]">progress</span>
            <span style={{ color: barColor }}>{pct}%</span>
          </div>
          <div className="h-1 rounded-full bg-white/10 overflow-hidden">
            <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: barColor, boxShadow: converged ? `0 0 6px ${barColor}` : undefined }} />
          </div>
        </div>
      )}
      <Row k="updates" v={`${s.updates}`} />
      <Row k="mean reward" v={s.updates > 0 ? s.meanReward.toFixed(2) : "\u2014"} tint={s.meanReward > -2 ? "#34d399" : s.meanReward > -5 ? "#fbbf24" : "#d1d5db"} />
      <Row k="upright" v={s.updates > 0 ? `${Math.round(s.uprightFrac * 100)}%` : "\u2014"} tint={s.uprightFrac > 0.5 ? "#34d399" : "#d1d5db"} />
      <Row k="epsilon" v={s.updates > 0 ? s.epsilon.toFixed(3) : "\u2014"} tint="#a78bfa" />
    </div>
  );
}

/* ================================================================
   PendulumScene — top-level exported component with guided UX
================================================================ */
export default function PendulumScene() {
  const [engine] = useState(() => { const e = getEngine(); stopBackground(e); return e; });
  const agentRef = useRef<DQNAgent>(engine.agent);
  const visibleEnvRef = useRef<PendState>({ ...engine.env });
  const convergedRef = useRef(engine.agent.converged);
  const grabRef = useRef<GrabState>({ active: false, wx: 0, wy: 0 });
  const grabActiveRef = useRef(false);
  const statsRef = useRef<TrainStats>({ ...engine.agent.stats });
  const [trainingStarted, setTrainingStarted] = useState(engine.trainingStarted);
  const [isGrabbing, setIsGrabbing] = useState(false);
  const [trainingDone, setTrainingDone] = useState(engine.convergenceCounted);
  const [showToast, setShowToast] = useState(false);
  const [overlayRoot, setOverlayRoot] = useState<HTMLElement | null>(null);
  const [trainMePos, setTrainMePos] = useState<{ x: number; y: number } | null>(null);
  const [warmStartPos, setWarmStartPos] = useState<{ x: number; y: number } | null>(null);
  const [warmStartApplied, setWarmStartApplied] = useState(false);
  const [isAbsorbing, setIsAbsorbing] = useState(false);
  const [absorbProgress, setAbsorbProgress] = useState(0);
  const [absorbCount, setAbsorbCount] = useState(0);
  const [warmPhase, setWarmPhase] = useState<"absorb" | "calibrate" | null>(null);
  const [devReplaySize, setDevReplaySize] = useState(0);
  const eventSourceRef = useRef<HTMLElement>(null!);

  useEffect(() => {
    agentRef.current = engine.agent;
    visibleEnvRef.current = { ...engine.env };
    convergedRef.current = engine.agent.converged;
    if (engine.trainingStarted) setTrainingStarted(true);
    if (engine.convergenceCounted) setTrainingDone(true);
    const hero = document.getElementById("top");
    if (hero) (eventSourceRef as React.MutableRefObject<HTMLElement>).current = hero;
    return () => { startBackground(engine); };
  }, [engine]);

  useEffect(() => {
    const id = setInterval(() => setIsGrabbing(grabRef.current.active), 50);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (process.env.NODE_ENV !== "development") return;
    const id = setInterval(() => setDevReplaySize(engine.agent.getReplaySize()), 500);
    return () => clearInterval(id);
  }, [engine]);

  useEffect(() => {
    const hero = document.getElementById("top");
    if (!hero) return;
    const el = document.createElement("div");
    el.className = "absolute inset-0 z-[5] pointer-events-none";
    hero.appendChild(el);
    setOverlayRoot(el);
    return () => { el.remove(); };
  }, []);

  const handleFirstRelease = useCallback(() => {
    setTrainingStarted(true);
    engine.trainingStarted = true;
    engine.trainingStartTime = Date.now();
  }, [engine]);

  const handleDownloadReplay = useCallback(() => {
    const transitions = engine.agent.exportReplay(10000);
    const weights = engine.agent.exportWeights();
    const payload = {
      _comment: `Captured from a ${engine.agent.converged ? "converged" : "in-progress"} run. Replace public/pendubot-warmstart-${PENDUBOT_MODE}.json with this file.`,
      mode: PENDUBOT_MODE,
      converged: engine.agent.converged,
      weights,
      transitions,
    };
    const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pendubot-warmstart-${PENDUBOT_MODE}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [engine]);

  const handleWarmStart = useCallback(async () => {
    if (warmStartApplied || isAbsorbing) return;
    setIsAbsorbing(true);
    setAbsorbProgress(0);
    setWarmPhase("absorb");
    try {
      const res = await fetch(`/pendubot-warmstart-${PENDUBOT_MODE}.json`, { cache: "no-cache" });
      if (!res.ok) throw new Error(`warm-start fetch ${res.status}`);
      const data = (await res.json()) as {
        transitions?: Array<{ o: number[]; a: number; r: number; n: number[] }>;
        weights?: { online: number[][]; target: number[][] };
      };
      const transitions = data.transitions ?? [];
      setAbsorbCount(transitions.length);
      // Do NOT call startTraining() yet. While trainingActive is false,
      // useFrame steps physics with torque=0 but skips recordVisibleStep,
      // so nothing flails against the pre-load network before weights are
      // swapped in. Training flips on at the end, after loadWeights +
      // physics snap land.
      const ABSORB_WEIGHT = data.weights ? 0.7 : 0.25;
      if (transitions.length > 0) {
        await engine.agent.injectBatch(transitions, {
          chunkSize: 500,
          onProgress: (done, total) =>
            setAbsorbProgress(total > 0 ? (done / total) * ABSORB_WEIGHT : ABSORB_WEIGHT),
        });
      } else {
        setAbsorbProgress(ABSORB_WEIGHT);
      }
      setWarmPhase("calibrate");
      if (data.weights && engine.agent.loadWeights(data.weights)) {
        // Weights loaded with a small sigma perturbation. No physics snap —
        // the agent picks up control from wherever the pendulum currently
        // is and swings up / stabilizes into balance under the loaded
        // policy. Start a fresh episode counter so the stabilization
        // window isn't polluted by pre-warm accumulated counters.
        engine.agent.resetVisibleEpisode();
        setAbsorbProgress(1);
      } else {
        // Fallback: no weights in the file → retrain the Q-net offline
        // from the transition data (slow path, same as the original demo).
        await engine.agent.warmStartTrain(6000, {
          updatesPerFrame: 8,
          onProgress: (done, total) =>
            setAbsorbProgress(ABSORB_WEIGHT + (total > 0 ? (done / total) * (1 - ABSORB_WEIGHT) : 0)),
        });
      }
      // Flip training on. From the next useFrame tick the agent acts under
      // the newly-loaded (sigma-perturbed) weights and recordVisibleStep
      // runs normal 16×/step updates, pulling the noisy weights back onto
      // the good optimum. Convergence fires organically after 3
      // consecutive episodes ≥ CONVERGE_FRAC (≈30 s at 20 Hz).
      if (!engine.agent.trainingActive) {
        engine.agent.startTraining();
        if (!engine.trainingStarted) {
          engine.trainingStarted = true;
          engine.trainingStartTime = Date.now();
          setTrainingStarted(true);
        }
      }
      setWarmStartApplied(true);
    } catch (e) {
      console.warn("warm-start failed:", e);
    } finally {
      setTimeout(() => {
        setIsAbsorbing(false);
        setAbsorbProgress(0);
        setWarmPhase(null);
      }, 400);
    }
  }, [engine, warmStartApplied, isAbsorbing]);

  useEffect(() => {
    if (trainingDone) return;
    const id = setInterval(() => {
      if (engine.agent.converged && !engine.convergenceCounted) {
        engine.convergenceCounted = true;
        setTrainingDone(true);
        setShowToast(true);
        setTimeout(() => setShowToast(false), 4200);
        (async () => {
          try {
            const elapsed = engine.trainingStartTime > 0
              ? Math.round((Date.now() - engine.trainingStartTime) / 1000)
              : null;
            const res = await fetch("/api/stats/trainings", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ timeSeconds: elapsed }),
            });
            if (!res.ok) return;
            const data = (await res.json()) as { trainings?: number; bestTime?: number | null };
            if (typeof data.trainings === "number") {
              window.dispatchEvent(new CustomEvent("pendulum-training-complete", {
                detail: { trainings: data.trainings, bestTime: data.bestTime ?? null },
              }));
            }
          } catch { /* best-effort */ }
        })();
      }
    }, 150);
    return () => clearInterval(id);
  }, [trainingDone, engine]);

  const showTrainMe = !trainingStarted && !isGrabbing;
  const showInvertMe = !trainingStarted && isGrabbing;

  return (
    <div className="relative w-full h-full" style={{ touchAction: "none" }}>
      {/* eslint-disable-next-line react/no-unknown-property */}
      <style>{`
        @keyframes trainMeBob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
        @keyframes trainMeArrow { 0%, 100% { transform: translateY(0); opacity: 1; } 50% { transform: translateY(6px); opacity: 0.6; } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes warmStartPulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(34,211,238,0.55), 0 0 18px rgba(34,211,238,0.25); } 50% { box-shadow: 0 0 0 8px rgba(34,211,238,0), 0 0 28px rgba(34,211,238,0.5); } }
        @keyframes absorbBorder { 0%, 100% { border-color: rgba(34,211,238,0.35); box-shadow: 0 0 12px rgba(34,211,238,0.25); } 50% { border-color: rgba(34,211,238,0.9); box-shadow: 0 0 22px rgba(34,211,238,0.6); } }
      `}</style>
      <Canvas
        className="absolute inset-0"
        camera={{ position: [0, -1.0, 5.6], fov: 48 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        eventSource={eventSourceRef}
        eventPrefix="client"
        style={{ pointerEvents: "none" }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[3, 3, 4]} intensity={1.5} color="#22d3ee" />
        <pointLight position={[-3, -2, 3]} intensity={1.1} color="#7c5cff" />
        <pointLight position={[0, 0, 2]} intensity={0.5} color="#ffffff" />
        <directionalLight position={[4, 5, 2]} intensity={0.9} color="#e8ecf5" />
        <hemisphereLight args={["#1a2540", "#0a0f1e", 0.25]} />
        <PivotProjector onPosition={setTrainMePos} />
        <BaseProjector onPosition={setWarmStartPos} />
        <SceneRunner
          agentRef={agentRef}
          visibleEnvRef={visibleEnvRef}
          convergedRef={convergedRef}
          grabRef={grabRef}
          grabActiveRef={grabActiveRef}
          statsRef={statsRef}
          onFirstRelease={handleFirstRelease}
        />
      </Canvas>

      {overlayRoot && createPortal(
        <>
          <div className="absolute bottom-5 right-5 flex flex-col gap-2 items-end pointer-events-none">
            {trainingStarted && <RewardPlot agentRef={agentRef} isAbsorbing={isAbsorbing} />}
            <StatsHud statsRef={statsRef} convergedRef={convergedRef} />
            {process.env.NODE_ENV === "development" && (
              <button
                onClick={handleDownloadReplay}
                disabled={devReplaySize < 500}
                className="pointer-events-auto font-mono text-[9px] uppercase tracking-[0.16em] rounded-md px-3 py-1.5 border backdrop-blur-sm transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                style={{
                  color: trainingDone ? "#fbbf24" : "#a78bfa",
                  borderColor: trainingDone ? "rgba(251,191,36,0.55)" : "rgba(167,139,250,0.45)",
                  background: "rgba(5,10,20,0.55)",
                }}
                title={`Download ${Math.min(10000, devReplaySize).toLocaleString()} sampled transitions as JSON (dev only)`}
              >
                ⬇ export replay · {devReplaySize.toLocaleString()}{trainingDone ? " ✦" : ""}
              </button>
            )}
          </div>

          {trainingStarted && !warmStartApplied && !trainingDone && warmStartPos && (
            <div
              className="absolute pointer-events-auto"
              style={{ left: `${warmStartPos.x}%`, top: `${warmStartPos.y}%`, transform: "translate(-50%, 0)" }}
            >
              <button
                onClick={handleWarmStart}
                disabled={isAbsorbing}
                className="font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.18em] rounded-full px-4 py-2 border backdrop-blur-sm transition-colors disabled:cursor-wait"
                style={{
                  color: "#22d3ee",
                  borderColor: "rgba(34,211,238,0.55)",
                  background: "rgba(5,10,20,0.55)",
                  animation: isAbsorbing ? undefined : "warmStartPulse 2.4s ease-in-out infinite",
                }}
                title="Inject pre-baked experiences into replay so training starts near-converged"
              >
                {isAbsorbing ? (
                  `${warmPhase === "calibrate" ? "Calibrating" : "Absorbing"}… ${Math.round(absorbProgress * 100)}%`
                ) : (
                  <span className="flex flex-col items-center leading-tight gap-0.5">
                    <span>⚡ Warm-Start</span>
                    <span className="text-[8px] sm:text-[9px] opacity-75 normal-case tracking-[0.12em]">(skip ahead)</span>
                  </span>
                )}
              </button>
            </div>
          )}

          {isAbsorbing && (
            <div className="absolute top-20 left-1/2 -translate-x-1/2 rounded-full bg-gradient-to-r from-[#22d3ee]/20 to-[#7c5cff]/20 border border-[#22d3ee]/60 backdrop-blur px-5 py-2 text-[11px] font-mono uppercase tracking-[0.22em] text-[#22d3ee] shadow-[0_0_24px_rgba(34,211,238,0.35)] pointer-events-none">
              {warmPhase === "calibrate"
                ? "⚡ Calibrating Q-values…"
                : `⚡ Absorbing ${absorbCount > 0 ? absorbCount.toLocaleString() + " " : ""}experiences…`}
            </div>
          )}

          {showTrainMe && trainMePos && (
            <div
              className="absolute pointer-events-none select-none"
              style={{ left: `${trainMePos.x}%`, top: `${trainMePos.y}%`, transform: "translate(-50%, -100%)" }}
            >
              <div className="flex flex-col items-center" style={{ animation: "trainMeBob 2s ease-in-out infinite" }}>
                <span
                  className="font-mono font-bold text-xl sm:text-2xl uppercase tracking-[0.18em]"
                  style={{ color: "#fbbf24", textShadow: "0 0 18px rgba(251,191,36,0.55), 0 0 40px rgba(251,191,36,0.25)" }}
                >
                  Train Me!
                </span>
                <span
                  className="text-2xl sm:text-3xl -mt-1"
                  style={{ color: "#fbbf24", animation: "trainMeArrow 1.2s ease-in-out infinite", filter: "drop-shadow(0 0 6px rgba(251,191,36,0.5))" }}
                >
                  &#x2193;
                </span>
              </div>
            </div>
          )}

          {showInvertMe && trainMePos && (
            <div
              className="absolute pointer-events-none select-none"
              style={{ left: `${trainMePos.x}%`, top: `${trainMePos.y}%`, transform: "translate(-50%, -100%)", animation: "fadeInUp 0.3s ease-out" }}
            >
              <span
                className="font-mono font-semibold text-lg sm:text-xl uppercase tracking-[0.14em]"
                style={{ color: "#22d3ee", textShadow: "0 0 14px rgba(34,211,238,0.5), 0 0 30px rgba(34,211,238,0.2)" }}
              >
                {PENDUBOT_MODE === "full" ? "Now Invert Me" : "Flip the Tip Up!"}
              </span>
            </div>
          )}

          {showToast && (
            <div className="absolute top-20 left-1/2 -translate-x-1/2 rounded-full bg-gradient-to-r from-[#fbbf24]/20 to-[#f59e0b]/20 border border-[#fbbf24]/60 backdrop-blur px-5 py-2 text-[11px] font-mono uppercase tracking-[0.22em] text-[#fbbf24] shadow-[0_0_30px_rgba(251,191,36,0.4)] animate-pulse pointer-events-none">
              ✦ RL converged — balance mastered
            </div>
          )}
        </>,
        overlayRoot,
      )}
    </div>
  );
}

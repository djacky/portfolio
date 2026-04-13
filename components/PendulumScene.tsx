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

const PEND_OFFSET_X = 1.8;
const PEND_OFFSET_Y = 0.9;

/* ================================================================
   Physics — single rigid-rod pendulum (OpenAI Gym Pendulum-v1)
   θ̈ = (3g/2L)·sin(θ) + (3/mL²)·τ
   θ = 0 is upright (unstable), θ = π is hanging (stable).
================================================================ */
const G_PHYS = 10;
const MASS = 1;
const LEN = 1;
const MAX_TORQUE = 2;
const DT = 0.05;
const MAX_SPEED = 8;
const EP_LEN = 200;

type PendState = { th: number; w: number };

function wrap(a: number): number {
  let x = (a + Math.PI) % (2 * Math.PI);
  if (x < 0) x += 2 * Math.PI;
  return x - Math.PI;
}

function pendDeriv(th: number, w: number, u: number): [number, number] {
  return [w, (3 * G_PHYS) / (2 * LEN) * Math.sin(th) + 3 / (MASS * LEN * LEN) * u];
}

function stepPend(s: PendState, u: number, dt: number = DT): PendState {
  const k1 = pendDeriv(s.th, s.w, u);
  const k2 = pendDeriv(s.th + k1[0] * dt / 2, s.w + k1[1] * dt / 2, u);
  const k3 = pendDeriv(s.th + k2[0] * dt / 2, s.w + k2[1] * dt / 2, u);
  const k4 = pendDeriv(s.th + k3[0] * dt, s.w + k3[1] * dt, u);
  return {
    th: s.th + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt / 6,
    w: Math.max(-MAX_SPEED, Math.min(MAX_SPEED,
      s.w + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt / 6)),
  };
}

function getObs(s: PendState): Float64Array {
  return Float64Array.of(Math.cos(s.th), Math.sin(s.th), s.w / MAX_SPEED);
}

function pendReward(s: PendState, u: number): number {
  const th = wrap(s.th);
  return -(th * th + 0.1 * s.w * s.w + 0.001 * u * u);
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
   DQN — Deep Q-Network for pendulum swing-up + balance
   Network: [3] → 32 → ReLU → 32 → ReLU → 7 Q-values
   Learns online from actual visible pendulum transitions.
================================================================ */
const OBS_DIM = 3;
const N_ACTIONS = 7;
const ACTION_TORQUES = Float64Array.from(
  { length: N_ACTIONS }, (_, i) => -MAX_TORQUE + (2 * MAX_TORQUE / (N_ACTIONS - 1)) * i,
);
const DQN_HIDDEN = 32;
const REPLAY_CAP = 20000;
const BATCH_SIZE = 32;
const UPDATES_PER_STEP = 8;
const DQN_LR = 0.003;
const GAMMA = 0.99;
const EPS_START = 1.0;
const EPS_END = 0.01;
const EPS_DECAY = 2000;
const TARGET_TAU = 0.005;
const MIN_REPLAY = 200;
const REWARD_SCALE = 0.1;
const CONVERGE_FRAC = 0.70;
const CONVERGE_HOLD = 3;
const MIN_CONVERGE_STEPS = 1000;
const PLOT_WINDOW = 40;

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

  private visSteps = 0;
  private visRewardAcc = 0;
  private visUprightAcc = 0;
  private visRewardWindow: number[] = [];
  private visUprightWindow: number[] = [];

  plotRewardHistory: number[] = [];
  plotUprightHistory: number[] = [];
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

    const isUp = Math.abs(wrap(next.th)) < 0.3 && Math.abs(next.w) < 1;

    this.plotRewardAcc += rawR;
    if (isUp) this.plotUprightAcc++;
    this.plotSteps++;
    if (this.plotSteps >= PLOT_WINDOW) {
      this.plotRewardHistory.push(this.plotRewardAcc / PLOT_WINDOW);
      this.plotUprightHistory.push(this.plotUprightAcc / PLOT_WINDOW);
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

  runBgSteps() { /* DQN learns inline via recordVisibleStep */ }

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

function bobPos(s: PendState): THREE.Vector3 {
  return new THREE.Vector3(LEN * Math.sin(s.th), LEN * Math.cos(s.th), 0);
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
   PendulumMesh — single rod + bob with glow + trail
================================================================ */
function PendulumMesh({
  envRef,
  convergedRef,
}: {
  envRef: MutableRefObject<PendState>;
  convergedRef: MutableRefObject<boolean>;
}) {
  const rodRef = useRef<THREE.Mesh>(null);
  const rodCoreRef = useRef<THREE.Mesh>(null);
  const bobGroupRef = useRef<THREE.Group>(null);
  const bobMeshRef = useRef<THREE.Mesh>(null);
  const bobRingRef = useRef<THREE.Mesh>(null);
  const bobGlowRef = useRef<THREE.Sprite>(null);
  const trailRef = useRef<THREE.Vector3[]>([]);
  const trailGeoRef = useRef<THREE.BufferGeometry>(null);
  const goldRef = useRef(0);
  const glowTex = useMemo(() => makeGlowTexture(), []);
  const TRAIL_LEN = 72;

  useFrame((_, dt) => {
    const s = envRef.current;
    const bob = bobPos(s);
    const pivot = new THREE.Vector3(0, 0, 0);

    if (rodRef.current) setRod(rodRef.current, pivot, bob);
    if (rodCoreRef.current) setRod(rodCoreRef.current, pivot, bob);
    if (bobGroupRef.current) bobGroupRef.current.position.copy(bob);
    if (bobRingRef.current) bobRingRef.current.rotation.z -= dt * 0.5;

    const target = convergedRef.current ? 1 : 0;
    const k = 1 - Math.exp(-dt * 2.2);
    goldRef.current += (target - goldRef.current) * k;
    const g = goldRef.current;
    const pulse = 0.9 + Math.sin(performance.now() * 0.0042) * 0.18;

    const lerpMat = (mesh: THREE.Mesh | null, base: THREE.Color, emI: number) => {
      if (!mesh) return;
      const mat = mesh.material as THREE.MeshPhysicalMaterial;
      mat.color.copy(base).lerp(COLOR_GOLD, g);
      mat.emissive.copy(base).lerp(COLOR_GOLD, g);
      mat.emissiveIntensity = emI + g * 0.6 * pulse;
    };
    lerpMat(rodRef.current, COLOR_ROD, 0.35);
    lerpMat(bobMeshRef.current, COLOR_BOB, 1.4);

    if (rodCoreRef.current)
      (rodCoreRef.current.material as THREE.MeshBasicMaterial).color.copy(COLOR_ROD).lerp(COLOR_GOLD, g);

    if (bobGlowRef.current) {
      (bobGlowRef.current.material as THREE.SpriteMaterial).color.copy(COLOR_BOB).lerp(COLOR_GOLD, g);
      bobGlowRef.current.scale.setScalar(1.2 * (1 + Math.sin(performance.now() * 0.0032) * 0.06));
    }

    trailRef.current.push(bob.clone());
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
      {/* rod outer */}
      <mesh ref={rodRef}>
        <cylinderGeometry args={[0.05, 0.06, 1, 18]} />
        <meshPhysicalMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={0.35} metalness={0.6} roughness={0.2} clearcoat={0.9} clearcoatRoughness={0.15} />
      </mesh>
      {/* rod inner glow core */}
      <mesh ref={rodCoreRef}>
        <cylinderGeometry args={[0.018, 0.018, 1.01, 10]} />
        <meshBasicMaterial color="#22d3ee" toneMapped={false} />
      </mesh>
      {/* bob assembly */}
      <group ref={bobGroupRef}>
        <sprite ref={bobGlowRef}>
          <spriteMaterial map={glowTex} color="#7c5cff" transparent depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
        </sprite>
        <mesh ref={bobRingRef}>
          <torusGeometry args={[0.19, 0.009, 12, 48]} />
          <meshBasicMaterial color="#7c5cff" toneMapped={false} />
        </mesh>
        <mesh ref={bobMeshRef}>
          <sphereGeometry args={[0.14, 28, 28]} />
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
   GrabPlane — invisible plane catching pointer events for the bob
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
        const bob = bobPos(envRef.current);
        if (Math.hypot(lx - bob.x, ly - bob.y) > 0.4) return;
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

/* ================================================================
   SceneRunner — game loop: visible env + background PPO training
================================================================ */
function SceneRunner({
  agentRef,
  visibleEnvRef,
  convergedRef,
  grabRef,
  statsRef,
  onFirstRelease,
}: {
  agentRef: MutableRefObject<DQNAgent | null>;
  visibleEnvRef: MutableRefObject<PendState>;
  convergedRef: MutableRefObject<boolean>;
  grabRef: MutableRefObject<GrabState>;
  statsRef: MutableRefObject<TrainStats>;
  onFirstRelease: () => void;
}) {
  const wasGrabRef = useRef(false);
  const prevGrabAngle = useRef(0);
  const grabVelRef = useRef(0);
  const accumRef = useRef(0);
  const physRef = useRef<PendState>({ ...visibleEnvRef.current });
  const physPrevRef = useRef<PendState>({ ...visibleEnvRef.current });

  useFrame((_, delta) => {
    const agent = agentRef.current;
    if (!agent) return;

    const nowGrab = grabRef.current.active;
    if (nowGrab) {
      if (!wasGrabRef.current) agent.resetVisibleEpisode();
      wasGrabRef.current = true;
      const { wx, wy } = grabRef.current;
      const newAngle = Math.atan2(wx, wy);
      const frameDt = Math.min(delta, 0.05);
      if (frameDt > 0) grabVelRef.current = wrap(newAngle - prevGrabAngle.current) / frameDt;
      prevGrabAngle.current = newAngle;
      const gs = { th: newAngle, w: grabVelRef.current };
      physRef.current = gs;
      physPrevRef.current = gs;
      visibleEnvRef.current = gs;
      accumRef.current = 0;
    } else {
      if (wasGrabRef.current) {
        wasGrabRef.current = false;
        const rs = { th: physRef.current.th, w: Math.max(-MAX_SPEED, Math.min(MAX_SPEED, grabVelRef.current)) };
        physRef.current = rs;
        physPrevRef.current = rs;
        visibleEnvRef.current = rs;
        grabVelRef.current = 0;
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
        if (agent.trainingActive) {
          agent.recordVisibleStep(prev, torque, physRef.current);
        }
      }

      const t = accumRef.current / DT;
      visibleEnvRef.current = {
        th: physPrevRef.current.th + (physRef.current.th - physPrevRef.current.th) * t,
        w: physPrevRef.current.w + (physRef.current.w - physPrevRef.current.w) * t,
      };
    }

    agent.refreshStats();
    statsRef.current = { ...agent.stats };
    if (agent.converged && !convergedRef.current) convergedRef.current = true;
  });

  return (
    <group position={[PEND_OFFSET_X, PEND_OFFSET_Y, 0]}>
      <PendulumMesh envRef={visibleEnvRef} convergedRef={convergedRef} />
      <GrabPlane envRef={visibleEnvRef} grabRef={grabRef} />
    </group>
  );
}

/* ================================================================
   HUD overlays
================================================================ */
function RewardPlot({ agentRef }: { agentRef: MutableRefObject<DQNAgent | null> }) {
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
    if (rH.length < 1) { canvas.width = 0; canvas.height = 0; return; }

    const dpr = window.devicePixelRatio || 1;
    const W = 220, H = 100;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const ml = 4, mr = 4, mt = 14, mb = 4, pw = W - ml - mr, ph = H - mt - mb;

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "7px monospace";
    ctx.textAlign = "left";
    ctx.fillText("reward / step", ml, 9);

    if (rH.length >= 2) {
      const rMin = Math.min(...rH) - 0.2;
      const rMax = Math.max(...rH) + 0.2;
      const rR = rMax - rMin || 1;
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
      ctx.fillStyle = "rgba(251,191,36,0.5)"; ctx.font = "6px monospace"; ctx.textAlign = "right";
      ctx.fillText("upright%", ml + pw, mt + 6);
    }
  }, [tick, agentRef]);

  return (
    <div className="rounded-md border border-white/10 bg-black/45 backdrop-blur-sm px-3 py-2 min-w-[220px]">
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
          {converged ? "RL converged" : s.trainingActive ? "RL training" : "grab the pendulum"}
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
   PendulumScene — top-level exported component
================================================================ */
export default function PendulumScene() {
  const agentRef = useRef<DQNAgent | null>(null);
  const visibleEnvRef = useRef<PendState>({ th: Math.PI + (Math.random() - 0.5) * 0.4, w: 0 });
  const convergedRef = useRef(false);
  const grabRef = useRef<GrabState>({ active: false, wx: 0, wy: 0 });
  const statsRef = useRef<TrainStats>({ ...INIT_STATS });
  const [ready, setReady] = useState(false);
  const [trainingStarted, setTrainingStarted] = useState(false);
  const [trainingDone, setTrainingDone] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [overlayRoot, setOverlayRoot] = useState<HTMLElement | null>(null);
  const [trainMePos, setTrainMePos] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const agent = new DQNAgent();
    if (!cancelled) { agentRef.current = agent; setReady(true); }
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const hero = document.getElementById("top");
    if (!hero) return;
    const el = document.createElement("div");
    el.className = "absolute inset-0 z-[5] pointer-events-none";
    hero.appendChild(el);
    setOverlayRoot(el);
    return () => { el.remove(); };
  }, []);

  const handleFirstRelease = useCallback(() => { setTrainingStarted(true); }, []);

  const countedRef = useRef(false);
  useEffect(() => {
    if (trainingDone) return;
    const id = setInterval(() => {
      if (statsRef.current.converged && !countedRef.current) {
        countedRef.current = true;
        setTrainingDone(true);
        setShowToast(true);
        setTimeout(() => setShowToast(false), 4200);
        (async () => {
          try {
            const res = await fetch("/api/stats/trainings", { method: "POST" });
            if (!res.ok) return;
            const data = (await res.json()) as { trainings?: number };
            if (typeof data.trainings === "number") {
              window.dispatchEvent(new CustomEvent("pendulum-training-complete", { detail: { trainings: data.trainings } }));
            }
          } catch { /* best-effort */ }
        })();
      }
    }, 150);
    return () => clearInterval(id);
  }, [trainingDone]);

  return (
    <div className="relative w-full h-full" style={{ touchAction: "none" }}>
      {/* eslint-disable-next-line react/no-unknown-property */}
      <style>{`
        @keyframes trainMeBob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
        @keyframes trainMeArrow { 0%, 100% { transform: translateY(0); opacity: 1; } 50% { transform: translateY(6px); opacity: 0.6; } }
      `}</style>
      <Canvas
        className="absolute inset-0"
        camera={{ position: [0, -1.0, 5.6], fov: 48 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[3, 3, 4]} intensity={1.5} color="#22d3ee" />
        <pointLight position={[-3, -2, 3]} intensity={1.1} color="#7c5cff" />
        <pointLight position={[0, 0, 2]} intensity={0.5} color="#ffffff" />
        <directionalLight position={[4, 5, 2]} intensity={0.9} color="#e8ecf5" />
        <hemisphereLight args={["#1a2540", "#0a0f1e", 0.25]} />
        <PivotProjector onPosition={setTrainMePos} />
        {ready && (
          <SceneRunner
            agentRef={agentRef}
            visibleEnvRef={visibleEnvRef}
            convergedRef={convergedRef}
            grabRef={grabRef}
            statsRef={statsRef}
            onFirstRelease={handleFirstRelease}
          />
        )}
      </Canvas>

      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center text-[10px] font-mono uppercase tracking-[0.22em] text-gray-500">
          initializing…
        </div>
      )}

      {overlayRoot && createPortal(
        <>
          <div className="absolute bottom-5 right-5 flex flex-col gap-2 items-end pointer-events-none">
            {trainingStarted && <RewardPlot agentRef={agentRef} />}
            <StatsHud statsRef={statsRef} convergedRef={convergedRef} />
          </div>

          {ready && !trainingStarted && trainMePos && (
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

          {showToast && (
            <div className="absolute top-6 left-1/2 -translate-x-1/2 rounded-full bg-gradient-to-r from-[#fbbf24]/20 to-[#f59e0b]/20 border border-[#fbbf24]/60 backdrop-blur px-5 py-2 text-[11px] font-mono uppercase tracking-[0.22em] text-[#fbbf24] shadow-[0_0_30px_rgba(251,191,36,0.4)] animate-pulse pointer-events-none">
              ✦ RL converged — swing-up mastered
            </div>
          )}
        </>,
        overlayRoot,
      )}
    </div>
  );
}

"use client";

/* ------------------------------------------------------------------
   PendulumScene — live reinforcement-learning demo.

   A 2-link pendulum rendered in Three.js, controlled by a
   REINFORCE policy-gradient agent trained *in the browser* using
   TensorFlow.js. The agent starts with random weights, so the
   pendulum flails; over ~60–120s of real time you watch it learn
   to keep the double pendulum upright.

   Physics: custom RK4 integration of the double-pendulum EOM
     (Lagrangian form, with added joint torques + joint damping).
   Agent:   small MLP (6 → 32 → 32 → 2 μ + 2 logσ), Gaussian policy,
            REINFORCE with standardized returns + entropy bonus,
            Adam optimizer.
   Visible env steps in real time (one physics step per frame).
   A separate training env runs several steps per frame, feeding
   the shared policy with experience. Watching the visible env =
   watching the current policy act.
------------------------------------------------------------------ */

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type MutableRefObject,
} from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";

const COLOR_ROD1 = new THREE.Color("#22d3ee");
const COLOR_ROD2 = new THREE.Color("#7c5cff");
const COLOR_GOLD = new THREE.Color("#fbbf24");

// Pendulum scene is shifted from world origin by this much (r3f group
// in SceneRunner). Grab-plane picking must subtract this to get back
// into pendulum-local coordinates where the pivot sits at (0,0).
const PEND_OFFSET_X = 1.8;
const PEND_OFFSET_Y = 0.9;
import * as tf from "@tensorflow/tfjs";

/* ================================================================
   physics — double pendulum with joint torques + damping
================================================================ */

const L1 = 1.0;
const L2 = 1.0;
const M1 = 1.0;
const M2 = 1.0;
const G = 9.81;
// Essentially frictionless — the controller is responsible for ALL
// energy dissipation. Without a working policy, the pendulum swings
// indefinitely; with one, it damps cleanly to rest.
const DAMPING = 0.0015;
const DT = 0.02;
const TORQUE_MAX = 8;

type EnvState = { th1: number; th2: number; w1: number; w2: number };

// Wrap angle to [-π, π] — used for reward/obs so small oscillations
// around rest read as small, not as ~2π.
function wrap(a: number): number {
  let x = (a + Math.PI) % (2 * Math.PI);
  if (x < 0) x += 2 * Math.PI;
  return x - Math.PI;
}

// Double-pendulum equations of motion — proper Lagrangian form with
// explicit mass matrix inversion. θ₁ and θ₂ are BOTH absolute angles
// measured from the downward vertical (so θ₁=θ₂=0 is hanging rest).
//
//   M(q) q̈ + C(q, q̇) + G(q) = Q
//
// with a pivot motor torque τ=u and viscous damping at each joint.
// Solved directly as  q̈ = M⁻¹ (Q − C − G).
function deriv(s: EnvState, u: number): [number, number, number, number] {
  const { th1, th2, w1, w2 } = s;
  const d = th1 - th2;
  const cd = Math.cos(d);
  const sd = Math.sin(d);

  // Mass matrix (symmetric)
  const M11 = (M1 + M2) * L1 * L1;
  const M12 = M2 * L1 * L2 * cd;
  const M22 = M2 * L2 * L2;
  const det = M11 * M22 - M12 * M12;

  // Coriolis / centrifugal
  const C1 = M2 * L1 * L2 * sd * w2 * w2;
  const C2 = -M2 * L1 * L2 * sd * w1 * w1;

  // Gravity (both angles measured from vertical → sin, not cos)
  const G1 = (M1 + M2) * G * L1 * Math.sin(th1);
  const G2 = M2 * G * L2 * Math.sin(th2);

  // Generalized forces — motor at pivot only, joint damping on both.
  const Q1 = u - DAMPING * w1;
  const Q2 = -DAMPING * w2;

  const rhs1 = Q1 - C1 - G1;
  const rhs2 = Q2 - C2 - G2;

  // θ̈ = M⁻¹ · rhs  (2×2 inverse)
  const a1 = (M22 * rhs1 - M12 * rhs2) / det;
  const a2 = (-M12 * rhs1 + M11 * rhs2) / det;

  return [w1, w2, a1, a2];
}

function stepEnv(s: EnvState, u: number, dt = DT): EnvState {
  const k1 = deriv(s, u);
  const s2 = {
    th1: s.th1 + (k1[0] * dt) / 2,
    th2: s.th2 + (k1[1] * dt) / 2,
    w1: s.w1 + (k1[2] * dt) / 2,
    w2: s.w2 + (k1[3] * dt) / 2,
  };
  const k2 = deriv(s2, u);
  const s3 = {
    th1: s.th1 + (k2[0] * dt) / 2,
    th2: s.th2 + (k2[1] * dt) / 2,
    w1: s.w1 + (k2[2] * dt) / 2,
    w2: s.w2 + (k2[3] * dt) / 2,
  };
  const k3 = deriv(s3, u);
  const s4 = {
    th1: s.th1 + k3[0] * dt,
    th2: s.th2 + k3[1] * dt,
    w1: s.w1 + k3[2] * dt,
    w2: s.w2 + k3[3] * dt,
  };
  const k4 = deriv(s4, u);
  return {
    th1: s.th1 + ((k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt) / 6,
    th2: s.th2 + ((k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt) / 6,
    w1: s.w1 + ((k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * dt) / 6,
    w2: s.w2 + ((k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * dt) / 6,
  };
}

function obsOf(s: EnvState): number[] {
  return [
    Math.sin(s.th1),
    Math.cos(s.th1),
    Math.sin(s.th2),
    Math.cos(s.th2),
    s.w1 / 6,
    s.w2 / 6,
  ];
}

/* ----------------------------------------------------------------
   Goal: Hanging stabilization.
     Target state = (θ₁=0, θ₂=0, ω₁=0, ω₂=0)
     → both links hanging at rest.

   The motor at joint 1 must actively damp the double pendulum to
   rest. Natural damping is nearly zero, so any energy in the system
   has to be removed by the controller. This is the task the neural
   net learns by behavioral cloning against a closed-form PD teacher.
---------------------------------------------------------------- */

// PD teacher — linearized around the hanging equilibrium.
// Gravity already supplies the restoring force on θ₁, so KP1 is small.
// The main job is velocity feedback on both joints to drain kinetic
// energy. This is a provably stable controller for the hanging goal.
const KP1 = 3.0;
const KD1 = 6.5;
const KD2 = 2.8;

function pdTeacher(s: EnvState): number {
  const u =
    -KP1 * Math.sin(wrap(s.th1)) -
    KD1 * s.w1 -
    KD2 * s.w2;
  return Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
}

// Success window — pendulum is "at rest" within these bounds.
function atTarget(s: EnvState): boolean {
  return (
    Math.abs(wrap(s.th1)) < 0.18 &&
    Math.abs(wrap(s.th2)) < 0.18 &&
    Math.abs(s.w1) < 0.6 &&
    Math.abs(s.w2) < 0.6
  );
}

function initState(): EnvState {
  // Moderate displacement from rest — "just got bumped".
  return {
    th1: (Math.random() - 0.5) * 1.8,
    th2: (Math.random() - 0.5) * 1.8,
    w1: (Math.random() - 0.5) * 1.2,
    w2: (Math.random() - 0.5) * 1.2,
  };
}

/* ---- 2-link inverse kinematics (for grabbing the tip) ---------------- */
// Given a target (tx, ty) for the tip in world coords (pivot at origin,
// +y = up; hanging rest = θ₁=θ₂=0), solve for absolute angles (θ₁, θ₂).
// Picks the elbow solution continuous with the previous pose.
function ik2(
  tx: number,
  ty: number,
  prevTh1: number,
  prevTh2: number,
): { th1: number; th2: number } {
  let r = Math.hypot(tx, ty);
  const rmax = L1 + L2 - 1e-3;
  if (r > rmax) {
    tx *= rmax / r;
    ty *= rmax / r;
    r = rmax;
  }
  const D = tx * tx + ty * ty;
  let cRel = (D - L1 * L1 - L2 * L2) / (2 * L1 * L2);
  cRel = Math.max(-1, Math.min(1, cRel));
  // Elbow angle (angle between rods), two branches.
  const relPos = Math.acos(cRel);
  const relNeg = -relPos;
  const prevRel = wrap(prevTh2 - prevTh1);
  const rel =
    Math.abs(wrap(relPos - prevRel)) < Math.abs(wrap(relNeg - prevRel))
      ? relPos
      : relNeg;
  const th1 =
    Math.atan2(tx, -ty) -
    Math.atan2(L2 * Math.sin(rel), L1 + L2 * Math.cos(rel));
  const th2 = th1 + rel; // absolute θ₂
  return { th1, th2 };
}

/* ================================================================
   agent — policy network trained via online behavioral cloning
   against the closed-form PD teacher. The narrative: the network
   watches the teacher control the pendulum and learns to reproduce
   its decisions, then takes over once its loss is low enough.
================================================================ */

type Mode = "idle" | "training" | "ready" | "control";

type AgentStats = {
  updates: number;
  loss: number;
  deployed: boolean;
  teacherAction: number;
  netAction: number;
  trialsTotal: number;
  trialsSuccess: number;
  successStreak: number;
  trainingComplete: boolean;
};

class Agent {
  W1!: tf.Variable;
  b1!: tf.Variable;
  W2!: tf.Variable;
  b2!: tf.Variable;
  Wm!: tf.Variable;
  bm!: tf.Variable;
  optim!: tf.Optimizer;
  ready = false;
  stats: AgentStats = {
    updates: 0,
    loss: 1,
    deployed: false,
    teacherAction: 0,
    netAction: 0,
    trialsTotal: 0,
    trialsSuccess: 0,
    successStreak: 0,
    trainingComplete: false,
  };

  async init() {
    await tf.setBackend("cpu");
    await tf.ready();
    const H = 32;
    // Glorot-scaled init for stable early gradients.
    this.W1 = tf.variable(tf.randomNormal([6, H], 0, Math.sqrt(2 / 38)));
    this.b1 = tf.variable(tf.zeros([H]));
    this.W2 = tf.variable(tf.randomNormal([H, H], 0, Math.sqrt(2 / 64)));
    this.b2 = tf.variable(tf.zeros([H]));
    this.Wm = tf.variable(tf.randomNormal([H, 1], 0, 0.1));
    this.bm = tf.variable(tf.zeros([1]));
    this.optim = tf.train.adam(1.5e-3);
    this.ready = true;
  }

  private netT(obsT: tf.Tensor2D): tf.Tensor2D {
    const h1 = tf.relu(tf.add(tf.matMul(obsT, this.W1), this.b1));
    const h2 = tf.relu(tf.add(tf.matMul(h1, this.W2), this.b2));
    return tf.add(tf.matMul(h2, this.Wm), this.bm) as tf.Tensor2D;
  }

  // Deterministic forward pass — used wherever we actually control
  // the pendulum with the network.
  actDet(o: number[]): number {
    return tf.tidy(() => {
      const obsT = tf.tensor2d([o]);
      return this.netT(obsT).dataSync()[0];
    });
  }

  // Supervised mini-batch update: mean squared error between the
  // network's output and the PD teacher's target torque.
  trainBatch(obsBatch: number[][], tgtBatch: number[]) {
    if (obsBatch.length === 0) return;
    const agent = this;
    let lossVal = 0;
    const lossTensor = this.optim.minimize(() => {
      const obsT = tf.tensor2d(obsBatch);
      const tgtT = tf.tensor2d(tgtBatch.map((a) => [a]));
      const pred = agent.netT(obsT);
      return tf.mean(tf.square(tf.sub(pred, tgtT))) as tf.Scalar;
    }, true);
    if (lossTensor) {
      lossVal = lossTensor.dataSync()[0];
      lossTensor.dispose();
    }
    this.stats.updates += 1;
    // Exponential moving average of the loss for a stable HUD reading.
    this.stats.loss = this.stats.loss * 0.9 + lossVal * 0.1;
  }

  // Re-randomize all weights — used by the "retrain" flow so the
  // viewer can watch the learning process from scratch again.
  resetWeights() {
    const H = 32;
    tf.tidy(() => {
      this.W1.assign(tf.randomNormal([6, H], 0, Math.sqrt(2 / 38)));
      this.b1.assign(tf.zeros([H]));
      this.W2.assign(tf.randomNormal([H, H], 0, Math.sqrt(2 / 64)));
      this.b2.assign(tf.zeros([H]));
      this.Wm.assign(tf.randomNormal([H, 1], 0, 0.1));
      this.bm.assign(tf.zeros([1]));
    });
    this.stats = {
      updates: 0,
      loss: 1,
      deployed: false,
      teacherAction: 0,
      netAction: 0,
      trialsTotal: 0,
      trialsSuccess: 0,
      successStreak: 0,
      trainingComplete: false,
    };
  }

  // Finite-difference the learned network at the hanging-rest state
  // to recover its EFFECTIVE linear controller gains:
  //   u ≈ k_th1 · θ₁ + k_th2 · θ₂ + k_w1 · ω₁ + k_w2 · ω₂
  // These are the quantities BC is implicitly learning, and they
  // should converge to the teacher's analytical values over training.
  gainsSnapshot(): { kTh1: number; kTh2: number; kW1: number; kW2: number } {
    const eps = 0.05;
    const obs0 = [0, 1, 0, 1, 0, 0];
    // θ₁ partial: perturb sin θ₁ ≈ ε, keep cos ≈ 1.
    const oThp = [Math.sin(eps), Math.cos(eps), 0, 1, 0, 0];
    const oThm = [Math.sin(-eps), Math.cos(-eps), 0, 1, 0, 0];
    const oTh2p = [0, 1, Math.sin(eps), Math.cos(eps), 0, 0];
    const oTh2m = [0, 1, Math.sin(-eps), Math.cos(-eps), 0, 0];
    // ω₁, ω₂ enter obs divided by 6.
    const oW1p = [0, 1, 0, 1, eps / 6, 0];
    const oW1m = [0, 1, 0, 1, -eps / 6, 0];
    const oW2p = [0, 1, 0, 1, 0, eps / 6];
    const oW2m = [0, 1, 0, 1, 0, -eps / 6];
    return tf.tidy(() => {
      const batch = tf.tensor2d([
        obs0, oThp, oThm, oTh2p, oTh2m, oW1p, oW1m, oW2p, oW2m,
      ]);
      const out = this.netT(batch).dataSync();
      const kTh1 = (out[1] - out[2]) / (2 * eps);
      const kTh2 = (out[3] - out[4]) / (2 * eps);
      const kW1 = (out[5] - out[6]) / (2 * eps);
      const kW2 = (out[7] - out[8]) / (2 * eps);
      return { kTh1, kTh2, kW1, kW2 };
    });
  }

  weightSnapshot(): { w1: Float32Array; w2: Float32Array; wm: Float32Array } {
    return {
      w1: this.W1.dataSync() as Float32Array,
      w2: this.W2.dataSync() as Float32Array,
      wm: this.Wm.dataSync() as Float32Array,
    };
  }
}

/* ================================================================
   rendering helpers
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

function jointPositions(s: EnvState) {
  // Both θ₁ and θ₂ are absolute angles from the downward vertical.
  const p0 = new THREE.Vector3(0, 0, 0);
  const p1 = new THREE.Vector3(
    L1 * Math.sin(s.th1),
    -L1 * Math.cos(s.th1),
    0,
  );
  const p2 = new THREE.Vector3(
    p1.x + L2 * Math.sin(s.th2),
    p1.y - L2 * Math.cos(s.th2),
    0,
  );
  return { p0, p1, p2 };
}

/* ================================================================
   pendulum mesh
================================================================ */

/* ---- fake-glow sprite texture (procedural radial gradient) ----
   Used as a post-processing-free halo behind joints / tip. Creating
   the texture once via CanvasTexture avoids shipping any asset and
   keeps the bundle small. Pattern adapted from fake-glow-material
   community examples for R3F. */
function makeGlowTexture(): THREE.Texture {
  const size = 128;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  const grad = ctx.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2,
  );
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

function PendulumMesh({
  envRef,
  convergedRef,
}: {
  envRef: MutableRefObject<EnvState>;
  convergedRef: MutableRefObject<boolean>;
}) {
  const rod1 = useRef<THREE.Mesh>(null);
  const rod2 = useRef<THREE.Mesh>(null);
  const rod1Core = useRef<THREE.Mesh>(null);
  const rod2Core = useRef<THREE.Mesh>(null);
  const j1Group = useRef<THREE.Group>(null);
  const j1Ring = useRef<THREE.Mesh>(null);
  const j1Sphere = useRef<THREE.Mesh>(null);
  const j1Glow = useRef<THREE.Sprite>(null);
  const tipGroup = useRef<THREE.Group>(null);
  const tip = useRef<THREE.Mesh>(null);
  const tipRing = useRef<THREE.Mesh>(null);
  const tipGlow = useRef<THREE.Sprite>(null);
  const trailRef = useRef<THREE.Vector3[]>([]);
  const trailGeoRef = useRef<THREE.BufferGeometry>(null);
  const goldRef = useRef(0);

  const glowTex = useMemo(() => makeGlowTexture(), []);
  const TRAIL_LEN = 72;

  useFrame((_, dt) => {
    const s = envRef.current;
    const { p0, p1, p2 } = jointPositions(s);
    if (rod1.current) setRod(rod1.current, p0, p1);
    if (rod2.current) setRod(rod2.current, p1, p2);
    if (rod1Core.current) setRod(rod1Core.current, p0, p1);
    if (rod2Core.current) setRod(rod2Core.current, p1, p2);
    if (j1Group.current) j1Group.current.position.copy(p1);
    if (tipGroup.current) tipGroup.current.position.copy(p2);

    // keep accent rings facing the camera so they read as flat discs
    if (j1Ring.current) j1Ring.current.rotation.z += dt * 0.4;
    if (tipRing.current) tipRing.current.rotation.z -= dt * 0.5;

    // Smoothly ease into the converged/gold celebration state.
    const target = convergedRef.current ? 1 : 0;
    const k = 1 - Math.exp(-dt * 2.2);
    goldRef.current += (target - goldRef.current) * k;
    const g = goldRef.current;
    const pulse = 0.9 + Math.sin(performance.now() * 0.0042) * 0.18;

    const lerpPhysical = (
      mesh: THREE.Mesh | null,
      base: THREE.Color,
      emissiveI: number,
    ) => {
      if (!mesh) return;
      const mat = mesh.material as THREE.MeshPhysicalMaterial;
      mat.color.copy(base).lerp(COLOR_GOLD, g);
      mat.emissive.copy(base).lerp(COLOR_GOLD, g);
      mat.emissiveIntensity = emissiveI + g * 0.6 * pulse;
    };
    lerpPhysical(rod1.current, COLOR_ROD1, 0.35);
    lerpPhysical(rod2.current, COLOR_ROD2, 0.35);
    lerpPhysical(j1Sphere.current, COLOR_ROD1, 0.9);
    lerpPhysical(tip.current, COLOR_ROD2, 1.4);

    // inner cores (thin glowing lines through each rod)
    const lerpCore = (mesh: THREE.Mesh | null, base: THREE.Color) => {
      if (!mesh) return;
      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.color.copy(base).lerp(COLOR_GOLD, g);
    };
    lerpCore(rod1Core.current, COLOR_ROD1);
    lerpCore(rod2Core.current, COLOR_ROD2);

    // halo sprite tint + breathing scale
    const haloBreath = 1 + Math.sin(performance.now() * 0.0032) * 0.06;
    const tintHalo = (spr: THREE.Sprite | null, base: THREE.Color) => {
      if (!spr) return;
      const mat = spr.material as THREE.SpriteMaterial;
      mat.color.copy(base).lerp(COLOR_GOLD, g);
    };
    tintHalo(j1Glow.current, COLOR_ROD1);
    tintHalo(tipGlow.current, COLOR_ROD2);
    if (j1Glow.current) j1Glow.current.scale.setScalar(0.75 * haloBreath);
    if (tipGlow.current) tipGlow.current.scale.setScalar(1.05 * haloBreath);

    // gradient trail — vertex colors fade from current (bright) to tail (0 alpha)
    trailRef.current.push(p2.clone());
    if (trailRef.current.length > TRAIL_LEN) trailRef.current.shift();
    const geo = trailGeoRef.current;
    if (geo) {
      const n = trailRef.current.length;
      const pos = new Float32Array(n * 3);
      const col = new Float32Array(n * 3);
      const baseTrail = new THREE.Color().copy(COLOR_ROD2).lerp(COLOR_GOLD, g);
      for (let i = 0; i < n; i++) {
        const v = trailRef.current[i];
        pos[i * 3] = v.x;
        pos[i * 3 + 1] = v.y;
        pos[i * 3 + 2] = v.z;
        const t = i / Math.max(1, n - 1); // 0 at tail, 1 at head
        col[i * 3] = baseTrail.r * t;
        col[i * 3 + 1] = baseTrail.g * t;
        col[i * 3 + 2] = baseTrail.b * t;
      }
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
      geo.setDrawRange(0, n);
      geo.computeBoundingSphere();
    }
  });

  return (
    <group>
      {/* ---- stepped mounting base (plinth) ---- */}
      <mesh position={[0, -0.02, -0.05]}>
        <cylinderGeometry args={[0.28, 0.34, 0.06, 32]} />
        <meshPhysicalMaterial
          color="#0b0f1a"
          metalness={0.85}
          roughness={0.28}
        />
      </mesh>
      <mesh position={[0, 0.03, -0.05]}>
        <cylinderGeometry args={[0.2, 0.22, 0.08, 32]} />
        <meshPhysicalMaterial
          color="#1a1f2e"
          metalness={0.9}
          roughness={0.2}
          clearcoat={0.6}
          clearcoatRoughness={0.2}
          emissive="#22d3ee"
          emissiveIntensity={0.12}
        />
      </mesh>
      {/* accent ring around the base */}
      <mesh position={[0, 0.07, -0.05]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.215, 0.008, 12, 48]} />
        <meshBasicMaterial color="#22d3ee" />
      </mesh>
      {/* pivot bolt */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.09, 24, 24]} />
        <meshPhysicalMaterial
          color="#e8ecf5"
          metalness={1}
          roughness={0.15}
          clearcoat={1}
          clearcoatRoughness={0.1}
          emissive="#ffffff"
          emissiveIntensity={0.35}
        />
      </mesh>

      {/* ---- rod 1: tapered physical + glowing core ---- */}
      <mesh ref={rod1}>
        <cylinderGeometry args={[0.048, 0.06, 1, 18]} />
        <meshPhysicalMaterial
          color="#22d3ee"
          emissive="#22d3ee"
          emissiveIntensity={0.35}
          metalness={0.6}
          roughness={0.2}
          clearcoat={0.9}
          clearcoatRoughness={0.15}
        />
      </mesh>
      <mesh ref={rod1Core}>
        <cylinderGeometry args={[0.018, 0.018, 1.01, 10]} />
        <meshBasicMaterial color="#22d3ee" toneMapped={false} />
      </mesh>

      {/* ---- rod 2: tapered physical + glowing core ---- */}
      <mesh ref={rod2}>
        <cylinderGeometry args={[0.042, 0.055, 1, 18]} />
        <meshPhysicalMaterial
          color="#7c5cff"
          emissive="#7c5cff"
          emissiveIntensity={0.35}
          metalness={0.6}
          roughness={0.2}
          clearcoat={0.9}
          clearcoatRoughness={0.15}
        />
      </mesh>
      <mesh ref={rod2Core}>
        <cylinderGeometry args={[0.016, 0.016, 1.01, 10]} />
        <meshBasicMaterial color="#7c5cff" toneMapped={false} />
      </mesh>

      {/* ---- elbow joint assembly: halo + accent ring + sphere ---- */}
      <group ref={j1Group}>
        <sprite ref={j1Glow}>
          <spriteMaterial
            map={glowTex}
            color="#22d3ee"
            transparent
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            toneMapped={false}
          />
        </sprite>
        <mesh ref={j1Ring} rotation={[0, 0, 0]}>
          <torusGeometry args={[0.135, 0.008, 10, 40]} />
          <meshBasicMaterial color="#22d3ee" toneMapped={false} />
        </mesh>
        <mesh ref={j1Sphere}>
          <sphereGeometry args={[0.1, 24, 24]} />
          <meshPhysicalMaterial
            color="#22d3ee"
            emissive="#22d3ee"
            emissiveIntensity={0.9}
            metalness={0.7}
            roughness={0.15}
            clearcoat={1}
            clearcoatRoughness={0.08}
          />
        </mesh>
      </group>

      {/* ---- tip assembly: big halo + accent ring + sphere ---- */}
      <group ref={tipGroup}>
        <sprite ref={tipGlow}>
          <spriteMaterial
            map={glowTex}
            color="#7c5cff"
            transparent
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            toneMapped={false}
          />
        </sprite>
        <mesh ref={tipRing}>
          <torusGeometry args={[0.17, 0.009, 12, 48]} />
          <meshBasicMaterial color="#7c5cff" toneMapped={false} />
        </mesh>
        <mesh ref={tip}>
          <sphereGeometry args={[0.13, 28, 28]} />
          <meshPhysicalMaterial
            color="#7c5cff"
            emissive="#7c5cff"
            emissiveIntensity={1.4}
            metalness={0.7}
            roughness={0.15}
            clearcoat={1}
            clearcoatRoughness={0.08}
          />
        </mesh>
      </group>

      {/* ---- tip trail (vertex-colored fade) ---- */}
      <line>
        <bufferGeometry ref={trailGeoRef} />
        <lineBasicMaterial
          vertexColors
          transparent
          opacity={0.85}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </line>
    </group>
  );
}

/* ================================================================
   training + visible env loop
================================================================ */

type GrabMode = "none" | "p1" | "p2";
type GrabState = { mode: GrabMode; wx: number; wy: number };

/* Mode toggle rendered inside the R3F scene via drei <Html>, so its
   screen position is driven by the pivot's projection and stays put
   across aspect-ratio changes. */
function PendulumModeToggle({
  mode,
  trainingDone,
  goTrain,
  goControl,
}: {
  mode: Mode;
  trainingDone: boolean;
  goTrain: () => void;
  goControl: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-1.5 w-max">
      {trainingDone && mode !== "control" && (
        <div className="text-[9px] font-mono uppercase tracking-[0.22em] text-[#fbbf24] whitespace-nowrap drop-shadow-[0_0_10px_rgba(251,191,36,0.6)] animate-pulse">
          ✦ training successful — now control!
        </div>
      )}
      <div className="flex gap-1" style={{ pointerEvents: "auto" }}>
        <button
          type="button"
          onClick={goTrain}
          className={`rounded-md px-2.5 py-1 text-[9px] font-mono uppercase tracking-[0.2em] border transition-colors ${
            mode === "training"
              ? "bg-accent2/20 border-accent2 text-accent2"
              : "bg-black/40 border-white/10 text-gray-400 hover:bg-white/10"
          }`}
        >
          ● train
        </button>
        <button
          type="button"
          onClick={goControl}
          className={`rounded-md px-2.5 py-1 text-[9px] font-mono uppercase tracking-[0.2em] border transition-all ${
            mode === "control"
              ? `bg-[#fbbf24]/20 border-[#fbbf24] text-[#fbbf24]${
                  trainingDone ? " shadow-[0_0_18px_rgba(251,191,36,0.55)] ilc-breathe" : ""
                }`
              : trainingDone
                ? "bg-[#fbbf24]/10 border-[#fbbf24]/60 text-[#fbbf24] shadow-[0_0_14px_rgba(251,191,36,0.45)] ilc-breathe"
                : "bg-black/40 border-white/10 text-gray-400 hover:bg-white/10"
          }`}
        >
          ✦ control
        </button>
      </div>
    </div>
  );
}

function SceneRunner({
  agentRef,
  visibleEnvRef,
  convergedRef,
  returnsRef,
  grabRef,
  statsRef,
  modeRef,
  hud,
}: {
  agentRef: MutableRefObject<Agent | null>;
  visibleEnvRef: MutableRefObject<EnvState>;
  convergedRef: MutableRefObject<boolean>;
  returnsRef: MutableRefObject<number[]>;
  grabRef: MutableRefObject<GrabState>;
  statsRef: MutableRefObject<AgentStats>;
  modeRef: MutableRefObject<Mode>;
  hud: {
    mode: Mode;
    trainingDone: boolean;
    goTrain: () => void;
    goControl: () => void;
  };
}) {
  // Replay buffer is fed exclusively from the visible env. The user's
  // interactions (and the initial kick from initState) are the sole
  // source of state diversity — no hidden parallel rollout.
  const bufObs = useRef<number[][]>([]);
  const bufTgt = useRef<number[]>([]);
  const holdRef = useRef(0);

  const BUF_MAX = 2400;
  const BATCH = 48;
  // For swing-up the teacher is switched (energy-pump vs LQR), so MSE
  // bottoms out higher than a pure-PD target. Use sustained hold-time
  // at the inverted target (judged on the visible env) as the real
  // completion signal instead.
  const HOLD_FRAMES_NEEDED = 90;

  useFrame(() => {
    const agent = agentRef.current;
    if (!agent || !agent.ready) return;
    const mode = modeRef.current;

    // ---- visible env: network in control, user can grab in any mode ----
    if (grabRef.current.mode !== "none") {
      const s = visibleEnvRef.current;
      const { wx, wy } = grabRef.current;
      if (grabRef.current.mode === "p1") {
        const a = Math.atan2(wx, -wy);
        visibleEnvRef.current = { th1: a, th2: s.th2, w1: 0, w2: 0 };
      } else {
        const { th1, th2 } = ik2(wx, wy, s.th1, s.th2);
        visibleEnvRef.current = { th1, th2, w1: 0, w2: 0 };
      }
    } else {
      // Training mode: the visible pendulum runs as FREE PHYSICS (zero
      // motor torque). The learned policy is NOT applied here — all
      // training happens invisibly in the hidden rollout env below.
      // This way "training" is honestly just the network watching and
      // learning, not secretly controlling.
      //
      // Control mode: the network drives the pendulum via its frozen
      // learned weights (actDet).
      let u = 0;
      if (mode === "control") {
        const o = obsOf(visibleEnvRef.current);
        u = agent.actDet(o);
        u = Math.max(-TORQUE_MAX, Math.min(TORQUE_MAX, u));
      }
      agent.stats.netAction = u;
      agent.stats.teacherAction = pdTeacher(visibleEnvRef.current);
      visibleEnvRef.current = stepEnv(visibleEnvRef.current, u);
    }
    // Safety clamp — never respawn, just clip runaway angular velocity
    // if the user yanks the tip absurdly fast.
    const vs = visibleEnvRef.current;
    if (Number.isFinite(vs.th1) && Math.abs(vs.w1) > 30) {
      visibleEnvRef.current = { ...vs, w1: Math.sign(vs.w1) * 30 };
    }
    if (Number.isFinite(vs.th2) && Math.abs(vs.w2) > 30) {
      visibleEnvRef.current = { ...visibleEnvRef.current, w2: Math.sign(vs.w2) * 30 };
    }
    statsRef.current.netAction = agent.stats.netAction;
    statsRef.current.teacherAction = agent.stats.teacherAction;

    // Control mode: no training, no data collection.
    if (mode !== "training") return;

    // ============= TRAINING DATA COLLECTION =============
    // Sample the visible env (which the user is actively disturbing) —
    // this way the user's interactions directly shape the training set.
    {
      const to = obsOf(visibleEnvRef.current);
      bufObs.current.push(to);
      bufTgt.current.push(pdTeacher(visibleEnvRef.current));
      if (bufObs.current.length > BUF_MAX) {
        bufObs.current.shift();
        bufTgt.current.shift();
      }
    }

    // ---- BC mini-batch update ----
    if (bufObs.current.length >= BATCH) {
      const obsB: number[][] = new Array(BATCH);
      const tgtB: number[] = new Array(BATCH);
      for (let k = 0; k < BATCH; k++) {
        const idx = Math.floor(Math.random() * bufObs.current.length);
        obsB[k] = bufObs.current[idx];
        tgtB[k] = bufTgt.current[idx];
      }
      agent.trainBatch(obsB, tgtB);
    }

    // ---- completion check (loss-based) ----
    // Training mode never runs the policy on the visible env, so we
    // can't judge by on-screen behavior. Instead: the student is
    // "done" when its BC loss (MSE vs teacher) stays low for a
    // sustained window AND it has seen enough updates.
    const LOSS_COMPLETE = 0.02;
    const LOSS_HOLD_FRAMES = 30;
    const UPDATES_MIN = 500;
    // Diversity gate: low loss on a degenerate buffer (e.g. user pinning
    // the tip at one angle) is meaningless. Require the replay to have
    // actually seen a range of states before we call training complete.
    const DIVERSITY_MIN = 0.9; // summed std across th1,th2,w1,w2
    const progress = Math.max(0, Math.min(1, 1 - agent.stats.loss / 0.4));
    returnsRef.current.push(progress);
    if (returnsRef.current.length > 160) returnsRef.current.shift();
    let diversity = 0;
    if (bufObs.current.length >= 64) {
      // obs layout: [sin θ1, cos θ1, sin θ2, cos θ2, w1, w2]
      const n = bufObs.current.length;
      let m0 = 0, m1 = 0, m4 = 0, m5 = 0;
      for (let i = 0; i < n; i++) {
        const o = bufObs.current[i];
        m0 += o[0]; m1 += o[2]; m4 += o[4]; m5 += o[5];
      }
      m0 /= n; m1 /= n; m4 /= n; m5 /= n;
      let v0 = 0, v1 = 0, v4 = 0, v5 = 0;
      for (let i = 0; i < n; i++) {
        const o = bufObs.current[i];
        v0 += (o[0] - m0) ** 2;
        v1 += (o[2] - m1) ** 2;
        v4 += (o[4] - m4) ** 2;
        v5 += (o[5] - m5) ** 2;
      }
      diversity =
        Math.sqrt(v0 / n) +
        Math.sqrt(v1 / n) +
        Math.sqrt(v4 / n) +
        Math.sqrt(v5 / n);
    }
    if (
      agent.stats.loss < LOSS_COMPLETE &&
      agent.stats.updates > UPDATES_MIN &&
      diversity > DIVERSITY_MIN
    ) {
      holdRef.current += 1;
    } else {
      holdRef.current = 0;
    }
    if (
      holdRef.current >= LOSS_HOLD_FRAMES &&
      !agent.stats.trainingComplete
    ) {
      agent.stats.trainingComplete = true;
      convergedRef.current = true;
    }

    // ---- mirror stats to HUD ----
    statsRef.current.updates = agent.stats.updates;
    statsRef.current.loss = agent.stats.loss;
    statsRef.current.deployed = agent.stats.deployed;
    statsRef.current.teacherAction = agent.stats.teacherAction;
    statsRef.current.netAction = agent.stats.netAction;
    statsRef.current.trialsTotal = agent.stats.trialsTotal;
    statsRef.current.trialsSuccess = agent.stats.trialsSuccess;
    statsRef.current.successStreak = agent.stats.successStreak;
    statsRef.current.trainingComplete = agent.stats.trainingComplete;
  });

  return (
    // Shift the whole pendulum scene 3cm right / 2cm up in screen
    // units. At the default camera (z=5.6, fov 48) the viewport is
    // ~5 world units tall, so ~0.83 world ≈ 3cm and ~0.56 world ≈ 2cm.
    // Applied as a group so the grab plane coordinates move with the
    // pendulum (clicks still land on the balls).
    <group position={[1.8, 0.9, 0]}>
      <PendulumMesh envRef={visibleEnvRef} convergedRef={convergedRef} />
      <GrabPlane envRef={visibleEnvRef} grabRef={grabRef} />
      {/* Mode toggle anchored to the pendulum pivot so aspect-ratio
          changes can't detach it.  0.42 world units above the pivot
          lands roughly where the user parked it at 16:9. */}
      <Html
        position={[0, 0.42, 0]}
        center
        zIndexRange={[20, 0]}
        style={{ pointerEvents: "none" }}
      >
        <PendulumModeToggle {...hud} />
      </Html>
    </group>
  );
}

/* ================================================================
   grab plane — invisible plane at z=0 that catches pointer events
   and lets the user pick & drag either ball in world space
================================================================ */

function GrabPlane({
  envRef,
  grabRef,
}: {
  envRef: MutableRefObject<EnvState>;
  grabRef: MutableRefObject<GrabState>;
}) {
  const pick = (wx: number, wy: number): GrabMode => {
    const { p1, p2 } = jointPositions(envRef.current);
    const d1 = Math.hypot(wx - p1.x, wy - p1.y);
    const d2 = Math.hypot(wx - p2.x, wy - p2.y);
    const R = 0.35;
    if (d2 < d1 && d2 < R) return "p2";
    if (d1 < R) return "p1";
    return "none";
  };
  return (
    <mesh
      position={[0, 0, 0]}
      onPointerDown={(e) => {
        // e.point is in WORLD space; the pendulum group is shifted by
        // (PEND_OFFSET_X, PEND_OFFSET_Y). Convert to pendulum-local.
        const lx = e.point.x - PEND_OFFSET_X;
        const ly = e.point.y - PEND_OFFSET_Y;
        const which = pick(lx, ly);
        if (which === "none") return;
        e.stopPropagation();
        grabRef.current.mode = which;
        grabRef.current.wx = lx;
        grabRef.current.wy = ly;
        (e.target as Element)?.setPointerCapture?.(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (grabRef.current.mode === "none") return;
        grabRef.current.wx = e.point.x - PEND_OFFSET_X;
        grabRef.current.wy = e.point.y - PEND_OFFSET_Y;
      }}
      onPointerUp={(e) => {
        if (grabRef.current.mode === "none") return;
        grabRef.current.mode = "none";
        try {
          (e.target as Element)?.releasePointerCapture?.(e.pointerId);
        } catch {
          /* no-op */
        }
      }}
      onPointerOut={() => {
        grabRef.current.mode = "none";
      }}
    >
      <planeGeometry args={[24, 24]} />
      <meshBasicMaterial transparent opacity={0} depthWrite={false} />
    </mesh>
  );
}

/* ================================================================
   HUD overlays — reward sparkline, stats, NN viz
================================================================ */

function RewardSparkline({
  returnsRef,
}: {
  returnsRef: MutableRefObject<number[]>;
}) {
  const [, forceRender] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => forceRender((x) => x + 1), 400);
    return () => window.clearInterval(id);
  }, []);
  const W = 200;
  const H = 54;
  const data = returnsRef.current;
  const pad = 4;
  if (data.length < 2)
    return (
      <svg width={W} height={H} className="opacity-80">
        <rect
          width={W}
          height={H}
          rx={6}
          fill="rgba(0,0,0,0.4)"
          stroke="rgba(255,255,255,0.1)"
        />
      </svg>
    );
  const min = 0;
  const max = 1;
  const pts = data
    .map((v, i) => {
      const x = pad + (i / (data.length - 1)) * (W - 2 * pad);
      const y =
        H - pad - ((v - min) / (max - min)) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const lastY =
    H - pad - ((data[data.length - 1] - min) / (max - min)) * (H - 2 * pad);
  return (
    <svg width={W} height={H} className="opacity-90">
      <rect
        width={W}
        height={H}
        rx={6}
        fill="rgba(0,0,0,0.45)"
        stroke="rgba(255,255,255,0.1)"
      />
      {/* zero line */}
      <line
        x1={pad}
        x2={W - pad}
        y1={H / 2}
        y2={H / 2}
        stroke="rgba(255,255,255,0.08)"
        strokeDasharray="2 3"
      />
      <polyline
        points={pts}
        fill="none"
        stroke="#22d3ee"
        strokeWidth={1.4}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx={W - pad} cy={lastY} r={2.2} fill="#7c5cff" />
      <text
        x={pad + 4}
        y={11}
        fill="rgba(255,255,255,0.55)"
        fontSize={8}
        fontFamily="monospace"
        letterSpacing="0.12em"
      >
        BC PROGRESS
      </text>
    </svg>
  );
}

function GainsHud({ agentRef }: { agentRef: MutableRefObject<Agent | null> }) {
  const [gains, setGains] = useState<{
    kTh1: number; kTh2: number; kW1: number; kW2: number;
  } | null>(null);
  useEffect(() => {
    const id = window.setInterval(() => {
      const a = agentRef.current;
      if (a && a.ready) setGains(a.gainsSnapshot());
    }, 180);
    return () => window.clearInterval(id);
  }, [agentRef]);

  // Ground-truth linearized gains of the PD teacher at the origin:
  //   u = -KP1·sin θ₁ - KD1·ω₁ - KD2·ω₂
  const targets = [
    { label: "k_θ₁", truth: -KP1, got: gains?.kTh1 ?? 0 },
    { label: "k_θ₂", truth: 0,   got: gains?.kTh2 ?? 0 },
    { label: "k_ω₁", truth: -KD1, got: gains?.kW1 ?? 0 },
    { label: "k_ω₂", truth: -KD2, got: gains?.kW2 ?? 0 },
  ];
  const W = 200;
  const rowH = 13;
  const H = 16 + rowH * targets.length + 4;
  const cx = 92; // zero line
  const scale = 9; // world-unit → px per gain unit
  return (
    <svg width={W} height={H} className="opacity-95">
      <rect
        width={W}
        height={H}
        rx={6}
        fill="rgba(0,0,0,0.45)"
        stroke="rgba(255,255,255,0.1)"
      />
      <text
        x={8}
        y={11}
        fill="rgba(255,255,255,0.55)"
        fontSize={8}
        fontFamily="monospace"
        letterSpacing="0.12em"
      >
        LEARNED GAINS
      </text>
      {/* zero axis */}
      <line
        x1={cx}
        x2={cx}
        y1={14}
        y2={H - 2}
        stroke="rgba(255,255,255,0.12)"
        strokeDasharray="2 2"
      />
      {targets.map((t, i) => {
        const y = 20 + i * rowH;
        const truthX = cx + t.truth * scale;
        const gotX = cx + t.got * scale;
        const err = Math.abs(t.got - t.truth);
        const close = err < 0.4 ? "#34d399" : err < 1.2 ? "#fbbf24" : "#9ca3af";
        return (
          <g key={t.label}>
            <text
              x={8}
              y={y + 3}
              fill="rgba(255,255,255,0.55)"
              fontSize={8}
              fontFamily="monospace"
            >
              {t.label}
            </text>
            {/* teacher target marker */}
            <circle
              cx={truthX}
              cy={y}
              r={2.2}
              fill="none"
              stroke="#7c5cff"
              strokeWidth={1}
            />
            {/* learned value bar */}
            <line
              x1={cx}
              y1={y}
              x2={gotX}
              y2={y}
              stroke={close}
              strokeWidth={1.6}
              strokeLinecap="round"
            />
            <circle cx={gotX} cy={y} r={1.9} fill={close} />
            <text
              x={W - 4}
              y={y + 3}
              textAnchor="end"
              fill={close}
              fontSize={8}
              fontFamily="monospace"
            >
              {t.got.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function NetworkViz({ agentRef }: { agentRef: MutableRefObject<Agent | null> }) {
  const [snap, setSnap] = useState<{
    w1: Float32Array;
    w2: Float32Array;
    wm: Float32Array;
  } | null>(null);

  useEffect(() => {
    const id = window.setInterval(() => {
      const a = agentRef.current;
      if (a && a.ready) setSnap(a.weightSnapshot());
    }, 500);
    return () => window.clearInterval(id);
  }, [agentRef]);

  const W = 200;
  const H = 110;
  const IN = 6;
  const HID_SHOW = 10; // visualize subset of hidden layer
  const OUT = 1;

  const cols = [
    { n: IN, x: 16 },
    { n: HID_SHOW, x: W / 2 - 10 },
    { n: HID_SHOW, x: W / 2 + 40 },
    { n: OUT, x: W - 16 },
  ];
  const nodePositions = cols.map((c) =>
    Array.from({ length: c.n }, (_, i) => ({
      x: c.x,
      y: 14 + ((H - 28) * (i + 0.5)) / c.n,
    })),
  );

  const edgesLayer = (
    from: number,
    to: number,
    raw: Float32Array | null,
    fromN: number,
    toN: number,
  ) => {
    const lines: JSX.Element[] = [];
    for (let i = 0; i < fromN; i++) {
      for (let j = 0; j < toN; j++) {
        const w = raw ? raw[i * toN + j] ?? 0 : 0;
        const mag = Math.min(1, Math.abs(w) * 2.5);
        if (mag < 0.05) continue;
        const p1 = nodePositions[from][i];
        const p2 = nodePositions[to][j];
        lines.push(
          <line
            key={`${from}-${i}-${j}`}
            x1={p1.x}
            y1={p1.y}
            x2={p2.x}
            y2={p2.y}
            stroke={w > 0 ? "#22d3ee" : "#7c5cff"}
            strokeOpacity={mag * 0.75}
            strokeWidth={0.5 + mag * 1.1}
          />,
        );
      }
    }
    return lines;
  };

  return (
    <svg width={W} height={H} className="opacity-90">
      <rect
        width={W}
        height={H}
        rx={6}
        fill="rgba(0,0,0,0.45)"
        stroke="rgba(255,255,255,0.1)"
      />
      <text
        x={8}
        y={11}
        fill="rgba(255,255,255,0.55)"
        fontSize={8}
        fontFamily="monospace"
        letterSpacing="0.12em"
      >
        POLICY NETWORK
      </text>
      {/* edges */}
      {edgesLayer(0, 1, snap?.w1 ?? null, IN, HID_SHOW)}
      {edgesLayer(1, 2, snap?.w2 ?? null, HID_SHOW, HID_SHOW)}
      {edgesLayer(2, 3, snap?.wm ?? null, HID_SHOW, OUT)}
      {/* nodes */}
      {nodePositions.flatMap((col, ci) =>
        col.map((p, i) => (
          <circle
            key={`n-${ci}-${i}`}
            cx={p.x}
            cy={p.y}
            r={2.2}
            fill="#ffffff"
            fillOpacity={0.9}
          />
        )),
      )}
    </svg>
  );
}

function StatsHud({
  statsRef,
  convergedRef,
}: {
  statsRef: MutableRefObject<AgentStats>;
  convergedRef: MutableRefObject<boolean>;
}) {
  const [, r] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => r((x) => x + 1), 250);
    return () => window.clearInterval(id);
  }, []);
  const s = statsRef.current;
  const converged = convergedRef.current;
  return (
    <div className="rounded-md border border-white/10 bg-black/45 backdrop-blur-sm px-3 py-2 font-mono text-[10px] leading-snug text-gray-300 min-w-[200px]">
      <div className="flex items-center gap-2 mb-1">
        <span
          className={`inline-block w-1.5 h-1.5 rounded-full ${
            converged ? "bg-good" : "bg-accent2"
          } animate-pulse`}
        />
        <span className="uppercase tracking-[0.2em] text-[9px] text-gray-500">
          {converged ? "policy learned" : "net training · net in control"}
        </span>
      </div>
      <Row k="updates" v={`${s.updates}`} />
      <Row
        k="BC loss"
        v={s.loss.toFixed(4)}
        tint={s.loss < 0.1 ? "#34d399" : "#d1d5db"}
      />
      <Row
        k="teacher u"
        v={s.teacherAction.toFixed(2)}
      />
      <Row
        k="net u"
        v={s.netAction.toFixed(2)}
        tint={converged ? "#34d399" : "#9ca3af"}
      />
    </div>
  );
}

function Row({ k, v, tint }: { k: string; v: string; tint?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 tabular-nums">
      <span className="text-gray-500 uppercase tracking-wider text-[9px]">
        {k}
      </span>
      <span style={{ color: tint ?? "#e5e7eb" }}>{v}</span>
    </div>
  );
}

/* ================================================================
   top-level scene
================================================================ */

export default function PendulumScene() {
  const agentRef = useRef<Agent | null>(null);
  const visibleEnvRef = useRef<EnvState>(initState());
  const convergedRef = useRef(false);
  const returnsRef = useRef<number[]>([]);
  const grabRef = useRef<GrabState>({ mode: "none", wx: 0, wy: 0 });
  const statsRef = useRef<AgentStats>({
    updates: 0,
    loss: 1,
    deployed: false,
    teacherAction: 0,
    netAction: 0,
    trialsTotal: 0,
    trialsSuccess: 0,
    successStreak: 0,
    trainingComplete: false,
  });
  const [agentReady, setAgentReady] = useState(false);
  const [mode, setMode] = useState<Mode>("training");
  const [trainingDone, setTrainingDone] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const modeRef = useRef<Mode>("training");
  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const a = new Agent();
      await a.init();
      if (cancelled) return;
      agentRef.current = a;
      // Give the user something to look at immediately — small displacement
      // so the network has motion to learn on even before the first grab.
      visibleEnvRef.current = initState();
      setAgentReady(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Poll for training completion — flips a one-shot notification toast
  // and unlocks the Control button.
  useEffect(() => {
    if (trainingDone) return;
    const id = window.setInterval(() => {
      if (statsRef.current.trainingComplete) {
        setTrainingDone(true);
        setShowToast(true);
        window.setTimeout(() => setShowToast(false), 4200);
      }
    }, 150);
    return () => window.clearInterval(id);
  }, [trainingDone]);

  const goTrain = () => setMode("training");
  const goControl = () => setMode("control");

  return (
    <div className="relative w-full h-full" style={{ touchAction: "none" }}>
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
        {/* sculptural rim light from upper-right for clearcoat highlight */}
        <directionalLight
          position={[4, 5, 2]}
          intensity={0.9}
          color="#e8ecf5"
        />
        {/* cool fill from below to keep underside from going pure black */}
        <hemisphereLight args={["#1a2540", "#0a0f1e", 0.25]} />
        {agentReady && (
          <SceneRunner
            agentRef={agentRef}
            visibleEnvRef={visibleEnvRef}
            convergedRef={convergedRef}
            returnsRef={returnsRef}
            grabRef={grabRef}
            statsRef={statsRef}
            modeRef={modeRef}
            hud={{ mode, trainingDone, goTrain, goControl }}
          />
        )}
      </Canvas>

      {/* HUD — bottom-right stack */}
      <div className="absolute bottom-5 right-5 z-10 flex flex-col gap-2 items-end pointer-events-none">
        <GainsHud agentRef={agentRef} />
        <RewardSparkline returnsRef={returnsRef} />
        <NetworkViz agentRef={agentRef} />
        <StatsHud statsRef={statsRef} convergedRef={convergedRef} />
      </div>

      {/* subtle continuous "ready" breathing animation for the control btn */}
      <style jsx>{`
        :global(.ilc-breathe) {
          animation: ilc-breathe 2.4s ease-in-out infinite;
        }
        @keyframes ilc-breathe {
          0%,
          100% {
            box-shadow: 0 0 10px rgba(251, 191, 36, 0.35),
              0 0 0 0 rgba(251, 191, 36, 0);
          }
          50% {
            box-shadow: 0 0 22px rgba(251, 191, 36, 0.7),
              0 0 0 4px rgba(251, 191, 36, 0.12);
          }
        }
      `}</style>

      {/* interaction hint — bottom-left */}
      <div className="absolute bottom-5 left-5 z-10 text-[10px] font-mono uppercase tracking-[0.22em] text-gray-500 pointer-events-none max-w-[280px] leading-relaxed">
        <div className="text-accent2 mb-1">goal · damp to rest</div>
        {mode === "training"
          ? "↳ uncontrolled physics — grab to disturb while the net learns"
          : "↳ grab either ball — the learned policy will recover it"}
      </div>

      {/* training-complete notification toast */}
      {showToast && (
        <div className="absolute top-6 left-1/2 -translate-x-1/2 z-20 rounded-full bg-gradient-to-r from-[#fbbf24]/20 to-[#f59e0b]/20 border border-[#fbbf24]/60 backdrop-blur px-5 py-2 text-[11px] font-mono uppercase tracking-[0.22em] text-[#fbbf24] shadow-[0_0_30px_rgba(251,191,36,0.4)] animate-pulse pointer-events-none">
          ✦ policy learned — control unlocked
        </div>
      )}

      {/* loading shim */}
      {!agentReady && (
        <div className="absolute inset-0 flex items-center justify-center text-[10px] font-mono uppercase tracking-[0.22em] text-gray-500">
          initializing controller…
        </div>
      )}
    </div>
  );
}

"use client";

/* ------------------------------------------------------------------
   AWSTopology3D — two-layer R3F scene for the live match simulator.

     ┌─────────── ARENA (y > 0) ────────────┐
     │   30 player orbs in a ring           │
     │   prize pool number floating center  │
     └────────── (vertical wires) ──────────┘
     ┌──── INFRASTRUCTURE (y < -2) ─────────┐
     │   [alb]  [gamelift]   (ingress)      │
     │     ↓                                │
     │   [ec2] → [sqs] → [postgres]         │
     │     ↘       ↓        ↑               │
     │           [lambda] → [s3]            │
     └──────────────────────────────────────┘

   Reads a Snapshot per frame (snapshotRef.current) and:
     - drives orb pulse / flash / shrink-on-elim
     - drives island heat → emissive
     - drives Lambda cluster materialization during distribute phase
     - moves a packet pool along source→sink straight lerps
------------------------------------------------------------------ */

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, RoundedBox, Text, Cylinder, Html } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { useMemo, useRef, type MutableRefObject } from "react";
import * as THREE from "three";
import { ISLAND_META, type IslandId, type Snapshot } from "@/lib/aws-sim";

const ARENA_Y = 0;
const INFRA_Y = -3.0;
const ARENA_RING_R = 5.6;

const ISLAND_POS: Record<IslandId, THREE.Vector3> = {
  // Front row (ingress tier) — closest to arena (high z).
  // ALB centered under the arena so the player→lobby flow reads top-down;
  // GameLift to the right as a peer front-tier that picks up telemetry.
  alb:      new THREE.Vector3( 0,   INFRA_Y,  6.5),
  gamelift: new THREE.Vector3( 6,   INFRA_Y,  6.5),
  // Core row: EC2 (compute) ← SQS (fan-out) → Postgres (state of record).
  ec2:      new THREE.Vector3(-6,   INFRA_Y,  2.3),
  sqs:      new THREE.Vector3( 0,   INFRA_Y,  2.3),
  postgres: new THREE.Vector3( 6,   INFRA_Y,  2.3),
  // Back row: Lambda (scoring/payouts) + S3 (replays + ledger).
  s3:       new THREE.Vector3(-5,   INFRA_Y, -2.5),
  lambda:   new THREE.Vector3( 4,   INFRA_Y, -2.5),
};

const MAX_PARTICLES = 140;

/* ============================================================
   particle pool
============================================================ */

function nodeWorldPos(
  kind: "player" | IslandId,
  idx: number,
  snap: Snapshot,
  out: THREE.Vector3,
): THREE.Vector3 | null {
  if (kind === "player") {
    const p = snap.players[idx];
    if (!p) return null;
    out.set(p.posX, ARENA_Y + 0.4, p.posZ);
    return out;
  }
  const v = ISLAND_POS[kind];
  if (!v) return null;
  out.set(v.x, v.y + 0.7, v.z);
  return out;
}

function ParticlePool({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  const meshes = useRef<(THREE.Mesh | null)[]>([]);
  const tmp = useMemo(() => new THREE.Vector3(), []);
  const tmp2 = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    const snap = snapshotRef.current;
    const list = snap?.particles ?? [];
    for (let i = 0; i < MAX_PARTICLES; i++) {
      const m = meshes.current[i];
      if (!m) continue;
      const p = list[i];
      if (!p || !snap) { m.visible = false; continue; }
      const a = nodeWorldPos(p.fromKind, p.fromIdx, snap, tmp);
      const b = nodeWorldPos(p.toKind,   p.toIdx,   snap, tmp2);
      if (!a || !b) { m.visible = false; continue; }
      const t = Math.min(1, Math.max(0, (snap.now - p.startedAt) / p.travelMs));
      // smoothstep for cleaner motion
      const u = t * t * (3 - 2 * t);
      m.position.set(
        a.x + (b.x - a.x) * u,
        a.y + (b.y - a.y) * u + Math.sin(t * Math.PI) * 0.6, // gentle arc up
        a.z + (b.z - a.z) * u,
      );
      m.visible = true;
      const mat = m.material as THREE.MeshBasicMaterial;
      mat.color.set(p.color);
      // batch transfer is the headline particle — it carries the whole
      // match session. Payout + refund shards next-biggest, then bulk writes.
      const baseScale =
        p.kind === "batch"                                     ? 0.32 :
        p.kind === "payout" || p.kind === "refund"             ? 0.22 :
        p.kind === "snapshot" || p.kind === "ledger" || p.kind === "balance" ? 0.18 :
        p.kind === "sns" || p.kind === "sqsDeliver"            ? 0.16 :
        0.13;
      m.scale.setScalar(baseScale);
    }
  });

  return (
    <>
      {Array.from({ length: MAX_PARTICLES }).map((_, i) => (
        <mesh key={i} ref={(el) => { meshes.current[i] = el; }} visible={false}>
          <icosahedronGeometry args={[1, 0]} />
          <meshBasicMaterial color="#22d3ee" toneMapped={false} />
        </mesh>
      ))}
    </>
  );
}

/* ============================================================
   arena floor + center prize pool
============================================================ */

function ArenaFloor() {
  return (
    <group position={[0, ARENA_Y - 0.05, 0]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[ARENA_RING_R - 0.4, ARENA_RING_R + 0.5, 64]} />
        <meshBasicMaterial color="#22d3ee" toneMapped={false} transparent opacity={0.18} side={THREE.DoubleSide} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, 0]}>
        <circleGeometry args={[ARENA_RING_R + 0.4, 64]} />
        <meshStandardMaterial color="#0b1220" metalness={0.2} roughness={0.85} />
      </mesh>
      {/* center decoration */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
        <ringGeometry args={[1.1, 1.18, 64]} />
        <meshBasicMaterial color="#7c5cff" toneMapped={false} transparent opacity={0.5} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

function PrizePoolCenter({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  const ref = useRef<HTMLDivElement>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  useFrame(() => {
    const snap = snapshotRef.current;
    if (!snap) return;
    if (ref.current) {
      ref.current.textContent = `$${Math.round(snap.prizePool).toLocaleString()}`;
      const glow = snap.prizePoolGlowUntil > snap.now;
      ref.current.style.color = glow ? "#fde68a" : "#ffffff";
      ref.current.style.textShadow = glow
        ? "0 0 24px rgba(253,224,71,0.85), 0 0 8px rgba(253,224,71,0.6)"
        : "0 0 12px rgba(34,211,238,0.6)";
      ref.current.style.transform = glow ? "scale(1.08)" : "scale(1)";
    }
    if (ringRef.current) {
      const mat = ringRef.current.material as THREE.MeshBasicMaterial;
      const glow = snap.prizePoolGlowUntil > snap.now;
      mat.opacity = glow ? 0.95 : 0.4;
      ringRef.current.scale.setScalar(glow ? 1.15 : 1);
    }
  });
  return (
    <group position={[0, ARENA_Y + 1.4, 0]}>
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.0, 0.04, 8, 64]} />
        <meshBasicMaterial color="#fbbf24" toneMapped={false} transparent opacity={0.4} />
      </mesh>
      <Html center distanceFactor={9} style={{ pointerEvents: "none" }}>
        <div style={{ textAlign: "center", fontFamily: "ui-monospace, Menlo, monospace" }}>
          <div style={{ fontSize: 8, letterSpacing: 2, color: "#94a3b8", textTransform: "uppercase" }}>
            prize pool
          </div>
          <div
            ref={ref}
            style={{
              fontSize: 22,
              fontWeight: 700,
              fontVariantNumeric: "tabular-nums",
              color: "#fff",
              transition: "color 0.2s, transform 0.2s, text-shadow 0.2s",
            }}
          >
            $0
          </div>
        </div>
      </Html>
    </group>
  );
}

/* ============================================================
   player orbs (30 of them)
============================================================ */

function PlayerOrbs({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  const refs = useRef<(THREE.Mesh | null)[]>([]);
  const matRefs = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const haloRefs = useRef<(THREE.Mesh | null)[]>([]);

  useFrame(() => {
    const snap = snapshotRef.current;
    if (!snap) return;
    for (let i = 0; i < snap.players.length; i++) {
      const p = snap.players[i];
      const m = refs.current[i];
      const mat = matRefs.current[i];
      const halo = haloRefs.current[i];
      if (!m || !mat) continue;

      // visibility: hidden until joined
      const joined = p.joinedAt !== null;
      const visible = joined;
      if (m.visible !== visible) m.visible = visible;
      if (halo) halo.visible = visible;

      // position
      m.position.set(p.posX, ARENA_Y + 0.4, p.posZ);
      if (halo) halo.position.set(p.posX, ARENA_Y + 0.05, p.posZ);

      // alive vs eliminated: shrink + sink
      const targetScale = !p.alive ? 0.18 : (p.isUser ? 0.34 : 0.26);
      const targetY = !p.alive ? ARENA_Y - 0.25 : ARENA_Y + 0.4;
      m.scale.setScalar(m.scale.x + (targetScale - m.scale.x) * 0.18);
      m.position.y += (targetY - m.position.y) * 0.18;

      // pulse
      const pulse = p.pulseUntil > snap.now ? (p.pulseUntil - snap.now) / 600 : 0;
      const flashOn = p.flashUntil > snap.now;
      const payoutOn = p.payoutFlashUntil > snap.now;

      // base color: user = cyan; alive = white; dead = grey
      let baseColor = "#e2e8f0";
      if (p.isUser) baseColor = "#22d3ee";
      if (!p.alive) baseColor = "#475569";
      mat.color.set(baseColor);
      // flash override
      if (flashOn && p.flashColor) mat.color.set(p.flashColor);
      if (payoutOn) mat.color.set("#34d399");

      // emissive — joined recently glows; pulse adds spike
      const recentJoin = joined && snap.now - (p.joinedAt ?? 0) < 600 ? 1 : 0;
      const target = (recentJoin + pulse) * 1.6 + (p.alive ? 0.6 : 0.2) + (payoutOn ? 1.5 : 0);
      mat.emissive.set(payoutOn ? "#34d399" : (flashOn && p.flashColor) ? p.flashColor : (p.isUser ? "#22d3ee" : "#fbbf24"));
      mat.emissiveIntensity += (target - mat.emissiveIntensity) * 0.18;

      // halo alpha
      if (halo) {
        const hMat = halo.material as THREE.MeshBasicMaterial;
        const haloTarget = p.isUser ? 0.45 : (payoutOn ? 0.6 : 0.18);
        hMat.opacity += (haloTarget - hMat.opacity) * 0.15;
      }
    }
  });

  return (
    <>
      {Array.from({ length: 30 }).map((_, i) => (
        <group key={i}>
          <mesh
            ref={(el) => { refs.current[i] = el; }}
            visible={false}
          >
            <sphereGeometry args={[1, 16, 16]} />
            <meshStandardMaterial
              ref={(el) => { matRefs.current[i] = el; }}
              color="#e2e8f0"
              emissive="#22d3ee"
              emissiveIntensity={0.4}
              metalness={0.3}
              roughness={0.4}
            />
          </mesh>
          <mesh
            ref={(el) => { haloRefs.current[i] = el; }}
            rotation={[-Math.PI / 2, 0, 0]}
            visible={false}
          >
            <ringGeometry args={[0.32, 0.4, 24]} />
            <meshBasicMaterial color="#22d3ee" toneMapped={false} transparent opacity={0.2} side={THREE.DoubleSide} />
          </mesh>
        </group>
      ))}
    </>
  );
}

/* ============================================================
   infrastructure islands
============================================================ */

function IslandBase({
  id, snapshotRef, children,
}: {
  id: IslandId;
  snapshotRef: MutableRefObject<Snapshot | null>;
  children: (matRef: MutableRefObject<THREE.MeshStandardMaterial | null>) => React.ReactNode;
}) {
  const matRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    const snap = snapshotRef.current;
    if (!snap || !matRef.current) return;
    const heat = snap.islandHeat[id] ?? 0;
    const target = 0.35 + heat * 2.2;
    matRef.current.emissiveIntensity += (target - matRef.current.emissiveIntensity) * 0.18;
    // lambda dim during phases 1 & 2
    if (id === "lambda") {
      const live = snap.health.lambda;
      const opacityTarget = live ? 1 : 0.18;
      matRef.current.transparent = !live;
      matRef.current.opacity += (opacityTarget - matRef.current.opacity) * 0.1;
      if (groupRef.current) {
        const sy = live ? 1 : 0.65;
        groupRef.current.scale.y += (sy - groupRef.current.scale.y) * 0.1;
      }
    }
  });

  return (
    <group ref={groupRef} position={ISLAND_POS[id]}>
      {/* island disc */}
      <mesh position={[0, -0.3, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.0, 1.4, 32]} />
        <meshBasicMaterial color={ISLAND_META[id].color} toneMapped={false} transparent opacity={0.22} side={THREE.DoubleSide} />
      </mesh>
      {children(matRef)}
      <Text position={[0, 1.5, 0]} fontSize={0.28} color="#fff" anchorX="center" outlineWidth={0.012} outlineColor="#000">
        {ISLAND_META[id].label}
      </Text>
      <Text position={[0, 1.22, 0]} fontSize={0.16} color={ISLAND_META[id].color} anchorX="center" outlineWidth={0.008} outlineColor="#000">
        {ISLAND_META[id].sub}
      </Text>
    </group>
  );
}

function Ec2Island({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  return (
    <IslandBase id="ec2" snapshotRef={snapshotRef}>
      {(matRef) => (
        <>
          <RoundedBox args={[1.5, 0.9, 1.5]} radius={0.12} smoothness={4}>
            <meshStandardMaterial ref={matRef} color="#1c1209" emissive={ISLAND_META.ec2.color} emissiveIntensity={0.4} metalness={0.55} roughness={0.4} />
          </RoundedBox>
          {/* CPU dots */}
          {[-0.35, 0, 0.35].flatMap((x) =>
            [-0.35, 0, 0.35].map((z) => (
              <mesh key={`${x}-${z}`} position={[x, 0.46, z]}>
                <boxGeometry args={[0.13, 0.04, 0.13]} />
                <meshBasicMaterial color={ISLAND_META.ec2.color} toneMapped={false} />
              </mesh>
            )),
          )}
        </>
      )}
    </IslandBase>
  );
}

function AlbIsland({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  // Horizontal routing slab with three rails fanning out — the
  // "load-balanced target groups" visual metaphor.
  return (
    <IslandBase id="alb" snapshotRef={snapshotRef}>
      {(matRef) => (
        <>
          <RoundedBox args={[1.9, 0.42, 1.3]} radius={0.1} smoothness={3}>
            <meshStandardMaterial ref={matRef} color="#051821" emissive={ISLAND_META.alb.color} emissiveIntensity={0.4} metalness={0.6} roughness={0.35} />
          </RoundedBox>
          {[-0.55, 0, 0.55].map((x) => (
            <mesh key={x} position={[x, 0.26, 0]}>
              <boxGeometry args={[0.22, 0.05, 0.9]} />
              <meshBasicMaterial color={ISLAND_META.alb.color} toneMapped={false} />
            </mesh>
          ))}
          <Text position={[0, 0.02, 0.68]} fontSize={0.32} color={ISLAND_META.alb.color} anchorX="center" anchorY="middle" outlineWidth={0.01} outlineColor="#000">
            ALB
          </Text>
        </>
      )}
    </IslandBase>
  );
}

function SqsIsland({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  // Horizontal pipe with discrete "message" slots — queue semantics.
  // Label shows live backlog + DLQ count so it feels instrumented.
  const bufferRef = useRef<HTMLDivElement>(null);
  useFrame(() => {
    const snap = snapshotRef.current;
    if (!snap || !bufferRef.current) return;
    const backlog = snap.fleet.sqsBacklog;
    const dlq = snap.fleet.sqsDlq;
    bufferRef.current.textContent = `${backlog} queued · ${dlq} DLQ`;
  });
  return (
    <IslandBase id="sqs" snapshotRef={snapshotRef}>
      {(matRef) => (
        <>
          <mesh rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[0.42, 0.42, 1.7, 18]} />
            <meshStandardMaterial ref={matRef} color="#07201a" emissive={ISLAND_META.sqs.color} emissiveIntensity={0.4} metalness={0.6} roughness={0.35} />
          </mesh>
          {[-0.6, -0.25, 0.1, 0.45].map((x) => (
            <mesh key={x} position={[x, 0, 0]}>
              <boxGeometry args={[0.16, 0.18, 0.18]} />
              <meshBasicMaterial color={ISLAND_META.sqs.color} toneMapped={false} />
            </mesh>
          ))}
          <Html center distanceFactor={10} position={[0, -0.55, 0]} style={{ pointerEvents: "none" }}>
            <div
              ref={bufferRef}
              style={{
                fontFamily: "ui-monospace, Menlo, monospace",
                fontSize: 9,
                letterSpacing: 1,
                color: ISLAND_META.sqs.color,
                textShadow: `0 0 6px ${ISLAND_META.sqs.color}80`,
                whiteSpace: "nowrap",
              }}
            >
              0 queued · 0 DLQ
            </div>
          </Html>
        </>
      )}
    </IslandBase>
  );
}

function PostgresIsland({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  return (
    <IslandBase id="postgres" snapshotRef={snapshotRef}>
      {(matRef) => (
        <>
          <Cylinder args={[0.85, 0.85, 1.4, 32]}>
            <meshStandardMaterial ref={matRef} color="#0c1424" emissive={ISLAND_META.postgres.color} emissiveIntensity={0.4} metalness={0.7} roughness={0.3} />
          </Cylinder>
          {[0.5, 0.1, -0.3].map((y) => (
            <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.83, 0.88, 36]} />
              <meshBasicMaterial color={ISLAND_META.postgres.color} toneMapped={false} side={THREE.DoubleSide} transparent opacity={0.85} />
            </mesh>
          ))}
        </>
      )}
    </IslandBase>
  );
}

function LambdaIsland({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  return (
    <IslandBase id="lambda" snapshotRef={snapshotRef}>
      {(matRef) => (
        <>
          {/* three node spheres */}
          {[
            [-0.55, 0,  0.0],
            [ 0.55, 0,  0.0],
            [ 0.0,  0,  0.6],
          ].map((p, i) => (
            <mesh key={i} position={[p[0], p[1], p[2]]}>
              <icosahedronGeometry args={[0.4, 0]} />
              <meshStandardMaterial
                ref={i === 0 ? matRef : undefined}
                color="#1f1605"
                emissive={ISLAND_META.lambda.color}
                emissiveIntensity={0.5}
                metalness={0.6}
                roughness={0.3}
                transparent
                opacity={0.5}
              />
            </mesh>
          ))}
          <Text position={[0, 0, 1.0]} fontSize={0.55} color={ISLAND_META.lambda.color} anchorX="center" anchorY="middle" outlineWidth={0.02} outlineColor="#000">
            λ
          </Text>
        </>
      )}
    </IslandBase>
  );
}

function GameLiftIsland({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  // Visualized as a small fleet of server pods — three rounded boxes in a
  // line — so it reads as a managed cluster rather than a single instance.
  // We let IslandBase drive pod[0]'s emissive via its own matRef; pods 1/2
  // track pod[0] in our own useFrame so the whole fleet glows together.
  const bufferRef = useRef<HTMLDivElement>(null);
  const matRefAll = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    const snap = snapshotRef.current;
    if (!snap) return;
    // sync pods 1 & 2 to pod 0's emissive intensity (which IslandBase drives)
    const lead = matRefAll.current[0];
    if (lead) {
      for (let i = 1; i < matRefAll.current.length; i++) {
        const m = matRefAll.current[i];
        if (m) m.emissiveIntensity = lead.emissiveIntensity;
      }
    }
    const live = snap.islandActive.gamelift;
    if (groupRef.current) {
      const targetScale = live ? 1 : 0.75;
      groupRef.current.scale.y += (targetScale - groupRef.current.scale.y) * 0.08;
    }
    if (bufferRef.current) {
      const heat = snap.islandHeat.gamelift ?? 0;
      const show = snap.phase === "match" || (snap.phase === "distribute" && heat > 0.2);
      bufferRef.current.style.opacity = show ? "1" : "0";
      bufferRef.current.textContent = `${snap.bufferedEvents} events buffered`;
    }
  });

  const offsets: [number, number, number][] = [
    [-0.9, 0, 0], [0, 0, 0], [0.9, 0, 0],
  ];

  return (
    <IslandBase id="gamelift" snapshotRef={snapshotRef}>
      {(mr) => (
        <group ref={groupRef}>
          {offsets.map((p, i) => (
            <RoundedBox key={i} args={[0.65, 0.75, 0.9]} radius={0.08} smoothness={3} position={p}>
              <meshStandardMaterial
                ref={(el) => {
                  matRefAll.current[i] = el;
                  if (i === 0) mr.current = el;
                }}
                color="#1a0c18"
                emissive={ISLAND_META.gamelift.color}
                emissiveIntensity={0.4}
                metalness={0.6}
                roughness={0.35}
              />
            </RoundedBox>
          ))}
          {offsets.map((p, i) => (
            <mesh key={`led-${i}`} position={[p[0], 0.42, 0.42]}>
              <boxGeometry args={[0.1, 0.04, 0.04]} />
              <meshBasicMaterial color={ISLAND_META.gamelift.color} toneMapped={false} />
            </mesh>
          ))}
          <Html center distanceFactor={10} position={[0, -0.75, 0]} style={{ pointerEvents: "none" }}>
            <div
              ref={bufferRef}
              style={{
                fontFamily: "ui-monospace, Menlo, monospace",
                fontSize: 9,
                letterSpacing: 1,
                color: ISLAND_META.gamelift.color,
                textShadow: `0 0 6px ${ISLAND_META.gamelift.color}80`,
                opacity: 0,
                transition: "opacity 0.2s",
                whiteSpace: "nowrap",
              }}
            >
              0 events buffered
            </div>
          </Html>
        </group>
      )}
    </IslandBase>
  );
}

function S3Island({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  return (
    <IslandBase id="s3" snapshotRef={snapshotRef}>
      {(matRef) => (
        <>
          {/* bucket frustum */}
          <mesh>
            <cylinderGeometry args={[0.55, 0.75, 0.95, 6]} />
            <meshStandardMaterial ref={matRef} color="#1a0f24" emissive={ISLAND_META.s3.color} emissiveIntensity={0.4} metalness={0.6} roughness={0.4} />
          </mesh>
          <mesh position={[0, 0.5, 0]}>
            <torusGeometry args={[0.55, 0.04, 8, 16]} />
            <meshBasicMaterial color={ISLAND_META.s3.color} toneMapped={false} />
          </mesh>
          <S3StackedCubes snapshotRef={snapshotRef} />
        </>
      )}
    </IslandBase>
  );
}

function S3StackedCubes({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  // visualize accumulated S3 objects as a stack of small amber cubes inside the bucket
  const groupRef = useRef<THREE.Group>(null);
  const cubes = useRef<(THREE.Mesh | null)[]>([]);
  const MAX = 12;
  useFrame(() => {
    const snap = snapshotRef.current;
    if (!snap) return;
    const objs = snap.s3Objects;
    const visible = Math.min(MAX, objs.length);
    for (let i = 0; i < MAX; i++) {
      const m = cubes.current[i];
      if (!m) continue;
      if (i < visible) {
        m.visible = true;
        const obj = objs[objs.length - visible + i];
        const isLedger = obj.kind === "ledger";
        const mat = m.material as THREE.MeshStandardMaterial;
        mat.color.set(isLedger ? "#fde047" : "#fbbf24");
        mat.emissive.set(isLedger ? "#fde047" : "#fbbf24");
        mat.emissiveIntensity = isLedger ? 0.9 : 0.45;
        // stack inside bucket
        m.position.set(0, -0.35 + i * 0.08, 0);
        m.scale.setScalar(isLedger ? 0.28 : 0.22);
      } else {
        m.visible = false;
      }
    }
  });
  return (
    <group ref={groupRef}>
      {Array.from({ length: MAX }).map((_, i) => (
        <mesh key={i} ref={(el) => { cubes.current[i] = el; }} visible={false}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={0.45} metalness={0.4} roughness={0.4} />
        </mesh>
      ))}
    </group>
  );
}

/* ============================================================
   wires between islands (static dim)
============================================================ */

const WIRES: Array<[IslandId, IslandId]> = [
  // ingress path
  ["alb", "ec2"],
  // synchronous state path
  ["ec2", "postgres"],
  // async fan-out path
  ["ec2", "sqs"],
  ["sqs", "lambda"],
  // scoring / payout / storage
  ["lambda", "postgres"],
  ["lambda", "s3"],
  // session fleet
  ["gamelift", "ec2"],
  ["gamelift", "s3"],
];

function InfraWires() {
  const tubes = useMemo(() => {
    return WIRES.map(([a, b]) => {
      const A = ISLAND_POS[a].clone(); A.y += 0.4;
      const B = ISLAND_POS[b].clone(); B.y += 0.4;
      const mid = A.clone().lerp(B, 0.5); mid.y += 0.2;
      const curve = new THREE.CatmullRomCurve3([A, mid, B]);
      const geom = new THREE.TubeGeometry(curve, 32, 0.025, 6, false);
      return { key: `${a}-${b}`, geom };
    });
  }, []);
  return (
    <>
      {tubes.map((t) => (
        <mesh key={t.key} geometry={t.geom}>
          <meshBasicMaterial color="#22d3ee" transparent opacity={0.15} toneMapped={false} />
        </mesh>
      ))}
    </>
  );
}


/* ============================================================
   scene
============================================================ */

function Scene({ snapshotRef }: { snapshotRef: MutableRefObject<Snapshot | null> }) {
  return (
    <>
      <ambientLight intensity={0.45} />
      <directionalLight position={[10, 12, 6]} intensity={0.55} />
      <directionalLight position={[-8, 8, -8]} intensity={0.3} color="#7c5cff" />
      <pointLight position={[0, 4, 0]} intensity={0.7} color="#22d3ee" distance={20} />
      <pointLight position={[4, -1, 4]} intensity={0.5} color="#fbbf24" distance={14} />

      <gridHelper args={[60, 60, "#1f2937", "#0b1220"]} position={[0, INFRA_Y - 0.85, 0]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, INFRA_Y - 0.86, 0]}>
        <planeGeometry args={[80, 80]} />
        <meshStandardMaterial color="#05070d" />
      </mesh>

      <ArenaFloor />
      <PrizePoolCenter snapshotRef={snapshotRef} />
      <PlayerOrbs snapshotRef={snapshotRef} />

      <InfraWires />
      <GameLiftIsland snapshotRef={snapshotRef} />
      <AlbIsland snapshotRef={snapshotRef} />
      <Ec2Island snapshotRef={snapshotRef} />
      <SqsIsland snapshotRef={snapshotRef} />
      <PostgresIsland snapshotRef={snapshotRef} />
      <LambdaIsland snapshotRef={snapshotRef} />
      <S3Island snapshotRef={snapshotRef} />

      <ParticlePool snapshotRef={snapshotRef} />

      <EffectComposer>
        <Bloom intensity={1.1} luminanceThreshold={0.22} luminanceSmoothing={0.9} mipmapBlur />
      </EffectComposer>

      <OrbitControls
        target={[0, -1.2, 0]}
        maxPolarAngle={Math.PI / 2 - 0.05}
        minDistance={6}
        maxDistance={40}
        enablePan={true}
        screenSpacePanning={true}
      />
    </>
  );
}

export default function AWSTopology3D({
  snapshotRef,
}: {
  snapshotRef: MutableRefObject<Snapshot | null>;
}) {
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [-4, 2, 18], fov: 42 }}
      style={{
        background: "radial-gradient(ellipse at center, #0b0f1a 0%, #05070d 70%)",
      }}
    >
      <Scene snapshotRef={snapshotRef} />
    </Canvas>
  );
}

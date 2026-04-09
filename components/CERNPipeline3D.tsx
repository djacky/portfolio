"use client";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { RoundedBox, Text, Html, Cylinder } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import { useRef, useMemo, useState, useEffect } from "react";
import * as THREE from "three";

/* ------------------------------------------------------------------
   CERN — backend request flight.
   One continuous camera-tracking shot following an HTTPS request as it
   traverses the live system: client → FastAPI → worker pool → H∞ solver
   → PostgreSQL → response back to client. Each node demonstrates a
   real backend skill in action (Pydantic validation, async routing,
   convex synthesis, transactional persistence).
------------------------------------------------------------------ */

/* ---------------- world layout ---------------- */

const POS = {
  client:    new THREE.Vector3(-14, 0,  5),
  gateway:   new THREE.Vector3( -8, 0,  0),
  workerC:   new THREE.Vector3( -2, 0, -3),  // worker pool center
  worker:    new THREE.Vector3( -2, 0, -3),  // selected worker (center cell)
  solver:    new THREE.Vector3(  5, 0.4, 1),
  postgres:  new THREE.Vector3( 11, 0,  4),
};

/* ---------------- phase timeline ---------------- */

const PHASES = {
  emit:    { start: 0.0,  end: 2.2  },
  gateway: { start: 2.2,  end: 4.6  },
  route:   { start: 4.6,  end: 6.6  },
  solve:   { start: 6.6,  end: 11.4 },
  persist: { start: 11.4, end: 13.4 },
  respond: { start: 13.4, end: 15.0 },
  done:    { start: 15.0, end: 16.6 },
};
const TOTAL_DURATION = 16.6;

/* ---------------- helpers ---------------- */

function smoothstep(a: number, b: number, x: number) {
  const t = Math.max(0, Math.min(1, (x - a) / Math.max(1e-6, b - a)));
  return t * t * (3 - 2 * t);
}
function lerpVec(a: THREE.Vector3, b: THREE.Vector3, t: number, out: THREE.Vector3) {
  out.copy(a).lerp(b, t);
  return out;
}

const _packetTmp = new THREE.Vector3();
function packetPos(t: number, out: THREE.Vector3): THREE.Vector3 {
  if (t < PHASES.emit.end) {
    return lerpVec(POS.client, POS.gateway, smoothstep(PHASES.emit.start, PHASES.emit.end, t), out);
  }
  if (t < PHASES.gateway.end) return out.copy(POS.gateway);
  if (t < PHASES.route.end) {
    return lerpVec(POS.gateway, POS.worker, smoothstep(PHASES.route.start, PHASES.route.end, t), out);
  }
  if (t < PHASES.solve.end) {
    const enterEnd = PHASES.solve.start + 0.7;
    if (t < enterEnd) {
      return lerpVec(POS.worker, POS.solver, smoothstep(PHASES.solve.start, enterEnd, t), out);
    }
    return out.copy(POS.solver);
  }
  if (t < PHASES.persist.end) {
    return lerpVec(POS.solver, POS.postgres, smoothstep(PHASES.persist.start, PHASES.persist.end, t), out);
  }
  if (t < PHASES.respond.end) {
    return lerpVec(POS.postgres, POS.client, smoothstep(PHASES.respond.start, PHASES.respond.end, t), out);
  }
  return out.copy(POS.client);
}

const _camPosTmp = new THREE.Vector3();
const _camLookTmp = new THREE.Vector3();
function cameraTarget(t: number, packet: THREE.Vector3, posOut: THREE.Vector3, lookOut: THREE.Vector3) {
  if (t < PHASES.emit.end) {
    posOut.set(packet.x + 1.5, packet.y + 4, packet.z + 8);
    lookOut.copy(packet);
    return;
  }
  if (t < PHASES.gateway.end) {
    const u = (t - PHASES.gateway.start) / (PHASES.gateway.end - PHASES.gateway.start);
    const ang = -0.4 + u * 0.5;
    posOut.set(POS.gateway.x + Math.sin(ang) * 7, 3, POS.gateway.z + Math.cos(ang) * 7);
    lookOut.set(POS.gateway.x + 1, 0.6, POS.gateway.z);
    return;
  }
  if (t < PHASES.route.end) {
    posOut.set(packet.x + 2, 4.5, packet.z + 6);
    lookOut.copy(packet);
    return;
  }
  if (t < PHASES.solve.end) {
    // hero orbit on solver
    const u = (t - PHASES.solve.start) / (PHASES.solve.end - PHASES.solve.start);
    const ang = -0.4 + u * 1.0;
    const r = 7.5 - u * 0.8;
    posOut.set(POS.solver.x + Math.sin(ang) * r, 3 + u * 0.6, POS.solver.z + Math.cos(ang) * r);
    lookOut.set(POS.solver.x, 1.4, POS.solver.z);
    return;
  }
  if (t < PHASES.persist.end) {
    posOut.set(packet.x + 1, 4, packet.z + 6.5);
    lookOut.copy(packet);
    return;
  }
  if (t < PHASES.respond.end) {
    posOut.set(packet.x, 5, packet.z + 9);
    lookOut.copy(packet);
    return;
  }
  // done — wide hero
  posOut.set(0, 6, 18);
  lookOut.set(-1, 0.5, 0);
}

/* ---------------- packet ---------------- */

function Packet({ tRef }: { tRef: React.MutableRefObject<number> }) {
  const ref = useRef<THREE.Mesh>(null);
  const tmp = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    if (!ref.current) return;
    packetPos(tRef.current, tmp);
    ref.current.position.copy(tmp);
    ref.current.position.y += 0.6 + Math.sin(performance.now() * 0.005) * 0.1;
    ref.current.rotation.y += 0.04;
    ref.current.rotation.x += 0.02;
  });
  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[0.22, 0]} />
      <meshStandardMaterial
        color="#22d3ee"
        emissive="#22d3ee"
        emissiveIntensity={3}
        toneMapped={false}
      />
    </mesh>
  );
}

/* ---------------- client terminal ---------------- */

function ClientNode({ tRef }: { tRef: React.MutableRefObject<number> }) {
  const responseRef = useRef<HTMLDivElement>(null);
  const [showResponse, setShowResponse] = useState(false);
  useFrame(() => {
    const inResp = tRef.current >= PHASES.respond.end - 0.1;
    if (inResp !== showResponse) setShowResponse(inResp);
  });
  return (
    <group position={POS.client}>
      <RoundedBox args={[1.6, 1.0, 1.0]} radius={0.08} smoothness={4}>
        <meshPhysicalMaterial
          color="#0f172a"
          emissive="#22d3ee"
          emissiveIntensity={0.3}
          metalness={0.6}
          roughness={0.35}
          clearcoat={1}
        />
      </RoundedBox>
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(1.6, 1.0, 1.0)]} />
        <lineBasicMaterial color="#22d3ee" transparent opacity={0.6} toneMapped={false} />
      </lineSegments>
      <Text position={[0, 1.0, 0]} fontSize={0.22} color="#9ca3af" anchorX="center">
        client
      </Text>
      <Text position={[0, 0.78, 0]} fontSize={0.14} color="#cbd5e1" anchorX="center" font="">
        commissioning UI
      </Text>
      {showResponse && (
        <Html position={[0, -0.8, 0]} distanceFactor={9} style={{ pointerEvents: "none", transform: "translate(-50%, -100%)" }}>
          <div ref={responseRef}
            style={{
              fontFamily: "ui-monospace, Menlo, monospace",
              fontSize: 9,
              color: "#34d399",
              background: "rgba(5,7,13,0.9)",
              border: "1px solid rgba(52,211,153,0.4)",
              borderRadius: 6,
              padding: "6px 9px",
              whiteSpace: "pre",
              boxShadow: "0 0 18px rgba(52,211,153,0.25)",
            }}>
{`200 OK
{ "R":[…], "S":[1,…],
  "T":[…], "id":"0x4f1a" }`}
          </div>
        </Html>
      )}
    </group>
  );
}

/* ---------------- FastAPI gateway with live Pydantic validation ---------------- */

const PYDANTIC_FIELDS = [
  "bandwidth_hz : float",
  "phase_margin : float",
  "gain_margin  : float",
  "sampling_ts  : float",
];

function GatewayNode({ tRef }: { tRef: React.MutableRefObject<number> }) {
  const [validated, setValidated] = useState(0);
  const matRef = useRef<THREE.MeshPhysicalMaterial>(null);
  useFrame(() => {
    const t = tRef.current;
    let n = 0;
    if (t > PHASES.gateway.start + 0.4) n = 1;
    if (t > PHASES.gateway.start + 0.9) n = 2;
    if (t > PHASES.gateway.start + 1.4) n = 3;
    if (t > PHASES.gateway.start + 1.9) n = 4;
    if (t < PHASES.gateway.start) n = 0;
    if (n !== validated) setValidated(n);
    if (matRef.current) {
      const active = t >= PHASES.emit.end - 0.2 && t <= PHASES.route.end;
      const target = active ? 1.6 : 0.35;
      matRef.current.emissiveIntensity += (target - matRef.current.emissiveIntensity) * 0.1;
    }
  });
  const showHud = validated > 0 || (tRef.current >= PHASES.gateway.start && tRef.current <= PHASES.route.end);
  return (
    <group position={POS.gateway}>
      {/* hexagonal slab */}
      <mesh rotation={[0, Math.PI / 6, 0]}>
        <cylinderGeometry args={[1.5, 1.5, 1.4, 6]} />
        <meshPhysicalMaterial
          ref={matRef}
          color="#0f172a"
          emissive="#22d3ee"
          emissiveIntensity={0.35}
          metalness={0.85}
          roughness={0.22}
          clearcoat={1}
          clearcoatRoughness={0.1}
        />
      </mesh>
      {/* edge ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0.71, 0]}>
        <ringGeometry args={[1.45, 1.55, 6]} />
        <meshBasicMaterial color="#22d3ee" toneMapped={false} side={THREE.DoubleSide} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, -0.71, 0]}>
        <ringGeometry args={[1.45, 1.55, 6]} />
        <meshBasicMaterial color="#22d3ee" toneMapped={false} side={THREE.DoubleSide} />
      </mesh>
      <Text position={[0, 1.4, 0]} fontSize={0.28} color="#ffffff" anchorX="center" outlineWidth={0.01} outlineColor="#000">
        FastAPI
      </Text>
      <Text position={[0, 1.08, 0]} fontSize={0.18} color="#e5e7eb" outlineWidth={0.008} outlineColor="#000" anchorX="center">
        gateway · async
      </Text>

      {/* Pydantic validation HUD */}
      <Html position={[0, -1.4, 0]} distanceFactor={9} style={{ pointerEvents: "none", transform: "translate(-50%, -100%)" }}>
        <div
          style={{
            fontFamily: "ui-monospace, Menlo, monospace",
            fontSize: 10,
            background: "rgba(5,7,13,0.88)",
            border: "1px solid rgba(34,211,238,0.45)",
            boxShadow: "0 0 22px rgba(34,211,238,0.25)",
            borderRadius: 8,
            padding: "8px 12px",
            color: "#cbd5e1",
            minWidth: 200,
            backdropFilter: "blur(4px)",
            opacity: showHud ? 1 : 0.25,
            transition: "opacity 0.3s",
          }}
        >
          <div style={{ color: "#22d3ee", fontWeight: 600, marginBottom: 4 }}>
            class SynthRequest(BaseModel):
          </div>
          {PYDANTIC_FIELDS.map((f, i) => {
            const ok = i < validated;
            return (
              <div key={f} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ color: ok ? "#34d399" : "#475569", width: 10 }}>
                  {ok ? "✓" : "·"}
                </span>
                <span style={{ color: ok ? "#d1d5db" : "#64748b" }}>{f}</span>
              </div>
            );
          })}
        </div>
      </Html>
    </group>
  );
}

/* ---------------- worker pool ---------------- */

function WorkerPool({ tRef }: { tRef: React.MutableRefObject<number> }) {
  const refs = useRef<(THREE.Mesh | null)[]>([]);
  const SIZE = 3;
  const SPACING = 1.15;
  const offsets = useMemo(() => {
    const arr: [number, number][] = [];
    for (let i = 0; i < SIZE; i++) {
      for (let j = 0; j < SIZE; j++) {
        arr.push([(i - 1) * SPACING, (j - 1) * SPACING]);
      }
    }
    return arr;
  }, []);
  const selectedIdx = 4; // center cell

  useFrame(() => {
    const t = tRef.current;
    const active = t >= PHASES.gateway.end && t <= PHASES.solve.start + 0.7;
    offsets.forEach((_, i) => {
      const m = refs.current[i];
      if (!m) return;
      const mat = m.material as THREE.MeshStandardMaterial;
      const isSelected = i === selectedIdx && active;
      const target = isSelected ? 2.2 : active ? 0.05 : 0.4;
      mat.emissiveIntensity += (target - mat.emissiveIntensity) * 0.12;
      const tgtY = isSelected ? 0.18 : 0;
      m.position.y += (tgtY - m.position.y) * 0.12;
    });
  });

  return (
    <group position={POS.workerC}>
      {offsets.map(([dx, dz], i) => (
        <mesh
          key={i}
          ref={(el) => { refs.current[i] = el; }}
          position={[dx, 0, dz]}
        >
          <boxGeometry args={[0.7, 0.4, 0.7]} />
          <meshStandardMaterial
            color="#1e293b"
            emissive="#a78bfa"
            emissiveIntensity={0.4}
            metalness={0.7}
            roughness={0.3}
          />
        </mesh>
      ))}
      <Text position={[0, 1.3, 0]} fontSize={0.22} color="#a78bfa" anchorX="center">
        worker pool
      </Text>
      <Text position={[0, 1.05, 0]} fontSize={0.13} color="#ddd6fe" outlineWidth={0.008} outlineColor="#000" anchorX="center">
        async · 9 workers · round-robin
      </Text>
    </group>
  );
}

/* ---------------- H∞ solver core with live Nyquist visualization ---------------- */

function NyquistCurve({ tRef }: { tRef: React.MutableRefObject<number> }) {
  const tubeRef = useRef<THREE.Mesh>(null);
  const uncRef = useRef<THREE.Mesh>(null);
  const N = 96;

  // Build a curve in (Re, Im, depth=log-frequency) parameterized by k ∈ [0,1].
  // k=0: poor design, loops near (-1, 0). k=1: optimized, sits well right of -1.
  const buildCurve = (k: number) => {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < N; i++) {
      const u = i / (N - 1);
      const theta = Math.PI * (0.15 + u * 1.7);
      const rInit = 1.5 - 0.3 * u;
      const rFin = 0.85 - 0.15 * u;
      const r = rInit * (1 - k) + rFin * k;
      const cxInit = -0.6;
      const cxFin = 0.4;
      const cx = cxInit * (1 - k) + cxFin * k;
      const re = cx + r * Math.cos(theta);
      const im = -r * Math.sin(theta);
      const z = (u - 0.5) * 1.6;
      pts.push(new THREE.Vector3(re, im, z));
    }
    return new THREE.CatmullRomCurve3(pts, false, "catmullrom", 0.5);
  };

  useFrame(() => {
    const t = tRef.current;
    let k = 0;
    if (t < PHASES.solve.start) k = 0;
    else if (t > PHASES.solve.end - 0.2) k = 1;
    else {
      const u = (t - PHASES.solve.start) / (PHASES.solve.end - PHASES.solve.start - 0.2);
      // ease-in-out
      k = u * u * (3 - 2 * u);
    }
    const curve = buildCurve(k);
    if (tubeRef.current) {
      const newGeom = new THREE.TubeGeometry(curve, 96, 0.04, 12, false);
      tubeRef.current.geometry.dispose();
      tubeRef.current.geometry = newGeom;
    }
    if (uncRef.current) {
      // ±3σ uncertainty band — wider tube around the mean
      const uncGeom = new THREE.TubeGeometry(curve, 96, 0.13, 12, false);
      uncRef.current.geometry.dispose();
      uncRef.current.geometry = uncGeom;
    }
  });

  return (
    <group position={[POS.solver.x, POS.solver.y + 2.6, POS.solver.z]} scale={[1.4, 1.4, 1.4]}>
      {/* axes hint */}
      <mesh>
        <sphereGeometry args={[0.07, 16, 16]} />
        <meshBasicMaterial color="#ef4444" toneMapped={false} />
      </mesh>
      <Text position={[-0.85, 0.08, 0]} fontSize={0.13} color="#ef4444" anchorX="center">
        −1
      </Text>
      {/* forbidden disk around -1 */}
      <mesh position={[-1, 0, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.28, 0.34, 32]} />
        <meshBasicMaterial color="#ef4444" toneMapped={false} side={THREE.DoubleSide} transparent opacity={0.7} />
      </mesh>
      <mesh position={[-1, 0, 0]}>
        <sphereGeometry args={[0.32, 24, 24]} />
        <meshBasicMaterial color="#ef4444" toneMapped={false} transparent opacity={0.18} />
      </mesh>

      {/* uncertainty wrapper */}
      <mesh ref={uncRef}>
        <tubeGeometry args={[new THREE.CatmullRomCurve3([new THREE.Vector3(), new THREE.Vector3(0.01,0,0)]), 8, 0.13, 8, false]} />
        <meshBasicMaterial color="#fbbf24" transparent opacity={0.15} toneMapped={false} />
      </mesh>
      {/* mean Nyquist curve */}
      <mesh ref={tubeRef}>
        <tubeGeometry args={[new THREE.CatmullRomCurve3([new THREE.Vector3(), new THREE.Vector3(0.01,0,0)]), 8, 0.04, 8, false]} />
        <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={2.2} toneMapped={false} />
      </mesh>
    </group>
  );
}

function SolverCore({ tRef }: { tRef: React.MutableRefObject<number> }) {
  const [cost, setCost] = useState(8.42);
  const matRef = useRef<THREE.MeshPhysicalMaterial>(null);
  useFrame(() => {
    const t = tRef.current;
    if (t >= PHASES.solve.start && t <= PHASES.solve.end) {
      const u = (t - PHASES.solve.start) / (PHASES.solve.end - PHASES.solve.start);
      const next = 8.42 * Math.exp(-3.2 * u) + 0.84;
      setCost(next);
    }
    if (matRef.current) {
      const active = t >= PHASES.solve.start && t <= PHASES.solve.end;
      const target = active ? 2.0 : 0.5;
      matRef.current.emissiveIntensity += (target - matRef.current.emissiveIntensity) * 0.1;
    }
  });
  const inSolve = tRef.current >= PHASES.solve.start - 0.1 && tRef.current <= PHASES.solve.end;
  return (
    <group position={POS.solver}>
      {/* core slab */}
      <RoundedBox args={[3.0, 1.6, 2.0]} radius={0.18} smoothness={6}>
        <meshPhysicalMaterial
          ref={matRef}
          color="#1c1917"
          emissive="#fbbf24"
          emissiveIntensity={0.5}
          metalness={0.85}
          roughness={0.2}
          clearcoat={1}
          clearcoatRoughness={0.1}
        />
      </RoundedBox>
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(3.0, 1.6, 2.0)]} />
        <lineBasicMaterial color="#fbbf24" transparent opacity={0.85} toneMapped={false} />
      </lineSegments>
      <Text position={[0, 1.35, 0]} fontSize={0.32} color="#ffffff" anchorX="center" outlineWidth={0.01} outlineColor="#000">
        H∞ solver
      </Text>
      <Text position={[0, 1.02, 0]} fontSize={0.18} color="#fde68a" outlineWidth={0.008} outlineColor="#000" anchorX="center">
        CVXPY · MOSEK · SDP
      </Text>

      <NyquistCurve tRef={tRef} />

      {/* HUD: cost ticker + solver state */}
      <Html position={[0, -1.5, 1.0]} distanceFactor={9} style={{ pointerEvents: "none", transform: "translate(-50%, -100%)" }}>
        <div
          style={{
            fontFamily: "ui-monospace, Menlo, monospace",
            fontSize: 10,
            background: "rgba(5,7,13,0.9)",
            border: "1px solid rgba(251,191,36,0.5)",
            boxShadow: "0 0 28px rgba(251,191,36,0.3)",
            borderRadius: 8,
            padding: "8px 12px",
            color: "#fbbf24",
            minWidth: 230,
            opacity: inSolve ? 1 : 0.3,
            transition: "opacity 0.3s",
          }}
        >
          <div style={{ color: "#fde68a", marginBottom: 4 }}>minimize ‖T_zw‖∞</div>
          <div style={{ color: "#cbd5e1" }}>
            cost ={" "}
            <span style={{ color: "#fbbf24", fontWeight: 700 }}>{cost.toFixed(3)}</span>
          </div>
          <div style={{ color: "#94a3b8", fontSize: 9, marginTop: 3 }}>
            s.t.  φ_m ≥ 45°,  ±3σ FRD robustness
          </div>
        </div>
      </Html>
    </group>
  );
}

/* ---------------- PostgreSQL node with INSERT animation ---------------- */

function PostgresNode({ tRef }: { tRef: React.MutableRefObject<number> }) {
  const matRef = useRef<THREE.MeshPhysicalMaterial>(null);
  const pulseRef = useRef<THREE.Mesh>(null);
  const [chars, setChars] = useState(0);
  const INSERT_TEXT =
    "INSERT INTO controllers\n  (R, S, T, bw_hz, pm_deg)\nVALUES ($1,$2,$3,$4,$5);";
  useFrame(() => {
    const t = tRef.current;
    const active = t >= PHASES.persist.start - 0.1 && t <= PHASES.persist.end;
    if (matRef.current) {
      const target = active ? 1.8 : 0.4;
      matRef.current.emissiveIntensity += (target - matRef.current.emissiveIntensity) * 0.12;
    }
    if (active) {
      const u = (t - PHASES.persist.start) / (PHASES.persist.end - PHASES.persist.start);
      const target = Math.floor(u * INSERT_TEXT.length);
      if (target !== chars) setChars(target);
    } else if (t < PHASES.persist.start) {
      if (chars !== 0) setChars(0);
    } else {
      if (chars !== INSERT_TEXT.length) setChars(INSERT_TEXT.length);
    }
    if (pulseRef.current) {
      // commit pulse — y rises through the cylinder during last 30% of persist
      const u = (t - (PHASES.persist.start + 0.6)) / (PHASES.persist.end - PHASES.persist.start - 0.6);
      if (u > 0 && u < 1) {
        pulseRef.current.position.y = -0.7 + u * 1.4;
        (pulseRef.current.material as THREE.MeshBasicMaterial).opacity = 0.85 * (1 - Math.abs(u - 0.5) * 2);
      } else {
        (pulseRef.current.material as THREE.MeshBasicMaterial).opacity = 0;
      }
    }
  });

  const showHud = tRef.current >= PHASES.persist.start - 0.2;

  return (
    <group position={POS.postgres}>
      {/* DB cylinder */}
      <Cylinder args={[0.95, 0.95, 1.5, 32]}>
        <meshPhysicalMaterial
          ref={matRef}
          color="#0c1424"
          emissive="#3b82f6"
          emissiveIntensity={0.4}
          metalness={0.8}
          roughness={0.25}
          clearcoat={1}
        />
      </Cylinder>
      {/* top + bottom edge rings */}
      {[0.75, -0.75].map((y) => (
        <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.93, 1.0, 48]} />
          <meshBasicMaterial color="#3b82f6" toneMapped={false} side={THREE.DoubleSide} />
        </mesh>
      ))}
      {/* horizontal disc bands suggesting DB rows */}
      {[0.35, 0.0, -0.35].map((y) => (
        <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.93, 0.97, 48]} />
          <meshBasicMaterial color="#1e3a8a" toneMapped={false} side={THREE.DoubleSide} transparent opacity={0.7} />
        </mesh>
      ))}
      {/* commit pulse disc */}
      <mesh ref={pulseRef} position={[0, -0.7, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.6, 0.95, 48]} />
        <meshBasicMaterial color="#34d399" toneMapped={false} side={THREE.DoubleSide} transparent opacity={0} />
      </mesh>

      <Text position={[0, 1.25, 0]} fontSize={0.26} color="#ffffff" anchorX="center" outlineWidth={0.01} outlineColor="#000">
        PostgreSQL
      </Text>
      <Text position={[0, 0.97, 0]} fontSize={0.16} color="#bfdbfe" outlineWidth={0.008} outlineColor="#000" anchorX="center">
        transactional persistence
      </Text>

      {showHud && (
        <Html position={[0, -1.5, 0]} distanceFactor={9} style={{ pointerEvents: "none", transform: "translate(-50%, -100%)" }}>
          <pre
            style={{
              fontFamily: "ui-monospace, Menlo, monospace",
              fontSize: 10,
              background: "rgba(5,7,13,0.9)",
              border: "1px solid rgba(59,130,246,0.5)",
              boxShadow: "0 0 22px rgba(59,130,246,0.3)",
              borderRadius: 8,
              padding: "8px 12px",
              color: "#cbd5e1",
              margin: 0,
              minWidth: 220,
            }}
          >
            <span style={{ color: "#60a5fa" }}>{INSERT_TEXT.slice(0, chars)}</span>
            <span style={{ color: "#3b82f6", animation: "blink 1s steps(2) infinite" }}>▍</span>
          </pre>
        </Html>
      )}
    </group>
  );
}

/* ---------------- connector tubes (just visual, no per-block activation) ---------------- */

function ConnectorTube({ from, to, color, dip = -0.5 }: { from: THREE.Vector3; to: THREE.Vector3; color: string; dip?: number }) {
  const curve = useMemo(() => {
    const m1 = from.clone().lerp(to, 0.33).add(new THREE.Vector3(0, dip, 0));
    const m2 = from.clone().lerp(to, 0.67).add(new THREE.Vector3(0, dip, 0));
    return new THREE.CatmullRomCurve3([from, m1, m2, to], false, "catmullrom", 0.4);
  }, [from, to, dip]);
  const inner = useMemo(() => new THREE.TubeGeometry(curve, 64, 0.025, 10, false), [curve]);
  const outer = useMemo(() => new THREE.TubeGeometry(curve, 64, 0.07, 10, false), [curve]);
  return (
    <group>
      <mesh geometry={outer}>
        <meshBasicMaterial color={color} transparent opacity={0.18} toneMapped={false} />
      </mesh>
      <mesh geometry={inner}>
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.4} toneMapped={false} transparent opacity={0.85} />
      </mesh>
    </group>
  );
}

/* ---------------- camera rig ---------------- */

function CameraRig({ tRef }: { tRef: React.MutableRefObject<number> }) {
  const { camera } = useThree();
  const lookAt = useRef(new THREE.Vector3());
  const pktTmp = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    packetPos(tRef.current, pktTmp);
    cameraTarget(tRef.current, pktTmp, _camPosTmp, _camLookTmp);
    camera.position.lerp(_camPosTmp, 0.06);
    lookAt.current.lerp(_camLookTmp, 0.06);
    camera.lookAt(lookAt.current);
  });
  return null;
}

/* ---------------- main scene ---------------- */

function Scene({ onComplete }: { onComplete: () => void }) {
  const tRef = useRef(0);
  const finished = useRef(false);
  const [doneText, setDoneText] = useState(false);

  useFrame((_, dt) => {
    tRef.current += dt;
    if (!doneText && tRef.current >= PHASES.done.start) setDoneText(true);
    if (!finished.current && tRef.current >= TOTAL_DURATION) {
      finished.current = true;
      onComplete();
    }
  });

  return (
    <>
      <CameraRig tRef={tRef} />

      {/* lights */}
      <ambientLight intensity={0.35} />
      <directionalLight position={[10, 12, 6]} intensity={0.7} />
      <directionalLight position={[-8, 6, -8]} intensity={0.4} color="#7c5cff" />
      <pointLight position={[POS.solver.x, 4, POS.solver.z + 2]} intensity={1.6} color="#fbbf24" distance={14} />
      <pointLight position={[POS.gateway.x, 3, POS.gateway.z + 2]} intensity={0.9} color="#22d3ee" distance={10} />
      <pointLight position={[POS.postgres.x, 3, POS.postgres.z]} intensity={0.9} color="#3b82f6" distance={10} />

      {/* floor grid */}
      <gridHelper args={[60, 60, "#1f2937", "#0b1220"]} position={[0, -1.0, 0]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.01, 0]}>
        <planeGeometry args={[80, 80]} />
        <meshStandardMaterial color="#05070d" />
      </mesh>

      {/* topology */}
      <ClientNode tRef={tRef} />
      <GatewayNode tRef={tRef} />
      <WorkerPool tRef={tRef} />
      <SolverCore tRef={tRef} />
      <PostgresNode tRef={tRef} />

      {/* connectors */}
      <ConnectorTube from={POS.client.clone().add(new THREE.Vector3(0.8, 0, 0))} to={POS.gateway.clone().add(new THREE.Vector3(-0.8, 0, 0))} color="#22d3ee" />
      <ConnectorTube from={POS.gateway.clone().add(new THREE.Vector3(0.8, 0, 0))} to={POS.workerC.clone().add(new THREE.Vector3(0, 0, 1))} color="#a78bfa" />
      <ConnectorTube from={POS.workerC.clone().add(new THREE.Vector3(0.8, 0, 0))} to={POS.solver.clone().add(new THREE.Vector3(-1.5, -0.4, 0))} color="#fbbf24" />
      <ConnectorTube from={POS.solver.clone().add(new THREE.Vector3(1.5, -0.4, 0))} to={POS.postgres.clone().add(new THREE.Vector3(-0.8, 0, 0))} color="#3b82f6" />
      <ConnectorTube from={POS.postgres.clone().add(new THREE.Vector3(0, 0.2, -0.5))} to={POS.client.clone().add(new THREE.Vector3(0.4, 0.4, 0))} color="#34d399" dip={2.5} />

      <Packet tRef={tRef} />

      {doneText && (
        <Text position={[0, 4.2, 0]} fontSize={0.85} color="#22d3ee" anchorX="center" anchorY="middle" outlineWidth={0.02} outlineColor="#0ea5b7">
          REQUEST COMPLETE
        </Text>
      )}

      <EffectComposer>
        <Bloom intensity={1.3} luminanceThreshold={0.18} luminanceSmoothing={0.9} mipmapBlur />
      </EffectComposer>
    </>
  );
}

export default function CERNPipeline3D({
  onComplete,
}: {
  onComplete: () => void;
}) {
  return (
    <Canvas
      dpr={[1, 2]}
      camera={{ position: [-12, 5, 14], fov: 42 }}
      style={{ background: "radial-gradient(ellipse at center, #0b0f1a 0%, #05070d 70%)" }}
    >
      <Scene onComplete={onComplete} />
    </Canvas>
  );
}

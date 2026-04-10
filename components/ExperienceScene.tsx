"use client";

/* ------------------------------------------------------------------ *
 *  ExperienceScene — 3D career graph rendered with React Three Fiber  *
 *  Replaces the desktop SVG graph in Experience.tsx                   *
 * ------------------------------------------------------------------ */

import { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

/* ── Types (mirrors Experience.tsx) ──────────────────────────────── */

type Metric = { value: string; label: string };
type Job = {
  icon: unknown;
  tint: string;
  period: string;
  role: string;
  org: string;
  location: string;
  headline: string;
  summary: string;
  metrics: Metric[];
  tags: string[];
  demo?: { href: string; label: string };
};

type Edge = { from: number; to: number; career: boolean; curve: number };

interface ExperienceSceneProps {
  jobs: Job[];
  edges: Edge[];
  selected: number | null;
  onSelect: (i: number) => void;
  onClose?: () => void;
  onMiss?: () => void;
}

/* ── 3D positions (hourglass layout) ─────────────────────────────── */

const POSITIONS_3D: [number, number, number][] = [
  [-2.4, 2, -0.4],  // 0 Disruptive Labs — top-left
  [2.4, 2, -0.4],   // 1 Eaton           — top-right
  [0, 0, 1.2],      // 2 CERN            — center, pushed forward
  [-2.4, -2, -0.4], // 3 Philips         — bottom-left
  [2.4, -2, -0.4],  // 4 Apple           — bottom-right
];

/* ── CareerOrb ───────────────────────────────────────────────────── */

function CareerOrb({
  position,
  tint,
  org,
  period,
  index,
  selected,
  onSelect,
}: {
  position: [number, number, number];
  tint: string;
  org: string;
  period: string;
  index: number;
  selected: number | null;
  onSelect: (i: number) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const isSel = selected === index;
  const isDim = selected !== null && !isSel;

  const color = useMemo(() => new THREE.Color(tint), [tint]);

  useFrame(() => {
    if (!meshRef.current) return;
    const target = isSel ? 1.2 : hovered ? 1.08 : 1;
    const cur = meshRef.current.scale.x;
    meshRef.current.scale.setScalar(cur + (target - cur) * 0.1);
  });

  const emissiveIntensity = isSel ? 0.4 : isDim ? 0.03 : 0.15;

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(index);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          setHovered(false);
          document.body.style.cursor = "auto";
        }}
      >
        <sphereGeometry args={[0.3, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          metalness={0.7}
          roughness={0.15}
          transparent
          opacity={isDim ? 0.35 : 1}
        />
      </mesh>

      {/* Orbital ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.36, 0.006, 12, 64]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity * 0.5}
          transparent
          opacity={isDim ? 0.08 : 0.2}
        />
      </mesh>

      {/* Label */}
      <Html
        position={[0, -0.55, 0]}
        center
        style={{ pointerEvents: "auto", userSelect: "none" }}
        distanceFactor={8}
      >
        <div
          onClick={(e) => {
            e.stopPropagation();
            onSelect(index);
          }}
          onMouseOver={() => {
            document.body.style.cursor = "pointer";
          }}
          onMouseOut={() => {
            document.body.style.cursor = "auto";
          }}
          style={{
            textAlign: "center",
            cursor: "pointer",
            opacity: isDim ? 0.3 : 1,
            transition: "opacity 0.3s",
          }}
        >
          <p
            style={{
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              fontSize: "11px",
              fontWeight: 600,
              color: tint,
              whiteSpace: "nowrap",
              margin: 0,
              lineHeight: 1.3,
            }}
          >
            {org}
          </p>
          <p
            style={{
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
              fontSize: "8px",
              color: "#6b7280",
              whiteSpace: "nowrap",
              margin: 0,
              marginTop: "2px",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            {period}
          </p>
        </div>
      </Html>
    </group>
  );
}

/* ── TubeEdge ────────────────────────────────────────────────────── */

function TubeEdge({
  edge,
  selected,
}: {
  edge: Edge;
  selected: number | null;
}) {
  const from = POSITIONS_3D[edge.from];
  const to = POSITIONS_3D[edge.to];

  const geometry = useMemo(() => {
    const start = new THREE.Vector3(...from);
    const end = new THREE.Vector3(...to);
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

    // Offset control point perpendicular for curvature
    const dir = new THREE.Vector3().subVectors(end, start);
    const perp = new THREE.Vector3(-dir.y, dir.x, 0).normalize();
    mid.add(perp.multiplyScalar(edge.curve * dir.length() * 0.5));
    // push midpoint forward slightly in z
    mid.z += 0.3;

    const curve = new THREE.CatmullRomCurve3([start, mid, end]);
    return new THREE.TubeGeometry(curve, 32, 0.012, 6, false);
  }, [from, to, edge.curve]);

  const connected = selected === edge.from || selected === edge.to;
  const dimmed = selected !== null && !connected;

  const baseOpacity = edge.career ? 0.7 : 0.3;
  const opacity = dimmed ? 0.05 : connected ? Math.min(baseOpacity * 1.6, 1) : baseOpacity;
  const edgeColor = edge.career ? "#22d3ee" : "#7c5cff";

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color={edgeColor}
        emissive={edgeColor}
        emissiveIntensity={connected ? 0.35 : 0.1}
        transparent
        opacity={opacity}
      />
    </mesh>
  );
}

/* ── Floating detail card ──────────────────────────────────────── */

function FloatingDetail({
  job,
  nodePos,
  onClose,
}: {
  job: Job;
  nodePos: [number, number, number];
  onClose: () => void;
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    requestAnimationFrame(() => setMounted(true));
  }, []);

  const side = nodePos[0] <= 0 ? 1 : -1;
  const cardPos: [number, number, number] = [
    nodePos[0] + side * 2.4,
    Math.max(Math.min(nodePos[1] + 0.3, 1.2), -0.3),
    nodePos[2] + 0.5,
  ];

  return (
    <Html
      position={cardPos}
      center
      style={{
        pointerEvents: "auto",
        opacity: mounted ? 1 : 0,
        transform: mounted ? "translateY(0)" : "translateY(10px)",
        transition: "opacity 0.35s ease-out, transform 0.35s ease-out",
      }}
      zIndexRange={[100, 0]}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 280,
          background: "rgba(8, 12, 24, 0.92)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: `1px solid ${job.tint}40`,
          borderTop: `2px solid ${job.tint}`,
          borderRadius: 14,
          padding: "16px 18px",
          fontFamily: "system-ui, -apple-system, sans-serif",
          boxShadow: `0 0 40px ${job.tint}10, 0 12px 40px rgba(0,0,0,0.5)`,
        }}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <p style={{ margin: 0, fontSize: 14, fontWeight: 600, color: "white", lineHeight: 1.3 }}>
              {job.role}
            </p>
            <p style={{ margin: "2px 0 0", fontSize: 11, color: job.tint, fontWeight: 500 }}>
              {job.org}
            </p>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onClose(); }}
            style={{
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 6,
              width: 22,
              height: 22,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: "pointer",
              color: "#6b7280",
              fontSize: 12,
              lineHeight: 1,
              padding: 0,
              flexShrink: 0,
            }}
          >
            ✕
          </button>
        </div>

        {/* Period & location */}
        <p style={{
          margin: "8px 0 0",
          fontSize: 10,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          color: "#6b7280",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
        }}>
          {job.period} · {job.location}
        </p>

        {/* Summary */}
        <p style={{
          margin: "10px 0 0",
          fontSize: 12,
          color: "#d1d5db",
          lineHeight: 1.5,
        }}>
          {job.summary}
        </p>

        {/* Metrics */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
          {job.metrics.map((m) => (
            <div
              key={m.label}
              style={{
                background: "rgba(0,0,0,0.4)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 6,
                padding: "4px 8px",
              }}
            >
              <span style={{ fontSize: 12, fontWeight: 600, color: job.tint }}>
                {m.value}
              </span>
              <span style={{
                marginLeft: 5,
                fontSize: 8,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "#6b7280",
              }}>
                {m.label}
              </span>
            </div>
          ))}
        </div>

        {/* Tags */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 8 }}>
          {job.tags.map((t) => (
            <span
              key={t}
              style={{
                fontSize: 8,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                padding: "2px 6px",
                borderRadius: 4,
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#6b7280",
              }}
            >
              {t}
            </span>
          ))}
        </div>

        {/* Demo link */}
        {job.demo && (
          <a
            href={job.demo.href}
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
              marginTop: 10,
              fontSize: 11,
              fontWeight: 500,
              color: job.tint,
              textDecoration: "none",
            }}
          >
            {job.demo.label} ↗
          </a>
        )}
      </div>
    </Html>
  );
}

/* ── Scene contents ──────────────────────────────────────────────── */

function SceneContents({
  jobs,
  edges,
  selected,
  onSelect,
  onClose,
}: ExperienceSceneProps) {
  return (
    <>
      {/* Lights */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />
      <directionalLight position={[-3, -2, 4]} intensity={0.3} color="#c4b5fd" />

      {/* Edges */}
      {edges.map((edge, i) => (
        <TubeEdge key={i} edge={edge} selected={selected} />
      ))}

      {/* Orbs */}
      {jobs.map((job, i) => (
        <CareerOrb
          key={job.org}
          position={POSITIONS_3D[i]}
          tint={job.tint}
          org={job.org}
          period={job.period}
          index={i}
          selected={selected}
          onSelect={onSelect}
        />
      ))}

      {/* Floating detail card */}
      {selected !== null && (
        <FloatingDetail
          key={selected}
          job={jobs[selected]}
          nodePos={POSITIONS_3D[selected]}
          onClose={onClose!}
        />
      )}

      {/* Controls — pause autoRotate while a card is open */}
      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate={selected === null}
        autoRotateSpeed={0.08}
        maxPolarAngle={Math.PI * 0.7}
        minPolarAngle={Math.PI * 0.3}
      />
    </>
  );
}

/* ── Visibility invalidator ──────────────────────────────────────── */

function Invalidator({ visible }: { visible: boolean }) {
  const { invalidate } = useThree();

  useFrame(() => {
    if (visible) invalidate();
  });

  return null;
}

/* ── Main component ──────────────────────────────────────────────── */

export default function ExperienceScene({
  jobs,
  edges,
  selected,
  onSelect,
  onClose,
  onMiss,
}: ExperienceSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold: 0.1 },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%" }}>
      <Canvas
        dpr={[1, 1.5]}
        gl={{ alpha: true, antialias: true }}
        camera={{ position: [0, 1.5, 8], fov: 40 }}
        frameloop="demand"
        style={{ background: "transparent", overflow: "visible" }}
        onPointerMissed={onMiss}
      >
        <Invalidator visible={visible} />
        <SceneContents
          jobs={jobs}
          edges={edges}
          selected={selected}
          onSelect={onSelect}
          onClose={onClose}
        />
      </Canvas>
    </div>
  );
}

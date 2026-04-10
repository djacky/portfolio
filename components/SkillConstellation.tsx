"use client";

/* ------------------------------------------------------------------ *
 *  SkillConstellation — 3D point-cloud backdrop for the Skills section *
 *  Four clusters (one per skill group) orbit a shared center.          *
 *  On hover of a skill card, that cluster brightens & pulls forward.   *
 * ------------------------------------------------------------------ */

import { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";

const POINTS_PER_CLUSTER = 35;
const CLUSTER_COUNT = 4;
const TOTAL = POINTS_PER_CLUSTER * CLUSTER_COUNT;

/* Tints match GROUPS order in Skills.tsx: ML&AI, Languages, Backend, Systems */
const CLUSTER_TINTS = ["#7c5cff", "#22d3ee", "#34d399", "#fbbf24"];

const ORBITS = [
  { radius: 3.0, speed: 0.12, phase: 0, ySpeed: 0.08, yPhase: 0 },
  { radius: 2.6, speed: 0.10, phase: Math.PI * 0.5, ySpeed: 0.09, yPhase: 1.2 },
  { radius: 3.2, speed: 0.09, phase: Math.PI, ySpeed: 0.07, yPhase: 2.4 },
  { radius: 2.8, speed: 0.11, phase: Math.PI * 1.5, ySpeed: 0.10, yPhase: 3.6 },
];

/* ── Constellation points ───────────────────────────────────────────── */

function ConstellationPoints({ hoveredGroup }: { hoveredGroup: number | null }) {
  const { geometry, offsets, baseColors } = useMemo(() => {
    const positions = new Float32Array(TOTAL * 3);
    const colors = new Float32Array(TOTAL * 3);
    const offsets = new Float32Array(TOTAL * 3);
    const baseColors = CLUSTER_TINTS.map((t) => new THREE.Color(t));

    for (let c = 0; c < CLUSTER_COUNT; c++) {
      const color = baseColors[c];
      for (let p = 0; p < POINTS_PER_CLUSTER; p++) {
        const i = c * POINTS_PER_CLUSTER + p;
        const i3 = i * 3;

        /* Spherical offset from cluster center */
        const phi = Math.random() * Math.PI * 2;
        const cosTheta = 2 * Math.random() - 1;
        const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
        const r = 0.3 + Math.random() * 1.2;

        offsets[i3] = r * sinTheta * Math.cos(phi);
        offsets[i3 + 1] = r * sinTheta * Math.sin(phi);
        offsets[i3 + 2] = r * cosTheta;

        positions[i3] = offsets[i3];
        positions[i3 + 1] = offsets[i3 + 1];
        positions[i3 + 2] = offsets[i3 + 2];

        const v = 0.85 + Math.random() * 0.3;
        colors[i3] = color.r * v;
        colors[i3 + 1] = color.g * v;
        colors[i3 + 2] = color.b * v;
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return { geometry, offsets, baseColors };
  }, []);

  /* Smooth lerped state for hover transitions */
  const currentAlphas = useRef(new Float32Array(CLUSTER_COUNT).fill(1));
  const currentZ = useRef(new Float32Array(CLUSTER_COUNT).fill(0));

  useFrame(({ clock }, delta) => {
    const t = clock.getElapsedTime();
    const dt = Math.min(delta, 0.05);
    const posArr = (geometry.attributes.position as THREE.BufferAttribute)
      .array as Float32Array;
    const colArr = (geometry.attributes.color as THREE.BufferAttribute)
      .array as Float32Array;

    const lerpSpeed = 5 * dt;

    for (let c = 0; c < CLUSTER_COUNT; c++) {
      /* Target brightness & depth based on hover */
      const targetAlpha =
        hoveredGroup === null ? 1 : hoveredGroup === c ? 1.5 : 0.25;
      const targetZ =
        hoveredGroup === null ? 0 : hoveredGroup === c ? 2.5 : -1.5;

      currentAlphas.current[c] +=
        (targetAlpha - currentAlphas.current[c]) * lerpSpeed;
      currentZ.current[c] +=
        (targetZ - currentZ.current[c]) * lerpSpeed;

      const orbit = ORBITS[c];
      const alpha = currentAlphas.current[c];
      const zShift = currentZ.current[c];
      const baseColor = baseColors[c];

      /* Cluster center: slow orbit */
      const cx = orbit.radius * Math.cos(t * orbit.speed + orbit.phase);
      const cy =
        orbit.radius * Math.sin(t * orbit.ySpeed + orbit.yPhase) * 0.5;
      const cz = zShift;

      for (let p = 0; p < POINTS_PER_CLUSTER; p++) {
        const i = c * POINTS_PER_CLUSTER + p;
        const i3 = i * 3;

        /* Slowly rotate each point's offset for organic drift */
        const rotSpeed = 0.12 + (p % 7) * 0.015;
        const angle = t * rotSpeed;
        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);

        const ox = offsets[i3];
        const oz = offsets[i3 + 2];

        posArr[i3] = cx + ox * cosA - oz * sinA;
        posArr[i3 + 1] = cy + offsets[i3 + 1];
        posArr[i3 + 2] = cz + ox * sinA + oz * cosA;

        /* Tint brightness tracks hover lerp */
        const v = 0.85 + ((p * 7) % 10) * 0.03;
        colArr[i3] = baseColor.r * v * alpha;
        colArr[i3 + 1] = baseColor.g * v * alpha;
        colArr[i3 + 2] = baseColor.b * v * alpha;
      }
    }

    (geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (geometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
  });

  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={0.055}
        vertexColors
        transparent
        opacity={0.65}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        sizeAttenuation
      />
    </points>
  );
}

/* ── Visibility invalidator ─────────────────────────────────────────── */

function Invalidator({ visible }: { visible: boolean }) {
  const { invalidate } = useThree();
  useFrame(() => {
    if (visible) invalidate();
  });
  return null;
}

/* ── Main ────────────────────────────────────────────────────────────── */

export default function SkillConstellation({
  hoveredGroup,
}: {
  hoveredGroup: number | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => setVisible(e.isIntersecting),
      { threshold: 0.1 },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className="absolute inset-0 overflow-hidden pointer-events-none"
      style={{ zIndex: 0 }}
    >
      <Canvas
        dpr={[1, 1.5]}
        gl={{ alpha: true, antialias: false }}
        camera={{ position: [0, 0, 8], fov: 50 }}
        frameloop="demand"
        style={{ background: "transparent" }}
      >
        <Invalidator visible={visible} />
        <ConstellationPoints hoveredGroup={hoveredGroup} />
        <EffectComposer>
          <Bloom intensity={0.6} luminanceThreshold={0.3} mipmapBlur />
        </EffectComposer>
      </Canvas>
    </div>
  );
}

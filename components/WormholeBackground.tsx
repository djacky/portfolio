"use client";

/* ------------------------------------------------------------------ *
 *  WormholeBackground — spiraling particle tunnel (R3F)               *
 *  Positioned behind the Contact CTA to draw the eye inward.         *
 * ------------------------------------------------------------------ */

import { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";

const DESKTOP_COUNT = 500;
const MOBILE_COUNT = 250;
const TUNNEL_DEPTH = 20;
const MAX_RADIUS = 4.5;
const SPIRAL_TWIST = 1.8;

/* ── Particle vortex ────────────────────────────────────────────────── */

function VortexParticles({ count }: { count: number }) {
  const { geometry, meta } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const meta = new Float32Array(count * 3); // theta, radiusFactor, speedFactor

    const accent = new THREE.Color("#7c5cff");
    const accent2 = new THREE.Color("#22d3ee");
    const tmp = new THREE.Color();

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const theta = Math.random() * Math.PI * 2;
      const radiusFactor = 0.15 + Math.random() * 0.85;
      const speedFactor = 0.3 + Math.random() * 0.7;

      meta[i3] = theta;
      meta[i3 + 1] = radiusFactor;
      meta[i3 + 2] = speedFactor;

      const z = -Math.random() * TUNNEL_DEPTH;
      const t = -z / TUNNEL_DEPTH;
      const r = MAX_RADIUS * Math.sqrt(t) * radiusFactor;
      const angle = theta + z * SPIRAL_TWIST;

      positions[i3] = r * Math.cos(angle);
      positions[i3 + 1] = r * Math.sin(angle);
      positions[i3 + 2] = z;

      tmp.lerpColors(accent, accent2, Math.random());
      colors[i3] = tmp.r;
      colors[i3 + 1] = tmp.g;
      colors[i3 + 2] = tmp.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return { geometry, meta };
  }, [count]);

  useFrame((_, delta) => {
    const posAttr = geometry.attributes.position as THREE.BufferAttribute;
    const arr = posAttr.array as Float32Array;
    const dt = Math.min(delta, 0.05);

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      let z = arr[i3 + 2];
      const theta = meta[i3];
      const radiusFactor = meta[i3 + 1];
      const speedFactor = meta[i3 + 2];

      z += (1.5 + speedFactor * 2.5) * dt;

      if (z > 1) z = -TUNNEL_DEPTH + Math.random() * 2;

      const t = Math.max(0, -z / TUNNEL_DEPTH);
      const r = MAX_RADIUS * Math.sqrt(t) * radiusFactor;
      const angle = theta + z * SPIRAL_TWIST;

      arr[i3] = r * Math.cos(angle);
      arr[i3 + 1] = r * Math.sin(angle);
      arr[i3 + 2] = z;
    }

    posAttr.needsUpdate = true;
  });

  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={0.06}
        vertexColors
        transparent
        opacity={0.7}
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

export default function WormholeBackground() {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [count] = useState(() =>
    typeof window !== "undefined" && window.innerWidth < 768
      ? MOBILE_COUNT
      : DESKTOP_COUNT,
  );

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
        camera={{ position: [0, 0, 4], fov: 65 }}
        frameloop="demand"
        style={{ background: "transparent" }}
      >
        <Invalidator visible={visible} />
        <VortexParticles count={count} />
        <EffectComposer>
          <Bloom intensity={0.8} luminanceThreshold={0.2} mipmapBlur />
        </EffectComposer>
      </Canvas>
    </div>
  );
}

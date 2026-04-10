"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

/* ------------------------------------------------------------------ *
 *  ParticleField                                                      *
 *  3D drifting particles with scroll parallax & mouse repulsion.      *
 *  Sits behind Hero text as an ambient background.                    *
 * ------------------------------------------------------------------ */

// Theme colours
const ACCENT_PURPLE = new THREE.Color("#7c5cff");
const ACCENT_CYAN = new THREE.Color("#22d3ee");

// Distribution bounds
const X_MIN = -8,
  X_MAX = 8;
const Y_MIN = -5,
  Y_MAX = 5;
const Z_MIN = -2,
  Z_MAX = 2;

const MOUSE_RADIUS = 2; // world units
const MOUSE_STRENGTH = 0.012;
const DRIFT_SPEED = 0.15;

/* ------------------------------------------------------------------ *
 *  Soft circle sprite texture (32x32 radial gradient)                 *
 * ------------------------------------------------------------------ */
function makeCircleTexture(): THREE.Texture {
  const size = 32;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const half = size / 2;
  const grad = ctx.createRadialGradient(half, half, 0, half, half, half);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.6)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

/* ------------------------------------------------------------------ *
 *  Particles inner component (lives inside the Canvas)                *
 * ------------------------------------------------------------------ */
interface ParticlesProps {
  count: number;
  scrollYRef: React.MutableRefObject<number>;
  mouseRef: React.MutableRefObject<{ x: number; y: number }>;
}

function Particles({ count, scrollYRef, mouseRef }: ParticlesProps) {
  const groupRef = useRef<THREE.Group>(null);
  const pointsRef = useRef<THREE.Points>(null);
  const { camera } = useThree();

  // Allocate all buffers once
  const { positions, colors, sizes, velocities } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const sz = new Float32Array(count);
    const vel = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      pos[i3] = X_MIN + Math.random() * (X_MAX - X_MIN);
      pos[i3 + 1] = Y_MIN + Math.random() * (Y_MAX - Y_MIN);
      pos[i3 + 2] = Z_MIN + Math.random() * (Z_MAX - Z_MIN);

      // Random purple or cyan with slight variation
      const c = Math.random() > 0.5 ? ACCENT_PURPLE : ACCENT_CYAN;
      const jitter = 0.9 + Math.random() * 0.2;
      col[i3] = c.r * jitter;
      col[i3 + 1] = c.g * jitter;
      col[i3 + 2] = c.b * jitter;

      sz[i] = 2 + Math.random() * 4;

      vel[i3] = (Math.random() - 0.5) * DRIFT_SPEED;
      vel[i3 + 1] = (Math.random() - 0.5) * DRIFT_SPEED;
      vel[i3 + 2] = (Math.random() - 0.5) * DRIFT_SPEED * 0.3;
    }
    return { positions: pos, colors: col, sizes: sz, velocities: vel };
  }, [count]);

  const texture = useMemo(() => makeCircleTexture(), []);

  // Reusable vectors for mouse raycasting
  const mouseNDC = useMemo(() => new THREE.Vector2(), []);
  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const hitPlane = useMemo(
    () => new THREE.Plane(new THREE.Vector3(0, 0, 1), 0),
    []
  );
  const mouseWorld = useMemo(() => new THREE.Vector3(), []);

  useFrame((_, delta) => {
    if (!pointsRef.current || !groupRef.current) return;
    const dt = Math.min(delta, 0.05); // clamp big spikes

    // Scroll parallax on group
    groupRef.current.position.y = scrollYRef.current * 0.0008;

    // Project mouse to world z=0
    mouseNDC.set(mouseRef.current.x, mouseRef.current.y);
    raycaster.setFromCamera(mouseNDC, camera);
    raycaster.ray.intersectPlane(hitPlane, mouseWorld);

    const posAttr = pointsRef.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const arr = posAttr.array as Float32Array;
    const radiusSq = MOUSE_RADIUS * MOUSE_RADIUS;

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;

      // Drift
      arr[i3] += velocities[i3] * dt;
      arr[i3 + 1] += velocities[i3 + 1] * dt;
      arr[i3 + 2] += velocities[i3 + 2] * dt;

      // Mouse repulsion (in world space, accounting for group offset)
      const px = arr[i3];
      const py = arr[i3 + 1] + groupRef.current!.position.y;
      const dx = px - mouseWorld.x;
      const dy = py - mouseWorld.y;
      const distSq = dx * dx + dy * dy;
      if (distSq < radiusSq && distSq > 0.001) {
        const dist = Math.sqrt(distSq);
        const force = MOUSE_STRENGTH / (distSq + 0.1);
        arr[i3] += (dx / dist) * force;
        arr[i3 + 1] += (dy / dist) * force;
      }

      // Wrap particles that leave bounds
      if (arr[i3] < X_MIN) arr[i3] = X_MAX;
      else if (arr[i3] > X_MAX) arr[i3] = X_MIN;
      if (arr[i3 + 1] < Y_MIN) arr[i3 + 1] = Y_MAX;
      else if (arr[i3 + 1] > Y_MAX) arr[i3 + 1] = Y_MIN;
      if (arr[i3 + 2] < Z_MIN) arr[i3 + 2] = Z_MAX;
      else if (arr[i3 + 2] > Z_MAX) arr[i3 + 2] = Z_MIN;
    }

    posAttr.needsUpdate = true;
  });

  return (
    <group ref={groupRef}>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            array={positions}
            count={count}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            array={colors}
            count={count}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          map={texture}
          vertexColors
          transparent
          blending={THREE.AdditiveBlending}
          sizeAttenuation
          depthWrite={false}
          size={4}
        />
      </points>
    </group>
  );
}

/* ------------------------------------------------------------------ *
 *  Outer wrapper: Canvas + visibility gating + event listeners        *
 * ------------------------------------------------------------------ */
export default function ParticleField() {
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollYRef = useRef(0);
  const mouseRef = useRef({ x: 0, y: 0 });
  const [visible, setVisible] = useState(true);
  const invalidateRef = useRef<(() => void) | null>(null);

  // Determine particle count once at mount
  const count = useMemo(() => {
    if (typeof window === "undefined") return 300;
    return window.innerWidth < 768 ? 150 : 300;
  }, []);

  // Scroll listener (passive, ref-only — no re-renders)
  useEffect(() => {
    function onScroll() {
      scrollYRef.current = window.scrollY;
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Mouse listener (NDC coordinates, ref-only)
  useEffect(() => {
    function onMove(e: PointerEvent) {
      mouseRef.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouseRef.current.y = -(e.clientY / window.innerHeight) * 2 + 1;
    }
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);

  // IntersectionObserver for visibility gating
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        setVisible(entry.isIntersecting);
      },
      { threshold: 0 }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // Continuously invalidate while visible (demand frameloop)
  const onCreated = useCallback(
    (state: { invalidate: () => void }) => {
      invalidateRef.current = state.invalidate;
    },
    []
  );

  useEffect(() => {
    if (!visible || !invalidateRef.current) return;
    let raf = 0;
    function loop() {
      invalidateRef.current?.();
      raf = requestAnimationFrame(loop);
    }
    loop();
    return () => cancelAnimationFrame(raf);
  }, [visible]);

  return (
    <div ref={containerRef} className="absolute inset-0 w-full h-full">
      <Canvas
        frameloop="demand"
        dpr={[1, 1.5]}
        gl={{ alpha: true, powerPreference: "low-power" }}
        camera={{ position: [0, 0, 6], fov: 50 }}
        style={{ pointerEvents: "none" }}
        onCreated={onCreated}
      >
        <Particles
          count={count}
          scrollYRef={scrollYRef}
          mouseRef={mouseRef}
        />
      </Canvas>
    </div>
  );
}

"use client";

/* ------------------------------------------------------------------ *
 *  AboutGraph — 3D knowledge graph for the About section               *
 *  Four pillar nodes connected by curved edges with particle flow.     *
 *  Mirrors patterns from ExperienceScene.tsx.                          *
 * ------------------------------------------------------------------ */

import { useRef, useState, useEffect, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";
import { PILLARS, EDGES, type Pillar, type GraphEdge } from "./aboutGraphData";
import { FONT_FAMILY } from "@/lib/fonts";

/* ── Procedural glow texture ──────────────────────────���───────────── */

let _glowTex: THREE.Texture | null = null;

function getGlowTexture(): THREE.Texture {
  if (_glowTex) return _glowTex;
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const half = size / 2;
  const grad = ctx.createRadialGradient(half, half, 0, half, half, half);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.4, "rgba(255,255,255,0.4)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  _glowTex = new THREE.CanvasTexture(canvas);
  return _glowTex;
}

/* ── PillarNode ───────────────────────────────────────────────────── */

function PillarNode({
  pillar,
  index,
  hovered,
  onHover,
  onUnhover,
  glowTex,
  draggingIndex,
  setDraggingIndex,
  positionsRef,
}: {
  pillar: Pillar;
  index: number;
  hovered: number | null;
  onHover: (i: number) => void;
  onUnhover: () => void;
  glowTex: THREE.Texture;
  draggingIndex: number | null;
  setDraggingIndex: (i: number | null) => void;
  positionsRef: React.MutableRefObject<THREE.Vector3[]>;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Sprite>(null);
  const groupRef = useRef<THREE.Group>(null);

  /* Spring-damper state */
  const dispRef = useRef(new THREE.Vector3());
  const velRef = useRef(new THREE.Vector3());
  const dragStartRef = useRef<{
    pointer: [number, number];
    disp: THREE.Vector3;
    lastPointer: [number, number];
    lastTime: number;
  } | null>(null);

  const { camera, size } = useThree();

  const basePos = useMemo(
    () => new THREE.Vector3(pillar.position[0], pillar.position[1], pillar.position[2]),
    [pillar.position],
  );

  const isDragging = draggingIndex === index;
  const isHovered = hovered === index;
  const isDimmed = hovered !== null && !isHovered && !isDragging;

  const color = useMemo(() => new THREE.Color(pillar.color), [pillar.color]);

  /* Window pointer listeners while being dragged */
  useEffect(() => {
    if (!isDragging) return;

    const handleMove = (e: PointerEvent) => {
      const start = dragStartRef.current;
      if (!start) return;

      const perspCam = camera as THREE.PerspectiveCamera;
      camera.updateMatrixWorld();

      const nodeWorldPos = basePos.clone().add(start.disp);
      const distToNode = camera.position.distanceTo(nodeWorldPos);
      const fovRad = (perspCam.fov * Math.PI) / 180;
      const visibleH = 2 * distToNode * Math.tan(fovRad / 2);
      const visibleW = visibleH * (size.width / size.height);

      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).normalize();
      const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1).normalize();

      const dxPx = e.clientX - start.pointer[0];
      const dyPx = e.clientY - start.pointer[1];
      const worldDx = (dxPx / size.width) * visibleW;
      const worldDy = -(dyPx / size.height) * visibleH;

      dispRef.current
        .copy(start.disp)
        .addScaledVector(right, worldDx)
        .addScaledVector(up, worldDy);

      /* Track velocity across recent motion for release inertia */
      const now = performance.now();
      const dtMs = now - start.lastTime;
      if (dtMs > 8) {
        const fDxPx = e.clientX - start.lastPointer[0];
        const fDyPx = e.clientY - start.lastPointer[1];
        const dtSec = dtMs / 1000;
        const vWx = ((fDxPx / size.width) * visibleW) / dtSec;
        const vWy = (-(fDyPx / size.height) * visibleH) / dtSec;
        velRef.current
          .set(0, 0, 0)
          .addScaledVector(right, vWx)
          .addScaledVector(up, vWy);
        start.lastPointer[0] = e.clientX;
        start.lastPointer[1] = e.clientY;
        start.lastTime = now;
      }
    };

    const handleUp = () => {
      dragStartRef.current = null;
      document.body.style.cursor = "auto";
      setDraggingIndex(null);
      onUnhover();
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    window.addEventListener("pointercancel", handleUp);

    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
      window.removeEventListener("pointercancel", handleUp);
    };
  }, [isDragging, basePos, camera, size, setDraggingIndex, onUnhover]);

  /* Spring-damper integration + scale/glow animation */
  useFrame((_, delta) => {
    if (meshRef.current) {
      const target = isDragging || isHovered ? 1.15 : isDimmed ? 0.92 : 1;
      const cur = meshRef.current.scale.x;
      meshRef.current.scale.setScalar(cur + (target - cur) * 0.1);
    }
    if (glowRef.current) {
      const target = isDragging || isHovered ? 1.4 : isDimmed ? 0.5 : 0.9;
      const cur = glowRef.current.scale.x;
      const s = cur + (target - cur) * 0.08;
      glowRef.current.scale.set(s, s, 1);
    }

    if (!isDragging) {
      /* Underdamped spring: k=40, c=4.5 → ~3 oscillations */
      const dt = Math.min(delta, 0.05);
      const k = 40;
      const c = 4.5;
      const d = dispRef.current;
      const v = velRef.current;
      v.x += (-k * d.x - c * v.x) * dt;
      v.y += (-k * d.y - c * v.y) * dt;
      v.z += (-k * d.z - c * v.z) * dt;
      d.x += v.x * dt;
      d.y += v.y * dt;
      d.z += v.z * dt;
    }

    if (groupRef.current) {
      groupRef.current.position.set(
        basePos.x + dispRef.current.x,
        basePos.y + dispRef.current.y,
        basePos.z + dispRef.current.z,
      );
      positionsRef.current[index].copy(groupRef.current.position);
    }
  });

  const emissiveIntensity = isDragging || isHovered ? 0.45 : isDimmed ? 0.04 : 0.18;
  const ringOpacity = isDimmed ? 0.06 : isDragging || isHovered ? 0.35 : 0.18;

  return (
    <group ref={groupRef} position={pillar.position}>
      {/* Primary sphere */}
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          if (draggingIndex !== null && draggingIndex !== index) return;
          onHover(index);
          document.body.style.cursor = isDragging ? "grabbing" : "grab";
        }}
        onPointerOut={() => {
          if (draggingIndex !== null) return;
          onUnhover();
          document.body.style.cursor = "auto";
        }}
        onPointerDown={(e) => {
          e.stopPropagation();
          dragStartRef.current = {
            pointer: [e.clientX, e.clientY],
            disp: dispRef.current.clone(),
            lastPointer: [e.clientX, e.clientY],
            lastTime: performance.now(),
          };
          velRef.current.set(0, 0, 0);
          setDraggingIndex(index);
          document.body.style.cursor = "grabbing";
        }}
      >
        <sphereGeometry args={[0.36, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity}
          metalness={0.7}
          roughness={0.15}
          transparent
          opacity={isDimmed ? 0.35 : 1}
        />
      </mesh>

      {/* Orbital ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.46, 0.006, 12, 64]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={emissiveIntensity * 0.5}
          transparent
          opacity={ringOpacity}
        />
      </mesh>

      {/* Glow halo sprite */}
      <sprite ref={glowRef}>
        <spriteMaterial
          map={glowTex}
          color={pillar.color}
          transparent
          opacity={isDimmed ? 0.08 : 0.25}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </sprite>

      {/* Label below the orb */}
      <Html
        position={[0, -0.7, 0]}
        center
        style={{ pointerEvents: "none", userSelect: "none" }}
        distanceFactor={8}
      >
        <div
          style={{
            textAlign: "center",
            opacity: isDimmed ? 0.25 : 1,
            transition: "opacity 0.3s",
          }}
        >
          <p
            style={{
              fontFamily: FONT_FAMILY,
              fontSize: "14px",
              fontWeight: 600,
              color: pillar.color,
              whiteSpace: "nowrap",
              margin: 0,
              lineHeight: 1.3,
            }}
          >
            {pillar.title}
          </p>
          <p
            style={{
              fontFamily: FONT_FAMILY,
              fontSize: "10px",
              color: "#9ca3af",
              whiteSpace: "nowrap",
              margin: "2px 0 0",
              lineHeight: 1.2,
              fontStyle: "italic",
            }}
          >
            {pillar.subtitle}
          </p>
        </div>
      </Html>

    </group>
  );
}

/* ── Edge particles (sprites traversing a curve) ──────────────────── */

function EdgeParticles({
  curve,
  colorA,
  colorB,
  active,
  dimmed,
  glowTex,
  count,
}: {
  curve: THREE.CatmullRomCurve3;
  colorA: string;
  colorB: string;
  active: boolean;
  dimmed: boolean;
  glowTex: THREE.Texture;
  count: number;
}) {
  const refs = useRef<(THREE.Sprite | null)[]>([]);
  const tRef = useRef<number[]>([]);
  const cA = useMemo(() => new THREE.Color(colorA), [colorA]);
  const cB = useMemo(() => new THREE.Color(colorB), [colorB]);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  /* Initialize evenly-spaced t values */
  if (tRef.current.length !== count) {
    tRef.current = Array.from({ length: count }, (_, i) => i / count);
  }

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.05);
    const speed = active ? 0.12 : 0.06;
    for (let i = 0; i < count; i++) {
      tRef.current[i] = (tRef.current[i] + speed * dt) % 1;
      const sprite = refs.current[i];
      if (!sprite) continue;
      const pt = curve.getPointAt(tRef.current[i]);
      sprite.position.copy(pt);
      tmpColor.copy(cA).lerp(cB, tRef.current[i]);
      (sprite.material as THREE.SpriteMaterial).color.copy(tmpColor);
    }
  });

  const opacity = dimmed ? 0.04 : active ? 0.55 : 0.2;

  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <sprite
          key={i}
          ref={(el) => {
            refs.current[i] = el;
          }}
          scale={[0.07, 0.07, 1]}
        >
          <spriteMaterial
            map={glowTex}
            transparent
            opacity={opacity}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            toneMapped={false}
          />
        </sprite>
      ))}
    </>
  );
}

/* ── GraphEdge ─────────────────────────────────��──────────────────── */

function GraphEdgeComp({
  edge,
  hovered,
  glowTex,
  positionsRef,
}: {
  edge: GraphEdge;
  hovered: number | null;
  glowTex: THREE.Texture;
  positionsRef: React.MutableRefObject<THREE.Vector3[]>;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const midRef = useRef(new THREE.Vector3());
  const midVelRef = useRef(new THREE.Vector3());

  /* Initial curve + geometry from static base pillar positions */
  const { initialGeometry, curve } = useMemo(() => {
    const staticFrom = PILLARS[edge.from].position;
    const staticTo = PILLARS[edge.to].position;
    const start = new THREE.Vector3(...staticFrom);
    const end = new THREE.Vector3(...staticTo);
    const dir = new THREE.Vector3().subVectors(end, start);
    const perp = new THREE.Vector3(-dir.y, dir.x, 0).normalize();
    const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
    mid.addScaledVector(perp, edge.curve * dir.length() * 0.5);
    mid.z += 0.3;

    midRef.current.copy(mid);
    const c = new THREE.CatmullRomCurve3([start, mid, end]);
    return {
      initialGeometry: new THREE.TubeGeometry(c, 48, 0.01, 6, false),
      curve: c,
    };
  }, [edge.from, edge.to, edge.curve]);

  const scratch = useMemo(
    () => ({
      dir: new THREE.Vector3(),
      perp: new THREE.Vector3(),
      targetMid: new THREE.Vector3(),
    }),
    [],
  );

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    const dt = Math.min(delta, 0.05);

    const start = positionsRef.current[edge.from];
    const end = positionsRef.current[edge.to];
    if (!start || !end) return;

    scratch.dir.subVectors(end, start);
    const len = scratch.dir.length() || 0.0001;
    scratch.perp.set(-scratch.dir.y, scratch.dir.x, 0).normalize();
    scratch.targetMid
      .addVectors(start, end)
      .multiplyScalar(0.5)
      .addScaledVector(scratch.perp, edge.curve * len * 0.5);
    scratch.targetMid.z += 0.3;

    /* Spring-damper on midpoint for floppy bend */
    const k = 28;
    const c = 5;
    const mid = midRef.current;
    const midV = midVelRef.current;
    midV.x += (-k * (mid.x - scratch.targetMid.x) - c * midV.x) * dt;
    midV.y += (-k * (mid.y - scratch.targetMid.y) - c * midV.y) * dt;
    midV.z += (-k * (mid.z - scratch.targetMid.z) - c * midV.z) * dt;
    mid.x += midV.x * dt;
    mid.y += midV.y * dt;
    mid.z += midV.z * dt;

    curve.points[0].copy(start);
    curve.points[1].copy(mid);
    curve.points[2].copy(end);
    curve.updateArcLengths();

    const newGeo = new THREE.TubeGeometry(curve, 48, 0.01, 6, false);
    meshRef.current.geometry.dispose();
    meshRef.current.geometry = newGeo;
  });

  const connected = hovered === edge.from || hovered === edge.to;
  const dimmed = hovered !== null && !connected;

  const opacity = dimmed ? 0.04 : connected ? 0.6 : 0.25;

  /* Blend colors from both endpoint pillars */
  const blendedColor = useMemo(() => {
    const c = new THREE.Color(PILLARS[edge.from].color);
    c.lerp(new THREE.Color(PILLARS[edge.to].color), 0.5);
    return c;
  }, [edge.from, edge.to]);

  return (
    <>
      <mesh ref={meshRef} geometry={initialGeometry}>
        <meshStandardMaterial
          color={blendedColor}
          emissive={blendedColor}
          emissiveIntensity={connected ? 0.3 : 0.08}
          transparent
          opacity={opacity}
        />
      </mesh>
      <EdgeParticles
        curve={curve}
        colorA={PILLARS[edge.from].color}
        colorB={PILLARS[edge.to].color}
        active={connected}
        dimmed={dimmed}
        glowTex={glowTex}
        count={3}
      />
    </>
  );
}

/* ── Detail panel (DOM overlay on hover, centered on container) ───── */

function DetailPanel({ pillar }: { pillar: Pillar }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  return (
    <div
      style={{
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: `translate(-50%, calc(-50% + ${mounted ? "0px" : "10px"}))`,
        opacity: mounted ? 1 : 0,
        transition: "opacity 0.35s ease-out, transform 0.35s ease-out",
        pointerEvents: "none",
        zIndex: 10,
      }}
    >
      <div
        style={{
          width: 300,
          background: "rgba(8, 12, 24, 0.92)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          border: `1px solid ${pillar.color}40`,
          borderTop: `2px solid ${pillar.color}`,
          borderRadius: 14,
          padding: "16px 18px",
          fontFamily: FONT_FAMILY,
          boxShadow: `0 0 40px ${pillar.color}10, 0 12px 40px rgba(0,0,0,0.5)`,
        }}
      >
        <p
          style={{
            margin: 0,
            fontSize: 10,
            fontWeight: 500,
            color: pillar.color,
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            fontFamily: FONT_FAMILY,
          }}
        >
          {pillar.subtitle}
        </p>
        <p
          style={{
            margin: "6px 0 0",
            fontSize: 15,
            fontWeight: 600,
            color: "white",
            lineHeight: 1.3,
          }}
        >
          {pillar.title}
        </p>
        <p
          style={{
            margin: "10px 0 0",
            fontSize: 12,
            color: "#d1d5db",
            lineHeight: 1.6,
          }}
        >
          {pillar.body}
        </p>
      </div>
    </div>
  );
}

/* ── Scene contents ───────────────────────────────────────────────── */

function SceneContents({
  hovered,
  onHover,
  onUnhover,
  draggingIndex,
  setDraggingIndex,
}: {
  hovered: number | null;
  onHover: (i: number) => void;
  onUnhover: () => void;
  draggingIndex: number | null;
  setDraggingIndex: (i: number | null) => void;
}) {
  const glowTex = useMemo(() => getGlowTexture(), []);

  /* Shared live pillar positions for dynamic edges */
  const positionsRef = useRef<THREE.Vector3[]>(
    PILLARS.map((p) => new THREE.Vector3(p.position[0], p.position[1], p.position[2])),
  );

  return (
    <>
      {/* Lights — matches ExperienceScene */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />
      <directionalLight position={[-3, -2, 4]} intensity={0.3} color="#c4b5fd" />

      {/* Edges */}
      {EDGES.map((edge, i) => (
        <GraphEdgeComp
          key={i}
          edge={edge}
          hovered={hovered}
          glowTex={glowTex}
          positionsRef={positionsRef}
        />
      ))}

      {/* Pillar nodes */}
      {PILLARS.map((p, i) => (
        <PillarNode
          key={p.id}
          pillar={p}
          index={i}
          hovered={hovered}
          onHover={onHover}
          onUnhover={onUnhover}
          glowTex={glowTex}
          draggingIndex={draggingIndex}
          setDraggingIndex={setDraggingIndex}
          positionsRef={positionsRef}
        />
      ))}

      {/* Camera controls — disable while dragging an orb */}
      <OrbitControls
        enabled={draggingIndex === null}
        enableZoom={false}
        enablePan={false}
        autoRotate={hovered === null && draggingIndex === null}
        autoRotateSpeed={0.06}
      />

      {/* Post-processing */}
      <EffectComposer>
        <Bloom intensity={0.5} luminanceThreshold={0.4} mipmapBlur />
      </EffectComposer>
    </>
  );
}

/* ── Visibility invalidator ──────────────────────────���────────────── */

function Invalidator({ visible }: { visible: boolean }) {
  const { invalidate } = useThree();
  useFrame(() => {
    if (visible) invalidate();
  });
  return null;
}

/* ── Main export ──────────────────────────────────────────────────── */

export default function AboutGraph() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [hovered, setHovered] = useState<number | null>(null);
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null);

  const handleHover = useCallback((i: number) => setHovered(i), []);
  const handleUnhover = useCallback(() => setHovered(null), []);

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
    <div ref={containerRef} style={{ width: "100%", height: "100%", position: "relative" }}>
      <Canvas
        dpr={[1, 1.5]}
        gl={{ alpha: true, antialias: true }}
        camera={{ position: [0, 0.5, 7.3], fov: 42 }}
        frameloop="demand"
        style={{ background: "transparent" }}
      >
        <Invalidator visible={visible} />
        <SceneContents
          hovered={hovered}
          onHover={handleHover}
          onUnhover={handleUnhover}
          draggingIndex={draggingIndex}
          setDraggingIndex={setDraggingIndex}
        />
      </Canvas>
      {hovered !== null && draggingIndex === null && (
        <DetailPanel key={hovered} pillar={PILLARS[hovered]} />
      )}
    </div>
  );
}

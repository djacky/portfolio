"use client";

/* ------------------------------------------------------------------
   Stack — Force-directed neural-network graph.

   The four capability cards are nodes in a graph, connected by
   spring constraints (Matter.js). Edges glow and carry activation
   pulses when a node is dragged. The graph self-organises through
   mutual repulsion + spring attraction + soft boundary forces.
   Cards are DOM-rendered for crisp text; a canvas layer draws edges.
------------------------------------------------------------------ */

import { useLayoutEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Brain, Code2, Cloud, Cpu, Wrench, Move } from "lucide-react";
import Typewriter from "./Typewriter";
import Matter from "matter-js";

/* =================== data =================== */

type Skill = { name: string; level: number };
type Group = {
  icon: typeof Brain;
  title: string;
  tint: string;
  tagline: string;
  skills: Skill[];
};

const GROUPS: Group[] = [
  {
    icon: Brain,
    title: "ML & AI",
    tint: "#7c5cff",
    tagline: "research-grade models, shipped to prod",
    skills: [
      { name: "PyTorch", level: 0.95 },
      { name: "Deep Learning", level: 0.92 },
      { name: "Reinforcement Learning", level: 0.9 },
      { name: "Convex Optimization", level: 0.95 },
      { name: "Model serving", level: 0.85 },
    ],
  },
  {
    icon: Code2,
    title: "Languages",
    tint: "#22d3ee",
    tagline: "a decade of shipping",
    skills: [
      { name: "Python", level: 0.98 },
      { name: "C++", level: 0.85 },
      { name: "Node.js", level: 0.85 },
      { name: "MATLAB", level: 0.9 },
      { name: "Solidity", level: 0.6 },
    ],
  },
  {
    icon: Cloud,
    title: "Backend & Cloud",
    tint: "#34d399",
    tagline: "distributed systems that don't wake you up at 3am",
    skills: [
      { name: "FastAPI · Pydantic", level: 0.95 },
      { name: "AWS · EC2/Lambda/S3/SNS", level: 0.92 },
      { name: "PostgreSQL", level: 0.85 },
      { name: "Auth · OAuth2", level: 0.88 },
      { name: "Docker · CI/CD", level: 0.88 },
    ],
  },
  {
    icon: Cpu,
    title: "Systems & Control",
    tint: "#fbbf24",
    tagline: "the math and silicon under the ML",
    skills: [
      { name: "Control theory", level: 0.92 },
      { name: "System ID", level: 0.9 },
      { name: "HIL / SIL", level: 0.85 },
      { name: "Embedded C++", level: 0.8 },
      { name: "Signal processing", level: 0.85 },
      { name: "MPC", level: 0.85 },
      { name: "H\u221E control", level: 0.85 },
    ],
  },
];

const TOOLS = [
  "Git",
  "Linux",
  "pytest",
  "Redis",
  "Celery",
];

/* ---- graph edges ---- */

type Edge = { a: number; b: number; tint: string };

const EDGES: Edge[] = [
  { a: 0, b: 3, tint: "#b08aff" }, // ML&AI <-> Systems&Control
  { a: 1, b: 2, tint: "#2be8b0" }, // Languages <-> Backend&Cloud
  { a: 0, b: 1, tint: "#52c4f0" }, // ML&AI <-> Languages
  { a: 2, b: 3, tint: "#8cda62" }, // Backend&Cloud <-> Systems&Control
  { a: 0, b: 2, tint: "#4ce0a0" }, // ML&AI <-> Backend (model serving)
];

/* =================== physics constants =================== */

const CARD_W = 280;
const CARD_H = 190;
const SANDBOX_H = 560;

type Pulse = {
  t: number;
  speed: number;
  direction: 1 | -1;
  alpha: number;
};

type BodyRef = {
  body: Matter.Body;
  homeX: number;
  homeY: number;
  phase: number;
  dragging: boolean;
};

type EdgeRef = {
  constraint: Matter.Constraint;
  a: number;
  b: number;
  tint: string;
  pulses: Pulse[];
};

/* =================== component =================== */

function NeuralGraph({ onHoverGroup }: { onHoverGroup: (i: number | null) => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);

  const [initialTargets] = useState(() => {
    const W =
      typeof window !== "undefined"
        ? Math.min(1100, window.innerWidth - 80)
        : 1000;
    const H = SANDBOX_H;
    const cols = W < 720 ? 1 : 2;
    const rows = Math.ceil(GROUPS.length / cols);
    const gapX = 28;
    const gapY = 24;
    const totalW = cols * CARD_W + (cols - 1) * gapX;
    const totalH = rows * CARD_H + (rows - 1) * gapY;
    const startX = (W - totalW) / 2;
    const startY = (H - totalH) / 2;
    return GROUPS.map((_, i) => {
      const c = i % cols;
      const r = Math.floor(i / cols);
      return {
        x: startX + c * (CARD_W + gapX),
        y: startY + r * (CARD_H + gapY),
      };
    });
  });

  useLayoutEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    let W = container.clientWidth || 1000;
    const H = SANDBOX_H;

    /* ---- canvas DPR setup ---- */
    const dpr = Math.min(devicePixelRatio || 1, 2);
    function sizeCanvas() {
      canvas!.width = W * dpr;
      canvas!.height = H * dpr;
      edgeCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    const edgeCtx = canvas.getContext("2d")!;
    sizeCanvas();

    /* ---- lattice targets ---- */
    const cols = W < 720 ? 1 : 2;
    const rows = Math.ceil(GROUPS.length / cols);
    const gapX = 28;
    const gapY = 24;

    function computeTargets(cw: number) {
      const c = cw < 720 ? 1 : 2;
      const r = Math.ceil(GROUPS.length / c);
      const tw = c * CARD_W + (c - 1) * gapX;
      const th = r * CARD_H + (r - 1) * gapY;
      const sx = (cw - tw) / 2 + CARD_W / 2;
      const sy = (H - th) / 2 + CARD_H / 2;
      return GROUPS.map((_, i) => ({
        x: sx + (i % c) * (CARD_W + gapX),
        y: sy + Math.floor(i / c) * (CARD_H + gapY),
      }));
    }

    const targets = computeTargets(W);

    /* ---- engine (zero-gravity, no walls) ---- */
    const engine = Matter.Engine.create({
      gravity: { x: 0, y: 0 },
      enableSleeping: false,
    });
    const world = engine.world;

    /* ---- card bodies ---- */
    const bodies: BodyRef[] = GROUPS.map((_, i) => {
      const t = targets[i];
      const body = Matter.Bodies.rectangle(t.x, t.y, CARD_W, CARD_H, {
        chamfer: { radius: 18 },
        friction: 0.1,
        frictionAir: 0.06,
        restitution: 0.3,
        density: 0.002,
      });
      return {
        body,
        homeX: t.x,
        homeY: t.y,
        phase: i * 1.7,
        dragging: false,
      };
    });
    Matter.World.add(
      world,
      bodies.map((b) => b.body),
    );

    /* ---- spring constraints (edges) ---- */
    const edgeRefs: EdgeRef[] = EDGES.map(({ a, b, tint }) => {
      const bodyA = bodies[a].body;
      const bodyB = bodies[b].body;
      const dx = bodyA.position.x - bodyB.position.x;
      const dy = bodyA.position.y - bodyB.position.y;
      const restLength = Math.hypot(dx, dy);

      const constraint = Matter.Constraint.create({
        bodyA,
        bodyB,
        length: restLength,
        stiffness: 0.004,
        damping: 0.05,
        render: { visible: false },
      });
      Matter.World.add(world, constraint);
      return { constraint, a, b, tint, pulses: [] };
    });

    /* ---- mouse drag ---- */
    const mouse = Matter.Mouse.create(container);
    // Allow page scrolling — detach wheel listeners that Matter.js adds
    const mw = (mouse as any).mousewheel;
    mouse.element.removeEventListener("mousewheel", mw);
    mouse.element.removeEventListener("DOMMouseScroll", mw);
    mouse.element.removeEventListener("wheel", mw);
    const mouseConstraint = Matter.MouseConstraint.create(engine, {
      mouse,
      constraint: {
        stiffness: 0.15,
        damping: 0.1,
        render: { visible: false },
      },
    });
    Matter.World.add(world, mouseConstraint);

    /* ---- drag events: fire activation pulses ---- */
    Matter.Events.on(mouseConstraint, "startdrag", (e: unknown) => {
      const evt = e as { body: Matter.Body };
      const idx = bodies.findIndex((b) => b.body === evt.body);
      if (idx < 0) return;
      bodies[idx].dragging = true;

      // fire pulses along connected edges
      for (const edge of edgeRefs) {
        if (edge.a === idx) {
          edge.pulses.push({ t: 0, speed: 0.018, direction: 1, alpha: 1 });
        } else if (edge.b === idx) {
          edge.pulses.push({ t: 1, speed: 0.018, direction: -1, alpha: 1 });
        }
      }
    });

    Matter.Events.on(mouseConstraint, "enddrag", (e: unknown) => {
      const evt = e as { body: Matter.Body };
      const idx = bodies.findIndex((b) => b.body === evt.body);
      if (idx >= 0) bodies[idx].dragging = false;
    });

    /* ---- tick loop ---- */
    const MAX_DT = 16.67;
    let rafId = 0;
    let last = performance.now();

    const tick = (now: number) => {
      const dt = Math.min(MAX_DT, now - last);
      last = now;

      /* -- custom forces -- */

      // 1) Mutual repulsion (prevent overlap)
      for (let i = 0; i < bodies.length; i++) {
        for (let j = i + 1; j < bodies.length; j++) {
          const a = bodies[i].body;
          const b = bodies[j].body;
          const dx = a.position.x - b.position.x;
          const dy = a.position.y - b.position.y;
          const dist = Math.max(Math.hypot(dx, dy), 1);
          const force = 0.8 / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          Matter.Body.applyForce(a, a.position, { x: fx, y: fy });
          Matter.Body.applyForce(b, b.position, { x: -fx, y: -fy });
        }
      }

      // 2) Soft boundary (replaces walls)
      const MARGIN = 60;
      const BOUNDARY_F = 0.0004;
      for (const b of bodies) {
        const { x, y } = b.body.position;
        let fx = 0;
        let fy = 0;
        if (x < MARGIN) fx = BOUNDARY_F * (MARGIN - x);
        if (x > W - MARGIN) fx = -BOUNDARY_F * (x - (W - MARGIN));
        if (y < MARGIN) fy = BOUNDARY_F * (MARGIN - y);
        if (y > H - MARGIN) fy = -BOUNDARY_F * (y - (H - MARGIN));
        if (fx || fy)
          Matter.Body.applyForce(b.body, b.body.position, { x: fx, y: fy });
      }

      // 3) Gentle homing (nudges graph back to centre)
      const HOME_F = 0.000005;
      for (const b of bodies) {
        if (b.dragging) continue;
        const dx = b.homeX - b.body.position.x;
        const dy = b.homeY - b.body.position.y;
        Matter.Body.applyForce(b.body, b.body.position, {
          x: dx * HOME_F,
          y: dy * HOME_F,
        });
      }

      /* -- update pulses -- */
      for (const edge of edgeRefs) {
        for (let i = edge.pulses.length - 1; i >= 0; i--) {
          const p = edge.pulses[i];
          p.t += p.speed * p.direction;
          // fade near the end
          if (p.direction === 1) {
            p.alpha = p.t > 0.7 ? 1 - (p.t - 0.7) / 0.3 : 1;
          } else {
            p.alpha = p.t < 0.3 ? p.t / 0.3 : 1;
          }
          if (p.t > 1 || p.t < 0) {
            // cascade: fire secondary pulses at destination node
            const destIdx = p.direction === 1 ? edge.b : edge.a;
            for (const other of edgeRefs) {
              if (other === edge) continue;
              if (other.a === destIdx) {
                other.pulses.push({
                  t: 0,
                  speed: 0.018,
                  direction: 1,
                  alpha: 0.6,
                });
              } else if (other.b === destIdx) {
                other.pulses.push({
                  t: 1,
                  speed: 0.018,
                  direction: -1,
                  alpha: 0.6,
                });
              }
            }
            edge.pulses.splice(i, 1);
          }
        }
        // cap pulses per edge to avoid runaway cascades
        if (edge.pulses.length > 4) edge.pulses.length = 4;
      }

      /* -- physics step -- */
      Matter.Engine.update(engine, dt);

      /* -- draw edges on canvas -- */
      edgeCtx.clearRect(0, 0, W, H);

      for (const edge of edgeRefs) {
        const posA = bodies[edge.a].body.position;
        const posB = bodies[edge.b].body.position;

        // base edge glow
        edgeCtx.save();
        edgeCtx.strokeStyle = edge.tint;
        edgeCtx.globalAlpha = 0.2;
        edgeCtx.lineWidth = 2;
        edgeCtx.shadowColor = edge.tint;
        edgeCtx.shadowBlur = 14;
        edgeCtx.beginPath();
        edgeCtx.moveTo(posA.x, posA.y);
        edgeCtx.lineTo(posB.x, posB.y);
        edgeCtx.stroke();

        // bright core
        edgeCtx.globalAlpha = 0.45;
        edgeCtx.lineWidth = 1;
        edgeCtx.shadowBlur = 6;
        edgeCtx.stroke();
        edgeCtx.restore();

        // draw pulses
        for (const pulse of edge.pulses) {
          const px = posA.x + (posB.x - posA.x) * pulse.t;
          const py = posA.y + (posB.y - posA.y) * pulse.t;
          const r = 7;
          const grad = edgeCtx.createRadialGradient(px, py, 0, px, py, r);
          grad.addColorStop(0, edge.tint);
          grad.addColorStop(1, "transparent");
          edgeCtx.save();
          edgeCtx.globalAlpha = pulse.alpha * 0.9;
          edgeCtx.shadowColor = edge.tint;
          edgeCtx.shadowBlur = 18;
          edgeCtx.fillStyle = grad;
          edgeCtx.beginPath();
          edgeCtx.arc(px, py, r, 0, Math.PI * 2);
          edgeCtx.fill();
          edgeCtx.restore();
        }
      }

      /* -- sync DOM transforms -- */
      const tSec = now * 0.001;
      bodies.forEach((b, i) => {
        const el = cardRefs.current[i];
        if (!el) return;
        let x = b.body.position.x;
        let y = b.body.position.y;
        let angle = b.body.angle;

        // subtle breathing that fades during motion
        const speed = Math.hypot(b.body.velocity.x, b.body.velocity.y);
        const breathe = Math.max(0, 1 - speed / 2);
        x += Math.cos(tSec * 0.6 + b.phase) * 2.5 * breathe;
        y += Math.sin(tSec * 0.75 + b.phase) * 3 * breathe;
        angle += Math.sin(tSec * 0.5 + b.phase) * 0.012 * breathe;

        el.style.transform = `translate(${x - CARD_W / 2}px, ${
          y - CARD_H / 2
        }px) rotate(${angle}rad)`;
      });

      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    // First-paint sync
    bodies.forEach((b, i) => {
      const el = cardRefs.current[i];
      if (!el) return;
      el.style.transform = `translate(${b.body.position.x - CARD_W / 2}px, ${
        b.body.position.y - CARD_H / 2
      }px)`;
      el.style.opacity = "1";
    });

    /* ---- resize ---- */
    let lastW = W;
    const ro = new ResizeObserver(() => {
      const newW = container.clientWidth;
      if (Math.abs(newW - lastW) < 4) return;
      lastW = newW;
      W = newW;
      sizeCanvas();
      const newTargets = computeTargets(newW);
      bodies.forEach((b, i) => {
        b.homeX = newTargets[i].x;
        b.homeY = newTargets[i].y;
      });
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      Matter.World.clear(world, false);
      Matter.Engine.clear(engine);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden cursor-grab active:cursor-grabbing"
      style={{
        height: SANDBOX_H,
        contain: "layout paint style",
        touchAction: "pan-y",
      }}
    >
      {/* edge canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ zIndex: 1 }}
      />

      {/* drag hint */}
      <div className="absolute top-4 left-4 z-10 inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.2em] text-gray-500 pointer-events-none">
        <Move className="w-3 h-3" />
        drag a node · watch the network respond
      </div>

      {GROUPS.map((g, i) => {
        const Icon = g.icon;
        const t = initialTargets[i];
        return (
          <div
            key={g.title}
            ref={(el) => {
              cardRefs.current[i] = el;
            }}
            className="absolute top-0 left-0 select-none"
            onMouseEnter={() => onHoverGroup(i)}
            onMouseLeave={() => onHoverGroup(null)}
            style={{
              width: CARD_W,
              height: CARD_H,
              transformOrigin: "center center",
              willChange: "transform",
              zIndex: 2,
              transform: `translate(${t.x}px, ${t.y}px)`,
            }}
          >
            <div
              className="relative w-full h-full rounded-[18px] p-5 overflow-hidden backdrop-blur-xl"
              style={{
                background: `linear-gradient(135deg, ${g.tint}18, rgba(10,12,20,0.85))`,
                border: `1px solid ${g.tint}55`,
                boxShadow: `0 20px 40px -20px rgba(0,0,0,0.8), 0 0 30px ${g.tint}22, inset 0 1px 0 rgba(255,255,255,0.06)`,
              }}
            >
              <div
                className="absolute -top-16 -right-16 w-40 h-40 rounded-full blur-3xl opacity-40 pointer-events-none"
                style={{ background: g.tint }}
              />
              <div className="relative flex items-start gap-2.5">
                <div
                  className="rounded-lg p-2 border"
                  style={{
                    background: `${g.tint}18`,
                    borderColor: `${g.tint}55`,
                  }}
                >
                  <Icon className="w-4 h-4" style={{ color: g.tint }} />
                </div>
                <div className="min-w-0">
                  <div className="text-[14px] font-semibold text-white leading-tight">
                    {g.title}
                  </div>
                  <div className="text-[10px] text-gray-500 leading-tight mt-0.5 truncate">
                    {g.tagline}
                  </div>
                </div>
              </div>
              <div className="relative mt-3 flex flex-wrap gap-1">
                {g.skills.map((s) => {
                  const fs = 10 + Math.round((s.level - 0.6) * 6);
                  return (
                    <span
                      key={s.name}
                      className="rounded-full border px-2 py-0.5 font-medium"
                      style={{
                        fontSize: `${fs}px`,
                        color:
                          s.level > 0.85 ? "#fff" : "rgba(229,231,235,0.85)",
                        borderColor: `${g.tint}44`,
                        background: `${g.tint}12`,
                      }}
                    >
                      {s.name}
                    </span>
                  );
                })}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* =================== section =================== */

export default function Skills() {
  return (
    <section id="skills" className="relative mx-auto max-w-6xl px-6 py-24">

      <header className="relative mb-10">
        <Typewriter
          text="Stack"
          as="p"
          speed={60}
          className="text-xs uppercase tracking-[0.25em] text-accent2"
        />
        <Typewriter
          text="The tools I reach for."
          as="h2"
          speed={30}
          delay={500}
          showCursor={false}
          className="mt-2 text-4xl md:text-5xl font-semibold text-gradient"
        />
        <Typewriter
          text="Four capability pillars, connected. Drag a node and watch the network respond."
          as="p"
          speed={18}
          delay={1200}
          showCursor={false}
          className="mt-3 max-w-2xl text-sm text-gray-400"
        />
      </header>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="relative"
      >
        <NeuralGraph onHoverGroup={() => {}} />
      </motion.div>

      {/* tooling strip */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="relative mt-8"
      >
        <div className="flex items-center gap-3 text-[10px] uppercase tracking-[0.25em] text-gray-500 mb-3">
          <Wrench className="w-3 h-3" />
          Day-to-day tooling
        </div>
        <div className="flex flex-wrap gap-1.5">
          {TOOLS.map((t) => (
            <span
              key={t}
              className="text-[11px] px-2.5 py-1 rounded-full border border-white/10 bg-white/[0.02] text-gray-500 hover:text-gray-200 hover:border-white/20 transition-colors"
            >
              {t}
            </span>
          ))}
        </div>
      </motion.div>
    </section>
  );
}

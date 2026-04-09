"use client";

/* ------------------------------------------------------------------
   Stack — Matter.js physics sandbox.

   The four capability cards are rigid bodies in a walled box. You
   can drag them, fling them, watch them bounce. After ~2 seconds of
   no interaction the cards smoothly settle back into a stable 2×2
   lattice so the page still *reads* as a structured section.

   Cards are rendered in the DOM (not on canvas) so all the glass,
   gradients, and text stay crisp — Matter.js just drives the
   position/rotation transforms each frame.
------------------------------------------------------------------ */

import { useLayoutEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Brain, Code2, Cloud, Cpu, Wrench, Move } from "lucide-react";
import Matter from "matter-js";

type Skill = { name: string; level: number; sub?: string };
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
      { name: "Reinforcement Learning", level: 0.88 },
      { name: "Optimization", level: 0.9 },
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
      { name: "TypeScript", level: 0.8 },
      { name: "MATLAB", level: 0.85 },
      { name: "Solidity", level: 0.6 },
    ],
  },
  {
    icon: Cloud,
    title: "Backend & Cloud",
    tint: "#34d399",
    tagline: "distributed systems that don't wake you up at 3am",
    skills: [
      { name: "FastAPI", level: 0.95 },
      { name: "AWS", level: 0.92 },
      { name: "PostgreSQL", level: 0.85 },
      { name: "Auth · OAuth2", level: 0.82 },
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
    ],
  },
];

const TOOLS = [
  "Git",
  "Linux",
  "pytest",
  "Grafana",
  "Redis",
  "Kubernetes",
  "Terraform",
  "Celery",
];

/* ---------------- physics sandbox ---------------- */

const CARD_W = 280;
const CARD_H = 190;
const SANDBOX_H = 560;
const RETURN_DELAY = 1500;

type CardMode = "idle" | "awake";
type BodyRef = {
  body: Matter.Body;
  targetX: number;
  targetY: number;
  mode: CardMode;
  phase: number;
  returning: boolean;
};

function PhysicsSandbox() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);
  // Compute initial lattice targets immediately so we can render cards
  // at their final positions *before* Matter.js mounts — this guarantees
  // something visible even if the physics engine has a hiccup.
  const [initialTargets] = useState(() => {
    // Rough SSR-safe guess; real positions get written on mount.
    const W = typeof window !== "undefined" ? Math.min(1100, window.innerWidth - 80) : 1000;
    const H = SANDBOX_H;
    const cols = W < 720 ? 1 : 2;
    const rows = Math.ceil(GROUPS.length / cols);
    const gapX = 28;
    const gapY = 24;
    const totalW = cols * CARD_W + (cols - 1) * gapX;
    const totalH = rows * CARD_H + (rows - 1) * gapY;
    // top-left corner coordinates (matches the `translate` we apply).
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
    if (!container) return;

    const W = container.clientWidth || 1000;
    const H = SANDBOX_H;

    // 2×2 lattice target positions (centered)
    const cols = W < 720 ? 1 : 2;
    const rows = Math.ceil(GROUPS.length / cols);
    const gapX = 28;
    const gapY = 24;
    const totalW = cols * CARD_W + (cols - 1) * gapX;
    const totalH = rows * CARD_H + (rows - 1) * gapY;
    const startX = (W - totalW) / 2 + CARD_W / 2;
    const startY = (H - totalH) / 2 + CARD_H / 2;

    const targets: { x: number; y: number }[] = GROUPS.map((_, i) => {
      const c = i % cols;
      const r = Math.floor(i / cols);
      return {
        x: startX + c * (CARD_W + gapX),
        y: startY + r * (CARD_H + gapY),
      };
    });

    // Engine — zero gravity, cards float by default and only move when
    // touched or hit by another awake card.
    const engine = Matter.Engine.create({
      gravity: { x: 0, y: 0 },
      enableSleeping: false,
    });
    const world = engine.world;

    // Walls
    const wallOpts: Matter.IChamferableBodyDefinition = {
      isStatic: true,
      render: { visible: false },
      friction: 0.2,
      restitution: 0.4,
    };
    const thickness = 200;
    const walls = [
      Matter.Bodies.rectangle(W / 2, -thickness / 2, W * 2, thickness, wallOpts),
      Matter.Bodies.rectangle(
        W / 2,
        H + thickness / 2,
        W * 2,
        thickness,
        wallOpts,
      ),
      Matter.Bodies.rectangle(
        -thickness / 2,
        H / 2,
        thickness,
        H * 2,
        wallOpts,
      ),
      Matter.Bodies.rectangle(
        W + thickness / 2,
        H / 2,
        thickness,
        H * 2,
        wallOpts,
      ),
    ];
    Matter.World.add(world, walls);

    // Card bodies — start at their lattice positions, all idle.
    const bodies: BodyRef[] = GROUPS.map((_, i) => {
      const t = targets[i];
      const body = Matter.Bodies.rectangle(t.x, t.y, CARD_W, CARD_H, {
        chamfer: { radius: 18 },
        friction: 0.15,
        frictionAir: 0.04,
        restitution: 0.5,
        density: 0.0022,
      });
      return {
        body,
        targetX: t.x,
        targetY: t.y,
        mode: "idle" as CardMode,
        phase: i * 1.7,
        returning: false,
      };
    });
    Matter.World.add(
      world,
      bodies.map((b) => b.body),
    );

    // Mouse drag
    const mouse = Matter.Mouse.create(container);
    const mouseConstraint = Matter.MouseConstraint.create(engine, {
      mouse,
      constraint: {
        stiffness: 0.15,
        damping: 0.1,
        render: { visible: false },
      },
    });
    Matter.World.add(world, mouseConstraint);

    // --- state machine ---
    //
    //   idle:  pinned at lattice target each frame, rendered with a
    //          small floating wobble. Pinning = zero velocity + setPosition
    //          so collisions can still *register* (the event fires) but the
    //          card doesn't drift.
    //   awake: free physics — responds to drag, walls, and other cards.
    //
    // Transitions:
    //   idle  → awake: user drags the card, OR an awake card collides with it.
    //   awake → idle : only via the "returning" phase below.
    //
    // Return behavior:
    //   On mouseleave(container), after RETURN_DELAY ms, every awake card
    //   is marked `returning`. In the tick loop we lerp it back to its
    //   lattice target and, once close enough, flip it back to idle.
    //   On mouseenter, cancel any pending return and un-flag returning.

    const wake = (ref: BodyRef) => {
      ref.mode = "awake";
      ref.returning = false;
    };

    Matter.Events.on(mouseConstraint, "startdrag", (e: unknown) => {
      const evt = e as { body: Matter.Body };
      const ref = bodies.find((b) => b.body === evt.body);
      if (ref) wake(ref);
    });

    Matter.Events.on(engine, "collisionStart", (e: unknown) => {
      const evt = e as { pairs: { bodyA: Matter.Body; bodyB: Matter.Body }[] };
      for (const pair of evt.pairs) {
        const a = bodies.find((b) => b.body === pair.bodyA);
        const b = bodies.find((bb) => bb.body === pair.bodyB);
        if (!a || !b) continue;
        if (a.mode === "awake" && b.mode === "idle") wake(b);
        else if (b.mode === "awake" && a.mode === "idle") wake(a);
      }
    });

    let returnTimer: number | null = null;
    const onMouseEnter = () => {
      if (returnTimer !== null) {
        window.clearTimeout(returnTimer);
        returnTimer = null;
      }
      // Do NOT clear `returning` flags here. If a card is already lerping
      // home, let it finish — otherwise it gets stuck in "awake" mode with
      // no velocity and no float offset (the "frozen box" bug). If the
      // user wants to grab a returning card, `startdrag` will wake it.
    };
    const onMouseLeave = () => {
      if (returnTimer !== null) window.clearTimeout(returnTimer);
      returnTimer = window.setTimeout(() => {
        for (const b of bodies) if (b.mode === "awake") b.returning = true;
      }, RETURN_DELAY);
    };
    container.addEventListener("mouseenter", onMouseEnter);
    container.addEventListener("mouseleave", onMouseLeave);

    // Runner — manual rAF so we can sync DOM in the same frame.
    // We cap the timestep at 16.67ms so tab-switching or a single slow
    // frame can't produce a giant integration step that lets bodies
    // tunnel through walls. Giant flings are the main failure mode.
    const MAX_DT = 16.67;
    const CAP_VEL = 45; // px/step — generous but bounded
    let rafId = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(MAX_DT, now - last);
      last = now;

      // Pin idle bodies to their lattice targets (collisions still fire;
      // they just don't drift). This is what kills the hover-out flicker:
      // idle bodies have no integration noise to amplify.
      for (const b of bodies) {
        if (b.mode !== "idle") continue;
        Matter.Body.setPosition(b.body, { x: b.targetX, y: b.targetY });
        Matter.Body.setVelocity(b.body, { x: 0, y: 0 });
        Matter.Body.setAngle(b.body, 0);
        Matter.Body.setAngularVelocity(b.body, 0);
      }

      // Returning phase: lerp awake bodies home, then flip to idle.
      for (const b of bodies) {
        if (!b.returning) continue;
        if (mouseConstraint.body === b.body) continue; // user grabbed it mid-return
        const pos = b.body.position;
        const nx = pos.x + (b.targetX - pos.x) * 0.1;
        const ny = pos.y + (b.targetY - pos.y) * 0.1;
        const na = b.body.angle * 0.88;
        Matter.Body.setPosition(b.body, { x: nx, y: ny });
        Matter.Body.setVelocity(b.body, { x: 0, y: 0 });
        Matter.Body.setAngle(b.body, na);
        Matter.Body.setAngularVelocity(b.body, 0);
        const dist = Math.hypot(b.targetX - nx, b.targetY - ny);
        if (dist < 0.6 && Math.abs(na) < 0.012) {
          b.mode = "idle";
          b.returning = false;
        }
      }

      // Clamp velocities before update so a fling can't tunnel a wall.
      for (const b of bodies) {
        if (b.mode !== "awake") continue;
        const vx = b.body.velocity.x;
        const vy = b.body.velocity.y;
        const speed = Math.hypot(vx, vy);
        if (speed > CAP_VEL) {
          const s = CAP_VEL / speed;
          Matter.Body.setVelocity(b.body, { x: vx * s, y: vy * s });
        }
      }

      Matter.Engine.update(engine, dt);

      // Safety net: escaped body → teleport back, zero velocity, idle.
      for (const b of bodies) {
        const { x, y } = b.body.position;
        const escaped =
          !Number.isFinite(x) ||
          !Number.isFinite(y) ||
          x < -CARD_W ||
          x > W + CARD_W ||
          y < -CARD_H ||
          y > H + CARD_H;
        if (escaped) {
          Matter.Body.setPosition(b.body, { x: b.targetX, y: b.targetY });
          Matter.Body.setVelocity(b.body, { x: 0, y: 0 });
          Matter.Body.setAngularVelocity(b.body, 0);
          Matter.Body.setAngle(b.body, 0);
          b.mode = "idle";
          b.returning = false;
        }
      }

      // Sync DOM transforms — idle cards get a subtle floating wobble
      // so the lattice feels alive even while nothing is happening.
      const tSec = now * 0.001;
      bodies.forEach((b, i) => {
        const el = cardRefs.current[i];
        if (!el) return;
        let x = b.body.position.x;
        let y = b.body.position.y;
        let angle = b.body.angle;
        if (b.mode === "idle") {
          x += Math.cos(tSec * 0.6 + b.phase) * 2.5;
          y += Math.sin(tSec * 0.75 + b.phase) * 3;
          angle += Math.sin(tSec * 0.5 + b.phase) * 0.012;
        }
        el.style.transform = `translate(${x - CARD_W / 2}px, ${
          y - CARD_H / 2
        }px) rotate(${angle}rad)`;
      });

      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    // Force an immediate sync so cards are visible on the very first paint.
    bodies.forEach((b, i) => {
      const el = cardRefs.current[i];
      if (!el) return;
      el.style.transform = `translate(${b.body.position.x - CARD_W / 2}px, ${
        b.body.position.y - CARD_H / 2
      }px)`;
      el.style.opacity = "1";
    });

    // Resize — simple reset to new targets on container width change
    let lastW = W;
    const ro = new ResizeObserver(() => {
      const newW = container.clientWidth;
      if (Math.abs(newW - lastW) < 4) return;
      lastW = newW;
      const newCols = newW < 720 ? 1 : 2;
      const newRows = Math.ceil(GROUPS.length / newCols);
      const tw = newCols * CARD_W + (newCols - 1) * gapX;
      const th = newRows * CARD_H + (newRows - 1) * gapY;
      const sx = (newW - tw) / 2 + CARD_W / 2;
      const sy = (H - th) / 2 + CARD_H / 2;
      bodies.forEach((b, i) => {
        const c = i % newCols;
        const r = Math.floor(i / newCols);
        b.targetX = sx + c * (CARD_W + gapX);
        b.targetY = sy + r * (CARD_H + gapY);
      });
    });
    ro.observe(container);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      if (returnTimer !== null) window.clearTimeout(returnTimer);
      container.removeEventListener("mouseenter", onMouseEnter);
      container.removeEventListener("mouseleave", onMouseLeave);
      Matter.World.clear(world, false);
      Matter.Engine.clear(engine);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full rounded-3xl border border-white/10 bg-gradient-to-b from-black/30 to-black/50 overflow-hidden cursor-grab active:cursor-grabbing"
      style={{
        height: SANDBOX_H,
        // Isolate this subtree from outer layout / paint — prevents the
        // floating cards' subpixel transforms from ever leaking into
        // document-level paint and causing the page to appear to shift.
        contain: "layout paint style",
        touchAction: "none",
      }}
    >
      {/* subtle grid floor */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.06]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* drag hint */}
      <div className="absolute top-4 left-4 inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.2em] text-gray-500 pointer-events-none">
        <Move className="w-3 h-3" />
        drag · fling · let go
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
            style={{
              width: CARD_W,
              height: CARD_H,
              transformOrigin: "center center",
              willChange: "transform",
              // Render at target position immediately — Matter.js will
              // overwrite this on the next frame.
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

/* ---------------- section ---------------- */

export default function Skills() {
  return (
    <section id="skills" className="relative mx-auto max-w-6xl px-6 py-24">
      <header className="mb-10">
        <p className="text-xs uppercase tracking-[0.25em] text-accent2">
          Stack
        </p>
        <h2 className="mt-2 text-4xl md:text-5xl font-semibold text-gradient">
          The tools I reach for.
        </h2>
        <p className="mt-3 max-w-2xl text-sm text-gray-400">
          Four capability pillars. Grab a card, fling it, let it settle — the
          lattice re-forms on its own.
        </p>
      </header>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.6 }}
      >
        <PhysicsSandbox />
      </motion.div>

      {/* tooling strip */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ delay: 0.2 }}
        className="mt-8"
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

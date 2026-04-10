"use client";

import { useEffect, useRef, useCallback } from "react";
import Matter from "matter-js";

const CANVAS_HEIGHT = 60;
const COLOR_LEFT = "#7c5cff";
const COLOR_RIGHT = "#22d3ee";

export default function ChainDivider() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  /* refs that survive across RAF frames */
  const engineRef = useRef<Matter.Engine | null>(null);
  const bodiesRef = useRef<Matter.Body[]>([]);
  const constraintsRef = useRef<Matter.Constraint[]>([]);
  const pinLeftRef = useRef<Matter.Constraint | null>(null);
  const pinRightRef = useRef<Matter.Constraint | null>(null);
  const rafRef = useRef<number>(0);
  const visibleRef = useRef(false);
  const scrollYRef = useRef(0);
  const prevScrollYRef = useRef(0);
  const prevTimeRef = useRef(0);

  /* ---------- build (or rebuild) the chain ---------- */
  const buildChain = useCallback((width: number) => {
    const engine = engineRef.current;
    if (!engine) return;

    /* clear previous bodies / constraints */
    Matter.Composite.clear(engine.world, false, true);
    bodiesRef.current = [];
    constraintsRef.current = [];
    pinLeftRef.current = null;
    pinRightRef.current = null;

    const isMobile = width < 768;
    const count = isMobile ? 15 : 25;
    const radius = 2.5;
    const gap = (width - radius * 2) / (count - 1);
    const yStart = CANVAS_HEIGHT * 0.35; // a bit above center so drape looks nice

    const bodies: Matter.Body[] = [];
    for (let i = 0; i < count; i++) {
      const body = Matter.Bodies.circle(radius + i * gap, yStart, radius, {
        mass: 0.05,
        friction: 0.02,
        frictionAir: 0.04,
        restitution: 0.1,
        render: { visible: false },
      });
      bodies.push(body);
    }
    Matter.Composite.add(engine.world, bodies);
    bodiesRef.current = bodies;

    /* link neighbours */
    const constraints: Matter.Constraint[] = [];
    for (let i = 0; i < count - 1; i++) {
      const c = Matter.Constraint.create({
        bodyA: bodies[i],
        bodyB: bodies[i + 1],
        length: gap,
        stiffness: 0.6,
        damping: 0.05,
        render: { visible: false },
      });
      constraints.push(c);
    }
    Matter.Composite.add(engine.world, constraints);
    constraintsRef.current = constraints;

    /* pin endpoints */
    const pinL = Matter.Constraint.create({
      pointA: { x: radius, y: yStart },
      bodyB: bodies[0],
      length: 0,
      stiffness: 1,
    });
    const pinR = Matter.Constraint.create({
      pointA: { x: width - radius, y: yStart },
      bodyB: bodies[count - 1],
      length: 0,
      stiffness: 1,
    });
    Matter.Composite.add(engine.world, [pinL, pinR]);
    pinLeftRef.current = pinL;
    pinRightRef.current = pinR;
  }, []);

  /* ---------- draw the chain ---------- */
  const draw = useCallback((ctx: CanvasRenderingContext2D, width: number) => {
    const bodies = bodiesRef.current;
    if (bodies.length < 2) return;

    ctx.clearRect(0, 0, width, CANVAS_HEIGHT);

    /* build gradient */
    const grad = ctx.createLinearGradient(0, 0, width, 0);
    grad.addColorStop(0, COLOR_LEFT);
    grad.addColorStop(1, COLOR_RIGHT);

    /* compute path once */
    const buildPath = () => {
      ctx.beginPath();
      ctx.moveTo(bodies[0].position.x, bodies[0].position.y);
      for (let i = 0; i < bodies.length - 1; i++) {
        const curr = bodies[i].position;
        const next = bodies[i + 1].position;
        const mx = (curr.x + next.x) / 2;
        const my = (curr.y + next.y) / 2;
        ctx.quadraticCurveTo(curr.x, curr.y, mx, my);
      }
      const last = bodies[bodies.length - 1].position;
      ctx.lineTo(last.x, last.y);
    };

    /* glow pass */
    ctx.save();
    ctx.strokeStyle = grad;
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.35;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    buildPath();
    ctx.stroke();
    ctx.restore();

    /* crisp pass */
    ctx.save();
    ctx.strokeStyle = grad;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    buildPath();
    ctx.stroke();
    ctx.restore();
  }, []);

  /* ---------- lifecycle ---------- */
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    /* engine */
    const engine = Matter.Engine.create({
      gravity: { x: 0, y: 0.6, scale: 0.001 },
    });
    engineRef.current = engine;

    /* initial sizing */
    const setSize = () => {
      const w = wrapper.clientWidth;
      canvas.width = w;
      canvas.height = CANVAS_HEIGHT;
      return w;
    };
    let width = setSize();
    buildChain(width);

    /* scroll tracking (passive) */
    scrollYRef.current = window.scrollY;
    prevScrollYRef.current = window.scrollY;

    const onScroll = () => {
      scrollYRef.current = window.scrollY;
    };
    window.addEventListener("scroll", onScroll, { passive: true });

    /* resize handler */
    const onResize = () => {
      width = setSize();
      buildChain(width);
    };
    window.addEventListener("resize", onResize);

    /* intersection observer */
    const observer = new IntersectionObserver(
      ([entry]) => {
        visibleRef.current = entry.isIntersecting;
        if (entry.isIntersecting) {
          prevTimeRef.current = performance.now();
          prevScrollYRef.current = scrollYRef.current;
          loop();
        }
      },
      { threshold: 0 }
    );
    observer.observe(wrapper);

    /* animation loop */
    const loop = () => {
      if (!visibleRef.current) return;

      const now = performance.now();
      const dt = Math.min((now - prevTimeRef.current) / 1000, 0.05); // cap delta
      prevTimeRef.current = now;

      /* scroll velocity => horizontal force */
      const scrollVel = dt > 0 ? (scrollYRef.current - prevScrollYRef.current) / (dt * 1000) : 0;
      prevScrollYRef.current = scrollYRef.current;

      const forceMag = scrollVel * 0.00004;
      const bodies = bodiesRef.current;
      for (let i = 1; i < bodies.length - 1; i++) {
        Matter.Body.applyForce(bodies[i], bodies[i].position, {
          x: forceMag,
          y: 0,
        });
      }

      Matter.Engine.update(engine, 1000 / 60);
      draw(ctx, width);

      rafRef.current = requestAnimationFrame(loop);
    };

    /* kick off if already visible (unlikely but safe) */
    if (visibleRef.current) {
      prevTimeRef.current = performance.now();
      loop();
    }

    /* cleanup */
    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onResize);
      observer.disconnect();
      Matter.Engine.clear(engine);
    };
  }, [buildChain, draw]);

  return (
    <div
      ref={wrapperRef}
      style={{ width: "100%", height: CANVAS_HEIGHT, position: "relative", pointerEvents: "none" }}
    >
      <canvas
        ref={canvasRef}
        style={{
          display: "block",
          width: "100%",
          height: CANVAS_HEIGHT,
        }}
      />
    </div>
  );
}

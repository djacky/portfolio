"use client";

import { useEffect, useRef } from "react";

/* ------------------------------------------------------------------ *
 *  Neural Constellation                                               *
 *  Floating glowing nodes with connections — the classic               *
 *  "connected nodes" AI motif.  Sits above the vignette in the Hero.  *
 * ------------------------------------------------------------------ */

const C1: [number, number, number] = [124, 92, 255]; // accent
const C2: [number, number, number] = [34, 211, 238]; // accent2
const MAX_DIST = 220;
const SPEED = 0.25;

interface Dot {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  a: number;
  c: [number, number, number];
  pulse: number;
  pulseSpeed: number;
}

export default function NeuralConstellation() {
  const cvs = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = cvs.current;
    if (!el) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;

    let w = 0;
    let h = 0;
    let raf = 0;
    let dots: Dot[] = [];
    const D2 = MAX_DIST * MAX_DIST;

    function resize() {
      const dpr = devicePixelRatio || 1;
      w = el!.offsetWidth;
      h = el!.offsetHeight;
      el!.width = w * dpr;
      el!.height = h * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function seed() {
      resize();
      const n = Math.min(Math.floor((w * h) / 9000), 150);
      dots = [];
      for (let i = 0; i < n; i++) {
        dots.push({
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * SPEED,
          vy: (Math.random() - 0.5) * SPEED,
          r: Math.random() * 2 + 1.5,
          a: Math.random() * 0.25 + 0.15,
          c: Math.random() > 0.5 ? C1 : C2,
          pulse: Math.random() * Math.PI * 2,
          pulseSpeed: Math.random() * 0.02 + 0.01,
        });
      }
    }

    function tick() {
      ctx!.clearRect(0, 0, w, h);

      for (const d of dots) {
        d.x += d.vx;
        d.y += d.vy;
        d.pulse += d.pulseSpeed;
        if (d.x < 0) d.x = w;
        else if (d.x > w) d.x = 0;
        if (d.y < 0) d.y = h;
        else if (d.y > h) d.y = 0;
      }

      // connections
      ctx!.lineWidth = 1;
      for (let i = 0; i < dots.length; i++) {
        for (let j = i + 1; j < dots.length; j++) {
          const dx = dots[i].x - dots[j].x;
          const dy = dots[i].y - dots[j].y;
          const dd = dx * dx + dy * dy;
          if (dd < D2) {
            const a = (1 - Math.sqrt(dd) / MAX_DIST) * 0.2;
            const [r, g, b] = dots[i].c;
            ctx!.strokeStyle = `rgba(${r},${g},${b},${a})`;
            ctx!.beginPath();
            ctx!.moveTo(dots[i].x, dots[i].y);
            ctx!.lineTo(dots[j].x, dots[j].y);
            ctx!.stroke();
          }
        }
      }

      // glowing dots
      for (const d of dots) {
        const pulseFactor = 0.7 + 0.3 * Math.sin(d.pulse);
        const alpha = d.a * pulseFactor;
        const [r, g, b] = d.c;

        // outer glow
        ctx!.shadowColor = `rgba(${r},${g},${b},${alpha})`;
        ctx!.shadowBlur = 8;
        ctx!.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx!.beginPath();
        ctx!.arc(d.x, d.y, d.r, 0, 6.2832);
        ctx!.fill();

        // bright core
        ctx!.shadowBlur = 0;
        ctx!.fillStyle = `rgba(${r},${g},${b},${Math.min(alpha * 1.5, 1)})`;
        ctx!.beginPath();
        ctx!.arc(d.x, d.y, d.r * 0.5, 0, 6.2832);
        ctx!.fill();
      }

      // reset shadow state
      ctx!.shadowColor = "transparent";
      ctx!.shadowBlur = 0;

      raf = requestAnimationFrame(tick);
    }

    seed();
    tick();
    window.addEventListener("resize", resize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={cvs}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  );
}

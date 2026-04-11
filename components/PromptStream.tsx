"use client";

import { useEffect, useRef } from "react";

/*
  Matrix rain background.
  Canvas-based: cascading katakana, binary, and code glyphs falling in
  columns. Each column has a bright leading head with a fading green trail.
  Runs at low opacity behind section content.
*/

/* ── glyph pools ─────────────────────────────────────── */
// Katakana (U+30A0 – U+30FF)
const KATAKANA = Array.from({ length: 96 }, (_, i) =>
  String.fromCharCode(0x30a0 + i),
);
const BINARY = ["0", "1"];
const CODE_TOKENS = [
  "{", "}", "(", ")", ";", "=", "<", ">", "/", "*",
  "+", "-", "&", "|", "!", "?", "#", "%", "^", "~",
  ":", ".", ",", "[", "]", "@", "$", "_", "\\",
];
const ALL_GLYPHS = [...KATAKANA, ...BINARY, ...BINARY, ...CODE_TOKENS];

function randomGlyph(): string {
  return ALL_GLYPHS[Math.floor(Math.random() * ALL_GLYPHS.length)];
}

/* ── config ──────────────────────────────────────────── */
const FONT_SIZE = 16;
const COL_WIDTH = FONT_SIZE + 2;      // px per column
const DROP_SPEED_MIN = 0.3;           // cells per frame
const DROP_SPEED_MAX = 1.0;
const TRAIL_LENGTH = 22;              // cells before fully fading
const HEAD_COLOR = [0, 255, 120];     // bright green-white head
const TRAIL_COLOR = [0, 180, 80];     // classic matrix green

interface Drop {
  col: number;        // column index
  y: number;          // current row (fractional)
  speed: number;      // cells per frame
  glyphs: string[];   // pre-generated trail glyphs
  trail: number;      // trail length for this drop
}

export default function MatrixRain() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d")!;
    const dpr = Math.min(devicePixelRatio || 1, 2);

    let W = 0;
    let H = 0;
    let cols = 0;
    let rows = 0;
    let drops: Drop[] = [];

    function createDrop(col: number, startAbove = true): Drop {
      const trail = Math.floor(TRAIL_LENGTH * (0.5 + Math.random() * 0.8));
      return {
        col,
        y: startAbove ? -(Math.random() * rows) : Math.random() * rows,
        speed: DROP_SPEED_MIN + Math.random() * (DROP_SPEED_MAX - DROP_SPEED_MIN),
        glyphs: Array.from({ length: trail + 4 }, () => randomGlyph()),
        trail,
      };
    }

    function resize() {
      const parent = canvas!.parentElement;
      W = parent?.clientWidth ?? window.innerWidth;
      H = parent?.clientHeight ?? window.innerHeight;
      canvas!.width = W * dpr;
      canvas!.height = H * dpr;
      canvas!.style.width = `${W}px`;
      canvas!.style.height = `${H}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      cols = Math.ceil(W / COL_WIDTH);
      rows = Math.ceil(H / FONT_SIZE);

      // Reset drops — ~60% column density for a natural look
      drops = [];
      for (let c = 0; c < cols; c++) {
        if (Math.random() < 0.6) {
          drops.push(createDrop(c, false));
        }
      }
    }

    resize();

    let rafId = 0;

    function tick() {
      rafId = requestAnimationFrame(tick);

      // Semi-transparent black overlay for trailing fade
      ctx.fillStyle = "rgba(5, 7, 13, 0.12)";
      ctx.fillRect(0, 0, W, H);

      ctx.font = `${FONT_SIZE}px "Martian Mono", "MS Gothic", "Meiryo", monospace`;
      ctx.textBaseline = "top";

      for (const drop of drops) {
        const headRow = Math.floor(drop.y);
        const x = drop.col * COL_WIDTH;

        // Draw trail cells
        for (let t = 0; t <= drop.trail; t++) {
          const row = headRow - t;
          if (row < -1 || row > rows) continue;

          const yPos = row * FONT_SIZE;
          const glyphIdx = t % drop.glyphs.length;

          if (t === 0) {
            // Bright head — white-green glow
            ctx.fillStyle = `rgba(${HEAD_COLOR[0]}, ${HEAD_COLOR[1]}, ${HEAD_COLOR[2]}, 0.95)`;
            ctx.shadowColor = `rgba(0, 255, 100, 0.8)`;
            ctx.shadowBlur = 12;
          } else {
            // Trail — fade from bright to dim
            const fade = 1 - t / drop.trail;
            const alpha = fade * fade * 0.75; // quadratic falloff
            ctx.fillStyle = `rgba(${TRAIL_COLOR[0]}, ${TRAIL_COLOR[1]}, ${TRAIL_COLOR[2]}, ${alpha})`;
            ctx.shadowBlur = 0;
          }

          // Occasionally mutate a glyph in the trail for shimmer
          if (t > 0 && Math.random() < 0.02) {
            drop.glyphs[glyphIdx] = randomGlyph();
          }

          ctx.fillText(drop.glyphs[glyphIdx], x, yPos);
        }

        ctx.shadowBlur = 0;

        // Advance
        drop.y += drop.speed;

        // Respawn when fully off screen
        if (headRow - drop.trail > rows) {
          drop.y = -(Math.random() * 10);
          drop.speed = DROP_SPEED_MIN + Math.random() * (DROP_SPEED_MAX - DROP_SPEED_MIN);
          drop.trail = Math.floor(TRAIL_LENGTH * (0.5 + Math.random() * 0.8));
          drop.glyphs = Array.from({ length: drop.trail + 4 }, () => randomGlyph());
        }
      }

      // Occasionally spawn new drops in empty columns
      if (Math.random() < 0.03) {
        const col = Math.floor(Math.random() * cols);
        const hasCol = drops.some((d) => d.col === col && d.y < rows * 0.3);
        if (!hasCol) {
          drops.push(createDrop(col, true));
        }
      }

      // Cap total drops
      if (drops.length > cols * 1.5) {
        drops.splice(0, drops.length - cols);
      }
    }

    rafId = requestAnimationFrame(tick);

    const ro = new ResizeObserver(resize);
    ro.observe(canvas.parentElement ?? canvas);

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ opacity: 0.25, zIndex: 0 }}
      aria-hidden="true"
    />
  );
}

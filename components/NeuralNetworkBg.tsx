"use client";

import { useEffect, useRef } from "react";

/*
  Matrix rain background — falling columns of katakana + code characters.
  Canvas-based, fixed behind all content at low opacity.
*/

const CHARS =
  "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン01ABCDEF#$%&@</>";

const FONT_SIZE = 13;
const DROP_SPEED = 0.5;
const RESET_CHANCE = 0.975;

export default function NeuralNetworkBg() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d")!;
    let cols = 0;
    let drops: number[] = [];

    function resize() {
      canvas!.width = window.innerWidth;
      canvas!.height = window.innerHeight;
      const newCols = Math.floor(canvas!.width / FONT_SIZE);
      // preserve existing drop positions, init new ones off-screen
      const next = Array.from({ length: newCols }, (_, i) =>
        i < drops.length ? drops[i] : Math.random() * -50,
      );
      drops = next;
      cols = newCols;
    }

    resize();

    const interval = setInterval(() => {
      ctx.fillStyle = "rgba(0,0,0,0.05)";
      ctx.fillRect(0, 0, canvas!.width, canvas!.height);

      ctx.font = `${FONT_SIZE}px monospace`;

      for (let i = 0; i < cols; i++) {
        const ch = CHARS[Math.floor(Math.random() * CHARS.length)];
        const x = i * FONT_SIZE;
        const y = drops[i] * FONT_SIZE;

        const brightness = Math.random();
        if (brightness > 0.95) {
          ctx.fillStyle = "#ffffff";
        } else if (brightness > 0.7) {
          ctx.fillStyle = "#00ff41";
        } else {
          ctx.fillStyle = "#003b00";
        }

        ctx.fillText(ch, x, y);

        if (y > canvas!.height && Math.random() > RESET_CHANCE) {
          drops[i] = 0;
        }
        drops[i] += DROP_SPEED;
      }
    }, 40);

    window.addEventListener("resize", resize);

    return () => {
      clearInterval(interval);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ opacity: 0.2, zIndex: 0 }}
    />
  );
}

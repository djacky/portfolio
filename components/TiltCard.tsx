"use client";
import { useCallback, useEffect, useRef, useState } from "react";

const MAX_TILT = 14;
const DEPTH = 8;
const LAYERS = 8;

interface TiltCardProps {
  children: React.ReactNode;
  className?: string;
}

export default function TiltCard({ children, className }: TiltCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const rectRef = useRef<DOMRect | null>(null);
  const [enabled, setEnabled] = useState(false);
  const [tilt, setTilt] = useState({ rX: 0, rY: 0 });
  const [active, setActive] = useState(false);
  const [highlight, setHighlight] = useState({ x: 50, y: 50 });

  useEffect(() => {
    const mq = window.matchMedia("(hover: hover)");
    setEnabled(mq.matches);
    const h = (e: MediaQueryListEvent) => setEnabled(e.matches);
    mq.addEventListener("change", h);
    return () => mq.removeEventListener("change", h);
  }, []);

  const handleMouseEnter = useCallback(() => {
    if (!enabled || !cardRef.current) return;
    rectRef.current = cardRef.current.getBoundingClientRect();
    setActive(true);
  }, [enabled]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!enabled || !rectRef.current) return;
      const rect = rectRef.current;
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;
      setTilt({
        rX: (0.5 - ny) * MAX_TILT * 2,
        rY: (nx - 0.5) * MAX_TILT * 2,
      });
      setHighlight({ x: nx * 100, y: ny * 100 });
    },
    [enabled],
  );

  const handleMouseLeave = useCallback(() => {
    rectRef.current = null;
    setTilt({ rX: 0, rY: 0 });
    setActive(false);
    setHighlight({ x: 50, y: 50 });
  }, []);

  /* Dynamic shadow shifts opposite to tilt direction */
  const sx = -tilt.rY * 1.5;
  const sy = tilt.rX * 1.5;
  const sb = 30 + (Math.abs(tilt.rX) + Math.abs(tilt.rY)) * 0.6;

  return (
    <div
      className={className}
      style={{ perspective: 1000 }}
      ref={cardRef}
      onMouseEnter={handleMouseEnter}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <div
        style={{
          transformStyle: "preserve-3d",
          transition: active
            ? "transform 100ms ease-out, box-shadow 100ms ease-out"
            : "transform 600ms cubic-bezier(0.23, 1, 0.32, 1), box-shadow 600ms cubic-bezier(0.23, 1, 0.32, 1)",
          transform: `rotateX(${tilt.rX}deg) rotateY(${tilt.rY}deg)`,
          position: "relative",
          borderRadius: "inherit",
          boxShadow: active
            ? `${sx}px ${sy}px ${sb}px rgba(0,0,0,0.45), 0 0 ${15 + (Math.abs(tilt.rX) + Math.abs(tilt.rY)) * 0.3}px rgba(124,92,255,0.06)`
            : "0 4px 20px rgba(0,0,0,0.25)",
        }}
      >
        {/* Depth layers — stacked behind to show card thickness on tilt */}
        {Array.from({ length: LAYERS }, (_, i) => {
          const z = -((i + 1) * (DEPTH / LAYERS));
          const edgeLight =
            Math.max(Math.abs(tilt.rY), Math.abs(tilt.rX)) / MAX_TILT;
          const borderAlpha = 0.03 + edgeLight * 0.06;

          return (
            <div
              key={i}
              aria-hidden
              style={{
                position: "absolute",
                inset: 0,
                transform: `translateZ(${z}px)`,
                background:
                  "linear-gradient(180deg, rgba(14,18,30,0.97), rgba(8,11,20,0.99))",
                borderRadius: "inherit",
                border: `1px solid rgba(255,255,255,${borderAlpha})`,
                pointerEvents: "none",
              }}
            />
          );
        })}

        {/* Main face */}
        <div style={{ position: "relative", transform: "translateZ(0px)" }}>
          {children}
        </div>

        {/* Specular glare overlay */}
        <div
          aria-hidden
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "inherit",
            pointerEvents: "none",
            zIndex: 2,
            opacity: active ? 1 : 0,
            transition: "opacity 400ms ease-out",
            background: `radial-gradient(ellipse at ${highlight.x}% ${highlight.y}%, rgba(255,255,255,0.08) 0%, transparent 65%)`,
          }}
        />
      </div>
    </div>
  );
}

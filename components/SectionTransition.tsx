"use client";

import { motion } from "framer-motion";
import { useEffect, useState, useMemo } from "react";

/* random hex/symbol burst lines */
function useCharBurst(lineCount: number, lineLen: number) {
  return useMemo(() => {
    const chars = "0123456789ABCDEFabcdef>><>::->=>fn0x//##$$%%&&";
    return Array.from({ length: lineCount }, () =>
      Array.from(
        { length: lineLen },
        () => chars[Math.floor(Math.random() * chars.length)],
      ).join(""),
    );
  }, [lineCount, lineLen]);
}

/* random glitch bar positions */
function useGlitchBars(count: number) {
  return useMemo(
    () =>
      Array.from({ length: count }, () => ({
        top: `${10 + Math.random() * 80}%`,
        width: `${30 + Math.random() * 50}%`,
        left: `${Math.random() * 20}%`,
        xOffset: (Math.random() - 0.5) * 16,
      })),
    [count],
  );
}

export default function SectionTransition() {
  const burstLines = useCharBurst(4, 48);
  const glitchBars = useGlitchBars(4);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      className="fixed inset-0 z-[90] pointer-events-none"
    >
      {/* brief blackout — holds opaque long enough to cover mode="wait" swap */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 0.85, 0.85, 0] }}
        transition={{ duration: 0.7, times: [0, 0.2, 0.72, 1] }}
        className="absolute inset-0 bg-bg"
      />

      {/* scanline sweep */}
      <motion.div
        initial={{ top: "-2%" }}
        animate={{ top: "102%" }}
        transition={{ duration: 0.45, ease: "easeInOut" }}
        className="absolute left-0 right-0 h-1.5"
        style={{
          background:
            "linear-gradient(180deg, transparent, #22d3ee80, #ffffff, #22d3ee80, transparent)",
          boxShadow: "0 0 20px 4px rgba(34,211,238,0.4)",
        }}
      />

      {/* character burst */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 0.7, 0.7, 0] }}
        transition={{ duration: 0.5, times: [0, 0.25, 0.45, 1] }}
        className="absolute inset-0 flex flex-col items-center justify-center gap-0.5 font-mono text-[10px] text-accent2/40 leading-tight"
      >
        {burstLines.map((line, i) => (
          <span key={i}>{line}</span>
        ))}
      </motion.div>

      {/* glitch bars */}
      {glitchBars.map((bar, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, x: bar.xOffset }}
          animate={{ opacity: [0, 0.8, 0], x: [bar.xOffset, 0, -bar.xOffset] }}
          transition={{
            duration: 0.35,
            delay: 0.05 + i * 0.04,
            ease: "easeOut",
          }}
          className="absolute h-[2px] bg-accent/60"
          style={{ top: bar.top, left: bar.left, width: bar.width }}
        />
      ))}
    </motion.div>
  );
}

"use client";

/* ------------------------------------------------------------------
   Disruptive Labs — Transformer Matchmaker.

   Each player is modeled as a SEQUENCE of past matches consumed by a
   small in-browser transformer (d_model=32, 2 heads, 2 layers, ~18K
   params).  The model outputs four heads per player: skill posterior
   (μ, σ), playstyle embedding, P(smurf), and a form modifier.

   Self-supervised next-match-placement prediction trains the skill
   head so ratings emerge from play history alone.  All state lives in
   a module-level MatchmakerEngine singleton so training persists
   across React mount/unmount.

   This component is a thin orchestrator that wires the 2D scene and
   the HUD, manages hover/pin state, and halts the engine when the
   browser tab is hidden.
------------------------------------------------------------------ */

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Users, Brain } from "lucide-react";
import { getEngine } from "@/lib/matchmaker-engine";
import MatchmakerScene2D from "./MatchmakerScene2D";
import MatchmakerHUD from "./MatchmakerHUD";

export default function MatchmakerDemo() {
  const engine = useMemo(() => getEngine(), []);
  const [hoveredId, setHoveredId] = useState<number | null>(null);
  const [pinnedId, setPinnedId] = useState<number | null>(null);

  useEffect(() => {
    engine.start();
    return () => engine.stop();
  }, [engine]);

  useEffect(() => {
    const onVis = () => {
      if (document.hidden) engine.stop();
      else engine.start();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [engine]);

  return (
    <div className="shimmer-border rounded-3xl">
      <div className="glass rounded-3xl p-6 md:p-8">
        {/* header */}
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-5">
          <div>
            <div className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.22em] text-accent2">
              <Users className="w-3.5 h-3.5" />
              Disruptive Labs · transformer matchmaker
            </div>
            <h3 className="mt-2 text-2xl md:text-3xl font-semibold text-gradient">
              Sequence-of-matches skill inference
            </h3>
            <p className="mt-2 text-sm text-gray-400 max-w-2xl leading-relaxed">
              A tiny transformer reads each player&apos;s past matches as a sequence and
              emits four signals at once: a skill posterior, a playstyle embedding, a
              smurf probability, and a short term form modifier. There are no ELO updates
              or labels; the ratings fall out of training the model to predict where each
              player lands in the next match.
            </p>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mt-6 grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-4"
        >
          <div>
            <MatchmakerScene2D
              hoveredId={hoveredId}
              pinnedId={pinnedId}
              onHover={setHoveredId}
              onPin={setPinnedId}
            />
            <div className="mt-3 text-[11px] font-mono text-gray-500 leading-relaxed px-1">
              <span className="text-accent2">transformer</span> · d_model=32 · 2 heads · 2 layers ·
              seq_len=20 · 4 heads (skill / playstyle / anomaly / form) · self-supervised next-match
              placement prediction · pure-TS forward + backward · Adam
            </div>
          </div>

          <MatchmakerHUD />
        </motion.div>
      </div>
    </div>
  );
}

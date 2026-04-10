"use client";

import { motion } from "framer-motion";
import { ArrowRight, MessageSquare } from "lucide-react";
import dynamic from "next/dynamic";
import NeuralConstellation from "./NeuralConstellation";
import { useContactDrawer } from "./ContactDrawer";

// Lazy-load the R3F canvas so it never blocks first paint and the
// heavy three.js bundle doesn't bloat the server render.
const PendulumScene = dynamic(() => import("./PendulumScene"), { ssr: false });

export default function Hero() {
  const { open: openContact } = useContactDrawer();
  return (
    <section
      id="top"
      className="relative min-h-screen flex items-center overflow-hidden"
    >
      {/* interactive 3D pendulum — full bleed behind content, biased right */}
      <div
        className="absolute inset-0 z-0"
        style={{ contain: "layout paint style" }}
      >
        <PendulumScene />
      </div>

      {/* soft vignette + base wash so text stays legible over the scene */}
      <div
        className="absolute inset-0 z-[1] pointer-events-none"
        style={{
          background:
            "radial-gradient(80% 60% at 25% 45%, rgba(5,7,13,0.85) 0%, rgba(5,7,13,0.55) 42%, rgba(5,7,13,0.1) 75%, transparent 100%)",
        }}
      />

      {/* AI constellation — above vignette so particles are clearly visible */}
      <div className="absolute inset-0 z-[2] pointer-events-none">
        <NeuralConstellation />
      </div>

      <div className="relative z-10 mx-auto max-w-6xl px-6 pt-28 pb-20 w-full pointer-events-none">
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.25em] text-accent2"
        >
          <span className="inline-block w-2 h-2 rounded-full bg-good animate-pulse" />
          Available for senior AI/ML roles
        </motion.p>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.6 }}
          className="mt-5 text-5xl sm:text-6xl md:text-7xl font-semibold leading-[1.02] tracking-tight"
        >
          <span className="text-gradient">Achille Nicoletti</span>
          <br />
          <span className="text-gray-300">Senior AI/ML Engineer.</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="mt-5 max-w-xl text-base md:text-lg text-gray-400 leading-relaxed"
        >
          <span className="text-white">PhD · EPFL.</span> A decade shipping
          production ML at <span className="text-white">CERN</span>,{" "}
          <span className="text-white">Eaton</span>, and as a{" "}
          <span className="text-white">founder</span>.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-9 flex flex-wrap items-center gap-4 pointer-events-auto"
        >
          <a
            href="#demos"
            className="group inline-flex items-center gap-2 rounded-full bg-accent px-5 py-3 text-sm font-medium text-white shadow-glow hover:bg-accent/90 transition-all"
          >
            Try the live demos
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </a>
          <button
            type="button"
            onClick={openContact}
            className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-black/30 backdrop-blur px-5 py-3 text-sm text-gray-200 hover:bg-white/10 transition-colors"
          >
            <MessageSquare className="w-4 h-4" /> Let&apos;s talk
          </button>
        </motion.div>

        {/* compact credential strip — no big glass tiles */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-12 flex flex-wrap gap-x-6 gap-y-2 text-[11px] font-mono uppercase tracking-[0.18em] text-gray-500"
        >
          <span>
            <span className="text-accent2">◆</span> 10+ yrs shipping
          </span>
          <span>
            <span className="text-accent2">◆</span> PPO · RL @ Eaton
          </span>
          <span>
            <span className="text-accent2">◆</span> H∞ control @ CERN
          </span>
          <span>
            <span className="text-accent2">◆</span> Python · PyTorch · C++ · AWS
          </span>
        </motion.div>

        {/* hint */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.7 }}
          transition={{ delay: 1.2, duration: 1.2 }}
          className="mt-16 text-[10px] font-mono uppercase tracking-[0.25em] text-gray-600"
        >
          ↳ grab the pendulum — teach it to swing up
        </motion.p>
      </div>
    </section>
  );
}

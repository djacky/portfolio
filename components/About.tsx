"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";
import { ChevronDown } from "lucide-react";
import Typewriter from "./Typewriter";
import { PILLARS, type Pillar } from "./aboutGraphData";

const AboutGraph = dynamic(() => import("./AboutGraph"), { ssr: false });

/* ── Mobile accordion card ────────────────────────────────────────── */

function MobileCard({
  pillar,
  isOpen,
  onToggle,
}: {
  pillar: Pillar;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <div
      className="rounded-xl border bg-white/[0.02] overflow-hidden transition-colors"
      style={{
        borderColor: isOpen ? `${pillar.color}55` : "rgba(255,255,255,0.08)",
        borderLeftWidth: 3,
        borderLeftColor: pillar.color,
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        className="w-full text-left px-4 py-3 flex items-center gap-3"
      >
        <div
          className="w-2 h-2 rounded-full shrink-0"
          style={{ backgroundColor: pillar.color }}
        />
        <div className="flex-1 min-w-0">
          <p className="text-[13px] font-semibold text-white truncate">
            {pillar.title}
          </p>
          <p className="text-[10px] italic text-gray-500 truncate mt-0.5">
            {pillar.subtitle}
          </p>
        </div>
        <ChevronDown
          className="w-4 h-4 text-gray-500 shrink-0 transition-transform"
          style={{ transform: isOpen ? "rotate(180deg)" : "none" }}
        />
      </button>

      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1">
              <p className="text-[13px] text-gray-300 leading-relaxed">
                {pillar.body}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ── About section ────────────────────────────────────────────────── */

export default function About() {
  const [openCard, setOpenCard] = useState<number | null>(null);

  return (
    <section id="about" className="relative mx-auto max-w-6xl px-6 py-24">
      <header className="mb-14">
        <Typewriter
          text="About"
          as="p"
          speed={60}
          className="text-xs uppercase tracking-[0.25em] text-accent2"
        />
        <Typewriter
          text="Where research-grade rigor meets production systems."
          as="h2"
          speed={30}
          delay={500}
          showCursor={false}
          className="mt-2 text-4xl font-semibold text-gradient"
        />
      </header>

      {/* Desktop: 3D knowledge graph */}
      <div className="hidden md:block">
        <div className="relative w-full" style={{ height: 560 }}>
          <AboutGraph />
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            transition={{ delay: 1.5, duration: 0.8 }}
            className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] font-mono uppercase tracking-[0.2em] text-gray-600 pointer-events-none z-10"
          >
            hover a node to read more · drag to orbit
          </motion.p>
        </div>
      </div>

      {/* Mobile: accordion cards */}
      <div className="md:hidden space-y-2">
        {PILLARS.map((pillar, i) => (
          <MobileCard
            key={pillar.id}
            pillar={pillar}
            isOpen={openCard === i}
            onToggle={() => setOpenCard(openCard === i ? null : i)}
          />
        ))}
      </div>
    </section>
  );
}

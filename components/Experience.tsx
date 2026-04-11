"use client";

/* ------------------------------------------------------------------ *
 *  Experience — interactive neural-network node graph (desktop)        *
 *  with a card-list fallback (mobile).                                *
 *                                                                     *
 *  Five roles arranged as a 2-1-2 network (hourglass). Career path    *
 *  highlighted in cyan; cross-connections in faint purple. Click a     *
 *  node to expand its detail panel below the graph.                   *
 * ------------------------------------------------------------------ */

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";
import {
  Rocket,
  Zap,
  Atom,
  Stethoscope,
  Apple,
  ArrowUpRight,
  ChevronDown,
} from "lucide-react";
import { useNavigation } from "./SectionRouter";
import Typewriter from "./Typewriter";

const ExperienceScene3D = dynamic(() => import("./ExperienceScene"), { ssr: false });

/* ── Data ─────────────────────────────────────────────────────────── */

type Metric = { value: string; label: string };
type Job = {
  icon: typeof Rocket;
  tint: string;
  period: string;
  role: string;
  org: string;
  location: string;
  headline: string;
  summary: string;
  metrics: Metric[];
  tags: string[];
  demo?: { href: string; label: string };
};

const JOBS: Job[] = [
  {
    icon: Rocket,
    tint: "#7c5cff",
    period: "Dec 2023 — Present",
    role: "Founder & Software Engineer",
    org: "Disruptive Labs",
    location: "Geneva, CH",
    headline:
      "Full-stack multiplayer game with an ML backbone for matchmaking and payouts.",
    summary:
      "Own the product end-to-end: distributed AWS backend, PyTorch models, live ops, compliance.",
    metrics: [
      { value: "250k+", label: "daily API req" },
      { value: "1TB+", label: "daily data" },
      { value: "SOC 2", label: "Type II" },
    ],
    tags: ["Python", "FastAPI", "PyTorch", "AWS", "PostgreSQL", "Docker"],
    demo: { href: "#demo-matchmaker", label: "Siamese matchmaker" },
  },
  {
    icon: Zap,
    tint: "#22d3ee",
    period: "Jan 2021 — Dec 2023",
    role: "Senior Research Engineer",
    org: "Eaton",
    location: "Lausanne, CH",
    headline:
      "Algorithm layer for Eaton's EV fleet load-balancing platform.",
    summary:
      "Trained a PPO policy that observes fleet state and learns an optimal current-allocation schedule, then shipped it to a global fleet of chargers via a FastAPI / AWS Lambda inference layer.",
    metrics: [
      { value: "weeks → days", label: "validation" },
      { value: "real-time", label: "inference" },
      { value: "OCPP", label: "protocol" },
    ],
    tags: ["PyTorch", "RL / PPO", "AWS Lambda", "FastAPI", "HIL / SIL"],
    demo: { href: "#demo-ev", label: "Live RL simulation" },
  },
  {
    icon: Atom,
    tint: "#34d399",
    period: "Jan 2018 — Jan 2021",
    role: "Senior Fellow",
    org: "CERN",
    location: "Geneva, CH",
    headline:
      "Data-driven control for the power converters behind the LHC experiments.",
    summary:
      "Developed Python APIs for frequency-response identification and H∞ / H2 synthesis. Implemented the resulting controllers on AVR8 microcontrollers in C++. Trained the department on the tooling.",
    metrics: [
      { value: "H∞ / H2", label: "synthesis" },
      { value: "AVR8 · C++", label: "real-time" },
    ],
    tags: ["C++", "Python", "Control", "System ID"],
    demo: { href: "#demo-cern", label: "Controller-synthesis pipeline" },
  },
  {
    icon: Stethoscope,
    tint: "#fbbf24",
    period: "May 2011 — Jul 2013",
    role: "Electrical Design Engineer II",
    org: "Philips Healthcare",
    location: "Highland Heights, USA",
    headline: "Motion control for the next generation of Philips CT scanners.",
    summary:
      "Modeled multidomain dynamics in MATLAB / Simulink, developed PLC control in CoDeSys (IEC 61131-3), shipped on production medical hardware.",
    metrics: [
      { value: "CT", label: "hardware" },
      { value: "IEC 61131-3", label: "PLC" },
    ],
    tags: ["MATLAB", "Simulink", "PLC", "CoDeSys"],
  },
  {
    icon: Apple,
    tint: "#f472b6",
    period: "Jun 2010 — Sep 2010",
    role: "Mixed-Signal Engineer (Intern)",
    org: "Apple",
    location: "Cupertino, USA",
    headline: "Transistor-level feasibility studies for mixed-signal systems.",
    summary:
      "Simulation and verification toward formal design reviews on unreleased hardware.",
    metrics: [{ value: "transistor-level", label: "mixed-signal" }],
    tags: ["Mixed-signal", "Analog"],
  },
];

type Edge = { from: number; to: number; career: boolean; curve: number };

const EDGES: Edge[] = [
  // career path (cyan, brighter)
  { from: 4, to: 3, career: true, curve: 0 },
  { from: 3, to: 2, career: true, curve: 0.15 },
  { from: 2, to: 1, career: true, curve: -0.15 },
  { from: 1, to: 0, career: true, curve: 0 },
  // cross-connections (purple, faint)
  { from: 4, to: 2, career: false, curve: -0.15 },
  { from: 2, to: 0, career: false, curve: 0.15 },
  { from: 3, to: 1, career: false, curve: -0.08 },
  { from: 4, to: 0, career: false, curve: 0.06 },
];

/* ── Main component ───────────────────────────────────────────────── */

export default function Experience() {
  const [sel, setSel] = useState<number | null>(null);
  const unhoverTimer = useRef<number | null>(null);
  const { goTo } = useNavigation();

  const handleHover = useCallback((i: number) => {
    if (unhoverTimer.current !== null) {
      window.clearTimeout(unhoverTimer.current);
      unhoverTimer.current = null;
    }
    setSel(i);
  }, []);

  const handleUnhover = useCallback(() => {
    if (unhoverTimer.current !== null) window.clearTimeout(unhoverTimer.current);
    unhoverTimer.current = window.setTimeout(() => {
      setSel(null);
      unhoverTimer.current = null;
    }, 400);
  }, []);

  useEffect(
    () => () => {
      if (unhoverTimer.current !== null) window.clearTimeout(unhoverTimer.current);
    },
    [],
  );

  const toggle = (i: number) => setSel(sel === i ? null : i);

  return (
    <section id="experience" className="relative mx-auto max-w-5xl px-6 py-24">
      {/* Header */}
      <header className="mb-10">
        <Typewriter
          text="Experience"
          as="p"
          speed={60}
          className="text-xs uppercase tracking-[0.25em] text-accent2"
        />
        <Typewriter
          text="A decade of production systems."
          as="h2"
          speed={30}
          delay={600}
          showCursor={false}
          className="mt-2 text-4xl md:text-5xl font-semibold text-gradient"
        />
        <Typewriter
          text="From transistor-level mixed-signal at Apple to reinforcement learning on a live EV fleet — every role has shipped to real hardware or real users."
          as="p"
          speed={18}
          delay={1600}
          showCursor={false}
          className="mt-3 max-w-2xl text-sm text-gray-400"
        />
      </header>

      {/* ── Desktop: 3D career graph ──────────────────────────── */}
      <div className="hidden md:block">
        <div className="relative w-full" style={{ height: 540 }}>
          <ExperienceScene3D
            jobs={JOBS}
            edges={EDGES}
            hovered={sel}
            onHover={handleHover}
            onUnhover={handleUnhover}
            onDemoClick={() => goTo("demos")}
          />
          <AnimatePresence>
            {sel === null && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.5 }}
                exit={{ opacity: 0 }}
                className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] font-mono uppercase tracking-[0.2em] text-gray-600 pointer-events-none z-10"
              >
                hover a node to explore · drag to orbit
              </motion.p>
            )}
          </AnimatePresence>
        </div>

      </div>

      {/* ── Mobile: card list ─────────────────────────────────── */}
      <div className="md:hidden space-y-2">
        {JOBS.map((job, i) => (
          <MobileCard
            key={job.org}
            job={job}
            isOpen={sel === i}
            onToggle={() => toggle(i)}
          />
        ))}
      </div>
    </section>
  );
}

/* ── Mobile card ──────────────────────────────────────────────────── */

function MobileCard({
  job,
  isOpen,
  onToggle,
}: {
  job: Job;
  isOpen: boolean;
  onToggle: () => void;
}) {
  const { goTo } = useNavigation();
  const Icon = job.icon;
  return (
    <div
      className="rounded-xl border bg-white/[0.02] overflow-hidden transition-colors"
      style={{
        borderColor: isOpen ? `${job.tint}55` : "rgba(255,255,255,0.08)",
        borderLeftWidth: 3,
        borderLeftColor: job.tint,
      }}
    >
      <button
        type="button"
        onClick={onToggle}
        className="w-full text-left px-4 py-3 flex items-center gap-3"
      >
        <Icon className="w-4 h-4 shrink-0" style={{ color: job.tint }} />
        <div className="flex-1 min-w-0">
          <p className="text-[13px] font-semibold text-white truncate">
            {job.org}
          </p>
          <p className="text-[11px] text-gray-500">{job.role}</p>
        </div>
        <span className="text-[10px] font-mono text-gray-600 shrink-0">
          {job.period}
        </span>
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
                {job.summary}
              </p>

              <div className="mt-3 flex flex-wrap gap-1.5">
                {job.metrics.map((m) => (
                  <div
                    key={m.label}
                    className="rounded-md border border-white/10 bg-black/30 px-2 py-1"
                  >
                    <span
                      className="text-[12px] font-semibold"
                      style={{ color: job.tint }}
                    >
                      {m.value}
                    </span>
                    <span className="ml-1.5 text-[9px] uppercase tracking-wider text-gray-500">
                      {m.label}
                    </span>
                  </div>
                ))}
              </div>

              <div className="mt-2 flex flex-wrap gap-1">
                {job.tags.map((t) => (
                  <span
                    key={t}
                    className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-white/10 text-gray-500"
                  >
                    {t}
                  </span>
                ))}
              </div>

              {job.demo && (
                <button
                  type="button"
                  onClick={() => goTo("demos")}
                  className="mt-3 inline-flex items-center gap-1 text-[11px] font-medium"
                  style={{ color: job.tint }}
                >
                  {job.demo.label}
                  <ArrowUpRight className="w-3 h-3" />
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

"use client";

/* ------------------------------------------------------------------
   Experience — compact single-expand accordion timeline.

   Each row is a dense one-liner: [dot] period · role · org — headline
   with metric pills and demo link inline. Click to expand in place for
   the longer summary and tag strip. Only one row open at a time; the
   first role is expanded by default.
------------------------------------------------------------------ */

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Rocket,
  Zap,
  Atom,
  Stethoscope,
  Apple,
  ArrowUpRight,
  ChevronDown,
} from "lucide-react";

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

export default function Experience() {
  const [openIdx, setOpenIdx] = useState(0);

  return (
    <section id="experience" className="relative mx-auto max-w-5xl px-6 py-24">
      <header className="mb-10">
        <p className="text-xs uppercase tracking-[0.25em] text-accent2">
          Experience
        </p>
        <h2 className="mt-2 text-4xl md:text-5xl font-semibold text-gradient">
          A decade of production systems.
        </h2>
        <p className="mt-3 max-w-2xl text-sm text-gray-400">
          From transistor-level mixed-signal at Apple to reinforcement learning
          on a live EV fleet — every role has shipped to real hardware or real
          users.
        </p>
      </header>

      <div className="relative">
        {/* timeline spine */}
        <div className="absolute left-[15px] top-3 bottom-3 w-px bg-gradient-to-b from-accent/60 via-white/10 to-transparent" />

        <ol className="space-y-1.5">
          {JOBS.map((j, i) => {
            const isOpen = openIdx === i;
            const Icon = j.icon;
            return (
              <li key={j.org + j.period} className="relative pl-12">
                {/* timeline dot */}
                <div
                  className="absolute left-0 top-3.5 w-[31px] h-[31px] rounded-xl border flex items-center justify-center transition-all"
                  style={{
                    background: isOpen ? `${j.tint}22` : `${j.tint}12`,
                    borderColor: isOpen ? `${j.tint}88` : `${j.tint}44`,
                    boxShadow: isOpen ? `0 0 18px ${j.tint}55` : "none",
                  }}
                >
                  <Icon className="w-3.5 h-3.5" style={{ color: j.tint }} />
                </div>

                <button
                  type="button"
                  onClick={() => setOpenIdx(isOpen ? -1 : i)}
                  className="w-full text-left rounded-xl border border-white/10 hover:border-white/20 bg-white/[0.02] hover:bg-white/[0.035] transition-colors px-4 py-3"
                  style={{
                    borderColor: isOpen
                      ? `${j.tint}55`
                      : "rgba(255,255,255,0.1)",
                    background: isOpen
                      ? `${j.tint}08`
                      : "rgba(255,255,255,0.02)",
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                        <span
                          className="text-[10px] font-mono uppercase tracking-[0.18em]"
                          style={{ color: j.tint }}
                        >
                          {j.period}
                        </span>
                        <span className="text-[11px] text-gray-600">·</span>
                        <span className="text-[13px] font-semibold text-white">
                          {j.role}
                        </span>
                        <span className="text-[12px] text-gray-500">
                          @ <span className="text-gray-300">{j.org}</span>
                        </span>
                      </div>
                      <p className="mt-1 text-[13px] text-gray-400 leading-snug">
                        {j.headline}
                      </p>
                    </div>
                    <ChevronDown
                      className="w-4 h-4 text-gray-500 shrink-0 mt-1 transition-transform"
                      style={{
                        transform: isOpen ? "rotate(180deg)" : "rotate(0deg)",
                      }}
                    />
                  </div>

                  <AnimatePresence initial={false}>
                    {isOpen && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.28, ease: "easeOut" }}
                        className="overflow-hidden"
                      >
                        <div className="pt-4 pb-1">
                          <p className="text-[13px] text-gray-300 leading-relaxed">
                            {j.summary}
                          </p>

                          <div className="mt-3 flex flex-wrap gap-1.5">
                            {j.metrics.map((m) => (
                              <div
                                key={m.label}
                                className="rounded-md border border-white/10 bg-black/30 px-2.5 py-1"
                              >
                                <span
                                  className="text-[12px] font-semibold tabular-nums"
                                  style={{ color: j.tint }}
                                >
                                  {m.value}
                                </span>
                                <span className="ml-1.5 text-[9px] uppercase tracking-wider text-gray-500">
                                  {m.label}
                                </span>
                              </div>
                            ))}
                          </div>

                          <div className="mt-2.5 flex flex-wrap gap-1">
                            {j.tags.map((t) => (
                              <span
                                key={t}
                                className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-white/10 text-gray-500"
                              >
                                {t}
                              </span>
                            ))}
                          </div>

                          <div className="mt-3 flex items-center gap-3 text-[11px] text-gray-500">
                            <span>{j.location}</span>
                            {j.demo && (
                              <>
                                <span className="text-gray-700">·</span>
                                <a
                                  href={j.demo.href}
                                  onClick={(e) => e.stopPropagation()}
                                  className="inline-flex items-center gap-1 font-medium hover:underline"
                                  style={{ color: j.tint }}
                                >
                                  {j.demo.label}
                                  <ArrowUpRight className="w-3 h-3" />
                                </a>
                              </>
                            )}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </button>
              </li>
            );
          })}
        </ol>
      </div>
    </section>
  );
}

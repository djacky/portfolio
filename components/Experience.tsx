"use client";

/* ------------------------------------------------------------------ *
 *  Experience — interactive neural-network node graph (desktop)        *
 *  with a card-list fallback (mobile).                                *
 *                                                                     *
 *  Five roles arranged as a 2-1-2 network (hourglass). Career path    *
 *  highlighted in cyan; cross-connections in faint purple. Click a     *
 *  node to expand its detail panel below the graph.                   *
 * ------------------------------------------------------------------ */

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Rocket,
  Zap,
  Atom,
  Stethoscope,
  Apple,
  ArrowUpRight,
  X,
  ChevronDown,
} from "lucide-react";

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

/* ── Graph layout (% of container) ────────────────────────────────── */

const POS = [
  { x: 22, y: 18 }, // 0 Disruptive Labs — top-left
  { x: 78, y: 18 }, // 1 Eaton           — top-right
  { x: 50, y: 48 }, // 2 CERN            — center
  { x: 22, y: 78 }, // 3 Philips         — bottom-left
  { x: 78, y: 78 }, // 4 Apple           — bottom-right
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

/** Quadratic bezier between two nodes. `c` offsets the control point perpendicular to the midpoint. */
function svgPath(x1: number, y1: number, x2: number, y2: number, c: number) {
  if (c === 0) return `M${x1} ${y1}L${x2} ${y2}`;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  return `M${x1} ${y1}Q${mx + dy * c} ${my - dx * c} ${x2} ${y2}`;
}

/* ── Main component ───────────────────────────────────────────────── */

export default function Experience() {
  const [sel, setSel] = useState<number | null>(null);
  const [hov, setHov] = useState<number | null>(null);
  const detailRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (sel !== null) {
      const t = setTimeout(
        () => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
        120,
      );
      return () => clearTimeout(t);
    }
  }, [sel]);

  const toggle = (i: number) => setSel(sel === i ? null : i);

  return (
    <section id="experience" className="relative mx-auto max-w-5xl px-6 py-24">
      {/* Header */}
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

      {/* ── Desktop: neural-net graph ─────────────────────────── */}
      <div className="hidden md:block">
        <div className="relative w-full" style={{ height: 420 }}>
          {/* SVG edges */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            fill="none"
          >
            {EDGES.map((e, i) => {
              const a = POS[e.from];
              const b = POS[e.to];
              const connected = sel === e.from || sel === e.to;
              const dimmed = sel !== null && !connected;
              const baseAlpha = e.career ? 0.22 : 0.08;
              const alpha = dimmed ? 0.03 : connected ? baseAlpha * 1.6 : baseAlpha;
              const color = e.career
                ? `rgba(34,211,238,${alpha})`
                : `rgba(124,92,255,${alpha})`;
              return (
                <path
                  key={i}
                  d={svgPath(a.x, a.y, b.x, b.y, e.curve)}
                  stroke={color}
                  strokeWidth={e.career ? 1.5 : 0.8}
                  vectorEffect="non-scaling-stroke"
                  strokeDasharray={e.career ? "6 10" : "3 9"}
                  className={e.career ? "edge-career" : "edge-neural"}
                  style={{ transition: "stroke 0.4s" }}
                />
              );
            })}
          </svg>

          {/* Nodes */}
          {JOBS.map((job, i) => {
            const pos = POS[i];
            const Icon = job.icon;
            const isSel = sel === i;
            const isHov = hov === i;
            const isDim = sel !== null && !isSel;
            return (
              <div
                key={job.org}
                className="absolute flex flex-col items-center gap-1.5 z-10"
                style={{
                  left: `${pos.x}%`,
                  top: `${pos.y}%`,
                  transform: "translate(-50%, -28px)",
                  opacity: isDim ? 0.3 : 1,
                  transition: "opacity 0.35s",
                }}
              >
                {/* circle */}
                <button
                  type="button"
                  onClick={() => toggle(i)}
                  onMouseEnter={() => setHov(i)}
                  onMouseLeave={() => setHov(null)}
                  className="relative w-14 h-14 rounded-full border-2 flex items-center justify-center cursor-pointer"
                  style={{
                    background: isSel
                      ? `${job.tint}20`
                      : isHov
                        ? `${job.tint}14`
                        : `${job.tint}0a`,
                    borderColor: isSel
                      ? `${job.tint}aa`
                      : isHov
                        ? `${job.tint}66`
                        : `${job.tint}33`,
                    boxShadow: isSel
                      ? `0 0 32px ${job.tint}55, 0 0 64px ${job.tint}22`
                      : isHov
                        ? `0 0 22px ${job.tint}33`
                        : `0 0 12px ${job.tint}15`,
                    transform: isHov || isSel ? "scale(1.12)" : "scale(1)",
                    transition: "all 0.3s ease",
                  }}
                >
                  <Icon className="w-5 h-5" style={{ color: job.tint }} />
                </button>

                {/* label */}
                <div className="text-center pointer-events-none select-none">
                  <p className="text-[12px] font-semibold text-white leading-tight">
                    {job.org}
                  </p>
                  <p className="text-[10px] font-mono uppercase tracking-wider text-gray-500 mt-0.5">
                    {job.period}
                  </p>
                </div>

                {/* hover: show role */}
                <AnimatePresence>
                  {(isHov || isSel) && (
                    <motion.p
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.18 }}
                      className="text-[10px] text-gray-400 text-center overflow-hidden leading-tight max-w-[160px]"
                    >
                      {job.role}
                    </motion.p>
                  )}
                </AnimatePresence>
              </div>
            );
          })}

          {/* hint */}
          <AnimatePresence>
            {sel === null && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.5 }}
                exit={{ opacity: 0 }}
                className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] font-mono uppercase tracking-[0.2em] text-gray-600"
              >
                click a node to explore
              </motion.p>
            )}
          </AnimatePresence>
        </div>

        {/* detail panel */}
        <AnimatePresence mode="wait">
          {sel !== null && (
            <motion.div
              ref={detailRef}
              key={sel}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.28, ease: "easeOut" }}
            >
              <JobDetail job={JOBS[sel]} onClose={() => setSel(null)} />
            </motion.div>
          )}
        </AnimatePresence>
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

/* ── Detail panel (desktop) ───────────────────────────────────────── */

function JobDetail({ job, onClose }: { job: Job; onClose: () => void }) {
  const Icon = job.icon;
  return (
    <div
      className="mt-4 rounded-2xl border bg-white/[0.02] p-6"
      style={{ borderColor: `${job.tint}33` }}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
            style={{
              background: `${job.tint}18`,
              border: `1px solid ${job.tint}44`,
            }}
          >
            <Icon className="w-[18px] h-[18px]" style={{ color: job.tint }} />
          </div>
          <div>
            <h3 className="text-[15px] font-semibold text-white">
              {job.role}{" "}
              <span className="text-gray-500 font-normal">@ {job.org}</span>
            </h3>
            <p className="text-[11px] font-mono uppercase tracking-wider text-gray-500">
              {job.period} &middot; {job.location}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-gray-500 hover:text-white transition-colors p-1 shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <p className="mt-4 text-[13px] text-gray-300 leading-relaxed">
        {job.summary}
      </p>

      <div className="mt-4 flex flex-wrap gap-2">
        {job.metrics.map((m) => (
          <div
            key={m.label}
            className="rounded-lg border border-white/10 bg-black/30 px-3 py-1.5"
          >
            <span
              className="text-[13px] font-semibold tabular-nums"
              style={{ color: job.tint }}
            >
              {m.value}
            </span>
            <span className="ml-2 text-[9px] uppercase tracking-wider text-gray-500">
              {m.label}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {job.tags.map((t) => (
          <span
            key={t}
            className="text-[9px] uppercase tracking-wider px-2 py-0.5 rounded-md border border-white/10 text-gray-500"
          >
            {t}
          </span>
        ))}
      </div>

      {job.demo && (
        <a
          href={job.demo.href}
          className="mt-4 inline-flex items-center gap-1.5 text-[12px] font-medium hover:underline"
          style={{ color: job.tint }}
        >
          {job.demo.label}
          <ArrowUpRight className="w-3.5 h-3.5" />
        </a>
      )}
    </div>
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
                <a
                  href={job.demo.href}
                  className="mt-3 inline-flex items-center gap-1 text-[11px] font-medium"
                  style={{ color: job.tint }}
                >
                  {job.demo.label}
                  <ArrowUpRight className="w-3 h-3" />
                </a>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

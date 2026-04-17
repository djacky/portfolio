"use client";

import { useState, useEffect, useRef, type ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, MessageSquare, Github, Linkedin, Mail, BookOpen } from "lucide-react";
import dynamic from "next/dynamic";
import NeuralConstellation from "./NeuralConstellation";
import { useContactDrawer } from "./ContactDrawer";
import { useNavigation } from "./SectionRouter";
import RollingTitle from "./RollingTitle";
import type { SiteStats } from "@/lib/siteStats";

// Lazy-load the R3F canvas so it never blocks first paint and the
// heavy three.js bundle doesn't bloat the server render.
const PendulumScene = dynamic(() => import("./PendulumScene"), { ssr: false });

const STATIC_CREDENTIALS: ReactNode[] = [
  "10+ yrs in R&D + production ML",
  "Python · PyTorch · FastAPI · C++ · AWS",
];

const CREDENTIAL_HOLD = 3200;

function formatInt(n: number): string {
  return n.toLocaleString("en-US");
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function buildCredentials(stats: SiteStats | null): ReactNode[] {
  if (!stats) return STATIC_CREDENTIALS;
  const trainLine = stats.bestTime
    ? `${formatInt(stats.trainings)} pendulums trained globally · best time ${formatTime(stats.bestTime)}`
    : `${formatInt(stats.trainings)} pendulums trained globally`;
  const live: ReactNode[] = [
    `${stats.publications} publications · ${formatInt(stats.citations)} citations`,
    trainLine,
    <>
      reading:{" "}
      <a
        href={stats.reading.link}
        target="_blank"
        rel="noopener noreferrer"
        className="text-accent2 hover:text-white underline-offset-2 hover:underline transition-colors pointer-events-auto"
      >
        {stats.reading.title}
      </a>
    </>,
  ];
  return [...STATIC_CREDENTIALS, ...live];
}

function FadingCredentials() {
  const [stats, setStats] = useState<SiteStats | null>(null);
  const [idx, setIdx] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* Fetch once on mount; failures fall back silently to the static list. */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/stats");
        if (!res.ok) return;
        const data = (await res.json()) as SiteStats;
        if (!cancelled) setStats(data);
      } catch {
        /* silent */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  /* Listen for live training-complete events from PendulumScene and
     patch the trainings count in place so the rotating credential
     reflects the new global total without a page refresh. */
  useEffect(() => {
    const onTrained = (e: Event) => {
      const detail = (e as CustomEvent<{ trainings?: number; bestTime?: number | null }>).detail;
      if (!detail || typeof detail.trainings !== "number") return;
      setStats((prev) =>
        prev ? { ...prev, trainings: detail.trainings!, bestTime: detail.bestTime ?? prev.bestTime } : prev,
      );
    };
    window.addEventListener("pendulum-training-complete", onTrained);
    return () =>
      window.removeEventListener("pendulum-training-complete", onTrained);
  }, []);

  const credentials = buildCredentials(stats);

  useEffect(() => {
    timerRef.current = setInterval(() => {
      setIdx((prev) => (prev + 1) % credentials.length);
    }, CREDENTIAL_HOLD);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [credentials.length]);

  const safeIdx = idx % credentials.length;

  return (
    <AnimatePresence mode="wait">
      <motion.span
        key={safeIdx}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.45, ease: "easeInOut" }}
        className="block text-sm font-mono uppercase tracking-[0.18em] text-gray-400"
      >
        <span className="text-accent2">◆</span> {credentials[safeIdx]}
      </motion.span>
    </AnimatePresence>
  );
}

export default function Hero() {
  const { open: openContact } = useContactDrawer();
  const { goTo } = useNavigation();
  return (
    <section
      id="top"
      className="relative h-screen flex items-center overflow-hidden"
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
          className="mt-5 text-4xl sm:text-5xl md:text-6xl font-semibold leading-[1.05] tracking-tight"
        >
          <span className="text-gradient">Achille Nicoletti</span>
          <br />
          <RollingTitle className="text-gray-300" />
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
          <button
            type="button"
            onClick={() => goTo("demos")}
            className="group inline-flex items-center gap-2 rounded-full bg-accent px-5 py-3 text-sm font-medium text-white shadow-glow hover:bg-accent/90 transition-all"
          >
            Try live demos
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
          <button
            type="button"
            onClick={openContact}
            className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-black/30 backdrop-blur px-5 py-3 text-sm text-gray-200 hover:bg-white/10 transition-colors"
          >
            <MessageSquare className="w-4 h-4" /> Let&apos;s talk
          </button>
        </motion.div>

        {/* social links */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mt-6 flex items-center gap-5 pointer-events-auto"
        >
          <a href="https://github.com/djacky" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-gray-500 hover:text-white transition-colors">
            <Github className="w-5 h-5" />
          </a>
          <a href="https://www.linkedin.com/in/achillen/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className="text-gray-500 hover:text-white transition-colors">
            <Linkedin className="w-5 h-5" />
          </a>
          <a href="mailto:globalminimum@protonmail.com" aria-label="Email" className="text-gray-500 hover:text-white transition-colors">
            <Mail className="w-5 h-5" />
          </a>
          <a href="https://scholar.google.com/citations?user=Fes_eScAAAAJ&hl=en" target="_blank" rel="noopener noreferrer" aria-label="Google Scholar" className="text-gray-500 hover:text-white transition-colors">
            <BookOpen className="w-5 h-5" />
          </a>
        </motion.div>

        {/* rotating credential */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-12 h-7"
        >
          <FadingCredentials />
        </motion.div>
      </div>
    </section>
  );
}

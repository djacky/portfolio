"use client";

/* ------------------------------------------------------------------
   DemoSwitcher — tabbed container for the three live demos.

   Only the active demo is mounted at any time. This keeps the page
   light (the EV sim runs a training loop, the CERN demo mounts a
   Three.js canvas) and gives deep links: #demo-ev, #demo-matchmaker,
   #demo-cern.
------------------------------------------------------------------ */

import { useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Users, Atom, Activity, Network, CircuitBoard } from "lucide-react";
import EVFleetDemo from "./EVFleetDemo";
import MatchmakerDemo from "./MatchmakerDemo";
import CERNDemo from "./CERNDemo";
import MagnetILCDemo from "./MagnetILCDemo";

// AWS + MPC are the heaviest demos — each pulls a 3D scene (transitively
// three.js / R3F). Defer their bundles until the user actually opens that
// tab so the initial section paint stays cheap.
const AWSDemo = dynamic(() => import("./AWSDemo"), { ssr: false });
const MPCDemo = dynamic(() => import("./MPCDemo"), { ssr: false });

type TabKey = "ev" | "matchmaker" | "cern" | "ilc" | "aws" | "mpc";

const TABS: {
  key: TabKey;
  hash: string;
  label: string;
  sub: string;
  icon: typeof Zap;
  tint: string;
  context: string;
}[] = [
  {
    key: "ev",
    hash: "demo-ev",
    label: "EV Fleet Load Balancer",
    sub: "Eaton · live RL training",
    icon: Zap,
    tint: "#22d3ee",
    context:
      "How do you split limited power across dozens of electric vehicles charging at once — without blowing the grid? This demo trains a reinforcement learning agent in your browser to make that decision in real time, the same way the production system does for Eaton's global charger fleet.",
  },
  {
    key: "matchmaker",
    hash: "demo-matchmaker",
    label: "Siamese Matchmaker",
    sub: "Disruptive Labs · embedding retrieval",
    icon: Users,
    tint: "#7c5cff",
    context:
      "Fair matchmaking means grouping players of similar skill — fast. This demo trains a neural network to map each player into a 2D \"skill fingerprint,\" then clusters similar players into lobbies. Watch the embeddings reorganize in real time as the model learns from match outcomes.",
  },
  {
    key: "cern",
    hash: "demo-cern",
    label: "CERN Controller Synthesis",
    sub: "LHC · H∞ pipeline",
    icon: Atom,
    tint: "#34d399",
    context:
      "The LHC's magnets need controllers tuned to sub-ppm precision — but each power converter behaves slightly differently. This demo recreates the automated pipeline I built at CERN: feed in a frequency response, pick your performance specs, and get back a ready-to-deploy controller.",
  },
  {
    key: "ilc",
    hash: "demo-ilc",
    label: "Magnet Current Ramp",
    sub: "CERN · data-driven ILC",
    icon: Activity,
    tint: "#fbbf24",
    context:
      "Particle physics experiments need magnet currents that follow precise ramp profiles, but real hardware drifts. This demo shows iterative learning control — the algorithm watches each ramp, learns from the tracking error, and converges on a correction signal that nails the reference.",
  },
  {
    key: "aws",
    hash: "demo-aws",
    label: "Live Match Backend",
    sub: "Disruptive Labs · 30-player sim",
    icon: Network,
    tint: "#fb923c",
    context:
      "Thirty players, one prize pool, fifteen minutes of real AWS plumbing. Lobby traffic — joins, bets, validation — hits FastAPI on EC2 and lands in Postgres. Once the match starts, player telemetry moves off the API tier entirely: every event buffers on an AWS GameLift session fleet. At match end, GameLift flushes the whole session to EC2 as a single bulk transfer, which validates and persists the dataset, then wakes a cold Lambda to score the field, split the pool, write the ledger to S3, and update every balance. Place a bet, inject a bad payload, or end the match early to drive it yourself.",
  },
  {
    key: "mpc",
    hash: "demo-mpc",
    label: "Boost Converter MPC",
    sub: "Eaton · real-time QP",
    icon: CircuitBoard,
    tint: "#f472b6",
    context:
      "An inductor-current loop for a grid-tied power converter, tracking a sinusoidal reference against hard current rails. The gold arc is the controller's predicted future — re-solved every millisecond as a convex QP in your browser, condensed-form MPC with Hildreth's dual solver. Watch the arc deflect away from the red rails as the state approaches them; that deflection is the core visual argument for MPC over PI. Drop a load step, push the reference 42% higher, or inject measurement noise — then toggle to the PI baseline and watch it clip through the rail.",
  },
];

export default function DemoSwitcher() {
  const [active, setActive] = useState<TabKey>("ev");

  const selectTab = (k: TabKey) => {
    setActive(k);
  };

  return (
    <div>
      {/* TAB BAR — orb buttons, max 3 per row */}
      <div
        role="tablist"
        aria-label="Live demos"
        className="flex flex-wrap justify-center gap-5 sm:gap-6 max-w-[34rem] mx-auto mb-10"
      >
        {TABS.map((tab) => {
          const isActive = active === tab.key;
          const Icon = tab.icon;
          return (
            <button
              key={tab.key}
              role="tab"
              aria-selected={isActive}
              aria-controls={`panel-${tab.key}`}
              onClick={() => selectTab(tab.key)}
              className="relative group w-[8.5rem] h-[8.5rem] sm:w-36 sm:h-36 rounded-full transition-transform hover:scale-[1.04] active:scale-[0.98] focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40"
            >
              {/* orb surface */}
              <div
                className="absolute inset-0 rounded-full transition-all duration-300"
                style={{
                  background: isActive
                    ? `radial-gradient(circle at 50% 22%, ${tab.tint}38, ${tab.tint}10 55%, rgba(11,15,26,0.75) 100%)`
                    : `radial-gradient(circle at 50% 22%, rgba(255,255,255,0.07), rgba(255,255,255,0.01) 60%), rgba(11,15,26,0.55)`,
                  border: `1px solid ${isActive ? tab.tint + "80" : "rgba(255,255,255,0.09)"}`,
                  boxShadow: isActive
                    ? `0 0 28px ${tab.tint}45, inset 0 1px 0 rgba(255,255,255,0.10), inset 0 -14px 28px rgba(0,0,0,0.35)`
                    : `inset 0 1px 0 rgba(255,255,255,0.05), inset 0 -10px 20px rgba(0,0,0,0.30)`,
                }}
              />

              {/* outer halo follows the active orb */}
              {isActive && (
                <motion.div
                  layoutId="demo-orb-halo"
                  className="absolute -inset-[3px] rounded-full pointer-events-none"
                  style={{
                    border: `1px solid ${tab.tint}55`,
                    boxShadow: `0 0 18px ${tab.tint}55`,
                  }}
                  transition={{ type: "spring", stiffness: 380, damping: 30 }}
                />
              )}

              {/* content — icon + label + sub, fully contained */}
              <div className="relative flex flex-col items-center justify-center h-full px-5 text-center gap-1.5">
                <Icon
                  className="w-5 h-5 shrink-0 transition-colors"
                  style={{ color: isActive ? tab.tint : "#9ca3af" }}
                />
                <div
                  className="text-[11px] sm:text-[11.5px] font-medium leading-[1.15] transition-colors"
                  style={{ color: isActive ? "#fff" : "#d1d5db" }}
                >
                  {tab.label}
                </div>
                <div
                  className="text-[8.5px] font-mono uppercase tracking-[0.12em] text-gray-500 leading-[1.1] line-clamp-1 max-w-full"
                  title={tab.sub}
                >
                  {tab.sub.split(" · ")[0]}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* CONTEXT — accessible description of what the active demo does */}
      <AnimatePresence mode="wait">
        <motion.p
          key={`ctx-${active}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.25 }}
          className="mx-auto max-w-3xl mb-8 text-sm text-gray-400 leading-relaxed text-center"
        >
          {TABS.find((t) => t.key === active)?.context}
        </motion.p>
      </AnimatePresence>

      {/* PANEL — only the active demo is mounted */}
      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          id={`panel-${active}`}
          role="tabpanel"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.3 }}
        >
          {active === "ev" && (
            <div id="demo-ev">
              <EVFleetDemo />
            </div>
          )}
          {active === "matchmaker" && (
            <div id="demo-matchmaker">
              <MatchmakerDemo />
            </div>
          )}
          {active === "cern" && (
            <div id="demo-cern">
              <CERNDemo />
            </div>
          )}
          {active === "ilc" && (
            <div id="demo-ilc">
              <MagnetILCDemo />
            </div>
          )}
          {active === "aws" && (
            <div id="demo-aws">
              <AWSDemo />
            </div>
          )}
          {active === "mpc" && (
            <div id="demo-mpc">
              <MPCDemo />
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

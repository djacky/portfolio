"use client";

/* ------------------------------------------------------------------
   DemoSwitcher — tabbed container for the three live demos.

   Only the active demo is mounted at any time. This keeps the page
   light (the EV sim runs a training loop, the CERN demo mounts a
   Three.js canvas) and gives deep links: #demo-ev, #demo-matchmaker,
   #demo-cern.
------------------------------------------------------------------ */

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Users, Atom, Activity } from "lucide-react";
import EVFleetDemo from "./EVFleetDemo";
import MatchmakerDemo from "./MatchmakerDemo";
import CERNDemo from "./CERNDemo";
import MagnetILCDemo from "./MagnetILCDemo";

type TabKey = "ev" | "matchmaker" | "cern" | "ilc";

const TABS: {
  key: TabKey;
  hash: string;
  label: string;
  sub: string;
  icon: typeof Zap;
  tint: string;
}[] = [
  {
    key: "ev",
    hash: "demo-ev",
    label: "EV Fleet Load Balancer",
    sub: "Eaton · live RL training",
    icon: Zap,
    tint: "#22d3ee",
  },
  {
    key: "matchmaker",
    hash: "demo-matchmaker",
    label: "Siamese Matchmaker",
    sub: "Disruptive Labs · embedding retrieval",
    icon: Users,
    tint: "#7c5cff",
  },
  {
    key: "cern",
    hash: "demo-cern",
    label: "CERN Controller Synthesis",
    sub: "LHC · H∞ pipeline",
    icon: Atom,
    tint: "#34d399",
  },
  {
    key: "ilc",
    hash: "demo-ilc",
    label: "Magnet Current Ramp",
    sub: "CERN · data-driven ILC",
    icon: Activity,
    tint: "#fbbf24",
  },
];

function hashToTab(hash: string): TabKey | null {
  const clean = hash.replace(/^#/, "");
  const found = TABS.find((t) => t.hash === clean);
  return found?.key ?? null;
}

export default function DemoSwitcher() {
  const [active, setActive] = useState<TabKey>("ev");

  // Deep-link support: if the URL lands on #demo-* or it changes while
  // the user is here, switch the active tab.
  useEffect(() => {
    const sync = () => {
      const k = hashToTab(window.location.hash);
      if (k) setActive(k);
    };
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  const selectTab = (k: TabKey) => {
    setActive(k);
    const tab = TABS.find((t) => t.key === k);
    if (tab && typeof window !== "undefined") {
      // update the hash without a scroll jump
      history.replaceState(null, "", `#${tab.hash}`);
    }
  };

  return (
    <div>
      {/* TAB BAR */}
      <div
        role="tablist"
        aria-label="Live demos"
        className="mx-auto max-w-5xl mb-10"
      >
        <div className="glass rounded-2xl p-2 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 relative">
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
                className="relative group rounded-xl px-4 py-3 text-left transition-colors"
              >
                {isActive && (
                  <motion.div
                    layoutId="demo-tab-pill"
                    className="absolute inset-0 rounded-xl"
                    style={{
                      background: `linear-gradient(135deg, ${tab.tint}22, ${tab.tint}08)`,
                      border: `1px solid ${tab.tint}55`,
                      boxShadow: `0 0 24px ${tab.tint}25, inset 0 0 18px ${tab.tint}14`,
                    }}
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                  />
                )}
                <div className="relative flex items-center gap-3">
                  <div
                    className="flex items-center justify-center w-9 h-9 rounded-lg shrink-0 transition-colors"
                    style={{
                      background: isActive ? `${tab.tint}20` : "rgba(255,255,255,0.04)",
                      color: isActive ? tab.tint : "#94a3b8",
                      border: `1px solid ${isActive ? tab.tint + "55" : "rgba(255,255,255,0.08)"}`,
                    }}
                  >
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <div
                      className="text-sm font-medium truncate transition-colors"
                      style={{ color: isActive ? "#fff" : "#d1d5db" }}
                    >
                      {tab.label}
                    </div>
                    <div className="text-[10px] font-mono uppercase tracking-wider text-gray-500 truncate">
                      {tab.sub}
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

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
            <div id="demo-ev" className="scroll-mt-24">
              <EVFleetDemo />
            </div>
          )}
          {active === "matchmaker" && (
            <div id="demo-matchmaker" className="scroll-mt-24">
              <MatchmakerDemo />
            </div>
          )}
          {active === "cern" && (
            <div id="demo-cern" className="scroll-mt-24">
              <CERNDemo />
            </div>
          )}
          {active === "ilc" && (
            <div id="demo-ilc" className="scroll-mt-24">
              <MagnetILCDemo />
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

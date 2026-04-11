"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { AnimatePresence } from "framer-motion";
import Hero from "./Hero";
import About from "./About";
import Experience from "./Experience";
import DemoSwitcher from "./DemoSwitcher";
import Skills from "./Skills";
import Publications from "./Publications";
import RecruiterMatch from "./RecruiterMatch";
import TickerNav from "./TickerNav";
import SectionPage from "./SectionPage";
import SectionTransition from "./SectionTransition";
import IntroTransition from "./IntroTransition";
import Typewriter from "./Typewriter";

/* ───────── types ───────── */
export type SectionId =
  | "hero"
  | "about"
  | "experience"
  | "demos"
  | "skills"
  | "publications"
  | "match";

interface NavigationContextValue {
  activeSection: SectionId;
  goTo: (target: SectionId) => void;
  introComplete: boolean;
}

const NavigationCtx = createContext<NavigationContextValue | null>(null);

export function useNavigation() {
  const ctx = useContext(NavigationCtx);
  if (!ctx) throw new Error("useNavigation must be used within SectionRouter");
  return ctx;
}

/* ───────── helpers ───────── */
const VALID_SECTIONS: SectionId[] = [
  "hero",
  "about",
  "experience",
  "demos",
  "skills",
  "publications",
  "match",
];

function readInitialSection(): SectionId {
  if (typeof window === "undefined") return "hero";
  const params = new URLSearchParams(window.location.search);
  const s = params.get("s") as SectionId | null;
  if (s && VALID_SECTIONS.includes(s)) return s;
  return "hero";
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/* ───────── demos section wrapper ───────── */
function DemosPage() {
  return (
    <section className="relative mx-auto max-w-6xl px-6 py-24">
      <header className="mb-12 text-center">
        <Typewriter
          text="Live Demos"
          as="p"
          speed={60}
          className="text-xs uppercase tracking-[0.25em] text-accent2"
        />
        <Typewriter
          text="Don't take my word for it"
          as="h2"
          speed={30}
          delay={600}
          showCursor={false}
          className="mt-2 text-4xl font-semibold text-gradient"
        />
        <Typewriter
          text="These aren't screenshots — they're the actual algorithms, running live in your browser. Train a policy, tweak a controller, watch embeddings converge."
          as="p"
          speed={18}
          delay={1400}
          showCursor={false}
          className="mt-3 text-gray-400 max-w-2xl mx-auto"
        />
      </header>
      <DemoSwitcher />
    </section>
  );
}

/* ───────── router ───────── */
export default function SectionRouter() {
  const [activeSection, setActiveSection] = useState<SectionId>("hero");
  const [introComplete, setIntroComplete] = useState(false);
  const [transitioning, setTransitioning] = useState(false);

  /* read ?s= param and sessionStorage intro flag on mount */
  useEffect(() => {
    const initial = readInitialSection();
    if (initial !== "hero") {
      setActiveSection(initial);
      setIntroComplete(true); // skip intro when deep-linking
    } else if (sessionStorage.getItem("intro-played")) {
      setIntroComplete(true);
    }
  }, []);

  const goTo = useCallback(
    async (target: SectionId) => {
      if (target === activeSection || transitioning) return;
      setTransitioning(true);
      // overlay covers the swap
      await sleep(200);
      setActiveSection(target);
      // update URL for shareability
      const url = target === "hero" ? window.location.pathname : `?s=${target}`;
      history.replaceState(null, "", url);
      // hold until blackout has covered the full mode="wait" exit+enter
      await sleep(500);
      setTransitioning(false);
    },
    [activeSection, transitioning],
  );

  const handleIntroComplete = useCallback(() => {
    setIntroComplete(true);
    sessionStorage.setItem("intro-played", "1");
  }, []);

  return (
    <NavigationCtx.Provider value={{ activeSection, goTo, introComplete }}>
      {/* intro overlay */}
      {!introComplete && <IntroTransition onComplete={handleIntroComplete} />}

      {/* ticker nav — visible once intro is done */}
      {introComplete && <TickerNav />}

      {/* section transition overlay */}
      <AnimatePresence>{transitioning && <SectionTransition />}</AnimatePresence>

      {/* active section */}
      <AnimatePresence mode="wait">
        {activeSection === "hero" && <Hero key="hero" />}
        {activeSection === "about" && (
          <SectionPage key="about" id="about">
            <About />
          </SectionPage>
        )}
        {activeSection === "experience" && (
          <SectionPage key="experience" id="experience">
            <Experience />
          </SectionPage>
        )}
        {activeSection === "demos" && (
          <SectionPage key="demos" id="demos">
            <DemosPage />
          </SectionPage>
        )}
        {activeSection === "skills" && (
          <SectionPage key="skills" id="skills">
            <Skills />
          </SectionPage>
        )}
        {activeSection === "publications" && (
          <SectionPage key="publications" id="publications">
            <Publications />
          </SectionPage>
        )}
        {activeSection === "match" && (
          <SectionPage key="match" id="match">
            <RecruiterMatch />
          </SectionPage>
        )}
      </AnimatePresence>
    </NavigationCtx.Provider>
  );
}

"use client";

import { useNavigation, type SectionId } from "./SectionRouter";
import { useContactDrawer } from "./ContactDrawer";
import { Home } from "lucide-react";
import { motion } from "framer-motion";

const SECTIONS: { id: SectionId; label: string }[] = [
  { id: "about", label: "about" },
  { id: "experience", label: "experience" },
  { id: "demos", label: "demos" },
  { id: "publications", label: "research" },
  { id: "skills", label: "stack" },
  { id: "match", label: "match" },
];

function TickerSet({
  activeSection,
  goTo,
  openContact,
}: {
  activeSection: SectionId;
  goTo: (id: SectionId) => void;
  openContact: () => void;
}) {
  return (
    <>
      {SECTIONS.map((s) => (
        <span key={s.id} className="inline-flex items-center shrink-0">
          <span className="text-accent2/30 select-none px-12" aria-hidden>·</span>
          <button
            type="button"
            onClick={() => goTo(s.id)}
            className={`transition-all whitespace-nowrap font-bold ${
              activeSection === s.id
                ? "text-accent2 glow-accent2-text"
                : "text-accent2/60 hover:text-accent2"
            }`}
          >
            {s.label}
          </button>
        </span>
      ))}
      <span className="inline-flex items-center shrink-0">
        <span className="text-accent2/30 select-none px-12" aria-hidden>·</span>
        <button
          type="button"
          onClick={openContact}
          className="transition-all whitespace-nowrap font-bold text-accent2/60 hover:text-accent2"
        >
          contact
        </button>
      </span>
    </>
  );
}

export default function TickerNav() {
  const { activeSection, goTo } = useNavigation();
  const { open: openContact } = useContactDrawer();

  return (
    <motion.nav
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="fixed top-5 left-0 right-0 z-50 h-10 flex items-center"
    >
      <div className="shrink-0 flex items-center gap-2 px-4">
        <button
          type="button"
          onClick={() => goTo("hero")}
          aria-label="Home"
          title="Home"
          className={`group relative flex items-center justify-center w-9 h-9 rounded-md border transition-all duration-200 ${
            activeSection === "hero"
              ? "border-accent2/60 bg-accent2/10 text-accent2 glow-accent2"
              : "border-accent2/25 bg-panel/40 text-accent2/70 hover:text-accent2 hover:border-accent2/60 hover:bg-accent2/10 hover:shadow-[0_0_14px_rgba(34,211,238,0.35)]"
          }`}
        >
          <Home className="w-[18px] h-[18px] transition-transform duration-200 group-hover:scale-110" />
          <span
            aria-hidden
            className="pointer-events-none absolute inset-0 rounded-md opacity-0 group-hover:opacity-100 transition-opacity duration-200"
            style={{
              background:
                "linear-gradient(120deg, rgba(124,92,255,0.18), rgba(34,211,238,0.18))",
              mixBlendMode: "screen",
            }}
          />
        </button>
      </div>

      <div
        className="flex-1 overflow-hidden relative group"
        style={{ maskImage: "linear-gradient(90deg, transparent, black 60px, black calc(100% - 60px), transparent)" }}
      >
        <div
          className="flex items-center font-mono text-[17px] uppercase tracking-[0.35em] animate-ticker-scroll group-hover:[animation-play-state:paused] whitespace-nowrap w-max"
        >
          <TickerSet activeSection={activeSection} goTo={goTo} openContact={openContact} />
          <TickerSet activeSection={activeSection} goTo={goTo} openContact={openContact} />
          <TickerSet activeSection={activeSection} goTo={goTo} openContact={openContact} />
        </div>
      </div>
    </motion.nav>
  );
}

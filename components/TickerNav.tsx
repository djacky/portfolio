"use client";

import { useNavigation, type SectionId } from "./SectionRouter";
import { useContactDrawer } from "./ContactDrawer";
import { Home, MessageSquare } from "lucide-react";
import { motion } from "framer-motion";

const SECTIONS: { id: SectionId; label: string }[] = [
  { id: "about", label: "about" },
  { id: "experience", label: "experience" },
  { id: "demos", label: "demos" },
  { id: "publications", label: "research" },
  { id: "skills", label: "stack" },
  { id: "match", label: "match" },
];

function TickerLabels({
  activeSection,
  goTo,
}: {
  activeSection: SectionId;
  goTo: (id: SectionId) => void;
}) {
  return (
    <>
      {SECTIONS.map((s, i) => (
        <span key={s.id} className="inline-flex items-center gap-8 shrink-0">
          {i > 0 && (
            <span className="text-accent2/30 select-none" aria-hidden>
              ·
            </span>
          )}
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

      {/* right side: contact + home */}
      <div className="shrink-0 flex items-center gap-2 px-3">
        <button
          type="button"
          onClick={() => goTo("hero")}
          aria-label="Home"
          className="flex items-center justify-center w-6 h-6 rounded text-gray-500 hover:text-accent2 transition-colors"
        >
          <Home className="w-4 h-4" />
        </button>
      </div>

      {/* ticker area */}
      <div
        className="flex-1 overflow-hidden relative group"
        style={{ maskImage: "linear-gradient(90deg, transparent, black 60px, black calc(100% - 60px), transparent)" }}
      >
        <div
          className="flex items-center gap-8 font-mono text-[17px] uppercase tracking-[0.35em] animate-ticker-scroll group-hover:[animation-play-state:paused] whitespace-nowrap w-max"
        >
          {/* triple the labels for infinite scroll illusion */}
          <TickerLabels activeSection={activeSection} goTo={goTo} />
          <span className="text-accent2/30 select-none px-8" aria-hidden>
            ·
          </span>
          <TickerLabels activeSection={activeSection} goTo={goTo} />
          <span className="text-accent2/30 select-none px-8" aria-hidden>
            ·
          </span>
          <TickerLabels activeSection={activeSection} goTo={goTo} />
        </div>
      </div>

      <div className="shrink-0 flex items-center gap-2 px-3">
        <button
          type="button"
          onClick={openContact}
          className="flex items-center gap-1.5 text-[10px] font-mono uppercase tracking-[0.2em] text-gray-500 hover:text-accent2 transition-colors"
        >
          <MessageSquare className="w-3 h-3" />
          <span className="hidden sm:inline">Contact</span>
        </button>
      </div>
    </motion.nav>
  );
}

"use client";

/* ------------------------------------------------------------------
   Nav — floating glass pill capsule with an animated active-section
   indicator driven by IntersectionObserver.

   Pattern is lifted from Linear / Vercel / Raycast / most recent
   Awwwards portfolio winners: a centered pill rather than a full-bleed
   bar, wordmark floating independently top-left, mobile collapses to
   a hamburger → full-screen glass overlay.
------------------------------------------------------------------ */

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X } from "lucide-react";
import { useContactDrawer } from "./ContactDrawer";

type Link = { href: string; label: string };

const LINKS: Link[] = [
  { href: "#match", label: "Match" },
  { href: "#about", label: "About" },
  { href: "#experience", label: "Experience" },
  { href: "#demos", label: "Demos" },
  { href: "#skills", label: "Stack" },
  { href: "#contact", label: "Contact" },
];

// Section IDs we want to track for the active indicator.
const TRACKED_IDS = LINKS.map((l) => l.href.replace(/^#/, ""));

export default function Nav() {
  const [active, setActive] = useState<string>("match");
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const { open: openContact } = useContactDrawer();

  // Scroll state for wordmark/pill subtle background blur ramp.
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Active-section tracking via IntersectionObserver.
  useEffect(() => {
    const sections = TRACKED_IDS.map((id) => document.getElementById(id)).filter(
      (el): el is HTMLElement => el != null,
    );
    if (sections.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        // Pick the most visible intersecting section.
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]) setActive(visible[0].target.id);
      },
      {
        // Trigger when a section's center-ish is in the viewport.
        rootMargin: "-40% 0px -55% 0px",
        threshold: [0, 0.25, 0.5, 0.75, 1],
      },
    );
    sections.forEach((s) => observer.observe(s));
    return () => observer.disconnect();
  }, []);

  // Lock body scroll while the mobile overlay is open.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  const go = (href: string) => {
    setOpen(false);
    // Contact link opens the drawer instead of scrolling.
    if (href === "#contact") {
      openContact();
      return;
    }
    setTimeout(() => {
      const id = href.replace(/^#/, "");
      const el = document.getElementById(id);
      el?.scrollIntoView({ behavior: "smooth", block: "start" });
      history.replaceState(null, "", href);
    }, 10);
  };

  return (
    <>
      {/* floating wordmark — top-left, independent of the pill */}
      <a
        href="#top"
        className={`fixed top-5 left-5 md:left-8 z-50 font-mono text-sm tracking-tight transition-opacity ${
          scrolled ? "opacity-100" : "opacity-90"
        }`}
      >
        <span className="text-accent">◆</span> nicoletti
        <span className="text-gray-500">.ai</span>
      </a>

      {/* desktop pill — centered */}
      <nav className="fixed top-4 left-1/2 -translate-x-1/2 z-50 hidden md:block">
        <div
          className="relative flex items-center gap-1 rounded-full border border-white/10 bg-black/40 backdrop-blur-xl px-2 py-1.5"
          style={{
            boxShadow:
              "0 0 0 1px rgba(255,255,255,0.02), 0 20px 40px -20px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06)",
          }}
        >
          {LINKS.map((l) => {
            const id = l.href.replace(/^#/, "");
            const isActive = active === id;
            return (
              <a
                key={l.href}
                href={l.href}
                onClick={(e) => {
                  e.preventDefault();
                  go(l.href);
                }}
                className="relative px-4 py-1.5 text-[13px] font-medium rounded-full transition-colors"
                style={{
                  color: isActive ? "#fff" : "rgba(209,213,219,0.75)",
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.color = "#fff";
                }}
                onMouseLeave={(e) => {
                  if (!isActive)
                    e.currentTarget.style.color = "rgba(209,213,219,0.75)";
                }}
              >
                {isActive && (
                  <motion.span
                    layoutId="nav-active-pill"
                    className="absolute inset-0 rounded-full -z-10"
                    style={{
                      background:
                        "linear-gradient(135deg, rgba(124,92,255,0.22), rgba(34,211,238,0.18))",
                      border: "1px solid rgba(124,92,255,0.35)",
                      boxShadow:
                        "0 0 18px rgba(124,92,255,0.25), inset 0 0 14px rgba(34,211,238,0.08)",
                    }}
                    transition={{ type: "spring", stiffness: 420, damping: 34 }}
                  />
                )}
                <span className="relative">{l.label}</span>
              </a>
            );
          })}
        </div>
      </nav>

      {/* mobile hamburger */}
      <button
        type="button"
        aria-label="Open menu"
        aria-expanded={open}
        onClick={() => setOpen(true)}
        className="fixed top-4 right-4 z-50 md:hidden h-10 w-10 rounded-full border border-white/10 bg-black/40 backdrop-blur-xl flex items-center justify-center text-gray-200 hover:text-white hover:border-white/20 transition-colors"
      >
        <Menu className="w-4 h-4" />
      </button>

      {/* mobile full-screen overlay */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="fixed inset-0 z-[60] md:hidden bg-black/80 backdrop-blur-xl"
          >
            <button
              type="button"
              aria-label="Close menu"
              onClick={() => setOpen(false)}
              className="absolute top-4 right-4 h-10 w-10 rounded-full border border-white/10 bg-black/40 flex items-center justify-center text-gray-200"
            >
              <X className="w-4 h-4" />
            </button>
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.05, duration: 0.35 }}
              className="h-full flex flex-col items-center justify-center gap-6"
            >
              {LINKS.map((l, i) => {
                const id = l.href.replace(/^#/, "");
                const isActive = active === id;
                return (
                  <motion.a
                    key={l.href}
                    href={l.href}
                    onClick={(e) => {
                      e.preventDefault();
                      go(l.href);
                    }}
                    initial={{ y: 14, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.08 + i * 0.05 }}
                    className="text-3xl font-semibold transition-colors"
                    style={{
                      color: isActive ? "#fff" : "rgba(209,213,219,0.7)",
                    }}
                  >
                    {l.label}
                  </motion.a>
                );
              })}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

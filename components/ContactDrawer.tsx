"use client";

/* ------------------------------------------------------------------
   ContactDrawer — slide-out right-anchored drawer with a contact form.

   Exposes a React context (`ContactDrawerProvider` + `useContactDrawer`)
   so any component on the page can open it:

     const { open } = useContactDrawer();
     <button onClick={open}>Let's talk</button>

   Behavior:
     - Slides in from the right with a spring.
     - Backdrop blur fades independently.
     - ESC closes, click-outside closes, focus trap.
     - Body scroll lock while open.
     - Submits to POST /api/contact (Resend-backed).
------------------------------------------------------------------ */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Send,
  Check,
  AlertTriangle,
  Mail,
  Briefcase,
  Users,
  Lightbulb,
  MessageSquare,
} from "lucide-react";

/* ---------------- context ---------------- */

type Ctx = { isOpen: boolean; open: () => void; close: () => void };
const ContactDrawerCtx = createContext<Ctx | null>(null);

export function useContactDrawer() {
  const ctx = useContext(ContactDrawerCtx);
  if (!ctx) throw new Error("useContactDrawer must be used within ContactDrawerProvider");
  return ctx;
}

export function ContactDrawerProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  return (
    <ContactDrawerCtx.Provider value={{ isOpen, open, close }}>
      {children}
      <ContactDrawer />
    </ContactDrawerCtx.Provider>
  );
}

/* ---------------- topic options ---------------- */

const TOPICS = [
  { key: "role", label: "Role opportunity", icon: Briefcase },
  { key: "collaboration", label: "Collaboration", icon: Users },
  { key: "consulting", label: "Consulting", icon: Lightbulb },
  { key: "general", label: "General", icon: MessageSquare },
] as const;
type TopicKey = (typeof TOPICS)[number]["key"];

/* ---------------- drawer ---------------- */

type Status = "idle" | "sending" | "success" | "error";

function ContactDrawer() {
  const { isOpen, close } = useContactDrawer();
  const panelRef = useRef<HTMLDivElement>(null);
  const firstFieldRef = useRef<HTMLInputElement>(null);

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [topic, setTopic] = useState<TopicKey>("role");
  const [message, setMessage] = useState("");
  const [honeypot, setHoneypot] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  // Body scroll lock while open.
  useEffect(() => {
    if (!isOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [isOpen]);

  // ESC to close + initial focus.
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    const t = window.setTimeout(() => firstFieldRef.current?.focus(), 300);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.clearTimeout(t);
    };
  }, [isOpen, close]);

  // Reset the form when closing after success (so the next open starts fresh).
  useEffect(() => {
    if (isOpen) return;
    if (status === "success") {
      const t = window.setTimeout(() => {
        setName("");
        setEmail("");
        setTopic("role");
        setMessage("");
        setStatus("idle");
        setError(null);
      }, 400);
      return () => window.clearTimeout(t);
    }
  }, [isOpen, status]);

  const charCount = message.length;
  const tooShort = charCount > 0 && charCount < 20;
  const tooLong = charCount > 4000;
  const canSubmit =
    name.trim().length > 0 &&
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim()) &&
    charCount >= 20 &&
    !tooLong &&
    status !== "sending";

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setStatus("sending");
    setError(null);
    try {
      const res = await fetch("/api/contact", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name, email, topic, message, honeypot }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data?.error ?? "Something went wrong. Please try again.");
        setStatus("error");
        return;
      }
      setStatus("success");
    } catch {
      setError("Network error. Please try again.");
      setStatus("error");
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* backdrop */}
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            onClick={close}
            className="fixed inset-0 z-[70] bg-black/70 backdrop-blur-md"
          />

          {/* panel */}
          <motion.aside
            key="panel"
            ref={panelRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="contact-drawer-title"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 240, damping: 32 }}
            className="fixed top-0 right-0 bottom-0 z-[80] w-full sm:w-[480px] bg-bg/95 border-l border-white/10 overflow-y-auto"
            style={{
              boxShadow: "-30px 0 80px -10px rgba(0,0,0,0.8)",
              backdropFilter: "blur(20px)",
            }}
          >
            {/* close button */}
            <button
              type="button"
              aria-label="Close contact form"
              onClick={close}
              className="absolute top-5 right-5 h-9 w-9 rounded-full border border-white/10 bg-black/40 hover:border-white/25 hover:bg-black/60 flex items-center justify-center text-gray-300 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="p-8 md:p-10">
              <p className="text-xs uppercase tracking-[0.25em] text-accent2">
                Contact
              </p>
              <h2
                id="contact-drawer-title"
                className="mt-2 text-3xl md:text-4xl font-semibold text-gradient leading-tight"
              >
                Let&apos;s work together.
              </h2>
              <p className="mt-3 text-sm text-gray-400 leading-relaxed">
                Tell me what you're building. I read every message and reply
                within a day or two.
              </p>

              {/* success state */}
              <AnimatePresence mode="wait">
                {status === "success" ? (
                  <motion.div
                    key="success"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mt-10 rounded-2xl border border-good/40 bg-good/5 p-6 text-center"
                  >
                    <div className="mx-auto w-12 h-12 rounded-full bg-good/15 border border-good/40 flex items-center justify-center mb-3">
                      <Check className="w-5 h-5 text-good" />
                    </div>
                    <div className="text-sm text-gray-100 font-medium">
                      Message sent.
                    </div>
                    <div className="mt-1.5 text-xs text-gray-400 leading-relaxed">
                      Thanks, {name || "there"} — I'll get back to you at{" "}
                      <span className="text-gray-200">{email}</span> soon.
                    </div>
                    <button
                      type="button"
                      onClick={close}
                      className="mt-5 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/[0.04] hover:bg-white/[0.08] px-5 py-2 text-xs text-gray-200 transition-colors"
                    >
                      Close
                    </button>
                  </motion.div>
                ) : (
                  <motion.form
                    key="form"
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    onSubmit={submit}
                    className="mt-8 space-y-5"
                  >
                    {/* honeypot */}
                    <input
                      type="text"
                      name="website"
                      value={honeypot}
                      onChange={(e) => setHoneypot(e.target.value)}
                      tabIndex={-1}
                      autoComplete="off"
                      className="absolute left-[-9999px] top-[-9999px]"
                      aria-hidden="true"
                    />

                    {/* name */}
                    <div>
                      <label
                        htmlFor="c-name"
                        className="text-[10px] uppercase tracking-[0.2em] font-mono text-gray-500"
                      >
                        Your name
                      </label>
                      <input
                        id="c-name"
                        ref={firstFieldRef}
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="Jane Recruiter"
                        maxLength={120}
                        className="mt-1.5 w-full rounded-xl bg-black/40 border border-white/10 hover:border-white/20 focus:border-accent2/60 focus:ring-2 focus:ring-accent2/20 focus:outline-none px-4 py-3 text-sm text-gray-100 transition-all"
                      />
                    </div>

                    {/* email */}
                    <div>
                      <label
                        htmlFor="c-email"
                        className="text-[10px] uppercase tracking-[0.2em] font-mono text-gray-500"
                      >
                        Email
                      </label>
                      <input
                        id="c-email"
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="jane@company.com"
                        maxLength={200}
                        className="mt-1.5 w-full rounded-xl bg-black/40 border border-white/10 hover:border-white/20 focus:border-accent2/60 focus:ring-2 focus:ring-accent2/20 focus:outline-none px-4 py-3 text-sm text-gray-100 transition-all"
                      />
                    </div>

                    {/* topic */}
                    <div>
                      <div className="text-[10px] uppercase tracking-[0.2em] font-mono text-gray-500 mb-2">
                        What's this about?
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {TOPICS.map((t) => {
                          const Icon = t.icon;
                          const isActive = topic === t.key;
                          return (
                            <button
                              key={t.key}
                              type="button"
                              onClick={() => setTopic(t.key)}
                              className="relative rounded-xl border px-3 py-2.5 text-left transition-all"
                              style={{
                                borderColor: isActive
                                  ? "rgba(34,211,238,0.5)"
                                  : "rgba(255,255,255,0.1)",
                                background: isActive
                                  ? "rgba(34,211,238,0.08)"
                                  : "rgba(0,0,0,0.25)",
                                boxShadow: isActive
                                  ? "0 0 18px rgba(34,211,238,0.15), inset 0 0 12px rgba(34,211,238,0.05)"
                                  : undefined,
                              }}
                            >
                              <Icon
                                className="w-3.5 h-3.5 mb-1"
                                style={{
                                  color: isActive ? "#22d3ee" : "#6b7280",
                                }}
                              />
                              <div
                                className="text-[12px] font-medium"
                                style={{ color: isActive ? "#fff" : "#d1d5db" }}
                              >
                                {t.label}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    {/* message */}
                    <div>
                      <label
                        htmlFor="c-msg"
                        className="flex items-center justify-between text-[10px] uppercase tracking-[0.2em] font-mono text-gray-500"
                      >
                        <span>Message</span>
                        <span className="tabular-nums normal-case tracking-normal text-gray-600">
                          {charCount} / 4000
                        </span>
                      </label>
                      <textarea
                        id="c-msg"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        placeholder="A few sentences on the role, team, or idea you have in mind."
                        rows={6}
                        maxLength={4000}
                        className="mt-1.5 w-full rounded-xl bg-black/40 border border-white/10 hover:border-white/20 focus:border-accent2/60 focus:ring-2 focus:ring-accent2/20 focus:outline-none px-4 py-3 text-sm text-gray-100 leading-relaxed font-mono resize-y transition-all"
                      />
                    </div>

                    {/* error */}
                    {(status === "error" || tooShort || tooLong) && (
                      <div className="flex items-start gap-2 text-xs text-bad">
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                        <span>
                          {error ??
                            (tooShort
                              ? "Message is a bit short — at least 20 characters."
                              : "Message is too long — 4000 characters max.")}
                        </span>
                      </div>
                    )}

                    {/* submit */}
                    <div className="pt-1">
                      <motion.button
                        type="submit"
                        disabled={!canSubmit}
                        whileTap={{ scale: 0.97 }}
                        className="w-full inline-flex items-center justify-center gap-2 rounded-full border px-6 py-3.5 text-sm font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                        style={{
                          background: "rgba(34,211,238,0.15)",
                          borderColor: "rgba(34,211,238,0.5)",
                          color: "#22d3ee",
                          boxShadow:
                            "0 0 28px rgba(34,211,238,0.18), inset 0 0 18px rgba(34,211,238,0.08)",
                        }}
                      >
                        {status === "sending" ? (
                          <>
                            <motion.span
                              animate={{ rotate: 360 }}
                              transition={{
                                duration: 0.9,
                                repeat: Infinity,
                                ease: "linear",
                              }}
                              className="inline-block w-4 h-4 rounded-full border-2 border-accent2/30 border-t-accent2"
                            />
                            Sending…
                          </>
                        ) : (
                          <>
                            <Send className="w-4 h-4" />
                            Send message
                          </>
                        )}
                      </motion.button>
                    </div>

                    <p className="text-[10px] font-mono text-gray-600 text-center leading-relaxed">
                      Sent via Resend · rate-limited to 5/hour per IP · not
                      stored.
                    </p>
                  </motion.form>
                )}
              </AnimatePresence>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

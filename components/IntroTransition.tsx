"use client";

import { useEffect, useRef, useState, useCallback } from "react";

interface IntroTransitionProps {
  onComplete: () => void;
}

/* ── helpers ─────────────────────────────────────────── */
function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

/* ── line / bar data ─────────────────────────────────── */
interface LineData {
  type: "line";
  cls: string;
  html: string;
  typewriter?: { speed: number };
}
interface BarData {
  type: "bar";
  label: string;
  duration: number;
}
interface BadgeData {
  type: "badge";
}
interface HintData {
  type: "hint";
}
type Entry = LineData | BarData | BadgeData | HintData;

/* ── ProgressBar component ───────────────────────────── */
function ProgressBar({
  label,
  duration,
  onDone,
}: {
  label: string;
  duration: number;
  onDone?: () => void;
}) {
  const fillRef = useRef<HTMLDivElement>(null);
  const pctRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    let cancelled = false;
    const steps = 60;
    const step = duration / steps;
    let i = 0;

    const interval = setInterval(() => {
      if (cancelled) return;
      i++;
      const p = Math.round((i / steps) * 100);
      if (fillRef.current) fillRef.current.style.width = `${p}%`;
      if (pctRef.current) pctRef.current.textContent = `${p}%`;
      if (i >= steps) {
        clearInterval(interval);
        onDone?.();
      }
    }, step);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [duration, onDone]);

  return (
    <div className="flex items-center gap-2.5 text-xs animate-fadeIn">
      <span
        className="min-w-[148px] whitespace-nowrap"
        style={{ color: "rgba(0,255,65,0.38)" }}
      >
        {label}
      </span>
      <div
        className="flex-1 h-1.5 rounded-sm overflow-hidden relative"
        style={{
          background: "#0e1a0e",
          border: "1px solid #1a3a1a",
        }}
      >
        <div
          ref={fillRef}
          className="h-full rounded-sm relative"
          style={{
            width: "0%",
            background: "linear-gradient(90deg, #005515, #00ff41)",
            transition: "width 0.06s linear",
          }}
        >
          <span
            className="absolute right-0 top-0 bottom-0 w-3"
            style={{
              background: "rgba(255,255,255,0.25)",
              filter: "blur(3px)",
            }}
          />
        </div>
      </div>
      <span
        ref={pctRef}
        className="min-w-[36px] text-right text-xs"
        style={{ color: "#00ff41" }}
      >
        0%
      </span>
    </div>
  );
}

/* ── Cursor component ────────────────────────────────── */
function Cursor() {
  return (
    <span
      className="inline-block w-2 h-3.5 align-middle ml-0.5"
      style={{
        background: "#00ff41",
        animation: "blink 1s step-end infinite",
      }}
    />
  );
}

/* ── main component ──────────────────────────────────── */
export default function IntroTransition({ onComplete }: IntroTransitionProps) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [fadingOut, setFadingOut] = useState(false);
  const [waitingForKey, setWaitingForKey] = useState(false);
  const [showCursor, setShowCursor] = useState(true);
  const bodyRef = useRef<HTMLDivElement>(null);
  const doneRef = useRef(false);
  const waitRef = useRef(false);

  const scrollBottom = useCallback(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, []);

  const addEntry = useCallback(
    (entry: Entry) => {
      setEntries((prev) => [...prev, entry]);
      setTimeout(scrollBottom, 20);
    },
    [scrollBottom],
  );

  /* ── run the terminal sequence ─────────────────────── */
  useEffect(() => {
    let cancelled = false;

    async function run() {
      await sleep(400);
      if (cancelled) return;

      // prompt
      addEntry({
        type: "line",
        cls: "line",
        html: '<span style="color:#00ff41">➜ nicoletti.dev</span> <span style="color:#2a4a2a">~/portfolio</span>',
      });
      await sleep(350);
      if (cancelled) return;

      // npm install command
      addEntry({
        type: "line",
        cls: "line",
        html: '<span style="color:#00ff41">$</span> <span style="color:#e2e8f0">npm install nicoletti</span>',
      });
      await sleep(500);
      if (cancelled) return;

      addEntry({ type: "line", cls: "dim", html: "" });
      addEntry({
        type: "line",
        cls: "info",
        html: 'npm <span style="color:#00ff41">notice</span>  created a lockfile as package-lock.json',
      });
      await sleep(180);
      if (cancelled) return;
      addEntry({
        type: "line",
        cls: "out",
        html: 'npm <span style="color:#ffb700">warn</span>   genius-level code detected — proceeding anyway',
      });
      await sleep(280);
      if (cancelled) return;
      addEntry({ type: "line", cls: "out", html: "" });

      // resolving
      addEntry({
        type: "line",
        cls: "out",
        html: '<span style="color:#00e5ff">⬢</span>  resolving dependency tree...',
      });
      await sleep(400);
      if (cancelled) return;

      // packages
      const pkgs = [
        ["nicoletti-core         ", 220],
        ["creative-problem-solver", 180],
        ["fullstack-wizardry      ", 260],
        ["coffee-addiction        ", 80],
        ["ml-superpowers          ", 320],
      ] as const;

      for (const [name, dur] of pkgs) {
        addEntry({
          type: "line",
          cls: "dim",
          html: `   <span style="color:#2a6a3a">+</span> <span style="color:#4a8a5a">${name}</span>`,
        });
        await sleep(dur * 0.35);
        if (cancelled) return;
      }

      await sleep(300);
      if (cancelled) return;
      addEntry({ type: "line", cls: "out", html: "" });
      addEntry({
        type: "line",
        cls: "out",
        html: '<span style="color:#00e5ff">⬢</span>  installing modules...',
      });
      await sleep(200);
      if (cancelled) return;

      // progress bars
      const bars = [
        ["core modules    ", 900],
        ["project assets  ", 1200],
        ["ML streams      ", 1600],
        ["demo builds     ", 1100],
        ["final bundle    ", 700],
      ] as const;

      for (const [label, dur] of bars) {
        addEntry({ type: "bar", label, duration: dur });
        await sleep(120);
        if (cancelled) return;
      }

      // wait for the longest bar
      await sleep(1700);
      if (cancelled) return;

      addEntry({ type: "line", cls: "out", html: "" });
      await sleep(300);
      if (cancelled) return;

      // output logs
      const logs: [string, string][] = [
        [
          "success",
          '✔  projects loaded          <span style="color:#2a4a2a">——————————</span>  <span style="color:#00ff41">12 projects</span>',
        ],
        [
          "success",
          '✔  generating block code     <span style="color:#2a4a2a">——————————</span>  <span style="color:#00ff41">ok</span>',
        ],
        [
          "success",
          '✔  loading demos             <span style="color:#2a4a2a">——————————</span>  <span style="color:#00ff41">8 demos</span>',
        ],
        [
          "warn",
          '⚠  generating ML streams     <span style="color:#2a4a2a">——————————</span>  <span style="color:#ffb700">high memory</span>',
        ],
        [
          "success",
          '✔  compiling personality     <span style="color:#2a4a2a">——————————</span>  <span style="color:#00ff41">charismatic</span>',
        ],
        [
          "success",
          '✔  indexing open source      <span style="color:#2a4a2a">——————————</span>  <span style="color:#00ff41">47 repos</span>',
        ],
        [
          "success",
          '✔  all streams loaded        <span style="color:#2a4a2a">——————————</span>  <span style="color:#00ff41">200 OK</span>',
        ],
      ];

      for (const [cls, text] of logs) {
        addEntry({ type: "line", cls, html: text });
        await sleep(160);
        if (cancelled) return;
      }

      await sleep(300);
      if (cancelled) return;
      addEntry({ type: "line", cls: "out", html: "" });

      // final summary box
      const boxLines = [
        "┌──────────────────────────────────────────────────┐",
        "│                                                  │",
        '│   <span style="color:#fff;font-weight:700">nicoletti@dev</span>  v1.0.0  ready                    │',
        "│   added 5 packages · 0 vulnerabilities           │",
        '│   boot time  <span style="color:#00e5ff">124ms</span>  ·  bundle  <span style="color:#00e5ff">2.4 kB</span>            │',
        "│                                                  │",
        "└──────────────────────────────────────────────────┘",
      ];
      for (const line of boxLines) {
        addEntry({ type: "line", cls: "success", html: line });
      }
      await sleep(400);
      if (cancelled) return;

      addEntry({ type: "line", cls: "out", html: "" });

      // show "press any key"
      addEntry({ type: "hint" });
      setWaitingForKey(true);
      waitRef.current = true;
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [addEntry]);

  /* ── handle keypress / click to exit ───────────────── */
  const handleExit = useCallback(async () => {
    if (doneRef.current || !waitRef.current) return;
    doneRef.current = true;
    setWaitingForKey(false);

    // type exit command
    addEntry({ type: "line", cls: "info", html: "" });
    addEntry({
      type: "line",
      cls: "line",
      html: '<span style="color:#00ff41">$</span> <span style="color:#e2e8f0">./enter --force</span>',
    });
    await sleep(200);
    addEntry({
      type: "line",
      cls: "success",
      html: "launching environment...",
    });
    setShowCursor(false);
    await sleep(500);

    // fade out
    setFadingOut(true);
    await sleep(900);
    onComplete();
  }, [addEntry, onComplete]);

  useEffect(() => {
    if (!waitingForKey) return;

    function onKey(e: KeyboardEvent) {
      // Ignore modifier-only keys
      if (["Shift", "Control", "Alt", "Meta"].includes(e.key)) return;
      handleExit();
    }
    function onClick() {
      handleExit();
    }

    document.addEventListener("keydown", onKey);
    document.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("click", onClick);
    };
  }, [waitingForKey, handleExit]);

  /* ── color map for line classes ────────────────────── */
  function lineColor(cls: string): string {
    if (cls.includes("success")) return "#00ff41";
    if (cls.includes("warn")) return "#ffb700";
    if (cls.includes("error")) return "#ff3b3b";
    if (cls.includes("info")) return "#00e5ff";
    if (cls.includes("dim")) return "#2a4a2a";
    return "rgba(0,255,65,0.38)";
  }

  return (
    <>
      {/* keyframe for cursor blink + fadeIn */}
      <style jsx global>{`
        @keyframes blink {
          50% {
            opacity: 0;
          }
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.15s ease-out forwards;
        }
        @keyframes pulse-hint {
          0%,
          100% {
            opacity: 0.4;
          }
          50% {
            opacity: 1;
          }
        }
      `}</style>

      <div
        className="fixed inset-0 z-[100] flex items-center justify-center"
        style={{
          background: "#000",
          transition: "opacity 0.9s ease",
          opacity: fadingOut ? 0 : 1,
          pointerEvents: fadingOut ? "none" : "auto",
        }}
      >
        {/* terminal window */}
        <div
          className="overflow-hidden"
          style={{
            width: "min(780px, 94vw)",
            border: "1px solid #005515",
            borderRadius: 8,
            boxShadow:
              "0 0 0 1px rgba(0,255,65,0.06), 0 0 60px rgba(0,255,65,0.07), 0 30px 80px rgba(0,0,0,0.8)",
          }}
        >
          {/* title bar */}
          <div
            className="flex items-center gap-2.5"
            style={{
              background: "#0d0d0d",
              borderBottom: "1px solid #1a1a1a",
              padding: "10px 16px",
            }}
          >
            <div className="flex gap-[7px]">
              <div
                className="w-3 h-3 rounded-full"
                style={{ background: "#ff5f57" }}
              />
              <div
                className="w-3 h-3 rounded-full"
                style={{ background: "#ffbd2e" }}
              />
              <div
                className="w-3 h-3 rounded-full"
                style={{ background: "#28c840" }}
              />
            </div>
            <div
              className="flex-1 text-center text-xs"
              style={{
                color: "rgba(255,255,255,0.25)",
                letterSpacing: "0.08em",
              }}
            >
              nicoletti@dev — zsh — 78×32
            </div>
          </div>

          {/* terminal body */}
          <div
            ref={bodyRef}
            className="relative overflow-y-auto"
            style={{
              background: "#060606",
              padding: "22px 24px 28px",
              minHeight: 380,
              maxHeight: "70vh",
            }}
          >
            {/* scan lines */}
            <div
              className="absolute inset-0 pointer-events-none"
              style={{
                background:
                  "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px)",
                zIndex: 1,
              }}
            />

            {/* content */}
            <div className="relative z-[2]">
              {entries.map((entry, i) => {
                if (entry.type === "line") {
                  return (
                    <div
                      key={i}
                      className="animate-fadeIn"
                      style={{
                        fontSize: 13,
                        lineHeight: 1.75,
                        whiteSpace: "pre",
                        color: lineColor(entry.cls),
                        minHeight: "1.75em",
                      }}
                      dangerouslySetInnerHTML={{ __html: entry.html }}
                    />
                  );
                }

                if (entry.type === "bar") {
                  return (
                    <ProgressBar
                      key={i}
                      label={entry.label}
                      duration={entry.duration}
                    />
                  );
                }

                if (entry.type === "hint") {
                  return (
                    <div
                      key={i}
                      className="text-center text-xs uppercase mt-4"
                      style={{
                        letterSpacing: "0.18em",
                        color: "rgba(0,255,65,0.38)",
                        animation: "pulse-hint 2s ease infinite",
                      }}
                    >
                      press any key to enter
                    </div>
                  );
                }

                return null;
              })}

              {/* blinking cursor at the end */}
              {showCursor && entries.length > 0 && <Cursor />}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

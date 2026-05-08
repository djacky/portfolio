"use client";

/* ------------------------------------------------------------------
   Recruiter-mode JD matcher.

   Self-contained component. To remove the feature entirely:
     1. Delete this file.
     2. Delete app/api/match/route.ts (and the empty parent dirs).
     3. Delete lib/candidate-dossier.ts, lib/match-schema.ts, lib/partial-json.ts.
     4. Remove the <RecruiterMatch /> import + element from app/page.tsx.
     5. Remove the "#match" link from components/Nav.tsx.

   How streaming works:
     - The /api/match route forwards Anthropic's SSE stream verbatim.
     - We read it with the fetch streams API + TextDecoder.
     - We extract every `input_json_delta` payload, append it to a
       buffer, and run tryParsePartialJSON after each delta.
     - Whenever the parser succeeds, we update React state with the
       (possibly incomplete) MatchAnalysis. The UI renders whichever
       fields are present and skeletons the rest. Sections fade in
       through AnimatePresence as they first become available.
------------------------------------------------------------------ */

import { useState, useRef, useEffect, Component, type ReactNode } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles,
  ArrowRight,
  RotateCcw,
  Check,
  Minus,
  X,
  AlertTriangle,
  ShieldCheck,
  Quote,
} from "lucide-react";
import type {
  MatchAnalysis,
  FitBand,
  ReqStatus,
  RubricRow,
  RequirementRow,
} from "@/lib/match-schema";
import Typewriter from "./Typewriter";

/* ---------------- sample JDs (chip row) ---------------- */

const SAMPLE_JDS: { label: string; jd: string }[] = [
  {
    label: "Staff ML Engineer",
    jd: `Staff Machine Learning Engineer — Applied Research

We are hiring a staff-level ML engineer to lead applied research on
reinforcement learning for autonomous control systems. You will own the
end-to-end pipeline: data ingestion, model training in PyTorch, deployment
of inference services on AWS, and integration with embedded targets.

Requirements:
- 7+ years of production Python
- Strong experience with PyTorch (training and inference)
- Reinforcement learning (PPO, SAC, or similar policy-gradient methods)
- Backend experience with FastAPI or similar async frameworks
- AWS (Lambda, EC2, S3) deployment experience
- Comfort with hardware-in-the-loop validation
- PhD or equivalent research experience preferred

Nice to have:
- C++ for embedded targets
- Convex optimization background
- Experience with CI/CD and Docker
- Prior work on power systems, robotics, or control`,
  },
  {
    label: "Applied Scientist",
    jd: `Applied Scientist — Foundation Models for Robotics

We're looking for an Applied Scientist to join our robotics foundation
model team. You'll train and fine-tune large multimodal models (vision +
proprioception + language) on robotic manipulation data and ship them to
real hardware.

Requirements:
- PhD in ML, robotics, control, or a related field
- Deep PyTorch experience including distributed training
- Track record publishing at NeurIPS / ICML / CoRL or shipping production ML
- Experience with imitation learning, RL, or behavior cloning
- Comfort with C++ and real-time systems
- Strong software engineering fundamentals (testing, CI, code review)

Nice to have:
- Experience deploying models on resource-constrained hardware
- Familiarity with MuJoCo, Isaac Gym, or similar simulators
- Background in classical control or system identification`,
  },
  {
    label: "ML Infrastructure",
    jd: `Senior ML Infrastructure Engineer

We are building the platform that lets our ML researchers ship models to
production without needing to be infra experts. You will own the training
job orchestration layer, the model serving stack, and the observability
on top of both.

Requirements:
- 5+ years of backend engineering at scale
- Deep Python and at least one other systems language (Go, Rust, C++)
- Experience operating Kubernetes in production
- Familiarity with model serving frameworks (TorchServe, Triton, KServe)
- Solid grasp of distributed systems and async I/O
- Experience with observability tooling (Grafana, Prometheus, OpenTelemetry)

Nice to have:
- Prior ML research or applied ML background
- Terraform / Pulumi for infra-as-code
- Experience with on-call rotations and incident response`,
  },
];

const MODEL_LABEL = "Claude Sonnet";

/* ---------------- color & status meta ---------------- */

const BAND_META: Record<FitBand, { label: string; color: string; bg: string; border: string }> = {
  strong: { label: "Strong fit", color: "#34d399", bg: "bg-good/10", border: "border-good/40" },
  partial: { label: "Partial fit", color: "#fbbf24", bg: "bg-warn/10", border: "border-warn/40" },
  weak: { label: "Weak fit", color: "#f87171", bg: "bg-bad/10", border: "border-bad/40" },
};

const STATUS_META: Record<ReqStatus, { color: string; icon: typeof Check; label: string }> = {
  met: { color: "#34d399", icon: Check, label: "Met" },
  partial: { color: "#fbbf24", icon: Minus, label: "Partial" },
  gap: { color: "#f87171", icon: X, label: "Gap" },
};

/* ---------------- ETA history (localStorage) ----------------
   We don't get progress events from the Anthropic non-streaming API,
   so the next-best UX is a learned ETA: time every successful call,
   keep the last 10 in localStorage, and show the median as the
   user-facing estimate while a request is in flight. */

const ETA_KEY = "match-eta-history-v1";
const ETA_MAX_SAMPLES = 10;
const ETA_DEFAULT_MS = 9000;

function loadEtaHistory(): number[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(ETA_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((n) => typeof n === "number" && n > 0) : [];
  } catch {
    return [];
  }
}
function saveEtaSample(ms: number) {
  if (typeof window === "undefined") return;
  try {
    const hist = loadEtaHistory();
    hist.push(ms);
    while (hist.length > ETA_MAX_SAMPLES) hist.shift();
    window.localStorage.setItem(ETA_KEY, JSON.stringify(hist));
  } catch {
    /* ignore quota errors */
  }
}
function estimateMs(history: number[]): number {
  if (history.length === 0) return ETA_DEFAULT_MS;
  const sorted = [...history].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function computeWeightedAverage(rubric: RubricRow[]): number {
  const tot = rubric.reduce((s, r) => s + r.weight, 0);
  if (tot === 0) return 0;
  return Math.round(rubric.reduce((s, r) => s + r.weight * r.score, 0) / tot);
}

/* score-band helpers */
function bandColorForScore(score: number): string {
  if (score >= 75) return "#34d399";
  if (score >= 50) return "#fbbf24";
  return "#f87171";
}
function bandForScore(score: number): FitBand {
  if (score >= 75) return "strong";
  if (score >= 50) return "partial";
  return "weak";
}

/* ============================================================
   MAIN COMPONENT
============================================================ */

type UIState = "idle" | "loading" | "done" | "error";

export default function RecruiterMatch() {
  const [jd, setJd] = useState("");
  const [state, setState] = useState<UIState>("idle");
  const [analysis, setAnalysis] = useState<MatchAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestStartedAt, setRequestStartedAt] = useState<number | null>(null);
  const [estimatedMs, setEstimatedMs] = useState<number>(ETA_DEFAULT_MS);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const submit = async () => {
    if (jd.trim().length < 80) {
      setError("Please paste a job description (at least 80 characters).");
      setState("error");
      return;
    }
    setState("loading");
    setError(null);
    setAnalysis(null);
    setEstimatedMs(estimateMs(loadEtaHistory()));
    const startedAt = Date.now();
    setRequestStartedAt(startedAt);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/match", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ jd }),
        signal: controller.signal,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data?.error ?? "Something went wrong. Please try again.");
        setState("error");
        return;
      }
      const a = data?.analysis as MatchAnalysis | undefined;
      if (
        !a ||
        typeof a !== "object" ||
        !Array.isArray(a.rubric) ||
        !Array.isArray(a.requirements) ||
        typeof a.summary !== "string"
      ) {
        console.error("[RecruiterMatch] malformed analysis:", data);
        setError("The analysis came back malformed. Please try again.");
        setState("error");
        return;
      }
      // backfill optional arrays so downstream rendering never crashes
      a.gaps = Array.isArray(a.gaps) ? a.gaps : [];
      a.talking_points = Array.isArray(a.talking_points) ? a.talking_points : [];
      saveEtaSample(Date.now() - startedAt);
      setAnalysis(a);
      setState("done");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError("Network error. Please try again.");
      setState("error");
    }
  };

  const reset = () => {
    abortRef.current?.abort();
    setState("idle");
    setAnalysis(null);
    setError(null);
    setJd("");
    setTimeout(() => taRef.current?.focus(), 100);
  };

  const fillSample = (jdText: string) => {
    setJd(jdText);
    setTimeout(() => taRef.current?.focus(), 0);
  };

  return (
    <section id="match" className="relative mx-auto max-w-6xl px-6 py-24">
      <header className="mb-14 text-center">
        <Typewriter
          text={`Recruiter mode · powered by ${MODEL_LABEL}`}
          as="p"
          speed={40}
          className="text-xs uppercase tracking-[0.25em] text-accent2"
        />
        <Typewriter
          text="Does this role fit?"
          as="h2"
          speed={30}
          delay={600}
          showCursor={false}
          className="mt-2 text-4xl md:text-5xl font-semibold text-gradient"
        />
        <Typewriter
          text="Paste a job description. An LLM compares it against my full experience and returns a segmented fit score, grounded evidence, and an honest list of gaps."
          as="p"
          speed={18}
          delay={1200}
          showCursor={false}
          className="mt-3 text-gray-400 max-w-2xl mx-auto"
        />
      </header>

      <div className="shimmer-border rounded-3xl">
        <div className="glass rounded-3xl p-6 md:p-8 relative overflow-hidden">
          <AnimatePresence mode="wait">
            {(state === "idle" || state === "error") && (
              <motion.div
                key="input"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.35 }}
              >
                <InputPanel
                  jd={jd}
                  setJd={setJd}
                  onSubmit={submit}
                  onSample={fillSample}
                  taRef={taRef}
                  error={state === "error" ? error : null}
                />
              </motion.div>
            )}

            {state === "loading" && (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <LoadingPanel
                  startedAt={requestStartedAt ?? Date.now()}
                  estimatedMs={estimatedMs}
                  hasHistory={loadEtaHistory().length > 0}
                />
              </motion.div>
            )}

            {state === "done" && analysis && (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.45 }}
              >
                <ResultsErrorBoundary onReset={reset}>
                  <ResultsPanel analysis={analysis} onReset={reset} />
                </ResultsErrorBoundary>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <p className="mt-4 text-center text-[11px] font-mono text-gray-600">
        Analysis generated by {MODEL_LABEL}. Verify against the rest of the site.
      </p>
    </section>
  );
}

/* ============================================================
   INPUT PANEL
============================================================ */

function InputPanel({
  jd,
  setJd,
  onSubmit,
  onSample,
  taRef,
  error,
}: {
  jd: string;
  setJd: (s: string) => void;
  onSubmit: () => void;
  onSample: (s: string) => void;
  taRef: React.RefObject<HTMLTextAreaElement>;
  error: string | null;
}) {
  const charCount = jd.length;
  const minChars = 80;
  const maxChars = 8000;
  const tooShort = charCount > 0 && charCount < minChars;
  const tooLong = charCount > maxChars;

  return (
    <div>
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <label
          htmlFor="jd-input"
          className="text-[10px] uppercase tracking-[0.2em] text-accent2"
        >
          Job description
        </label>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-gray-600 mr-1">
            try:
          </span>
          {SAMPLE_JDS.map((s) => (
            <button
              key={s.label}
              type="button"
              onClick={() => onSample(s.jd)}
              className="text-[11px] font-mono px-2.5 py-1 rounded-full border border-white/10 bg-white/[0.03] text-gray-400 hover:text-accent2 hover:border-accent2/40 transition-colors"
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative">
        <textarea
          id="jd-input"
          ref={taRef}
          value={jd}
          onChange={(e) => setJd(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onSubmit();
          }}
          placeholder="Paste a job description here. The more detail, the better the match.&#10;&#10;Tip: ⌘/Ctrl + Enter to analyze."
          className="w-full min-h-[260px] rounded-2xl bg-black/40 border border-white/10 hover:border-white/20 focus:border-accent2/60 focus:ring-2 focus:ring-accent2/20 focus:outline-none p-5 text-sm text-gray-100 leading-relaxed font-mono resize-y transition-all"
          style={{ boxShadow: "inset 0 0 30px rgba(34,211,238,0.04)" }}
        />
        <div className="absolute bottom-3 right-4 text-[10px] font-mono text-gray-600 tabular-nums pointer-events-none">
          {charCount.toLocaleString()} / {maxChars.toLocaleString()}
        </div>
      </div>

      {(tooShort || tooLong || error) && (
        <div className="mt-3 flex items-center gap-2 text-xs text-bad">
          <AlertTriangle className="w-3.5 h-3.5" />
          <span>
            {error ??
              (tooShort
                ? `Need at least ${minChars} characters.`
                : `Too long — please trim to under ${maxChars.toLocaleString()} characters.`)}
          </span>
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-[11px] font-mono text-gray-500">
          <ShieldCheck className="w-3.5 h-3.5 text-good" />
          Sent to the analysis service. Not stored.
        </div>
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={onSubmit}
          disabled={charCount < minChars || tooLong}
          className="inline-flex items-center gap-2 rounded-full bg-accent2/15 border border-accent2/50 hover:bg-accent2/25 hover:border-accent2 disabled:opacity-40 disabled:cursor-not-allowed transition-all px-6 py-3 text-sm font-semibold text-accent2"
          style={{
            boxShadow:
              "0 0 28px rgba(34,211,238,0.18), inset 0 0 18px rgba(34,211,238,0.08)",
          }}
        >
          <Sparkles className="w-4 h-4" />
          Analyze fit
          <ArrowRight className="w-4 h-4" />
        </motion.button>
      </div>
    </div>
  );
}

/* ============================================================
   LOADING PANEL — premium throbber that occupies the same
   visual slot the segmented score ring will occupy when results
   arrive, so the transition feels continuous instead of swapping
   between two unrelated layouts.
============================================================ */

function LoadingPanel({
  startedAt,
  estimatedMs,
  hasHistory,
}: {
  startedAt: number;
  estimatedMs: number;
  hasHistory: boolean;
}) {
  // tick every 100ms so the elapsed counter feels live without being jittery
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(id);
  }, []);

  const elapsed = Math.max(0, now - startedAt);
  // Asymptote at 95% so the bar never claims to be done before it actually is.
  const rawPct = elapsed / estimatedMs;
  const pct = Math.min(0.95, 1 - Math.exp(-rawPct * 1.6));
  const overdue = elapsed > estimatedMs;
  const fmt = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
  const remaining = Math.max(0, estimatedMs - elapsed);

  return (
    <div className="grid md:grid-cols-[auto,1fr] gap-8 items-center min-h-[260px]">
      <CometRing />
      <div className="space-y-4 max-w-md">
        <div className="text-[10px] uppercase tracking-[0.2em] font-mono text-accent2">
          Analyzing
        </div>
        <div className="text-base md:text-lg text-gray-200 leading-relaxed">
          Comparing the job description against the full candidate dossier.
        </div>

        {/* progress bar */}
        <div className="pt-1">
          <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{
                background: overdue
                  ? "linear-gradient(90deg, rgba(251,191,36,0.5), #fbbf24)"
                  : "linear-gradient(90deg, rgba(34,211,238,0.4), #22d3ee)",
                boxShadow: overdue
                  ? "0 0 12px rgba(251,191,36,0.5)"
                  : "0 0 12px rgba(34,211,238,0.5)",
              }}
              animate={{ width: `${pct * 100}%` }}
              transition={{ duration: 0.2, ease: "linear" }}
            />
          </div>

          {/* elapsed / ETA line */}
          <div className="mt-2 flex items-center justify-between text-[11px] font-mono">
            <span className="text-gray-400 tabular-nums">
              {fmt(elapsed)} elapsed
            </span>
            <span
              className={`tabular-nums ${overdue ? "text-warn" : "text-gray-500"}`}
            >
              {overdue
                ? "taking longer than usual…"
                : `~${fmt(remaining)} remaining`}
            </span>
          </div>
          <div className="mt-1 text-[10px] font-mono text-gray-600">
            {hasHistory
              ? `ETA learned from your last ${
                  loadEtaHistory().length
                } request${loadEtaHistory().length === 1 ? "" : "s"} (~${fmt(
                  estimatedMs,
                )} median).`
              : `First-run estimate: ~${fmt(estimatedMs)}. Future calls will use your actual history.`}
          </div>
        </div>
      </div>
    </div>
  );
}

function CometRing() {
  const SIZE = 180;
  const R = 76;
  const C = 2 * Math.PI * R;
  return (
    <div
      className="relative mx-auto md:mx-0"
      style={{ width: SIZE, height: SIZE }}
    >
      {/* faint base ring */}
      <svg
        width={SIZE}
        height={SIZE}
        className="absolute inset-0"
        style={{ overflow: "visible" }}
      >
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={R}
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={10}
          fill="none"
        />
      </svg>

      {/* rotating gradient comet */}
      <svg
        width={SIZE}
        height={SIZE}
        className="absolute inset-0 animate-[spin_2.4s_linear_infinite]"
        style={{ overflow: "visible" }}
      >
        <defs>
          <linearGradient id="comet-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="1" />
          </linearGradient>
        </defs>
        <circle
          cx={SIZE / 2}
          cy={SIZE / 2}
          r={R}
          stroke="url(#comet-grad)"
          strokeWidth={10}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={`${C * 0.55} ${C * 0.45}`}
          style={{ filter: "drop-shadow(0 0 14px rgba(34,211,238,0.55))" }}
        />
      </svg>

      {/* breathing inner pulse */}
      <motion.div
        className="absolute rounded-full"
        style={{
          left: SIZE / 2 - 26,
          top: SIZE / 2 - 26,
          width: 52,
          height: 52,
          background:
            "radial-gradient(circle, rgba(34,211,238,0.35) 0%, rgba(34,211,238,0) 70%)",
        }}
        animate={{ scale: [0.85, 1.15, 0.85], opacity: [0.5, 0.9, 0.5] }}
        transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
      />

      {/* center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <Sparkles className="w-5 h-5 text-accent2" />
        <div className="mt-1 text-[10px] uppercase tracking-[0.2em] text-accent2 font-mono">
          analyzing
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   ERROR BOUNDARY — without this, a render crash inside
   ResultsPanel silently unmounts the whole subtree and the
   AnimatePresence parent shows an empty panel.
============================================================ */

class ResultsErrorBoundary extends Component<
  { children: ReactNode; onReset: () => void },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: unknown) {
    console.error("[RecruiterMatch] ResultsPanel crash:", error, info);
  }
  render() {
    if (this.state.error) {
      return (
        <div className="p-6 text-center space-y-4">
          <div className="inline-flex items-center gap-2 text-bad text-sm">
            <AlertTriangle className="w-4 h-4" />
            The analysis came back in an unexpected shape and could not be rendered.
          </div>
          <div className="text-[11px] font-mono text-gray-500 break-all">
            {this.state.error.message}
          </div>
          <button
            onClick={this.props.onReset}
            className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/[0.04] hover:bg-white/[0.08] transition-colors px-5 py-2.5 text-sm text-gray-200"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ============================================================
   RESULTS PANEL
============================================================ */

function ResultsPanel({
  analysis,
  onReset,
}: {
  analysis: MatchAnalysis;
  onReset: () => void;
}) {
  // Sanity-check the headline against the rubric arithmetic; if the
  // model drifts more than ±5 from the weighted mean, trust the math.
  const recomputed = computeWeightedAverage(analysis.rubric);
  const score =
    Math.abs(recomputed - analysis.score) <= 5 ? analysis.score : recomputed;
  // Map band from the (possibly corrected) score so band and ring agree.
  const band = bandForScore(score);
  const meta = BAND_META[band];

  // Highlight a rubric category when its ring segment is hovered (and vice versa).
  const [hoveredCategory, setHoveredCategory] = useState<number | null>(null);

  return (
    <div className="space-y-10">
      {/* HEADLINE — band pill + summary, full width */}
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center max-w-3xl mx-auto"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border ${meta.border} ${meta.bg} text-xs font-mono uppercase tracking-wider`}
          style={{ color: meta.color }}
        >
          <span
            className="inline-block w-1.5 h-1.5 rounded-full animate-pulse"
            style={{ background: meta.color }}
          />
          {meta.label}
        </motion.div>
        <motion.p
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="mt-4 text-base md:text-lg text-gray-100 leading-relaxed"
        >
          {analysis.summary}
        </motion.p>
      </motion.div>

      {/* CENTERPIECE — big segmented ring */}
      <SegmentedScoreRing
        rubric={analysis.rubric}
        score={score}
        band={band}
        hovered={hoveredCategory}
        onHover={setHoveredCategory}
      />

      {/* REQUIREMENTS */}
      <RequirementsTable requirements={analysis.requirements} />

      {/* GAPS */}
      {analysis.gaps && analysis.gaps.length > 0 && (
        <Card tint="#fbbf24" title="Honest gaps" icon={AlertTriangle} delay={0.3}>
          <ul className="space-y-2">
            {analysis.gaps.map((g, i) => (
              <li key={i} className="text-sm text-gray-300 flex gap-2">
                <span className="text-warn mt-0.5">·</span>
                <span>{g}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* RESET */}
      <div className="flex justify-center pt-2">
        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/[0.04] hover:bg-white/[0.08] hover:border-white/30 transition-colors px-5 py-2.5 text-sm text-gray-200"
        >
          <RotateCcw className="w-3.5 h-3.5" /> Try another job description
        </button>
      </div>
    </div>
  );
}

/* ============================================================
   SEGMENTED SCORE RING
   The ring is divided into N arcs, one per rubric category.
   Each arc's angular extent is proportional to the category's
   weight; each arc's color is the band color of that category's
   own score (good/warn/bad). Hovering an arc highlights the
   corresponding row in the rubric breakdown below.
============================================================ */

function SegmentedScoreRing({
  rubric,
  score,
  band,
  hovered,
  onHover,
}: {
  rubric: RubricRow[];
  score: number;
  band: FitBand;
  hovered: number | null;
  onHover: (i: number | null) => void;
}) {
  const SIZE = 460;
  const CX = SIZE / 2;
  const CY = SIZE / 2;
  const R = 192;
  const STROKE = 26;
  const GAP_DEG = 3.5; // gap between segments in degrees
  const meta = BAND_META[band];
  const totalWeight = rubric.reduce((s, r) => s + r.weight, 0) || 1;

  // Build segment angles. Start at -90° (top) and walk clockwise.
  let cursor = -90;
  const segments = rubric.map((row, i) => {
    const span = (row.weight / totalWeight) * 360 - GAP_DEG;
    const start = cursor + GAP_DEG / 2;
    const end = start + span;
    cursor += (row.weight / totalWeight) * 360;
    return { row, start, end, index: i };
  });

  const hoveredRow = hovered !== null ? rubric[hovered] : null;
  const hoveredColor = hoveredRow ? bandColorForScore(hoveredRow.score) : null;
  const hoveredPct = hoveredRow ? (hoveredRow.weight / totalWeight) * 100 : 0;

  return (
    <div className="flex flex-col items-center">
      <div className="text-[10px] uppercase tracking-[0.25em] font-mono text-gray-500 mb-3">
        Hover a segment · weight = arc length · color = score
      </div>
      <div
        className="relative"
        style={{ width: SIZE, height: SIZE }}
        onMouseLeave={() => onHover(null)}
      >
        {/* soft halo behind the ring */}
        <div
          className="absolute inset-8 rounded-full pointer-events-none transition-colors duration-300"
          style={{
            background: `radial-gradient(circle, ${
              hoveredColor ?? meta.color
            }18 0%, transparent 70%)`,
            filter: "blur(8px)",
          }}
        />

        <svg
          width={SIZE}
          height={SIZE}
          className="absolute inset-0"
          style={{ overflow: "visible" }}
        >
          {/* faint base ring under everything */}
          <circle
            cx={CX}
            cy={CY}
            r={R}
            stroke="rgba(255,255,255,0.04)"
            strokeWidth={STROKE}
            fill="none"
          />
          {/* WIDE invisible hit-areas so the user can hit segments easily */}
          {segments.map((seg) => (
            <path
              key={`hit-${seg.index}`}
              d={describeArc(CX, CY, R, seg.start, seg.end)}
              stroke="transparent"
              strokeWidth={STROKE + 24}
              fill="none"
              style={{ cursor: "pointer", pointerEvents: "stroke" }}
              onMouseEnter={() => onHover(seg.index)}
            />
          ))}
          {segments.map((seg) => {
            const color = bandColorForScore(seg.row.score);
            const isHovered = hovered === seg.index;
            const isDimmed = hovered !== null && !isHovered;
            return (
              <motion.path
                key={seg.index}
                d={describeArc(CX, CY, R, seg.start, seg.end)}
                stroke={color}
                strokeWidth={isHovered ? STROKE + 5 : STROKE}
                strokeLinecap="round"
                fill="none"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{
                  pathLength: 1,
                  opacity: isDimmed ? 0.18 : 1,
                }}
                transition={{
                  pathLength: { duration: 0.9, delay: 0.1 + seg.index * 0.12, ease: "easeOut" },
                  opacity: { duration: 0.25 },
                  strokeWidth: { duration: 0.2 },
                }}
                style={{
                  filter: isHovered
                    ? `drop-shadow(0 0 22px ${color}cc)`
                    : `drop-shadow(0 0 10px ${color}55)`,
                  pointerEvents: "none",
                }}
              />
            );
          })}
        </svg>

        {/* center: hovered category detail OR headline score
            Text must fit inside the inner circle (radius R - STROKE/2).
            Keep content narrow and short enough that its corners never
            punch through the ring. */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div
            className="text-center"
            style={{ width: R * 1.35, maxWidth: SIZE - STROKE * 4 }}
          >
            {hoveredRow ? (
                <motion.div
                  key={`cat-${hovered}`}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18 }}
                >
                  <div
                    className="text-[11px] uppercase tracking-[0.22em] font-mono mb-1.5"
                    style={{ color: hoveredColor ?? "#94a3b8" }}
                  >
                    {hoveredRow.category}
                  </div>
                  <div
                    className="text-6xl font-semibold tabular-nums leading-none"
                    style={{ color: hoveredColor ?? meta.color }}
                  >
                    {hoveredRow.score}
                  </div>
                  <div className="mt-1 text-[10px] uppercase tracking-[0.2em] font-mono text-gray-500">
                    / 100 · weight {hoveredRow.weight} ({hoveredPct.toFixed(0)}%)
                  </div>
                  {hoveredRow.note && (
                    <div
                      className="mt-2.5 text-[12px] text-gray-300 leading-snug px-1"
                      style={{
                        display: "-webkit-box",
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden",
                      }}
                    >
                      {hoveredRow.note}
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="headline"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.2 }}
                >
                  <div className="text-[12px] uppercase tracking-[0.25em] font-mono text-gray-500 mb-2">
                    Overall fit
                  </div>
                  <div
                    className="text-7xl font-semibold tabular-nums leading-none"
                    style={{ color: meta.color }}
                  >
                    {score}
                  </div>
                  <div className="mt-2 text-[12px] uppercase tracking-[0.22em] font-mono text-gray-500">
                    / 100
                  </div>
                </motion.div>
              )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* SVG arc helpers — draws an arc on the circle of radius r centered at (cx,cy)
   from startAngleDeg to endAngleDeg, clockwise, in standard SVG coords. */
function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const a = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
}
function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
}

/* ---------------- Generic colored card ---------------- */

function Card({
  tint,
  title,
  icon: Icon,
  children,
  delay,
}: {
  tint: string;
  title: string;
  icon: typeof Check;
  children: React.ReactNode;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="rounded-2xl border p-5"
      style={{ borderColor: `${tint}33`, background: `${tint}0a` }}
    >
      <div
        className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] font-mono mb-3"
        style={{ color: tint }}
      >
        <Icon className="w-3.5 h-3.5" />
        {title}
      </div>
      {children}
    </motion.div>
  );
}

/* ---------------- Skeleton rows ---------------- */

function SkeletonRows({ count }: { count: number }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 overflow-hidden">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={`p-5 grid grid-cols-[auto,1fr] gap-4 ${
            i < count - 1 ? "border-b border-white/5" : ""
          }`}
        >
          <div className="w-7 h-7 rounded-lg bg-white/5 animate-pulse" />
          <div className="space-y-2">
            <div className="h-4 w-2/3 rounded bg-white/5 animate-pulse" />
            <div className="h-3 w-1/2 rounded bg-white/5 animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}

/* ---------------- Rubric breakdown ---------------- */

/* ---------------- Requirements table ---------------- */

function RequirementsTable({ requirements }: { requirements: RequirementRow[] }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2, duration: 0.5 }}
    >
      <div className="text-[10px] uppercase tracking-[0.2em] font-mono text-gray-500 mb-3">
        Requirement-by-requirement · click any row to see the source quote
      </div>
      <div className="rounded-2xl border border-white/10 bg-black/20 overflow-hidden">
        {requirements.map((r, i) => (
          <RequirementRowItem
            key={i}
            row={r}
            index={i}
            isLast={i === requirements.length - 1}
          />
        ))}
      </div>
    </motion.div>
  );
}

function RequirementRowItem({
  row,
  index,
  isLast,
}: {
  row: RequirementRow;
  index: number;
  isLast: boolean;
}) {
  const meta = STATUS_META[row.status];
  const Icon = meta.icon;
  const [open, setOpen] = useState(false);
  const hasQuote = !!row.evidence_quote && row.evidence_quote.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.05 * index }}
      className={`${!isLast ? "border-b border-white/5" : ""} ${
        hasQuote ? "cursor-pointer" : ""
      } hover:bg-white/[0.015] transition-colors`}
      onClick={() => hasQuote && setOpen((o) => !o)}
    >
      <div className="p-4 md:p-5 grid grid-cols-[auto,1fr] gap-4">
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center border shrink-0"
          style={{
            background: `${meta.color}14`,
            borderColor: `${meta.color}55`,
            boxShadow: `0 0 12px ${meta.color}33`,
          }}
        >
          <Icon className="w-3.5 h-3.5" style={{ color: meta.color }} />
        </div>
        <div className="min-w-0">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="text-sm font-medium text-gray-100">{row.requirement}</div>
            <span
              className="text-[9px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full border"
              style={{
                color: meta.color,
                borderColor: `${meta.color}55`,
                background: `${meta.color}10`,
              }}
            >
              {meta.label}
            </span>
          </div>
          <div className="mt-1.5 text-xs text-gray-400 leading-relaxed">{row.evidence}</div>
          <div className="mt-2 flex items-center gap-3 flex-wrap">
            {row.project_ref && (
              <span className="inline-flex items-center gap-1 text-[11px] font-mono text-gray-500">
                <span className="opacity-60">project:</span> {row.project_ref}
              </span>
            )}
            {hasQuote && (
              <span className="inline-flex items-center gap-1 text-[11px] font-mono text-gray-500">
                <Quote className="w-3 h-3" />
                {open ? "hide source" : "show source quote"}
              </span>
            )}
          </div>
          <AnimatePresence>
            {open && hasQuote && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div
                  className="mt-3 rounded-lg border-l-2 pl-3 py-2 pr-3 text-xs italic text-gray-300 bg-white/[0.02]"
                  style={{ borderColor: meta.color }}
                >
                  &ldquo;{row.evidence_quote}&rdquo;
                  <div className="mt-1 text-[10px] not-italic font-mono text-gray-600">
                    verbatim from candidate dossier
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}

"use client";

/* ------------------------------------------------------------------
   AWSDemo — "Live Match Backend"

   Three-phase narrative simulation of the gaming platform backend
   I built at Disruptive Labs: 30 players join, place bets, play a
   compressed match, and the prize pool is split by an ML scoring
   Lambda. Every player action is a real API call that hits a real
   schema and a real DB row — and the user can see all of it.

   Layout:
     ┌─ header + phase strip ─────────────────────────────────┐
     │                                                         │
     │  ┌─── 3D arena + infra ──────────┐  ┌─ leaderboard ──┐ │
     │  │  health · clock · log · rps   │  │  rank | name   │ │
     │  │  (absolute overlays)          │  │                │ │
     │  └───────────────────────────────┘  └────────────────┘ │
     │                                                         │
     │  ┌─ controls: bet · bad payload · end match · reset ──┐│
     └─────────────────────────────────────────────────────────┘
------------------------------------------------------------------ */

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import {
  Network, AlertTriangle, RotateCcw, FastForward, ShieldOff,
  Activity, Server, CloudOff, Timer,
} from "lucide-react";
import {
  MatchSim, ISLAND_META, ISLAND_IDS,
  type Snapshot, type Phase, type IslandId, type LogEntry, type PlayerState,
  type FleetState, type ChaosBanner,
} from "@/lib/aws-sim";

const Topology3D = dynamic(() => import("./AWSTopology3D"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center text-xs font-mono text-gray-500">
      booting arena…
    </div>
  ),
});

const HUD_HZ = 8;

interface HudState {
  phase: Phase;
  phaseProgress: number;
  matchClockMs: number;
  matchClockTotalMs: number;
  prizePool: number;
  leaderboard: PlayerState[];
  health: Snapshot["health"];
  islandActive: Snapshot["islandActive"];
  islandHeat: Snapshot["islandHeat"];
  log: LogEntry[];
  rps: number;
  rpsHistory: number[];
  s3Count: number;
  userPlayer: PlayerState | null;
  fleet: FleetState;
  banners: ChaosBanner[];
}

const EMPTY_FLEET: FleetState = {
  concurrentMatches: 47,
  ec2Pods: 8,
  ec2PodsDesired: 8,
  albRps: 2_400,
  albReroutes: 0,
  sqsBacklog: 12,
  sqsDlq: 0,
};

const EMPTY_HUD: HudState = {
  phase: "lobby",
  phaseProgress: 0,
  matchClockMs: 0,
  matchClockTotalMs: 1,
  prizePool: 0,
  leaderboard: [],
  health: { alb: true, ec2: true, postgres: true, sqs: true, s3: true, lambda: false, gamelift: true },
  islandActive: { alb: false, ec2: false, postgres: false, sqs: false, lambda: false, s3: false, gamelift: false },
  islandHeat:   { alb: 0,     ec2: 0,     postgres: 0,     sqs: 0,     lambda: 0,     s3: 0,     gamelift: 0     },
  log: [],
  rps: 0,
  rpsHistory: [],
  s3Count: 0,
  userPlayer: null,
  fleet: EMPTY_FLEET,
  banners: [],
};

export default function AWSDemo() {
  const simRef = useRef<MatchSim | null>(null);
  const snapshotRef = useRef<Snapshot | null>(null);

  const [bet, setBet] = useState(50);
  const [hud, setHud] = useState<HudState>(EMPTY_HUD);

  /* ---------- init + raf loop ---------- */
  useEffect(() => {
    const sim = new MatchSim();
    sim.setUserBet(50);
    simRef.current = sim;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      sim.step(dt);
      snapshotRef.current = sim.snapshot();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  /* ---------- HUD refresh (8Hz) ---------- */
  useEffect(() => {
    const id = window.setInterval(() => {
      const sim = simRef.current;
      const snap = snapshotRef.current;
      if (!sim || !snap) return;
      sim.pollRpsHistory();
      const userIdx = snap.userPlayerIdx;
      setHud({
        phase: snap.phase,
        phaseProgress: snap.phaseProgress,
        matchClockMs: snap.matchClockMs,
        matchClockTotalMs: snap.matchClockTotalMs,
        prizePool: snap.prizePool,
        leaderboard: snap.leaderboard,
        health: snap.health,
        islandActive: snap.islandActive,
        islandHeat: snap.islandHeat,
        log: snap.log,
        rps: snap.rps,
        rpsHistory: snap.rpsHistory,
        s3Count: snap.s3Objects.length,
        userPlayer: userIdx !== null ? snap.players[userIdx] : null,
        fleet: snap.fleet,
        banners: snap.chaosBanners,
      });
    }, 1000 / HUD_HZ);
    return () => clearInterval(id);
  }, []);

  /* ---------- bet slider → sim (lobby only) ---------- */
  useEffect(() => {
    if (hud.phase === "lobby") simRef.current?.setUserBet(bet);
  }, [bet, hud.phase]);

  /* ---------- handlers ---------- */
  const onReset = () => simRef.current?.reset();
  const onBadPayload = () => simRef.current?.triggerBadPayload();
  const onEndMatch = () => simRef.current?.endMatchEarly();
  const onGameLiftDrop = () => simRef.current?.triggerGameLiftDrop();
  const onLambdaTimeout = () => simRef.current?.triggerLambdaTimeout();
  const onEc2Fail = () => simRef.current?.triggerEc2PodFail();

  return (
    <div className="shimmer-border rounded-3xl">
      <div className="glass rounded-3xl p-5 md:p-7">
        <Header />

        <PhaseStrip
          phase={hud.phase}
          phaseProgress={hud.phaseProgress}
        />

        <PlatformBar fleet={hud.fleet} />

        <div className="mt-3 grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-4">
          <div className="relative rounded-2xl bg-black/50 border border-white/5 overflow-hidden h-[460px] md:h-[560px]">
            <Topology3D snapshotRef={snapshotRef} />
            {/* overlays */}
            <InfraHealthOverlay health={hud.health} active={hud.islandActive} />
            <MatchClockOverlay phase={hud.phase} ms={hud.matchClockMs} totalMs={hud.matchClockTotalMs} />
            <RequestLogOverlay log={hud.log} />
            <ThroughputOverlay history={hud.rpsHistory} rps={hud.rps} />
            <PhaseAnnouncementOverlay phase={hud.phase} progress={hud.phaseProgress} />
            <ChaosBannerOverlay banners={hud.banners} />
          </div>

          <Leaderboard players={hud.leaderboard} phase={hud.phase} />
        </div>

        <ControlBar
          phase={hud.phase}
          bet={bet}
          setBet={setBet}
          userPlayer={hud.userPlayer}
          onBadPayload={onBadPayload}
          onEndMatch={onEndMatch}
          onReset={onReset}
          onGameLiftDrop={onGameLiftDrop}
          onLambdaTimeout={onLambdaTimeout}
          onEc2Fail={onEc2Fail}
        />

        <Legend />
      </div>
    </div>
  );
}

/* ====================================================================
   HEADER + PHASE STRIP
==================================================================== */

function Header() {
  return (
    <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-5">
      <div>
        <div className="flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.22em] text-[#fb923c]">
          <Network className="w-3.5 h-3.5" />
          Disruptive Labs · live match backend
        </div>
        <h3 className="mt-2 text-2xl md:text-3xl font-semibold text-gradient">
          30 Players · One Pool · GameLift Fleet → EC2 → Prize Pipeline
        </h3>
        <p className="mt-2 text-sm text-gray-400 max-w-2xl leading-relaxed">
          A compressed run of the real gaming platform I shipped at Disruptive Labs. Thirty
          players land on the API, a managed{" "}
          <span className="text-[#f472b6]">AWS GameLift</span> fleet buffers the live session,
          and once the match ends EC2 and a Lambda fan the payouts out through RDS, SQS, and S3.
        </p>
      </div>
    </div>
  );
}

const PHASES: { key: Phase; label: string; sub: string }[] = [
  { key: "lobby",      label: "Lobby",      sub: "joins · bets · pool" },
  { key: "match",      label: "Match",      sub: "gamelift · session telemetry" },
  { key: "distribute", label: "Distribute", sub: "batch → ec2 → λ → payouts" },
  { key: "summary",    label: "Summary",    sub: "auto-loop" },
];

function PhaseStrip({ phase, phaseProgress }: { phase: Phase; phaseProgress: number }) {
  const activeIdx = PHASES.findIndex((p) => p.key === phase);
  return (
    <div className="mt-4 flex items-stretch gap-1.5">
      {PHASES.map((p, i) => {
        const isActive = i === activeIdx;
        const isDone = i < activeIdx;
        const fill = isActive ? phaseProgress * 100 : isDone ? 100 : 0;
        const tint = isActive ? "#22d3ee" : isDone ? "#34d399" : "#475569";
        return (
          <div
            key={p.key}
            className="flex-1 rounded-lg p-2 relative overflow-hidden border"
            style={{
              borderColor: isActive ? "rgba(34,211,238,0.4)" : "rgba(255,255,255,0.05)",
              background: isActive ? "rgba(34,211,238,0.08)" : "rgba(11,15,26,0.5)",
            }}
          >
            <div
              className="absolute left-0 bottom-0 h-0.5 transition-all duration-300"
              style={{ width: `${fill}%`, background: tint }}
            />
            <div className="text-[10px] uppercase tracking-[0.2em] flex items-center gap-1.5" style={{ color: tint }}>
              {i + 1}. {p.label}
            </div>
            <div className="text-[10px] font-mono text-gray-500 mt-0.5">{p.sub}</div>
          </div>
        );
      })}
    </div>
  );
}

/* ====================================================================
   3D OVERLAYS
==================================================================== */

const ICONS_FOR: Record<IslandId, string> = {
  gamelift: "GL", alb: "ALB", ec2: "EC2", sqs: "SQS", postgres: "RDS", lambda: "λ", s3: "S3",
};

function InfraHealthOverlay({
  health, active,
}: {
  health: Snapshot["health"];
  active: Snapshot["islandActive"];
}) {
  return (
    <div className="absolute top-3 left-3 px-2.5 py-1.5 rounded-lg bg-black/60 backdrop-blur border border-white/5">
      <div className="text-[8.5px] uppercase tracking-wider text-gray-500 mb-1">infra health</div>
      <div className="flex items-center gap-2.5">
        {ISLAND_IDS.map((id) => {
          const live = health[id];
          const hot = active[id];
          const color = !live ? "#374151" : hot ? "#34d399" : "#94a3b8";
          return (
            <div key={id} className="flex items-center gap-1 text-[9.5px] font-mono">
              <span
                className="inline-block w-1.5 h-1.5 rounded-full"
                style={{ background: color, boxShadow: hot ? `0 0 6px ${color}` : "none" }}
              />
              <span style={{ color: live ? "#cbd5e1" : "#475569" }}>{ICONS_FOR[id]}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MatchClockOverlay({ phase, ms, totalMs }: { phase: Phase; ms: number; totalMs: number }) {
  // simulated clock: present 15:00 → 0:00
  const realMin = 15;
  const remainingMin = (ms / totalMs) * realMin;
  const m = Math.floor(remainingMin);
  const s = Math.floor((remainingMin - m) * 60);
  const label = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  const tint = phase === "match" ? "#22d3ee"
             : phase === "lobby" ? "#94a3b8"
             : phase === "distribute" ? "#fbbf24" : "#34d399";
  return (
    <div className="absolute top-3 right-3 px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur border border-white/5 text-right">
      <div className="text-[8.5px] uppercase tracking-wider text-gray-500">match clock</div>
      <div className="font-mono tabular-nums text-lg" style={{ color: tint }}>{label}</div>
    </div>
  );
}

function RequestLogOverlay({ log }: { log: LogEntry[] }) {
  return (
    <div className="absolute bottom-3 left-3 w-[295px] rounded-lg bg-black/70 backdrop-blur border border-white/5 p-2 font-mono text-[9.5px]">
      <div className="text-[8.5px] uppercase tracking-wider text-gray-500 mb-1 flex items-center gap-1">
        <Activity className="w-2.5 h-2.5" /> request log
      </div>
      <div className="space-y-0.5 h-[110px] overflow-hidden">
        <AnimatePresence initial={false}>
          {log.slice(0, 9).map((e) => (
            <motion.div
              key={e.id}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="flex items-center gap-1.5"
            >
              <span style={{ color: methodColor(e.method) }}>{e.method}</span>
              <span className="text-gray-300 truncate flex-1">{e.path}</span>
              <span style={{ color: statusColor(e.status) }}>{e.status}</span>
              <span className="text-gray-500 tabular-nums w-6 text-right">{e.latencyMs}ms</span>
            </motion.div>
          ))}
        </AnimatePresence>
        {log.length === 0 && <div className="text-gray-600">awaiting requests…</div>}
      </div>
    </div>
  );
}

function methodColor(m: "POST" | "GET" | "PUT" | "GL") {
  return m === "POST" ? "#fb923c"
       : m === "GET"  ? "#22d3ee"
       : m === "PUT"  ? "#a78bfa"
       : /* GL */      "#f472b6";
}
function statusColor(s: number) {
  return s >= 500 ? "#f87171" : s >= 400 ? "#fbbf24" : "#34d399";
}

function ThroughputOverlay({ history, rps }: { history: number[]; rps: number }) {
  const W = 180;
  const H = 50;
  const max = Math.max(4, ...history);
  const path = useMemo(() => {
    if (history.length < 2) return "";
    const N = history.length;
    return "M" + history.map((v, i) => {
      const x = (i / (N - 1)) * W;
      const y = H - (v / max) * (H - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" L");
  }, [history, max]);
  return (
    <div className="absolute bottom-3 right-3 rounded-lg bg-black/70 backdrop-blur border border-white/5 p-2">
      <div className="flex items-center justify-between gap-2 text-[8.5px] uppercase tracking-wider text-gray-500">
        <span>throughput</span>
        <span className="font-mono text-white tabular-nums">{rps} rps</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block mt-1" preserveAspectRatio="none">
        {[0.33, 0.66].map((f) => (
          <line key={f} x1={0} x2={W} y1={H * f} y2={H * f} stroke="#1f2937" strokeWidth={0.5} />
        ))}
        <path d={path} fill="none" stroke="#22d3ee" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
        {/* fill below */}
        {path && (
          <path d={`${path} L ${W} ${H} L 0 ${H} Z`} fill="#22d3ee" opacity={0.12} />
        )}
      </svg>
    </div>
  );
}

function PhaseAnnouncementOverlay({ phase, progress }: { phase: Phase; progress: number }) {
  // brief announcement text at the top center when a phase starts
  const announcements: Record<Phase, string> = {
    lobby: "PRE-MATCH LOBBY",
    match: "MATCH IN PROGRESS",
    distribute: "λ COLD START · DISTRIBUTING PRIZE POOL",
    summary: "MATCH COMPLETE",
  };
  const visible = progress < 0.18; // first ~18% of phase
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key={phase}
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className="absolute top-24 left-1/2 -translate-x-1/2 px-4 py-1.5 rounded-full bg-black/70 backdrop-blur border border-white/10 text-[10px] font-mono uppercase tracking-[0.25em]"
          style={{
            color: phase === "distribute" ? "#fbbf24" : phase === "match" ? "#22d3ee" : "#cbd5e1",
            boxShadow: phase === "distribute" ? "0 0 24px rgba(251,191,36,0.35)" : "none",
          }}
        >
          {announcements[phase]}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/* ====================================================================
   PLATFORM BAR — concurrent matches, ALB, ASG, SQS/DLQ.
   Sits above the 3D canvas so this match reads as one drop in the
   platform bucket. Values update live; autoscale + DLQ flash on chaos.
==================================================================== */

function PlatformBar({ fleet }: { fleet: FleetState }) {
  const autoscaling = fleet.ec2Pods !== fleet.ec2PodsDesired;
  return (
    <div className="mt-3 rounded-xl bg-black/40 border border-white/5 px-4 py-2 flex flex-wrap items-center gap-x-5 gap-y-2">
      <div className="flex items-center gap-1.5 pr-2">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#34d399] animate-pulse" />
        <span className="text-[9px] uppercase tracking-[0.2em] text-gray-500">platform · us-east-1</span>
      </div>
      <FleetStat label="live matches" value={fleet.concurrentMatches.toString()} hint="total" tint="#22d3ee" />
      <FleetStat label="ALB" value={fleet.albRps.toLocaleString()} hint="rps" tint="#fb923c" />
      <FleetStat
        label="EC2 ASG"
        value={`${fleet.ec2Pods}/${fleet.ec2PodsDesired}`}
        hint={autoscaling ? "scaling" : "healthy"}
        tint={autoscaling ? "#fbbf24" : "#34d399"}
        pulse={autoscaling}
      />
      <FleetStat
        label="reroutes"
        value={fleet.albReroutes.toString()}
        hint="ALB"
        tint={fleet.albReroutes > 0 ? "#fbbf24" : "#475569"}
      />
      <FleetStat
        label="SQS"
        value={fleet.sqsBacklog.toString()}
        hint="queued"
        tint={fleet.sqsBacklog > 40 ? "#fbbf24" : "#94a3b8"}
      />
      <FleetStat
        label="DLQ"
        value={fleet.sqsDlq.toString()}
        hint="parked"
        tint={fleet.sqsDlq > 0 ? "#f87171" : "#475569"}
        pulse={fleet.sqsDlq > 0}
      />
    </div>
  );
}

function FleetStat({
  label, value, hint, tint, pulse,
}: {
  label: string; value: string; hint: string; tint: string; pulse?: boolean;
}) {
  return (
    <div className="flex flex-col items-start leading-tight">
      <div className="text-[8px] uppercase tracking-[0.18em] text-gray-500">{label}</div>
      <div className="flex items-baseline gap-1.5">
        <motion.span
          key={value}
          initial={{ opacity: 0.6, y: -2 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
          className="font-mono tabular-nums text-[13px] font-semibold"
          style={{ color: tint, textShadow: pulse ? `0 0 10px ${tint}80` : "none" }}
        >
          {value}
        </motion.span>
        <span className="text-[9px] font-mono text-gray-500">{hint}</span>
      </div>
    </div>
  );
}

/* ====================================================================
   CHAOS BANNER — failure + recovery narration, stacked under platform HUD
==================================================================== */

function ChaosBannerOverlay({ banners }: { banners: ChaosBanner[] }) {
  return (
    <div className="absolute top-14 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1.5 pointer-events-none z-[2]">
      <AnimatePresence>
        {banners.map((b) => {
          const tone = b.tone === "danger"
            ? { fg: "#fecaca", bg: "rgba(127,29,29,0.55)", br: "rgba(248,113,113,0.55)", glow: "rgba(248,113,113,0.35)" }
            : b.tone === "warn"
            ? { fg: "#fde68a", bg: "rgba(120,53,15,0.55)", br: "rgba(251,191,36,0.55)", glow: "rgba(251,191,36,0.28)" }
            : { fg: "#bbf7d0", bg: "rgba(6,78,59,0.55)", br: "rgba(52,211,153,0.55)", glow: "rgba(52,211,153,0.25)" };
          return (
            <motion.div
              key={b.id}
              layout
              initial={{ opacity: 0, y: -8, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -4, scale: 0.98 }}
              transition={{ duration: 0.25, ease: [0.2, 0.7, 0.2, 1] }}
              className="px-3 py-1.5 rounded-md backdrop-blur-md flex items-center gap-2 min-w-[340px] max-w-[460px]"
              style={{
                background: tone.bg,
                border: `1px solid ${tone.br}`,
                boxShadow: `0 0 24px ${tone.glow}`,
              }}
            >
              <AlertTriangle
                className="w-3 h-3 shrink-0"
                style={{ color: tone.fg }}
              />
              <div className="flex-1 min-w-0">
                <div className="text-[10px] font-mono uppercase tracking-[0.15em] truncate" style={{ color: tone.fg }}>
                  {b.headline}
                </div>
                <div className="text-[9.5px] font-mono text-gray-300/90 truncate">{b.detail}</div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

/* ====================================================================
   LEADERBOARD
==================================================================== */

function Leaderboard({ players, phase }: { players: PlayerState[]; phase: Phase }) {
  const showPayouts = phase === "distribute" || phase === "summary";
  return (
    <div className="rounded-2xl bg-bg/60 border border-white/5 p-3 flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] uppercase tracking-[0.2em] text-gray-400">leaderboard</div>
        <div className="text-[9px] font-mono text-gray-500">{showPayouts ? "by payout" : "by score"}</div>
      </div>
      <div className="space-y-1 overflow-y-auto" style={{ maxHeight: "560px" }}>
        {players.map((p, i) => {
          const dead = !p.alive;
          const tint = i === 0 ? "#fbbf24" : i === 1 ? "#cbd5e1" : i === 2 ? "#fb923c" : "#475569";
          return (
            <div
              key={p.id}
              className="flex items-center gap-2 px-1.5 py-1 rounded text-[10.5px] font-mono"
              style={{
                background: p.isUser ? "rgba(34,211,238,0.10)" : i < 3 ? "rgba(255,255,255,0.025)" : "transparent",
                borderLeft: p.isUser ? "2px solid #22d3ee" : "2px solid transparent",
              }}
            >
              <span className="tabular-nums w-5 text-right" style={{ color: tint }}>{i + 1}</span>
              <span className={dead ? "text-gray-600 line-through" : p.isUser ? "text-cyan-300" : "text-gray-300"}>
                {p.name}
              </span>
              <span className="text-gray-600 ml-auto">{p.kills}k · {p.damage}d</span>
              {showPayouts ? (
                <span className="tabular-nums w-12 text-right" style={{ color: "#fbbf24" }}>
                  ${p.payout.toFixed(0)}
                </span>
              ) : (
                <span className="tabular-nums w-12 text-right text-gray-500">
                  ${p.bet}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ====================================================================
   CONTROL BAR
==================================================================== */

function ControlBar({
  phase, bet, setBet, userPlayer,
  onBadPayload, onEndMatch, onReset,
  onGameLiftDrop, onLambdaTimeout, onEc2Fail,
}: {
  phase: Phase;
  bet: number;
  setBet: (n: number) => void;
  userPlayer: PlayerState | null;
  onBadPayload: () => void;
  onEndMatch: () => void;
  onReset: () => void;
  onGameLiftDrop: () => void;
  onLambdaTimeout: () => void;
  onEc2Fail: () => void;
}) {
  const inLobby = phase === "lobby";
  const inMatch = phase === "match";
  const inDistribute = phase === "distribute";
  // Bad-payload chaos only lands on EC2/Pydantic during lobby — during
  // match, player traffic goes to GameLift, not our HTTP API.
  const canBadPayload = inLobby;

  return (
    <div className="mt-4 rounded-2xl bg-bg/60 border border-white/5 p-3 space-y-3">
      {/* Row 1 — You / bet */}
      <div className="flex items-center gap-3">
        <div className="text-[10px] uppercase tracking-[0.2em] text-[#22d3ee] whitespace-nowrap">
          you · player #1
        </div>
        <div className="flex-1 min-w-[160px]">
          <input
            type="range" min={5} max={250} step={5}
            value={bet}
            onChange={(e) => setBet(+e.target.value)}
            disabled={!inLobby}
            className="w-full accent-[#22d3ee] disabled:opacity-50"
            aria-label="Your bet"
          />
        </div>
        <div className="font-mono text-white tabular-nums text-sm w-14 text-right">
          ${userPlayer?.bet ?? bet}
        </div>
        {!inLobby && userPlayer && (
          <div className="font-mono text-[10px] text-gray-500 whitespace-nowrap">
            {userPlayer.alive ? "alive" : "OUT"} · {userPlayer.kills}k · ${userPlayer.payout.toFixed(0)} payout
          </div>
        )}
      </div>

      {/* Row 2 — chaos bar, grouped + labeled. */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="inline-flex items-center gap-2">
          <span className="text-[9px] uppercase tracking-[0.2em] text-gray-500">inject</span>
          <ChaosBtn
            label="Bad payload"
            icon={<ShieldOff className="w-3 h-3" />}
            onClick={onBadPayload}
            tone="warn"
            disabled={!canBadPayload}
            tooltip="POST /match/join with amount: -50 — Pydantic rejects, nothing reaches the DB"
          />
          <ChaosBtn
            label="GameLift crash"
            icon={<CloudOff className="w-3 h-3" />}
            onClick={onGameLiftDrop}
            tone="danger"
            disabled={!inMatch}
            tooltip="Kill a GameLift fleet instance mid-match. Without the full session buffer we can't score fairly — match is VOIDED and all 30 stakes are refunded idempotently via SNS → SQS → Lambda."
          />
          <ChaosBtn
            label="EC2 pod fail"
            icon={<Server className="w-3 h-3" />}
            onClick={onEc2Fail}
            tone="danger"
            tooltip="Kill an EC2 pod. If a bet was in-flight, watch the 502 bounce back to the player, then ALB retry on a healthy sibling with the same Idempotency-Key — commits exactly once. ASG spawns the replacement in parallel."
          />
          <ChaosBtn
            label="Lambda timeout"
            icon={<Timer className="w-3 h-3" />}
            onClick={onLambdaTimeout}
            tone="warn"
            disabled={!inDistribute}
            tooltip="Force a payout Lambda to hit its 30s timeout. DLQ catches the payload; retry honors the idempotency key — no double-pay."
          />
        </div>

        <div className="ml-auto inline-flex items-center gap-2">
          <span className="text-[9px] uppercase tracking-[0.2em] text-gray-500">flow</span>
          <ChaosBtn
            label="End match early"
            icon={<FastForward className="w-3 h-3" />}
            onClick={onEndMatch}
            tone="warn"
            disabled={!inMatch}
            tooltip="Trigger phase 3 — the Lambda cold-start moment"
          />
          <ChaosBtn
            label="Reset"
            icon={<RotateCcw className="w-3 h-3" />}
            onClick={onReset}
            tone="neutral"
          />
        </div>
      </div>
    </div>
  );
}

function ChaosBtn({
  label, icon, onClick, tone, disabled, tooltip,
}: {
  label: string; icon: React.ReactNode; onClick: () => void;
  tone: "danger" | "warn" | "good" | "neutral";
  disabled?: boolean;
  tooltip?: string;
}) {
  const palette = {
    danger:  { fg: "#f87171", bg: "rgba(248,113,113,0.10)", br: "rgba(248,113,113,0.45)" },
    warn:    { fg: "#fbbf24", bg: "rgba(251,191,36,0.10)",  br: "rgba(251,191,36,0.45)" },
    good:    { fg: "#34d399", bg: "rgba(52,211,153,0.10)",  br: "rgba(52,211,153,0.45)" },
    neutral: { fg: "#cbd5e1", bg: "rgba(255,255,255,0.05)", br: "rgba(255,255,255,0.15)" },
  }[tone];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={tooltip}
      className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-mono transition-all hover:brightness-125 disabled:opacity-40 disabled:cursor-not-allowed"
      style={{ color: palette.fg, background: palette.bg, border: `1px solid ${palette.br}` }}
    >
      {icon}{label}
    </button>
  );
}

/* ====================================================================
   LEGEND
==================================================================== */

function Legend() {
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 text-[10px] font-mono">
      <span className="text-gray-500 mr-1 uppercase tracking-wider">stack:</span>
      {(["gamelift", "alb", "ec2", "sqs", "postgres", "lambda", "s3"] as IslandId[]).map((id) => (
        <span
          key={id}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-white/5 border border-white/5"
          style={{ color: ISLAND_META[id].color }}
        >
          <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: ISLAND_META[id].color }} />
          {ISLAND_META[id].label}
        </span>
      ))}
    </div>
  );
}

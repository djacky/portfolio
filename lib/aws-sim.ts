/* ------------------------------------------------------------------
   match-sim — narrative simulation of a 30-player live match against
   the production backend (FastAPI / Pydantic / RDS Postgres / S3 /
   Lambda). Drives both the two-layer 3D scene and the HUD.

   Phases:
     lobby      ~8s   players materialize, bets validated, pool fills
     match      ~20s  performance events stream in; eliminations punctuate
                      S3 chunks dropped every 5s; Lambda stays cold
     distribute ~8s   Lambda cold-start, scoring, 30 payouts fly out,
                      S3 ledger archive, balances updated
     summary    ~∞    final state until user resets / auto-loops
------------------------------------------------------------------ */

export type Phase = "lobby" | "match" | "distribute" | "summary";

export type IslandId = "ec2" | "pydantic" | "postgres" | "lambda" | "s3" | "gamelift";

export const ISLAND_IDS: IslandId[] = ["gamelift", "ec2", "pydantic", "postgres", "lambda", "s3"];

export interface PlayerState {
  id: number;
  name: string;
  bet: number;
  /** circle radius position (XZ plane) */
  posX: number;
  posZ: number;
  /** perf */
  kills: number;
  damage: number;
  survivalMs: number;
  /** status */
  alive: boolean;
  eliminatedAt: number | null;
  rank: number | null;
  payout: number;
  finalScore: number;
  /** visual */
  joinedAt: number | null;
  pulseUntil: number;
  flashColor: string | null;
  flashUntil: number;
  payoutFlashUntil: number;
  /** is this slot the user? */
  isUser: boolean;
}

export interface Particle {
  id: number;
  kind:
    | "join"        // player → ec2
    | "perf"        // player → gamelift (session telemetry)
    | "valid"       // ec2 → pydantic (intermediate)
    | "write"       // pydantic → postgres
    | "elim"        // player → gamelift (session event)
    | "s3chunk"     // gamelift → s3 (replay chunk upload)
    | "batch"       // gamelift → ec2 (match-end bulk transfer)
    | "snapshot"    // postgres → lambda (match dataset on end)
    | "payout"      // lambda → player
    | "ledger"      // lambda → s3
    | "balance"     // lambda → postgres
    | "reject";     // pydantic → ec2 (rejected validation, bounces back)
  fromKind: "player" | IslandId;
  fromIdx: number; // when fromKind=player
  toKind: "player" | IslandId;
  toIdx: number;
  startedAt: number;
  travelMs: number;
  color: string;
  label?: string;
  amount?: number; // for payout shards
}

export interface LogEntry {
  id: number;
  ts: number;
  method: "POST" | "GET" | "PUT" | "GL";
  path: string;
  status: number; // 200 | 422 | 503
  latencyMs: number;
  detail?: string; // optional inline detail e.g. "$50"
}

export interface S3Object {
  id: number;
  createdAt: number;
  label: string;
  kind: "chunk" | "ledger";
}

export interface InfraHealth {
  ec2: boolean;
  pydantic: boolean;
  postgres: boolean;
  s3: boolean;
  lambda: boolean;   // false during phases 1 & 2; true during phase 3
  gamelift: boolean; // managed fleet — always provisioned, active during match
}

export interface Snapshot {
  now: number;            // sim time ms (continuously increasing)
  phase: Phase;
  phaseProgress: number;  // 0..1 of current phase
  /** match clock counts DOWN from MATCH_DURATION_MS during phase 2 */
  matchClockMs: number;
  matchClockTotalMs: number;

  prizePool: number;
  prizePoolGlowUntil: number;

  /** events buffered in the GameLift session (drains at match end) */
  bufferedEvents: number;

  players: PlayerState[];
  leaderboard: PlayerState[]; // sorted by finalScore desc (or current score during match)
  particles: Particle[];
  islandHeat: Record<IslandId, number>;
  islandActive: Record<IslandId, boolean>;
  health: InfraHealth;

  log: LogEntry[];        // newest first
  rps: number;
  rpsHistory: number[];

  s3Objects: S3Object[];

  userPlayerIdx: number | null;

  /** UI hint flags */
  showColdStartBurst: boolean;
  showPoolLockFlash: boolean;

  /** platform-level fleet state (one match is a drop in this bucket) */
  fleet: FleetState;

  /** active chaos banners (rendered at top of canvas) */
  chaosBanners: ChaosBanner[];
}

/** platform-wide fleet state — animates under load + chaos */
export interface FleetState {
  concurrentMatches: number;      // total live matches on the platform
  ec2Pods: number;                 // current ASG pod count behind ALB
  ec2PodsDesired: number;          // ASG desired count (target after scaling)
  albRps: number;                  // requests/sec at the load balancer
  albReroutes: number;             // running counter of ALB reroute events
  sqsBacklog: number;              // messages waiting in wallet/notify queues
  sqsDlq: number;                  // messages parked in the DLQ
}

export interface ChaosBanner {
  id: number;
  kind: "gamelift-drop" | "lambda-timeout" | "ec2-fail";
  headline: string;                // short, e.g. "GAMELIFT FLEET-02 CRASHED"
  detail: string;                  // "restoring 47 events from replay_chunk_T"
  startedAt: number;
  durationMs: number;
  tone: "danger" | "warn" | "good";
}

const PLAYER_COUNT = 30;
const ARENA_RADIUS = 5.6;
const MATCH_DURATION_MS = 22_000;     // compressed "15 min"
const LOBBY_DURATION_MS = 8_500;
const DISTRIBUTE_DURATION_MS = 11_000; // extra room for the gamelift→ec2 batch beat
const SUMMARY_AUTO_LOOP_MS = 12_000;

const PARTICLE_TRAVEL_MS = 800;

const NAMES = [
  "aurora", "neo", "kairo", "vex", "luma", "rune", "pyra", "echo",
  "nova", "zephyr", "halix", "orin", "sable", "vale", "trix", "kestrel",
  "onyx", "jett", "ramen", "fable", "wren", "cipher", "mosaic", "ember",
  "drift", "quill", "kobalt", "nyx", "talon", "rift",
];

const ISLAND_HEAT_DECAY = 0.88;
const RPS_HISTORY_LEN = 80;
const LOG_LIMIT = 18;

function rand(min: number, max: number) { return min + Math.random() * (max - min); }
function clamp01(x: number) { return Math.max(0, Math.min(1, x)); }

export class MatchSim {
  private sim_now = 0;
  private phase: Phase = "lobby";
  private phaseStarted = 0;
  private players: PlayerState[];
  private particles: Particle[] = [];
  private particleCounter = 0;
  private logCounter = 0;
  private s3Counter = 0;
  private prizePool = 0;
  private prizePoolGlowUntil = 0;
  private islandHeat: Record<IslandId, number> = { ec2: 0, pydantic: 0, postgres: 0, lambda: 0, s3: 0, gamelift: 0 };
  private log: LogEntry[] = [];
  private completionsForRps: number[] = [];
  private rpsHistory: number[] = [];
  private s3Objects: S3Object[] = [];
  private userPlayerIdx: number | null = null;
  private showPoolLockFlash = false;
  private poolLockFlashStarted = 0;
  private showColdStartBurst = false;
  private coldStartBurstAt = 0;
  /** scripted scheduling cursors */
  private nextJoinAt = 0;
  private joinIdx = 0;
  private nextPerfEventAt = 0;
  private nextS3ChunkAt = 0;
  private elimSchedule: { at: number; playerIdx: number }[] = [];
  private elimCursor = 0;
  private distributePlannedAt = 0;
  private distributeKickedOff = false;
  private batchTransferAt: number | null = null;
  private bulkWriteAt: number | null = null;
  private payoutsFiredAt: number | null = null;
  private summaryEnteredAt = 0;
  private autoLoop = true;
  private bufferedEvents = 0;
  /** platform fleet — drift baseline + reactive to chaos */
  private fleet: FleetState = {
    concurrentMatches: 47,
    ec2Pods: 8,
    ec2PodsDesired: 8,
    albRps: 2_400,
    albReroutes: 0,
    sqsBacklog: 12,
    sqsDlq: 0,
  };
  private fleetTickAccum = 0;
  private chaosBanners: ChaosBanner[] = [];
  private bannerCounter = 0;
  /** chaos state — drives recovery animations */
  private gameliftRecoveryAt: number | null = null;
  private lambdaRetryAt: number | null = null;
  private ec2FailAt: number | null = null;
  private ec2RespawnAt: number | null = null;

  constructor() {
    this.players = this.makePlayers();
    this.scheduleLobby();
  }

  /* -------- public API -------- */

  setUserBet(bet: number) {
    // pre-lobby/lobby: assigns or updates user-controlled slot at index 0
    if (this.phase !== "lobby") return;
    const userIdx = 0;
    this.players[userIdx].bet = Math.max(1, Math.min(500, Math.round(bet)));
    this.players[userIdx].name = "you";
    this.players[userIdx].isUser = true;
    this.userPlayerIdx = userIdx;
  }

  triggerBadPayload() {
    // fires a rejected-bet particle: ec2 → pydantic (red) → bounces back.
    // Only meaningful in lobby — during match, telemetry goes to GameLift, not our API.
    if (this.phase !== "lobby") return;
    this.bumpHeat("ec2", 0.55);
    this.bumpHeat("pydantic", 0.85);
    this.spawnParticle({
      kind: "reject",
      fromKind: "ec2", fromIdx: 0,
      toKind: "pydantic", toIdx: 0,
      travelMs: 350,
      color: "#f87171",
      label: "amount: -50",
    });
    this.appendLog({
      method: "POST", path: "/match/join",
      status: 422, latencyMs: Math.round(rand(3, 7)),
      detail: "ValidationError: amount > 0",
    });
  }

  endMatchEarly() {
    if (this.phase !== "match") return;
    // collapse remaining match time → flush to distribute phase soon
    this.transition("distribute");
  }

  /* ---- chaos triggers (senior-signal scenarios) ---- */

  /** GameLift fleet instance crashes mid-match. Recovery: buffered events
   *  restored from last S3 replay checkpoint, session resumes on a sibling. */
  triggerGameLiftDrop() {
    if (this.phase !== "match") return;
    if (this.gameliftRecoveryAt !== null) return; // don't stack
    const restoredEvents = Math.min(this.bufferedEvents, Math.floor(rand(30, 80)));
    this.pushBanner({
      kind: "gamelift-drop",
      headline: "GAMELIFT FLEET-02 CRASHED",
      detail: `restoring ${restoredEvents} events from replay_chunk_T · session migrating to fleet-05`,
      durationMs: 4200,
      tone: "danger",
    });
    this.appendLog({
      method: "GL", path: "fleet-02.crash",
      status: 503, latencyMs: 0,
      detail: "EC2 health-check fail",
    });
    // flash — gamelift heat spikes then dips
    this.bumpHeat("gamelift", 1);
    this.fleet.albReroutes += 1;
    // recovery: 1.5s later, an S3 → GameLift restore particle fires
    this.gameliftRecoveryAt = this.sim_now + 1500;
  }

  /** Lambda payout invocation times out. Recovery: message parks in SQS DLQ,
   *  next invocation reads the same payload with its idempotency key, retries
   *  successfully — no double-pay. */
  triggerLambdaTimeout() {
    if (this.phase !== "distribute") return;
    if (this.lambdaRetryAt !== null) return;
    this.pushBanner({
      kind: "lambda-timeout",
      headline: "LAMBDA TIMEOUT · 30s",
      detail: "routed to SQS DLQ · retry with idempotency key payout_match_7f3",
      durationMs: 4500,
      tone: "warn",
    });
    this.appendLog({
      method: "POST", path: "/lambda/distribute",
      status: 504, latencyMs: 30_000,
      detail: "timeout · → DLQ",
    });
    this.fleet.sqsDlq += 1;
    this.bumpHeat("lambda", 1);
    this.lambdaRetryAt = this.sim_now + 2000;
  }

  /** EC2 pod fails health check. Recovery: ALB stops routing to it,
   *  ASG spawns a replacement, DesiredCapacity holds steady. */
  triggerEc2PodFail() {
    if (this.ec2FailAt !== null) return;
    this.ec2FailAt = this.sim_now;
    this.fleet.ec2Pods = Math.max(1, this.fleet.ec2Pods - 1);
    this.fleet.albReroutes += 1;
    this.pushBanner({
      kind: "ec2-fail",
      headline: `EC2 POD api-${(Math.random() * 1000).toFixed(0).padStart(3, "0")} UNHEALTHY`,
      detail: `ALB draining · ASG scaling ${this.fleet.ec2Pods}→${this.fleet.ec2PodsDesired} · replacement spawning`,
      durationMs: 5000,
      tone: "warn",
    });
    this.appendLog({
      method: "GET", path: "/health",
      status: 503, latencyMs: 29,
      detail: "ALB target drained",
    });
    // recovery: ASG spawns replacement in 2.5s
    this.ec2RespawnAt = this.sim_now + 2500;
  }

  reset() {
    this.sim_now = 0;
    this.phase = "lobby";
    this.phaseStarted = 0;
    this.players = this.makePlayers();
    this.particles = [];
    this.prizePool = 0;
    this.prizePoolGlowUntil = 0;
    this.islandHeat = { ec2: 0, pydantic: 0, postgres: 0, lambda: 0, s3: 0, gamelift: 0 };
    this.log = [];
    this.completionsForRps = [];
    this.rpsHistory = [];
    this.s3Objects = [];
    this.userPlayerIdx = null;
    this.showColdStartBurst = false;
    this.showPoolLockFlash = false;
    this.distributeKickedOff = false;
    this.batchTransferAt = null;
    this.bulkWriteAt = null;
    this.payoutsFiredAt = null;
    this.bufferedEvents = 0;
    this.chaosBanners = [];
    this.gameliftRecoveryAt = null;
    this.lambdaRetryAt = null;
    this.ec2FailAt = null;
    this.ec2RespawnAt = null;
    this.fleet = {
      concurrentMatches: 47,
      ec2Pods: 8,
      ec2PodsDesired: 8,
      albRps: 2_400,
      albReroutes: 0,
      sqsBacklog: 12,
      sqsDlq: 0,
    };
    this.fleetTickAccum = 0;
    this.scheduleLobby();
  }

  setAutoLoop(v: boolean) { this.autoLoop = v; }

  step(dtMs: number) {
    if (dtMs <= 0) return;
    const dt = Math.min(dtMs, 250);
    this.sim_now += dt;

    // decay island heat
    for (const k of ISLAND_IDS) {
      this.islandHeat[k] *= ISLAND_HEAT_DECAY;
      if (this.islandHeat[k] < 0.005) this.islandHeat[k] = 0;
    }

    // update player drift + survival
    for (const p of this.players) {
      if (p.alive && this.phase === "match") {
        p.survivalMs += dt;
      }
      // tiny circular drift around home position
      const wob = Math.sin(this.sim_now * 0.001 + p.id) * 0.03;
      p.posX = p.posX * 0.995 + (Math.cos((p.id / PLAYER_COUNT) * Math.PI * 2) * ARENA_RADIUS) * 0.005;
      p.posZ = p.posZ * 0.995 + (Math.sin((p.id / PLAYER_COUNT) * Math.PI * 2) * ARENA_RADIUS) * 0.005;
      p.posX += wob;
    }

    // retire old particles
    this.particles = this.particles.filter((p) => this.sim_now - p.startedAt < p.travelMs);

    // phase logic
    switch (this.phase) {
      case "lobby":      this.tickLobby(); break;
      case "match":      this.tickMatch(); break;
      case "distribute": this.tickDistribute(); break;
      case "summary":    this.tickSummary(); break;
    }

    // update RPS rolling window
    while (this.completionsForRps.length && this.sim_now - this.completionsForRps[0] > 1000) {
      this.completionsForRps.shift();
    }

    // platform fleet drift + autoscale + chaos recovery
    this.tickFleet(dt);
    // retire expired chaos banners
    this.chaosBanners = this.chaosBanners.filter(
      (b) => this.sim_now - b.startedAt < b.durationMs,
    );
    if (this.showPoolLockFlash && this.sim_now - this.poolLockFlashStarted > 1500) {
      this.showPoolLockFlash = false;
    }
    if (this.showColdStartBurst && this.sim_now - this.coldStartBurstAt > 1200) {
      this.showColdStartBurst = false;
    }
  }

  /** call this at HUD tick rate to keep the throughput chart smooth */
  pollRpsHistory() {
    this.rpsHistory.push(this.completionsForRps.length);
    if (this.rpsHistory.length > RPS_HISTORY_LEN) {
      this.rpsHistory.splice(0, this.rpsHistory.length - RPS_HISTORY_LEN);
    }
  }

  snapshot(): Snapshot {
    const phaseDur = this.phaseDuration();
    const phaseProgress = clamp01((this.sim_now - this.phaseStarted) / phaseDur);
    const matchClock = this.phase === "match"
      ? Math.max(0, MATCH_DURATION_MS - (this.sim_now - this.phaseStarted))
      : (this.phase === "lobby" ? MATCH_DURATION_MS : 0);
    const leaderboard = [...this.players].sort((a, b) => {
      // sort by score (post-match) or kills + damage (during match) desc
      const sa = a.finalScore || (a.kills * 100 + a.damage + a.survivalMs * 0.001);
      const sb = b.finalScore || (b.kills * 100 + b.damage + b.survivalMs * 0.001);
      return sb - sa;
    });
    return {
      now: this.sim_now,
      phase: this.phase,
      phaseProgress,
      matchClockMs: matchClock,
      matchClockTotalMs: MATCH_DURATION_MS,
      prizePool: this.prizePool,
      prizePoolGlowUntil: this.prizePoolGlowUntil,
      bufferedEvents: this.bufferedEvents,
      players: this.players,
      leaderboard,
      particles: this.particles,
      islandHeat: { ...this.islandHeat },
      islandActive: {
        ec2: this.islandHeat.ec2 > 0.05,
        pydantic: this.islandHeat.pydantic > 0.05,
        postgres: this.islandHeat.postgres > 0.05,
        lambda: this.phase === "distribute" || this.phase === "summary",
        s3: this.islandHeat.s3 > 0.05,
        gamelift: this.phase === "match" || this.islandHeat.gamelift > 0.05,
      },
      health: {
        ec2: true, pydantic: true, postgres: true, s3: true,
        lambda: this.phase === "distribute" || this.phase === "summary",
        gamelift: true, // managed fleet always provisioned
      },
      log: [...this.log],
      rps: this.completionsForRps.length,
      rpsHistory: [...this.rpsHistory],
      s3Objects: [...this.s3Objects],
      userPlayerIdx: this.userPlayerIdx,
      showColdStartBurst: this.showColdStartBurst,
      showPoolLockFlash: this.showPoolLockFlash,
      fleet: { ...this.fleet },
      chaosBanners: [...this.chaosBanners],
    };
  }

  /* -------- internals: lifecycle -------- */

  private makePlayers(): PlayerState[] {
    const out: PlayerState[] = [];
    for (let i = 0; i < PLAYER_COUNT; i++) {
      const angle = (i / PLAYER_COUNT) * Math.PI * 2;
      // tiny radial jitter so the ring doesn't look mechanical
      const r = ARENA_RADIUS + (Math.sin(i * 1.3) * 0.15);
      out.push({
        id: i,
        name: i === 0 ? "you" : NAMES[i],
        bet: i === 0 ? 50 : Math.round(rand(20, 120)),
        posX: Math.cos(angle) * r,
        posZ: Math.sin(angle) * r,
        kills: 0, damage: 0, survivalMs: 0,
        alive: true,
        eliminatedAt: null,
        rank: null,
        payout: 0,
        finalScore: 0,
        joinedAt: null,
        pulseUntil: 0,
        flashColor: null,
        flashUntil: 0,
        payoutFlashUntil: 0,
        isUser: i === 0,
      });
    }
    return out;
  }

  private scheduleLobby() {
    this.phase = "lobby";
    this.phaseStarted = this.sim_now;
    this.nextJoinAt = this.sim_now + 200;
    this.joinIdx = 0;
  }

  private scheduleMatch() {
    this.phase = "match";
    this.phaseStarted = this.sim_now;
    this.nextPerfEventAt = this.sim_now + 200;
    this.nextS3ChunkAt = this.sim_now + 4500;
    this.distributeKickedOff = false;
    this.payoutsFiredAt = null;
    this.bufferedEvents = 0;
    // schedule ~14 eliminations across the match window
    this.elimSchedule = [];
    this.elimCursor = 0;
    const ELIM_COUNT = 14;
    for (let i = 0; i < ELIM_COUNT; i++) {
      const u = (i + 1) / (ELIM_COUNT + 1);
      // back-loaded: eliminations cluster late
      const t = u * u * MATCH_DURATION_MS * 0.95;
      this.elimSchedule.push({ at: this.sim_now + t, playerIdx: -1 });
    }
  }

  private scheduleDistribute() {
    this.phase = "distribute";
    this.phaseStarted = this.sim_now;
    this.distributeKickedOff = true;
    this.batchTransferAt = null;
    this.bulkWriteAt = null;
    this.payoutsFiredAt = null;
    this.coldStartBurstAt = this.sim_now;
    this.showColdStartBurst = true;
    this.appendLog({
      method: "POST", path: "/internal/distribute",
      status: 200, latencyMs: 1,
      detail: `session.end · ${this.bufferedEvents} evts`,
    });
  }

  private scheduleSummary() {
    this.phase = "summary";
    this.phaseStarted = this.sim_now;
    this.summaryEnteredAt = this.sim_now;
  }

  private transition(to: Phase) {
    if (to === "match") this.scheduleMatch();
    else if (to === "distribute") this.scheduleDistribute();
    else if (to === "summary") this.scheduleSummary();
    else if (to === "lobby") this.scheduleLobby();
  }

  private phaseDuration() {
    return this.phase === "lobby" ? LOBBY_DURATION_MS
         : this.phase === "match" ? MATCH_DURATION_MS
         : this.phase === "distribute" ? DISTRIBUTE_DURATION_MS
         : SUMMARY_AUTO_LOOP_MS;
  }

  /* -------- phase ticks -------- */

  private tickLobby() {
    // stagger 30 joins across the lobby window
    while (this.joinIdx < PLAYER_COUNT && this.sim_now >= this.nextJoinAt) {
      this.fireJoin(this.joinIdx);
      this.joinIdx += 1;
      this.nextJoinAt += 220;
    }
    // after all joined + a beat, lock the pool then move to match
    const allJoined = this.joinIdx >= PLAYER_COUNT;
    if (allJoined && !this.showPoolLockFlash && this.players.every((p) => p.joinedAt !== null && this.sim_now - (p.joinedAt ?? 0) > 350)) {
      this.showPoolLockFlash = true;
      this.poolLockFlashStarted = this.sim_now;
      this.prizePoolGlowUntil = this.sim_now + 1200;
    }
    if (allJoined && this.sim_now - this.phaseStarted > LOBBY_DURATION_MS) {
      this.transition("match");
    }
  }

  private tickMatch() {
    const phaseElapsed = this.sim_now - this.phaseStarted;
    // stream performance events at increasing rate (heat-up)
    const intensity = clamp01(phaseElapsed / MATCH_DURATION_MS);
    const perfPeriodMs = Math.max(80, 380 - intensity * 280);
    while (this.sim_now >= this.nextPerfEventAt) {
      this.firePerfEvent();
      this.nextPerfEventAt += perfPeriodMs;
    }
    // S3 chunk every ~5s of sim time
    while (this.sim_now >= this.nextS3ChunkAt) {
      this.fireS3Chunk();
      this.nextS3ChunkAt += 5000;
    }
    // eliminations
    while (this.elimCursor < this.elimSchedule.length && this.sim_now >= this.elimSchedule[this.elimCursor].at) {
      this.fireElimination();
      this.elimCursor += 1;
    }
    // end of match → distribute
    if (phaseElapsed >= MATCH_DURATION_MS) {
      this.transition("distribute");
    }
  }

  private tickDistribute() {
    const elapsed = this.sim_now - this.phaseStarted;
    const eventCount = this.bufferedEvents;

    // Beat 1 (400..1400ms): GameLift session ends → bulk match snapshot
    // ships to EC2. Single large "batch" particle is the key architectural
    // moment — the only time match data touches our HTTP API tier.
    if (elapsed > 400 && this.batchTransferAt === null) {
      this.batchTransferAt = this.sim_now;
      this.spawnParticle({
        kind: "batch",
        fromKind: "gamelift", fromIdx: 0,
        toKind: "ec2", toIdx: 0,
        travelMs: 1000, color: "#f472b6",
        label: `match_summary · ${eventCount} evts`,
      });
      this.bumpHeat("gamelift", 1);
      this.appendLog({
        method: "POST", path: "/match/bulk-ingest",
        status: 200, latencyMs: Math.round(rand(110, 180)),
        detail: `${eventCount} events`,
      });
    }

    // Beat 2 (1400..2800ms): EC2 receives → pyd validate → postgres bulk
    // insert. One validation hop, one bulk write — not 2k of them.
    if (this.batchTransferAt !== null && this.bulkWriteAt === null && elapsed > 1450) {
      this.bulkWriteAt = this.sim_now;
      this.bumpHeat("ec2", 0.9);
      this.spawnParticle({
        kind: "valid",
        fromKind: "ec2", fromIdx: 0,
        toKind: "pydantic", toIdx: 0,
        travelMs: 400, color: "#34d399",
      });
      setTimeoutSim(this, 450, () => {
        this.bumpHeat("pydantic", 1);
        this.spawnParticle({
          kind: "write",
          fromKind: "pydantic", fromIdx: 0,
          toKind: "postgres", toIdx: 0,
          travelMs: 500, color: "#3b82f6",
          label: "INSERT events ×N",
        });
        this.appendLog({
          method: "PUT", path: "/db/match_events×N",
          status: 200, latencyMs: Math.round(rand(55, 95)),
          detail: `bulk ${eventCount}`,
        });
      });
      setTimeoutSim(this, 1000, () => {
        this.bumpHeat("postgres", 1);
      });
    }

    // Beat 3 (3200..4200ms): cold Lambda pulls the match dataset from pg.
    if (elapsed > 3200 && this.particles.every((p) => p.kind !== "snapshot") && this.payoutsFiredAt === null && elapsed < 3700) {
      this.spawnParticle({
        kind: "snapshot",
        fromKind: "postgres", fromIdx: 0,
        toKind: "lambda", toIdx: 0,
        travelMs: 900, color: "#3b82f6",
        label: "match dataset",
      });
      this.bumpHeat("postgres", 0.9);
      this.bumpHeat("lambda", 0.5);
      this.appendLog({ method: "GET", path: "/db/match.snapshot", status: 200, latencyMs: 28 });
    }

    // Beat 4 (~5000ms): compute payouts, fire 30 shards.
    if (elapsed > 4800 && this.payoutsFiredAt === null) {
      this.computePayouts();
      this.firePayoutShards();
      this.fireLedger();
      this.fireBalanceUpdate();
      this.payoutsFiredAt = this.sim_now;
    }

    // wrap → summary
    if (elapsed >= DISTRIBUTE_DURATION_MS) {
      this.transition("summary");
    }
  }

  private tickSummary() {
    const elapsed = this.sim_now - this.phaseStarted;
    if (this.autoLoop && elapsed >= SUMMARY_AUTO_LOOP_MS) {
      this.reset();
    }
  }

  /* -------- event spawners -------- */

  private fireJoin(idx: number) {
    const p = this.players[idx];
    p.joinedAt = this.sim_now;
    p.pulseUntil = this.sim_now + 600;

    // particle: player → ec2
    this.spawnParticle({
      kind: "join",
      fromKind: "player", fromIdx: idx,
      toKind: "ec2", toIdx: 0,
      travelMs: PARTICLE_TRAVEL_MS, color: "#22d3ee",
      label: `join +$${p.bet}`,
    });
    // chain: ec2 → pydantic → postgres after small delay
    setTimeoutSim(this, PARTICLE_TRAVEL_MS, () => {
      this.bumpHeat("ec2", 0.6);
      this.spawnParticle({
        kind: "valid", fromKind: "ec2", fromIdx: 0,
        toKind: "pydantic", toIdx: 0,
        travelMs: 350, color: "#34d399",
      });
    });
    setTimeoutSim(this, PARTICLE_TRAVEL_MS + 400, () => {
      this.bumpHeat("pydantic", 0.7);
      this.spawnParticle({
        kind: "write", fromKind: "pydantic", fromIdx: 0,
        toKind: "postgres", toIdx: 0,
        travelMs: 500, color: "#3b82f6",
      });
    });
    setTimeoutSim(this, PARTICLE_TRAVEL_MS + 900, () => {
      this.bumpHeat("postgres", 0.6);
      this.prizePool += p.bet;
      this.appendLog({
        method: "POST", path: "/match/join",
        status: 200, latencyMs: Math.round(rand(8, 16)),
        detail: `${p.name} +$${p.bet}`,
      });
      this.completionsForRps.push(this.sim_now);
    });
  }

  private firePerfEvent() {
    // Match-phase telemetry flows player → GameLift session server.
    // No HTTP API traffic during a live match — events buffer in the fleet
    // and drain to EC2 as a single bulk transfer when the session ends.
    const alive = this.players.filter((p) => p.alive);
    if (alive.length === 0) return;
    const player = alive[Math.floor(Math.random() * alive.length)];
    if (Math.random() < 0.18) player.kills += 1;
    player.damage += Math.round(rand(20, 95));
    player.pulseUntil = this.sim_now + 350;

    this.spawnParticle({
      kind: "perf",
      fromKind: "player", fromIdx: player.id,
      toKind: "gamelift", toIdx: 0,
      travelMs: PARTICLE_TRAVEL_MS * 0.7, color: "#f472b6",
    });
    setTimeoutSim(this, 600, () => {
      this.bumpHeat("gamelift", 0.35);
      this.bufferedEvents += 1;
      this.appendLog({
        method: "GL", path: "session.telemetry",
        status: 200, latencyMs: Math.round(rand(1, 4)),
        detail: `${player.name} dmg+`,
      });
      this.completionsForRps.push(this.sim_now);
    });
  }

  private fireS3Chunk() {
    // GameLift fleet periodically offloads replay chunks directly to S3 —
    // keeps session memory bounded and gives us a replay trail independent
    // of the backend DB.
    this.spawnParticle({
      kind: "s3chunk",
      fromKind: "gamelift", fromIdx: 0,
      toKind: "s3", toIdx: 0,
      travelMs: 700, color: "#fbbf24",
      label: "replay.chunk",
    });
    setTimeoutSim(this, 750, () => {
      this.bumpHeat("s3", 0.7);
      this.bumpHeat("gamelift", 0.25);
      this.s3Objects.push({
        id: ++this.s3Counter,
        createdAt: this.sim_now,
        kind: "chunk",
        label: `replay_chunk_${(this.sim_now / 1000).toFixed(0)}.json`,
      });
      this.appendLog({
        method: "PUT", path: "/s3/match.replay",
        status: 200, latencyMs: Math.round(rand(35, 60)),
      });
      this.completionsForRps.push(this.sim_now);
    });
  }

  private fireElimination() {
    const alive = this.players.filter((p) => p.alive);
    if (alive.length <= 2) return;
    // pick the lowest-performing alive (with some randomness so it's not predictable)
    alive.sort((a, b) => (a.kills * 100 + a.damage) - (b.kills * 100 + b.damage));
    const candidatePool = alive.slice(0, Math.max(3, Math.floor(alive.length * 0.4)));
    const target = candidatePool[Math.floor(Math.random() * candidatePool.length)];
    target.alive = false;
    target.eliminatedAt = this.sim_now;
    target.flashColor = "#f87171";
    target.flashUntil = this.sim_now + 700;

    // Elimination is just another session event during live play — it
    // buffers in GameLift alongside perf telemetry until the match ends.
    this.spawnParticle({
      kind: "elim",
      fromKind: "player", fromIdx: target.id,
      toKind: "gamelift", toIdx: 0,
      travelMs: PARTICLE_TRAVEL_MS * 0.7, color: "#f87171",
    });
    setTimeoutSim(this, 600, () => {
      this.bumpHeat("gamelift", 0.6);
      this.bufferedEvents += 1;
      this.appendLog({
        method: "GL", path: "session.elimination",
        status: 200, latencyMs: Math.round(rand(2, 5)),
        detail: `${target.name} OUT`,
      });
      this.completionsForRps.push(this.sim_now);
    });
  }

  private computePayouts() {
    // assign final score: kills*100 + damage + survival(s)*5
    const survivors = this.players.filter((p) => p.alive);
    // survivors get a fat survival bonus
    for (const p of this.players) {
      const survSec = p.alive ? MATCH_DURATION_MS / 1000 : (p.survivalMs / 1000);
      const survBonus = p.alive ? 250 : 0;
      p.finalScore = p.kills * 100 + p.damage + survSec * 5 + survBonus;
    }
    const total = this.players.reduce((s, p) => s + p.finalScore, 0) || 1;
    for (const p of this.players) {
      p.payout = (p.finalScore / total) * this.prizePool;
    }
    // assign ranks
    const sorted = [...this.players].sort((a, b) => b.finalScore - a.finalScore);
    sorted.forEach((p, i) => { p.rank = i + 1; });
  }

  private firePayoutShards() {
    // 30 shards lambda → each player.
    // In production: lambda publishes to an SNS topic `match.ended`; SQS
    // queues (wallet, notify, ledger) fan out in parallel — each with its
    // own retry policy + DLQ. Here we narrate that via the log/fleet HUD
    // and keep the scene a direct render so the causal chain stays legible.
    this.bumpHeat("lambda", 1);
    this.fleet.sqsBacklog += 60; // 30 wallet + 30 notify, briefly
    this.appendLog({
      method: "POST", path: "/sns/match.ended",
      status: 200, latencyMs: 4,
      detail: "fan-out → wallet+notify+ledger",
    });
    for (let i = 0; i < this.players.length; i++) {
      const p = this.players[i];
      const offset = i * 25;
      setTimeoutSim(this, offset, () => {
        this.spawnParticle({
          kind: "payout",
          fromKind: "lambda", fromIdx: 0,
          toKind: "player", toIdx: i,
          travelMs: 900 + i * 12,
          color: "#fbbf24",
          amount: p.payout,
        });
      });
      setTimeoutSim(this, offset + 920 + i * 12, () => {
        p.payoutFlashUntil = this.sim_now + 1100;
        p.flashColor = "#34d399";
        p.flashUntil = this.sim_now + 900;
      });
    }
  }

  private fireLedger() {
    setTimeoutSim(this, 200, () => {
      this.spawnParticle({
        kind: "ledger",
        fromKind: "lambda", fromIdx: 0,
        toKind: "s3", toIdx: 0,
        travelMs: 700, color: "#fbbf24",
        label: "ledger.json",
      });
    });
    setTimeoutSim(this, 950, () => {
      this.bumpHeat("s3", 1);
      this.s3Objects.push({
        id: ++this.s3Counter,
        createdAt: this.sim_now,
        kind: "ledger",
        label: "match.payout_ledger.json",
      });
      this.appendLog({
        method: "PUT", path: "/s3/match.payouts",
        status: 200, latencyMs: 42,
      });
      this.completionsForRps.push(this.sim_now);
    });
  }

  private fireBalanceUpdate() {
    setTimeoutSim(this, 400, () => {
      this.spawnParticle({
        kind: "balance",
        fromKind: "lambda", fromIdx: 0,
        toKind: "postgres", toIdx: 0,
        travelMs: 700, color: "#3b82f6",
        label: "UPDATE balances ×30",
      });
    });
    setTimeoutSim(this, 1150, () => {
      this.bumpHeat("postgres", 1);
      this.appendLog({
        method: "PUT", path: "/db/users.balance×30", status: 200, latencyMs: 38,
      });
      this.completionsForRps.push(this.sim_now);
    });
  }

  /* -------- helpers -------- */

  private pushBanner(b: Omit<ChaosBanner, "id" | "startedAt">) {
    this.chaosBanners.push({
      ...b,
      id: ++this.bannerCounter,
      startedAt: this.sim_now,
    });
  }

  /** Platform fleet drift + autoscale + chaos recovery beats.
   *  Runs every step — feeds the "platform HUD" at the top of the canvas.
   *  This is what makes one match feel like a drop in a much bigger bucket. */
  private tickFleet(dt: number) {
    this.fleetTickAccum += dt;
    // update at ~4 Hz
    if (this.fleetTickAccum < 250) {
      this.handleChaosRecovery();
      return;
    }
    this.fleetTickAccum = 0;

    // baseline drift — concurrentMatches breathes between 40..55 to feel alive.
    const t = this.sim_now / 1000;
    const wave = Math.sin(t * 0.11) * 4 + Math.cos(t * 0.23) * 2;
    this.fleet.concurrentMatches = Math.max(30, Math.round(47 + wave + rand(-1, 1)));

    // ALB rps breathes with concurrent matches
    this.fleet.albRps = Math.round(
      this.fleet.concurrentMatches * rand(48, 56) + rand(-80, 120),
    );

    // SQS backlog drifts — drain rate > produce rate in steady state
    this.fleet.sqsBacklog = Math.max(
      0,
      Math.round(this.fleet.sqsBacklog * 0.92 + rand(0, 4)),
    );

    // Autoscale — if backlog or albRps spikes, bump desired.
    const rpsPerPod = this.fleet.albRps / Math.max(1, this.fleet.ec2Pods);
    if (rpsPerPod > 360 || this.fleet.sqsBacklog > 60) {
      this.fleet.ec2PodsDesired = Math.min(16, this.fleet.ec2PodsDesired + 1);
    } else if (rpsPerPod < 180 && this.fleet.ec2PodsDesired > 6) {
      this.fleet.ec2PodsDesired = Math.max(6, this.fleet.ec2PodsDesired - 1);
    }
    // pods converge toward desired (slowly — ASG takes time)
    if (this.fleet.ec2Pods < this.fleet.ec2PodsDesired && Math.random() < 0.35) {
      this.fleet.ec2Pods += 1;
    } else if (this.fleet.ec2Pods > this.fleet.ec2PodsDesired && Math.random() < 0.25) {
      this.fleet.ec2Pods -= 1;
    }

    // DLQ decays very slowly (operator drains it)
    if (this.fleet.sqsDlq > 0 && Math.random() < 0.08) {
      this.fleet.sqsDlq -= 1;
    }

    this.handleChaosRecovery();
  }

  /** Fires the recovery beats once the chaos timers elapse. */
  private handleChaosRecovery() {
    // GameLift: S3 replay → fleet restore particle
    if (this.gameliftRecoveryAt !== null && this.sim_now >= this.gameliftRecoveryAt) {
      this.gameliftRecoveryAt = null;
      this.spawnParticle({
        kind: "s3chunk", // reuse visual — yellow wire
        fromKind: "s3", fromIdx: 0,
        toKind: "gamelift", toIdx: 0,
        travelMs: 900, color: "#fbbf24",
        label: "replay restore",
      });
      this.bumpHeat("s3", 0.8);
      setTimeoutSim(this, 950, () => {
        this.bumpHeat("gamelift", 0.9);
        this.pushBanner({
          kind: "gamelift-drop",
          headline: "SESSION MIGRATED · fleet-05",
          detail: "replay restored · no data loss · match resuming",
          durationMs: 3500,
          tone: "good",
        });
        this.appendLog({
          method: "GL", path: "fleet-05.resume",
          status: 200, latencyMs: 142,
          detail: "state restored",
        });
      });
    }

    // Lambda: retry with idempotency key succeeds
    if (this.lambdaRetryAt !== null && this.sim_now >= this.lambdaRetryAt) {
      this.lambdaRetryAt = null;
      this.pushBanner({
        kind: "lambda-timeout",
        headline: "RETRY OK · idempotency honored",
        detail: "payout_match_7f3 re-read from DLQ · no double-pay",
        durationMs: 3500,
        tone: "good",
      });
      this.appendLog({
        method: "POST", path: "/lambda/distribute",
        status: 200, latencyMs: Math.round(rand(180, 240)),
        detail: "retry · idemp_key match",
      });
      this.fleet.sqsDlq = Math.max(0, this.fleet.sqsDlq - 1);
      this.bumpHeat("lambda", 0.9);
    }

    // EC2: ASG spawns replacement
    if (this.ec2RespawnAt !== null && this.sim_now >= this.ec2RespawnAt) {
      this.ec2RespawnAt = null;
      this.ec2FailAt = null;
      this.fleet.ec2Pods += 1;
      this.pushBanner({
        kind: "ec2-fail",
        headline: "EC2 REPLACEMENT HEALTHY",
        detail: `ASG scaled back to ${this.fleet.ec2Pods} · ALB re-attached`,
        durationMs: 3500,
        tone: "good",
      });
      this.appendLog({
        method: "GET", path: "/health",
        status: 200, latencyMs: 4,
        detail: "new target healthy",
      });
      this.bumpHeat("ec2", 0.6);
    }
  }

  private spawnParticle(p: Omit<Particle, "id" | "startedAt"> & Partial<Pick<Particle, "startedAt">>) {
    this.particles.push({
      ...p,
      id: ++this.particleCounter,
      startedAt: this.sim_now,
    });
  }

  private bumpHeat(id: IslandId, amount: number) {
    this.islandHeat[id] = Math.min(1, this.islandHeat[id] + amount);
  }

  private appendLog(entry: Omit<LogEntry, "id" | "ts">) {
    this.log.unshift({ ...entry, id: ++this.logCounter, ts: this.sim_now });
    if (this.log.length > LOG_LIMIT) this.log.length = LOG_LIMIT;
  }
}

/* -------- micro setTimeout-equivalent driven by sim time -------- */

interface Pending { fireAt: number; fn: () => void }
const pending = new WeakMap<MatchSim, Pending[]>();

function setTimeoutSim(sim: MatchSim, delayMs: number, fn: () => void) {
  let arr = pending.get(sim);
  if (!arr) { arr = []; pending.set(sim, arr); }
  // accessing private field via cast — small concession for cleanliness
  const now = (sim as unknown as { sim_now: number }).sim_now;
  arr.push({ fireAt: now + delayMs, fn });
}

// patch step() to drain pending callbacks
const origStep = MatchSim.prototype.step;
MatchSim.prototype.step = function (dtMs: number) {
  origStep.call(this, dtMs);
  const arr = pending.get(this);
  if (!arr || arr.length === 0) return;
  const now = (this as unknown as { sim_now: number }).sim_now;
  let writeIdx = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i].fireAt <= now) {
      try { arr[i].fn(); } catch { /* swallow */ }
    } else {
      arr[writeIdx++] = arr[i];
    }
  }
  arr.length = writeIdx;
};

/* -------- island display metadata -------- */

export const ISLAND_META: Record<IslandId, { label: string; sub: string; color: string }> = {
  gamelift: { label: "AWS GameLift",    sub: "session telemetry", color: "#f472b6" },
  ec2:      { label: "EC2 · FastAPI",   sub: "uvicorn / async",   color: "#fb923c" },
  pydantic: { label: "Pydantic",        sub: "schema validate",   color: "#34d399" },
  postgres: { label: "RDS Postgres",    sub: "SQLAlchemy ORM",    color: "#3b82f6" },
  lambda:   { label: "Lambda Cluster",  sub: "rank() · payout()", color: "#fbbf24" },
  s3:       { label: "S3 Vault",        sub: "replays · ledger",  color: "#a78bfa" },
};

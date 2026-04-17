/* ------------------------------------------------------------------
   POST /api/stats/trainings — atomically increments the global
   pendulum training counter and tracks the best (minimum)
   convergence time. Called once per visitor when the DQN agent
   converges. Client-side localStorage dedupe prevents a single
   browser from spamming the endpoint.

   Body (optional JSON):
     { "timeSeconds": 342 }   — convergence wall-clock time.
     If provided and lower than the stored best, updates it.

   Env vars:
     REDIS_URL — standard redis:// connection string. Auto-injected
     when you attach Redis Cloud via the Vercel Marketplace. Pull
     to local with `npx vercel env pull .env.development.local`.

   In local dev without REDIS_URL, POST returns { trainings: 0 }
   so the UI still works.
------------------------------------------------------------------ */

import { NextRequest, NextResponse } from "next/server";
import { getRedis } from "@/lib/redis";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TRAININGS_KEY = "pendulum:trainings";
const BEST_TIME_KEY = "pendulum:best_time";

export async function POST(req: NextRequest) {
  const redis = await getRedis();
  if (!redis) {
    return NextResponse.json({ trainings: 0, bestTime: null, kv: false });
  }
  try {
    let timeSeconds: number | null = null;
    try {
      const body = await req.json();
      if (typeof body.timeSeconds === "number" && body.timeSeconds > 0) {
        timeSeconds = Math.round(body.timeSeconds);
      }
    } catch { /* no body or invalid JSON — that's fine */ }

    const next = await redis.incr(TRAININGS_KEY);

    let bestTime: number | null = null;
    if (timeSeconds !== null) {
      const current = await redis.get(BEST_TIME_KEY);
      const currentBest = current !== null ? Number(current) : Infinity;
      if (timeSeconds < currentBest) {
        await redis.set(BEST_TIME_KEY, String(timeSeconds));
        bestTime = timeSeconds;
      } else {
        bestTime = Number.isFinite(currentBest) ? currentBest : null;
      }
    } else {
      const v = await redis.get(BEST_TIME_KEY);
      bestTime = v !== null && Number.isFinite(Number(v)) ? Number(v) : null;
    }

    return NextResponse.json({ trainings: next, bestTime, kv: true });
  } catch (err) {
    console.error("trainings counter increment failed:", err);
    return NextResponse.json(
      { trainings: 0, bestTime: null, kv: false, error: "increment failed" },
      { status: 500 },
    );
  }
}

export async function GET() {
  const redis = await getRedis();
  if (!redis) {
    return NextResponse.json({ trainings: 0 });
  }
  try {
    const v = await redis.get(TRAININGS_KEY);
    const n = Number(v);
    return NextResponse.json({ trainings: Number.isFinite(n) ? n : 0 });
  } catch {
    return NextResponse.json({ trainings: 0 });
  }
}

/* ------------------------------------------------------------------
   POST /api/stats/trainings — atomically increments the global
   pendulum training counter. Called once per visitor when the
   behavioral-cloning agent converges. Client-side localStorage
   dedupe prevents a single browser from spamming the endpoint.

   Env vars:
     REDIS_URL — standard redis:// connection string. Auto-injected
     when you attach Redis Cloud via the Vercel Marketplace. Pull
     to local with `npx vercel env pull .env.development.local`.

   In local dev without REDIS_URL, POST returns { trainings: 0 }
   so the UI still works.
------------------------------------------------------------------ */

import { NextResponse } from "next/server";
import { getRedis } from "@/lib/redis";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TRAININGS_KEY = "pendulum:trainings";

export async function POST() {
  const redis = await getRedis();
  if (!redis) {
    return NextResponse.json({ trainings: 0, kv: false });
  }
  try {
    const next = await redis.incr(TRAININGS_KEY);
    return NextResponse.json({ trainings: next, kv: true });
  } catch (err) {
    console.error("trainings counter increment failed:", err);
    return NextResponse.json(
      { trainings: 0, kv: false, error: "increment failed" },
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

/* ------------------------------------------------------------------
   POST /api/stats/trainings — atomically increments the global
   pendulum training counter. Called once per visitor when the
   behavioral-cloning agent converges. Client-side localStorage
   dedupe prevents a single browser from spamming the endpoint.

   Env vars (set in Vercel → Project → Settings → Environment
   Variables, or auto-injected when you attach an Upstash Redis
   integration via the Vercel Marketplace):
     UPSTASH_REDIS_REST_URL
     UPSTASH_REDIS_REST_TOKEN

   In local dev without these env vars, POST returns { trainings: 0 }
   so the UI still works.
------------------------------------------------------------------ */

import { NextResponse } from "next/server";
import { Redis } from "@upstash/redis";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TRAININGS_KEY = "pendulum:trainings";

export async function POST() {
  if (!process.env.UPSTASH_REDIS_REST_URL || !process.env.UPSTASH_REDIS_REST_TOKEN) {
    return NextResponse.json({ trainings: 0, kv: false });
  }
  try {
    const redis = Redis.fromEnv();
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
  if (!process.env.UPSTASH_REDIS_REST_URL || !process.env.UPSTASH_REDIS_REST_TOKEN) {
    return NextResponse.json({ trainings: 0 });
  }
  try {
    const redis = Redis.fromEnv();
    const v = await redis.get<number>(TRAININGS_KEY);
    return NextResponse.json({ trainings: typeof v === "number" ? v : 0 });
  } catch {
    return NextResponse.json({ trainings: 0 });
  }
}

/* ------------------------------------------------------------------
   GET /api/stats — consolidated site stats for the hero rotation
   and the Publications section stats strip.

   Data sources:
     - publications, citations, h-index, i10-index, reading:
       lib/siteStats.json (single source of truth, edit + push).
       Google Scholar has no public API, so these are maintained
       manually — update whenever the numbers meaningfully change.
     - trainings:  global counter in Redis, incremented by
       /api/stats/trainings.

   Env vars (only needed for the trainings counter):
     REDIS_URL — auto-injected when you attach Redis Cloud via
     the Vercel Marketplace. For local dev, pull with
     `npx vercel env pull .env.development.local`.
------------------------------------------------------------------ */

import { NextResponse } from "next/server";
import { getRedis } from "@/lib/redis";
import siteStatsJson from "@/lib/siteStats.json";
import type { SiteStats } from "@/lib/siteStats";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TRAININGS_KEY = "pendulum:trainings";
const BEST_TIME_KEY = "pendulum:best_time";

async function readTrainings(): Promise<number> {
  const redis = await getRedis();
  if (!redis) return 0;
  try {
    const v = await redis.get(TRAININGS_KEY);
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

async function readBestTime(): Promise<number | null> {
  const redis = await getRedis();
  if (!redis) return null;
  try {
    const v = await redis.get(BEST_TIME_KEY);
    if (v === null) return null;
    const n = Number(v);
    return Number.isFinite(n) && n > 0 ? n : null;
  } catch {
    return null;
  }
}

export async function GET() {
  const [trainings, bestTime] = await Promise.all([readTrainings(), readBestTime()]);

  const body: SiteStats = {
    publications: siteStatsJson.publications,
    citations: siteStatsJson.citations,
    hIndex: siteStatsJson.hIndex,
    i10Index: siteStatsJson.i10Index,
    trainings,
    bestTime,
    reading: {
      title: siteStatsJson.reading.title,
      authors: siteStatsJson.reading.authors,
      link: siteStatsJson.reading.link,
    },
  };

  return NextResponse.json(body, {
    headers: {
      "Cache-Control": "public, s-maxage=60, stale-while-revalidate=300",
    },
  });
}

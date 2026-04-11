/* ------------------------------------------------------------------
   GET /api/stats — consolidated site stats for the hero rotation
   and the Publications section stats strip.

   Data sources:
     - publications, citations, h-index, i10-index, reading:
       lib/siteStats.json (single source of truth, edit + push).
       Google Scholar has no public API, so these are maintained
       manually — update whenever the numbers meaningfully change.
     - trainings:  global counter in Upstash Redis, incremented by
       /api/stats/trainings.

   Env vars (only needed for the trainings counter):
     UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN — auto-set
     when you attach Upstash Redis via the Vercel Marketplace.
------------------------------------------------------------------ */

import { NextResponse } from "next/server";
import { Redis } from "@upstash/redis";
import siteStatsJson from "@/lib/siteStats.json";
import type { SiteStats } from "@/lib/siteStats";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TRAININGS_KEY = "pendulum:trainings";

async function readTrainings(): Promise<number> {
  if (!process.env.UPSTASH_REDIS_REST_URL || !process.env.UPSTASH_REDIS_REST_TOKEN) {
    return 0;
  }
  try {
    const redis = Redis.fromEnv();
    const v = await redis.get<number>(TRAININGS_KEY);
    return typeof v === "number" ? v : 0;
  } catch {
    return 0;
  }
}

export async function GET() {
  const trainings = await readTrainings();

  const body: SiteStats = {
    publications: siteStatsJson.publications,
    citations: siteStatsJson.citations,
    hIndex: siteStatsJson.hIndex,
    i10Index: siteStatsJson.i10Index,
    trainings,
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

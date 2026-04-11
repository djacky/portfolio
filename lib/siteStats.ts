/* ------------------------------------------------------------------ *
 *  Site stats — shared types and helpers for the hero live counters.   *
 *                                                                      *
 *  publications, citations, hIndex, i10Index:  static JSON, edit       *
 *    lib/siteStats.json and push to update.                            *
 *  currentReading:  lib/currentReading.json                            *
 *  trainings:       global counter backed by Upstash Redis. See        *
 *    app/api/stats/trainings/route.ts for the KV wiring.               *
 * ------------------------------------------------------------------ */

export type SiteStats = {
  publications: number;
  citations: number;
  hIndex: number;
  i10Index: number;
  trainings: number;
  reading: {
    title: string;
    authors: string;
    link: string;
  };
};

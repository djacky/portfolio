/* ------------------------------------------------------------------
   Request-side helpers shared by /api/contact and /api/match.
------------------------------------------------------------------ */

import type { NextRequest } from "next/server";

/** IPv4 (dotted quad) or IPv6 — loose but sufficient to reject obviously
 *  malformed `x-forwarded-for` values that an attacker could inject to
 *  fragment per-IP rate-limit buckets. */
const IP_LIKE = /^(?:[\da-fA-F:.]{2,45})$/;

/**
 * Best-effort client IP for rate limiting. Trusts only the first
 * `x-forwarded-for` hop (the upstream proxy's view of the client) and
 * validates it shapes like an IP. Long XFF chains are common from open
 * proxies and unreliable, so we cap at 4 hops before discarding the
 * header altogether.
 *
 * Falls back to `x-real-ip`, then a constant "anonymous" bucket — the
 * latter rate-limits unknown-origin traffic together rather than letting
 * each missing-header request slip through.
 */
export function clientIp(req: NextRequest): string {
  const xff = req.headers.get("x-forwarded-for");
  if (xff) {
    const hops = xff.split(",");
    if (hops.length <= 4) {
      const first = hops[0]?.trim() ?? "";
      if (first && IP_LIKE.test(first)) return first;
    }
  }
  const real = req.headers.get("x-real-ip")?.trim();
  if (real && IP_LIKE.test(real)) return real;
  return "anonymous";
}

/** Common prompt-injection markers. Conservative — these phrases very
 *  rarely appear in legitimate job descriptions, so a hit strongly
 *  suggests adversarial intent. The match endpoint already wraps JD
 *  text in delimiters and forces structured tool output, so this is
 *  defense-in-depth rather than the primary control. */
const INJECTION_PATTERNS: RegExp[] = [
  /ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts?)/i,
  /disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts?)/i,
  /(?:reveal|print|output|show|repeat)\s+(?:your|the)\s+(?:system|initial)\s+prompt/i,
  /you\s+are\s+now\s+(?:a|an)\s+/i,
  /\bact\s+as\s+(?:if\s+)?(?:you|a|an)\b/i,
  /<\/?\s*(?:system|assistant|instructions?)\s*>/i,
  /\[\s*end\s+of\s+(?:job|description|jd)\s*\]/i,
];

export function looksLikePromptInjection(s: string): boolean {
  for (const re of INJECTION_PATTERNS) {
    if (re.test(s)) return true;
  }
  return false;
}

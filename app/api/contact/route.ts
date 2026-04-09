/* ------------------------------------------------------------------
   POST /api/contact — send a contact-form message via Resend.

   Environment variables:
     RESEND_API_KEY    — required. From https://resend.com/api-keys
     CONTACT_TO_EMAIL  — required. The address that receives messages.
     CONTACT_FROM_EMAIL — optional. Verified sender. Defaults to
       "Portfolio <onboarding@resend.dev>" which works out of the box
       on Resend's sandbox domain but can only send to the account
       owner's email. Swap to a verified domain for production.

   Rate limit: in-memory token bucket, same pattern as /api/match.
------------------------------------------------------------------ */

import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_NAME = 120;
const MAX_EMAIL = 200;
const MAX_SUBJECT = 200;
const MAX_MESSAGE = 4000;
const MIN_MESSAGE = 20;
const RATE_LIMIT_PER_HOUR = 5;

/* ---------------- in-memory rate limiter ---------------- */

type Bucket = { windowStart: number; count: number };
const buckets = new Map<string, Bucket>();
const WINDOW_MS = 60 * 60 * 1000;

function rateLimit(ip: string): { ok: boolean; reason?: string } {
  const now = Date.now();
  const b = buckets.get(ip) ?? { windowStart: now, count: 0 };
  if (now - b.windowStart > WINDOW_MS) {
    b.windowStart = now;
    b.count = 0;
  }
  if (b.count >= RATE_LIMIT_PER_HOUR) {
    return {
      ok: false,
      reason: "Too many messages in the last hour. Please try again later.",
    };
  }
  b.count += 1;
  buckets.set(ip, b);
  return { ok: true };
}

/* ---------------- handler ---------------- */

const TOPIC_LABELS: Record<string, string> = {
  role: "Role opportunity",
  collaboration: "Collaboration",
  consulting: "Consulting",
  general: "General",
};

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export async function POST(req: NextRequest) {
  const apiKey = process.env.RESEND_API_KEY;
  const toEmail = process.env.CONTACT_TO_EMAIL;
  const fromEmail =
    process.env.CONTACT_FROM_EMAIL ?? "Portfolio <onboarding@resend.dev>";

  if (!apiKey || !toEmail) {
    return NextResponse.json(
      {
        error:
          "Server misconfigured: RESEND_API_KEY or CONTACT_TO_EMAIL is not set.",
      },
      { status: 500 },
    );
  }

  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0].trim() ??
    req.headers.get("x-real-ip") ??
    "anonymous";
  const limit = rateLimit(ip);
  if (!limit.ok) {
    return NextResponse.json({ error: limit.reason }, { status: 429 });
  }

  let body: {
    name?: unknown;
    email?: unknown;
    topic?: unknown;
    message?: unknown;
    honeypot?: unknown;
  };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  // Simple honeypot — bots auto-fill hidden fields.
  if (typeof body.honeypot === "string" && body.honeypot.length > 0) {
    return NextResponse.json({ ok: true }, { status: 200 });
  }

  const name = typeof body.name === "string" ? body.name.trim() : "";
  const email = typeof body.email === "string" ? body.email.trim() : "";
  const topicRaw = typeof body.topic === "string" ? body.topic : "general";
  const topic = TOPIC_LABELS[topicRaw] ?? TOPIC_LABELS.general;
  const message = typeof body.message === "string" ? body.message.trim() : "";

  if (!name || name.length > MAX_NAME) {
    return NextResponse.json(
      { error: "Please provide your name." },
      { status: 400 },
    );
  }
  // Simple email shape check — we don't need a full RFC parser.
  const emailOk = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  if (!emailOk || email.length > MAX_EMAIL) {
    return NextResponse.json(
      { error: "Please provide a valid email address." },
      { status: 400 },
    );
  }
  if (message.length < MIN_MESSAGE) {
    return NextResponse.json(
      {
        error: `Message is a bit short — please write at least ${MIN_MESSAGE} characters.`,
      },
      { status: 400 },
    );
  }
  if (message.length > MAX_MESSAGE) {
    return NextResponse.json(
      { error: `Message is too long (max ${MAX_MESSAGE} characters).` },
      { status: 400 },
    );
  }

  const subject = `[Portfolio · ${topic}] ${name}`.slice(0, MAX_SUBJECT);

  const textBody = [
    `New portfolio contact form message`,
    ``,
    `Name:    ${name}`,
    `Email:   ${email}`,
    `Topic:   ${topic}`,
    ``,
    `Message:`,
    message,
  ].join("\n");

  const htmlBody = `
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #111;">
      <h2 style="margin: 0 0 16px 0; font-size: 18px;">New portfolio contact</h2>
      <table style="border-collapse: collapse; margin-bottom: 16px; font-size: 14px;">
        <tr><td style="padding: 4px 12px 4px 0; color: #666;">Name</td><td>${escapeHtml(name)}</td></tr>
        <tr><td style="padding: 4px 12px 4px 0; color: #666;">Email</td><td><a href="mailto:${escapeHtml(email)}">${escapeHtml(email)}</a></td></tr>
        <tr><td style="padding: 4px 12px 4px 0; color: #666;">Topic</td><td>${escapeHtml(topic)}</td></tr>
      </table>
      <div style="white-space: pre-wrap; padding: 16px; background: #f6f6f6; border-radius: 8px; font-size: 14px; line-height: 1.55;">${escapeHtml(message)}</div>
    </div>
  `.trim();

  try {
    const upstream = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        from: fromEmail,
        to: [toEmail],
        reply_to: email,
        subject,
        text: textBody,
        html: htmlBody,
      }),
    });

    if (!upstream.ok) {
      const text = await upstream.text().catch(() => "");
      console.error("[/api/contact] resend error:", upstream.status, text);
      return NextResponse.json(
        { error: "Could not send your message. Please try again shortly." },
        { status: 502 },
      );
    }

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (err) {
    console.error("[/api/contact] fetch failed:", err);
    return NextResponse.json(
      { error: "Could not reach the email service. Try again in a moment." },
      { status: 502 },
    );
  }
}

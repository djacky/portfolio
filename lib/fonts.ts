/* ------------------------------------------------------------------ *
 *  Site font theme — single source of truth.                           *
 *                                                                      *
 *  To preview a different font across every page, swap the two         *
 *  marked lines below. Examples:                                       *
 *                                                                      *
 *    import { Inter } from "next/font/google";                         *
 *    export const siteFont = Inter({ ... });                           *
 *                                                                      *
 *    import { JetBrains_Mono } from "next/font/google";                *
 *    export const siteFont = JetBrains_Mono({ ... });                  *
 *                                                                      *
 *    import { Space_Grotesk } from "next/font/google";                 *
 *    export const siteFont = Space_Grotesk({ ... });                   *
 *                                                                      *
 *  Everything downstream — Tailwind classes (font-sans/font-mono),     *
 *  inline fontFamily in R3F <Html> panels — resolves through the       *
 *  CSS variable and FONT_FAMILY constant, so you only edit this file.  *
 * ------------------------------------------------------------------ */

// ────── swap these two lines to change the site font ──────
import { Martian_Mono } from "next/font/google";
export const siteFont = Martian_Mono({
  subsets: ["latin"],
  variable: "--font-site",
  display: "swap",
});
// ───────────────────────────────────────────────────────────

/** fontFamily string for inline style={{ fontFamily: FONT_FAMILY }} usage. */
export const FONT_FAMILY =
  "var(--font-site), ui-monospace, SFMono-Regular, Menlo, monospace";

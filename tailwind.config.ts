import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#05070d",
        panel: "#0b0f1a",
        panel2: "#111726",
        line: "#1f2937",
        accent: "#7c5cff",
        accent2: "#22d3ee",
        good: "#34d399",
        warn: "#fbbf24",
        bad: "#f87171",
      },
      fontFamily: {
        display: ["var(--font-site)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        body: ["var(--font-site)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        sans: ["var(--font-site)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        mono: ["var(--font-site)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        glow: "0 0 40px -10px rgba(124,92,255,0.45)",
      },
      keyframes: {
        float: { "0%,100%": { transform: "translateY(0)" }, "50%": { transform: "translateY(-6px)" } },
        shimmer: { "0%": { backgroundPosition: "-200% 0" }, "100%": { backgroundPosition: "200% 0" } },
        "ticker-scroll": { from: { transform: "translateX(0)" }, to: { transform: "translateX(-33.333%)" } },
      },
      animation: {
        float: "float 6s ease-in-out infinite",
        shimmer: "shimmer 3s linear infinite",
        "ticker-scroll": "ticker-scroll 30s linear infinite",
      },
    },
  },
  plugins: [],
};
export default config;

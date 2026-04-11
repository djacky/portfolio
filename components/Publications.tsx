"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ExternalLink, BookOpen } from "lucide-react";
import Typewriter from "./Typewriter";
import type { SiteStats } from "@/lib/siteStats";

const SCHOLAR_URL =
  "https://scholar.google.com/citations?user=Fes_eScAAAAJ&hl=en";

/* Initial placeholder values shown while /api/stats is loading.
   Replaced by live OpenAlex data once the fetch resolves. */
const INITIAL_STATS = {
  hIndex: 9,
  i10Index: 9,
  papers: 20,
  citations: 340,
};

type Paper = {
  title: string;
  authors: string;
  venue: string;
  year: number;
};

const SELECTED: Paper[] = [
  {
    title:
      "On active disturbance rejection based control design for superconducting RF cavities",
    authors:
      "J Vincent, D Morris, N Usher, Z Gao, S Zhao, A Nicoletti, Q Zheng",
    venue:
      "Nuclear Instruments and Methods in Physics Research Section A",
    year: 2011,
  },
  {
    title:
      "Robust H\u221E controller design using frequency-domain data via convex optimization",
    authors: "A Karimi, A Nicoletti, Y Zhu",
    venue:
      "International Journal of Robust and Nonlinear Control",
    year: 2018,
  },
  {
    title:
      "Robust Smith Predictor design for time-delay systems with H\u221E performance",
    authors: "V De Oliveira, A Nicoletti, A Karimi",
    venue: "Recent Results on Time-Delay Systems: Analysis and Control",
    year: 2016,
  },
  {
    title:
      "A data-driven approach to model-reference control with applications to particle accelerator power converters",
    authors: "A Nicoletti, M Martino, A Karimi",
    venue: "Control Engineering Practice",
    year: 2019,
  },
  {
    title:
      "A robust data-driven controller design methodology with applications to particle accelerator power converters",
    authors: "A Nicoletti, M Martino, A Karimi",
    venue: "IEEE Transactions on Control Systems Technology",
    year: 2018,
  },
  {
    title:
      "Data-driven approach to iterative learning control via convex optimisation",
    authors: "A Nicoletti, M Martino, D Aguglia",
    venue: "IET Control Theory & Applications",
    year: 2020,
  },
  {
    title:
      "A data-driven frequency-domain approach for robust controller design via convex optimization",
    authors: "A Nicoletti",
    venue: "PhD Thesis, EPFL",
    year: 2018,
  },
];

export default function Publications() {
  const [stats, setStats] = useState(INITIAL_STATS);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/stats");
        if (!res.ok) return;
        const data = (await res.json()) as SiteStats;
        if (cancelled) return;
        setStats({
          papers: data.publications,
          citations: data.citations,
          hIndex: data.hIndex,
          i10Index: data.i10Index,
        });
      } catch {
        /* keep placeholders */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section
      id="publications"
      className="relative mx-auto max-w-6xl px-6 py-24"
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <header className="mb-10">
          <Typewriter
            text="Research"
            as="p"
            speed={60}
            className="text-xs uppercase tracking-[0.25em] text-accent2"
          />
          <Typewriter
            text="Published work"
            as="h2"
            speed={30}
            delay={500}
            showCursor={false}
            className="mt-2 text-4xl md:text-5xl font-semibold text-gradient"
          />
          <Typewriter
            text="Selected publications from a decade of research at EPFL and CERN."
            as="p"
            speed={18}
            delay={1100}
            showCursor={false}
            className="mt-3 text-gray-400 max-w-2xl"
          />
        </header>

        {/* stats strip */}
        <div className="grid grid-cols-4 gap-4 mb-10">
          {[
            { label: "Publications", value: stats.papers },
            { label: "Citations", value: stats.citations.toLocaleString("en-US") },
            { label: "h-index", value: stats.hIndex },
            { label: "i10-index", value: stats.i10Index },
          ].map((s) => (
            <div
              key={s.label}
              className="glass rounded-xl px-5 py-4 text-center"
            >
              <div className="text-2xl md:text-3xl font-display font-semibold text-white">
                {s.value}
              </div>
              <div className="mt-1 text-[11px] font-mono uppercase tracking-[0.18em] text-gray-500">
                {s.label}
              </div>
            </div>
          ))}
        </div>

        {/* paper list */}
        <div className="space-y-3">
          {SELECTED.map((p, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass rounded-xl px-5 py-4 group hover:border-accent/30 transition-colors"
            >
              <div className="flex items-start gap-3">
                <BookOpen className="w-4 h-4 mt-1 shrink-0 text-accent2/60" />
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-medium text-gray-200 leading-snug">
                    {p.title}
                  </h3>
                  <p className="mt-1 text-xs text-gray-500 truncate">
                    {p.authors}
                  </p>
                  <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] font-mono text-gray-500">
                    <span className="text-accent2/70">{p.venue}</span>
                    <span>{p.year}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* link to full profile */}
        <div className="mt-8 text-center">
          <a
            href={SCHOLAR_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-sm text-gray-400 hover:text-accent2 transition-colors"
          >
            View all {stats.papers} publications on Google Scholar
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </motion.div>
    </section>
  );
}

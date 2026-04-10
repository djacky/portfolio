"use client";

import { motion } from "framer-motion";
import { ExternalLink, BookOpen } from "lucide-react";

const SCHOLAR_URL =
  "https://scholar.google.com/citations?user=Fes_eScAAAAJ&hl=en";

const STATS = {
  citations: 357,
  hIndex: 9,
  i10Index: 9,
  papers: 20,
};

type Paper = {
  title: string;
  authors: string;
  venue: string;
  year: number;
  citations: number;
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
    citations: 140,
  },
  {
    title:
      "Robust H\u221E controller design using frequency-domain data via convex optimization",
    authors: "A Karimi, A Nicoletti, Y Zhu",
    venue:
      "International Journal of Robust and Nonlinear Control",
    year: 2018,
    citations: 71,
  },
  {
    title:
      "Robust Smith Predictor design for time-delay systems with H\u221E performance",
    authors: "V De Oliveira, A Nicoletti, A Karimi",
    venue: "Recent Results on Time-Delay Systems: Analysis and Control",
    year: 2016,
    citations: 29,
  },
  {
    title:
      "A data-driven approach to model-reference control with applications to particle accelerator power converters",
    authors: "A Nicoletti, M Martino, A Karimi",
    venue: "Control Engineering Practice",
    year: 2019,
    citations: 22,
  },
  {
    title:
      "A robust data-driven controller design methodology with applications to particle accelerator power converters",
    authors: "A Nicoletti, M Martino, A Karimi",
    venue: "IEEE Transactions on Control Systems Technology",
    year: 2018,
    citations: 22,
  },
  {
    title:
      "Data-driven approach to iterative learning control via convex optimisation",
    authors: "A Nicoletti, M Martino, D Aguglia",
    venue: "IET Control Theory & Applications",
    year: 2020,
    citations: 11,
  },
  {
    title:
      "A data-driven frequency-domain approach for robust controller design via convex optimization",
    authors: "A Nicoletti",
    venue: "PhD Thesis, EPFL",
    year: 2018,
    citations: 3,
  },
];

export default function Publications() {
  return (
    <section
      id="publications"
      className="relative mx-auto max-w-6xl px-6 py-24"
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
      >
        <header className="mb-10">
          <p className="text-xs uppercase tracking-[0.25em] text-accent2">
            Research
          </p>
          <h2 className="mt-2 text-4xl md:text-5xl font-semibold text-gradient">
            Published work
          </h2>
          <p className="mt-3 text-gray-400 max-w-2xl">
            Selected publications from a decade of research at EPFL and CERN.
            Full list on{" "}
            <a
              href={SCHOLAR_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent2 hover:underline"
            >
              Google&nbsp;Scholar
            </a>
            .
          </p>
        </header>

        {/* stats strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-10">
          {[
            { label: "Publications", value: STATS.papers },
            { label: "Citations", value: `${STATS.citations}+` },
            { label: "h-index", value: STATS.hIndex },
            { label: "i10-index", value: STATS.i10Index },
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
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
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
                    <span>
                      {p.citations > 0
                        ? `${p.citations} citation${p.citations !== 1 ? "s" : ""}`
                        : ""}
                    </span>
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
            View all {STATS.papers} publications on Google Scholar
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </motion.div>
    </section>
  );
}

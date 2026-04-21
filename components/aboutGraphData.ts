/* ------------------------------------------------------------------ *
 *  aboutGraphData — shared data for the About 3D knowledge graph       *
 *  Consumed by AboutGraph.tsx (desktop 3D) and About.tsx (mobile).     *
 * ------------------------------------------------------------------ */

export interface Pillar {
  id: string;
  title: string;
  subtitle: string;
  body: string;
  color: string;
  position: [number, number, number];
}

export interface GraphEdge {
  from: number;
  to: number;
  curve: number;
}

export const PILLARS: Pillar[] = [
  {
    id: "origin",
    title: "The Origin",
    subtitle: "Where I come from",
    body: "I started in control systems — EPFL PhD, Harvard post-doc, CERN Fellow — designing feedback loops that keep systems stable under uncertainty. But control theory at that scale doesn't stay on paper: deploying it inside particle accelerators and EV fleets meant mastering the full stack, from algorithm to infrastructure to database management, in Python and AWS. Then I realized that neural networks are just control systems that learn their own parameters. That insight bridged classical engineering with modern AI, and it changed everything.",
    color: "#7c5cff",
    position: [-1.85, 1.1, 0],
  },
  {
    id: "engineer",
    title: "The Engineer",
    subtitle: "What I actually build",
    body: "I design and ship systems end to end — from convex optimization algorithms and deep/reinforcement learning agents to distributed AWS infrastructure handling hundreds of thousands of daily requests. Owned production-grade Python backends, real-time ML/deep learning pipelines, performance critical C++ systems, and data-driven robust controller synthesis for uncertain systems. My control systems background means I think about stability, latency, and failure modes from day one \u2014 not as an afterthought.",
    color: "#22d3ee",
    position: [1.85, 1.1, -0.3],
  },
  {
    id: "edge",
    title: "The Edge",
    subtitle: "What makes me different",
    body: "Most engineers either research or ship. I do both. My background in control theory, spectral analysis, and constrained optimization gives me a layer of analytical depth that's rare in pure software engineers - and my decade of production deployments gives me the pragmatism that's rare in researchers. I can design the algorithm, architect the infrastructure, and make the tradeoffs that hold up under pressure. That combination is what lets me work on problems where the stakes are high and the margin for error is low.",
    color: "#34d399",
    position: [-1.85, -1.1, 0.5],
  },
  {
    id: "person",
    title: "The Person",
    subtitle: "Who you\u2019re actually hiring",
    body: "I take ownership seriously. Not in the buzzword sense — in the sense that I stay until the problem is solved, I push back when something is wrong, and I'm honest about what I don't know. I've worked alone as a founder, as part of small research teams at CERN, and inside large engineering organizations. I adapt, but my standard doesn't. I'm looking for a team that builds things worth building, and wants someone who brings both rigor and drive to the table.",
    color: "#fbbf24",
    position: [1.85, -1.1, 0.2],
  },
];

/* Fully-connected K4 — all 6 edges.
   curve = perpendicular offset so tubes don't overlap. */
export const EDGES: GraphEdge[] = [
  { from: 0, to: 1, curve: -0.12 }, // Origin <-> Engineer
  { from: 0, to: 2, curve: 0.15 },  // Origin <-> Edge
  { from: 0, to: 3, curve: 0.2 },   // Origin <-> Person (diagonal)
  { from: 1, to: 2, curve: -0.2 },  // Engineer <-> Edge (diagonal)
  { from: 1, to: 3, curve: 0.1 },   // Engineer <-> Person
  { from: 2, to: 3, curve: -0.1 },  // Edge <-> Person
];

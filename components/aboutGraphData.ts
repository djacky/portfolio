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
    body: "I started in control systems \u2014 EPFL PhD, Harvard post-doc, CERN Fellow \u2014 designing feedback loops that keep machines stable under uncertainty. Then I realized that neural networks are just control systems that learn their own parameters. That insight bridged classical engineering with modern AI, and it changed everything.",
    color: "#7c5cff",
    position: [-1.85, 1.1, 0],
  },
  {
    id: "engineer",
    title: "The Engineer",
    subtitle: "What I actually build",
    body: "I ship AI/ML systems end-to-end \u2014 from model architecture to production monitoring. Reinforcement learning agents for real-time EV fleet management, Siamese networks for matchmaking, and distributed microservices handling 250k+ daily requests on AWS. My control systems background means I think about stability, latency, and failure modes from day one \u2014 not as an afterthought.",
    color: "#22d3ee",
    position: [1.85, 1.1, -0.3],
  },
  {
    id: "edge",
    title: "The Edge",
    subtitle: "What makes me different",
    body: "Most ML engineers think in probabilities. I also think in dynamics \u2014 how systems behave over time, how errors propagate, where instability hides. I\u2019ve applied formal stability analysis to training loops, used H\u221e robustness methods on model drift, and brought a decade of engineering rigor to production ML. That dual lens catches what pure data-science thinking misses.",
    color: "#34d399",
    position: [-1.85, -1.1, 0.5],
  },
  {
    id: "person",
    title: "The Person",
    subtitle: "Who you\u2019re actually hiring",
    body: "I care about craft. Clean APIs because the next engineer\u2019s time matters. Documented models because unmaintainable systems are liabilities. A patent, peer-reviewed publications, and Six Sigma certification \u2014 because rigor isn\u2019t just for code. Currently open to senior roles where AI/ML is a core product bet, not a side experiment. Bonus if the problem is genuinely hard.",
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

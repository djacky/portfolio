/* ------------------------------------------------------------------
   Candidate dossier — the structured context the LLM uses to score
   incoming job descriptions.

   This file is the *only* place your bio + projects live for the
   recruiter-match feature. Edit freely. Keep it dense and factual:
   the model scores against what it can read here. Avoid marketing
   adjectives. Prefer outcomes ("trained PPO policy on a global EV
   fleet, weeks → days validation cycle") over self-ratings.

   Aim for ~2k–3k tokens total. This whole string is sent on every
   request and is prompt-cached server-side, so the cost is paid once
   per ~5 min window, not per request.
------------------------------------------------------------------ */

export const CANDIDATE_DOSSIER = `
# CANDIDATE: Achille Nicoletti
PhD, EPFL Lausanne · Based in Geneva, CH
Open to: senior IC roles across three buckets — Control Systems,
AI/ML Research Engineer, and Python Backend.

## ONE-LINE POSITIONING
PhD control theorist transitioning into AI/ML and backend engineering.
Published research in control (H∞, data-driven synthesis, ILC);
shipped production RL (Eaton EV fleet) and production Python backends
(FastAPI / AWS, 250k+ daily requests).

## ROLE ARCHETYPES (score JDs against the bucket they fit)

STRONG FIT — Control Systems / Research Engineer:
  Controls, MPC, H∞/H2, system identification, convex optimization,
  embedded C++, real-time loops, HIL/SIL, power electronics, robotics,
  autonomous vehicles, accelerator/physics instrumentation.
  This is the candidate's primary specialization — PhD + 10 years
  published research + shipped hardware. Score generously.

STRONG FIT — AI/ML Engineer / Applied Research:
  Reinforcement learning (PPO, actor-critic), deep learning (PyTorch,
  Transformers, Siamese nets), convex/stochastic optimization, applied
  research, ML infra, model serving, inference at scale.
  Note: transitioning from controls — has shipped RL and deep learning
  to production but no NeurIPS/ICML publications. Treat production ML
  track record (Eaton PPO, Disruptive transformer matchmaker) as the
  evidence, not papers. Score as STRONG when the role is applied /
  research-engineering rather than pure ML academic research.

STRONG FIT — Backend / Python Engineer:
  Python, FastAPI, Pydantic, AWS (Lambda, EC2, S3, SNS, SQS, DynamoDB,
  Cognito), PostgreSQL, Docker, CI/CD, distributed systems, API design,
  auth (OAuth2, JWT), observability, compliance (SOC 2 Type II).
  Decade of shipping Python in production, founder-level ownership of
  distributed AWS backends. Score as STRONG for backend-Python roles,
  especially ones with an ML/data/research angle.

PARTIAL FIT:
  - Pure frontend / React-only (TypeScript is used for tooling, not primary)
  - Kubernetes / Terraform (day-to-day but not deep specialty)
  - LLMs, RAG, agentic systems (recent applied work, not multi-year)
  - Pure ML-academic roles expecting NeurIPS/ICML publication record

GAP:
  - iOS / Swift / native mobile
  - Frontend-only (designer-developer, marketing sites)
  - Pure data engineering / Spark / Snowflake without ML angle
  - Management-only roles with no IC component

## CORE STRENGTHS
- Deep learning & RL in PyTorch (PPO, REINFORCE, actor-critic, Siamese networks)
- Distributed Python backends on AWS (FastAPI, Pydantic, Lambda, DynamoDB, S3, Cognito, EC2)
- Convex optimization & data-driven control (H∞, H2, CVXPY, MOSEK)
- Performance-critical C++ (embedded AVR8, real-time control loops)
- End-to-end product delivery: CI/CD, Docker, QA, live ops, SOC 2 Type II compliance
- Research-grade rigor with production discipline (10+ years shipping to real users / real hardware)

## SKILLS MATRIX
ML / AI:
  - PyTorch — training + inference, 0.95
  - Deep learning — CNN, Transformer, Siamese, 0.92
  - Reinforcement learning — PPO, REINFORCE, actor-critic, 0.88
  - Optimization — convex, H∞, H2, 0.90
  - Model serving — ONNX, AWS Lambda, 0.85

Languages:
  - Python — 10+ years, primary, 0.98
  - C++ — embedded, AVR8, performance, 0.85
  - MATLAB / Simulink — controls, 0.85
  - TypeScript — React, Next.js, 0.80
  - Solidity, 0.60

Backend & Cloud:
  - FastAPI / Pydantic — 250k+ requests/day in prod, 0.95
  - AWS — Lambda, EC2, DynamoDB, S3, Cognito, 0.92
  - PostgreSQL — Celery, pipelines, 0.85
  - Auth — OAuth2, JWT, SOC 2 Type II, 0.82
  - Docker / CI/CD, 0.88

Systems & Control:
  - Control theory — PhD, EPFL, 0.92
  - System identification, 0.90
  - HIL / SIL validation pipelines, 0.85
  - Embedded C++ — AVR8, real-time, 0.80
  - Signal processing, 0.85

Day-to-day tooling: Git, Linux, pytest, Grafana, Redis, Kubernetes, Terraform, Celery.

## EDUCATION
PhD, EPFL (École Polytechnique Fédérale de Lausanne) — control theory and
data-driven controller synthesis for power-electronic systems.

## EXPERIENCE (most recent first)

### Founder & Software Engineer — Disruptive Labs · Geneva, CH
Dec 2023 — Present
Owns the product end-to-end: distributed AWS backend, PyTorch models for
matchmaking and payouts, live ops, compliance.
- Built and runs a full-stack multiplayer game with an ML backbone.
- 250k+ daily API requests, 1TB+ daily pipeline data.
- Achieved SOC 2 Type II compliance.
- Stack: Python, FastAPI, PyTorch, AWS (Lambda, DynamoDB, S3, Cognito), PostgreSQL, Docker.
- Trained a Siamese network for player matchmaking — embeds player history
  into a learned similarity space, retrieves nearest neighbors at request time.

### Senior Research Engineer — Eaton · Lausanne, CH
Jan 2021 — Dec 2023
Built the algorithm layer for Eaton's EV fleet load-balancing platform.
- Trained a PPO policy that observes fleet state (charger occupancy,
  vehicle SoC, grid constraints) and outputs an optimal current-allocation
  schedule across the fleet.
- Shipped the policy to a global fleet of chargers via a FastAPI / AWS
  Lambda inference layer. Real-time fleet-wide inference.
- OCPP protocol integration with the charger firmware.
- Reduced validation cycle time from weeks to days through HIL/SIL automation.
- Stack: PyTorch, RL (PPO), AWS Lambda, FastAPI, HIL/SIL pipelines.

### Senior Fellow — CERN · Geneva, CH
Jan 2018 — Jan 2021
Data-driven control for the power converters behind the LHC experiments.
- Developed Python APIs for frequency-response identification and
  H∞ / H2 controller synthesis (CVXPY + MOSEK convex solvers).
- Turned a manual, expert-driven commissioning step into a one-click workflow:
  upload measured frequency response → pick desired closed-loop specs →
  service returns RST polynomial ready to flash onto the converter.
- Implemented the resulting controllers on AVR8 microcontrollers in C++.
- Trained the department on the tooling.
- Stack: C++, Python, control theory, convex optimization, system ID.

### Electrical Design Engineer II — Philips Healthcare · Highland Heights, USA
May 2011 — Jul 2013
Motion control for the next generation of Philips CT scanners.
- Modeled multidomain dynamics in MATLAB / Simulink.
- Developed PLC control in CoDeSys (IEC 61131-3).
- Shipped on production medical hardware.

### Mixed-Signal Engineer (Intern) — Apple · Cupertino, USA
Jun 2010 — Sep 2010
Transistor-level feasibility studies for mixed-signal systems on
unreleased hardware. Simulation and verification toward formal design reviews.

## SIGNATURE PROJECTS (link these in evidence when relevant)

### CERN power converter controller synthesis — link: #demos (CERN block)
Python service that automates LHC power-converter commissioning.
Ingests measured frequency response, runs ARX system identification,
solves an H∞ synthesis problem (CVXPY + MOSEK), returns an RST polynomial.
Replaced a manual, expert-only workflow.

### Eaton EV fleet RL controller — link: #demos (EV Fleet block)
PPO-trained reinforcement learning policy for charger current allocation
across a global EV fleet. Real-time inference via FastAPI on AWS Lambda.
OCPP protocol integration.

### Disruptive Labs Siamese matchmaker — link: #demos (Matchmaker block)
Siamese embedding network for player matchmaking in a multiplayer game.
Production system handling 250k+ daily API requests with sub-second
matchmaking latency.

## MATCHING GUIDANCE

Before scoring, identify which of the three STRONG-FIT archetypes above
the JD most resembles (Controls, AI/ML, or Backend-Python). Score the
JD's requirements against that bucket's strengths — don't penalize a
controls JD for lacking ML requirements or vice versa.

For cross-bucket JDs (e.g., "RL for robotics", "ML engineer with
controls background", "Python backend for scientific computing"),
this is the sweet spot: score as STRONG and explicitly call out the
bridge in the rationale.
`.trim();

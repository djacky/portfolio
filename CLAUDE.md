# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev      # next dev — http://localhost:3000
npm run build    # next build
npm run start    # serve built app
npm run lint     # next lint
```

No test runner is configured.

## Stack

Next.js 14 (app router) · React 18 · TypeScript · Tailwind · Framer Motion. Interactive demos use `@react-three/fiber` + `three`, `@tensorflow/tfjs` (in-browser training), `matter-js` (2D physics), and `recharts`. Path alias `@/*` maps to the repo root (see `tsconfig.json`).

## Architecture

Single-page portfolio. `app/page.tsx` composes section components from `components/` in order (Nav, Hero, RecruiterMatch, About, Experience, a `#demos` section rendering `DemoSwitcher`, Skills, Contact). `app/layout.tsx` holds global metadata + fonts; `app/globals.css` + `tailwind.config.ts` define the theme (custom `accent`, `accent2`, `good` colors, `text-gradient`, `shadow-glow`).

### API routes (`app/api/`)
- `contact/` — contact form submission endpoint used by `ContactDrawer`.
- `match/` — recruiter job-description matching endpoint used by `RecruiterMatch`. Schema in `lib/match-schema.ts`, candidate data in `lib/candidate-dossier.ts`.

### Interactive demos
All demos run entirely client-side and are dynamically imported with `ssr: false` so three.js / tfjs never hit the server bundle. When adding or editing a demo, preserve this pattern — importing these statically will blow up SSR and first-paint.

- **`PendulumScene.tsx`** (Hero background) — R3F canvas with a **single inverted pendulum** controlled by DQN (Deep Q-Network). The agent learns swing-up + balance entirely in-browser from actual visible pendulum transitions — no background simulation. Network: `[3]→32→ReLU→32→ReLU→7 Q-values` (discrete torques). Important invariants:
  - Physics: rigid-rod pendulum `θ̈ = (3g/2L)sin(θ) + (3/mL²)τ`, RK4 integration at dt=0.05s. θ=0 is upright (unstable), θ=π is hanging (stable). Torque clipped to ±2 Nm, angular velocity to ±8 rad/s.
  - The agent learns **only from real visible pendulum movements** instigated by the user. No training happens until the user grabs and releases the bob. DQN stores (s,a,r,s') transitions in a 20K replay buffer and runs 8 gradient updates per environment step with Adam optimizer. Pure MSE loss (no Huber/gradient clipping — critical for convergence).
  - DQN: 7 discrete actions (torques from -2 to +2 Nm), ε-greedy exploration (1.0→0.01 over 2000 steps), soft target update τ=0.005, LR=0.003, γ=0.99, batch size 32. Reward: `-(θ² + 0.1·ω² + 0.001·τ²)` scaled by 0.1. Convergence in ~3800-6400 steps (~1-2 min at 60fps).
  - The user can **grab the bob** at any time; on release the agent resumes control with the user's release velocity. Training auto-starts on first grab-release (no button). "Train Me!" animated gold text appears above the base until training begins.
  - After convergence (upright > 70% for 3 consecutive 200-step episodes, minimum 1000 total steps), everything transitions to gold, toast notification appears. Training continues at reduced rate (2 updates/step) to maintain policy without catastrophic forgetting.
  - The whole scene is wrapped in `<group position={[PEND_OFFSET_X, PEND_OFFSET_Y, 0]}>` to shift it right/up behind the Hero text. `GrabPlane` converts world-space `e.point` to pendulum-local by subtracting these constants — **if you change the group position, update `PEND_OFFSET_X/Y` too**, otherwise grab picking silently breaks.
  - HUDs: `RewardPlot` shows reward/upright curves, `StatsHud` shows updates/reward/upright%/epsilon. Overlays are portaled to a `z-[5]` container inside the Hero `#top` section to escape the `z-0` canvas stacking context.
- **`EVFleetDemo.tsx`** — PPO current-allocation policy simulation (EV fleet load balancing).
- **`PrizePoolDemo.tsx`** — softmax-based prize-pool distributor with tunable feature weights / temperature.
- **`CERNDemo.tsx` / `CERNPipeline3D.tsx`**, **`MatchmakerDemo.tsx`** — additional domain demos.
- **`DemoSwitcher.tsx`** — tabbed switcher for the `#demos` section.

### Shared UI
`ContactDrawer.tsx` exposes a `useContactDrawer()` hook consumed by `Hero` and `Contact` to open a global contact overlay — prefer this over ad-hoc modals.

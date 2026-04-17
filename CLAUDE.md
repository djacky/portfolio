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

- **`PendulumScene.tsx`** (Hero background) — R3F canvas with a **double pendulum (pendubot)** controlled by DQN (Deep Q-Network). Only joint 1 is actuated; the agent learns balance-only entirely in-browser. Training begins on first grab-release. Network: `[6]→64→ReLU→64→ReLU→21 Q-values` (discrete torques). Important invariants:
  - Physics: coupled Lagrangian EOM with mass matrix inversion (M1=1, M2=0.5, L1=0.6, L2=0.6), RK4 integration at dt=0.05s. θ=0 is upright (unstable), θ=π is hanging (stable). Torque clipped to ±15 Nm (joint 1 only), angular velocity to ±15 rad/s. Damping is zero (`DAMP1 = DAMP2 = 0`).
  - Training is driven **solely from visible-tick transitions** (no ghost/parallel rollout). Each physics step feeds (s, a, r, s') into a 50K replay buffer and triggers `UPDATES_PER_STEP = 16` gradient updates with Adam; post-convergence this drops to 2 updates/step to maintain policy. Pure MSE loss (no Huber/gradient clipping — critical for convergence). No training happens until the user first grabs the tip bob and releases it — that gates `agent.trainingActive`.
  - While the user is **grabbing**, the scene additionally calls `agent.injectDemo()` each frame: it runs inverse dynamics (`inverseDynJ1`) on the IK-driven state to recover the torque the user is implicitly applying, snaps it to the nearest discrete action, and pushes (prev, a, r, next) into replay as a demonstration — giving the agent free supervised signal from the user's balancing attempt.
  - DQN: 21 discrete actions (torques from -15 to +15 Nm), ε-greedy exploration (1.0→0.01 over 3000 steps), soft target update τ=0.008, LR=0.003, γ=0.99, batch size 32. Reward: `(cos(θ₁)+1)/2 · (cos(θ₂)+1)/2 - 0.002·(ω₁²+ω₂²) - 0.001·τ²` — multiplicative structure ensures both links must be upright for high reward, preventing local optima where only link 1 balances.
  - **Catastrophic-forgetting guard**: on every episode end where uprightFrac ≥ CONVERGE_FRAC and beats the prior best, the agent snapshots online weights into `bestNet`. Post-convergence, if 3 consecutive episodes drop below 20% upright, it reloads `bestNet` into both online and target networks.
  - **Two modes** controlled by `PENDUBOT_MODE` constant at the top of the file:
    - `"full"` — both links inverted (θ₁≈0, θ₂≈0). Reward: `(cos θ₁+1)/2 · (cos θ₂+1)/2`. This is the default.
    - `"partial"` — link 1 hangs (θ₁≈π), link 2 inverted (θ₂≈0). Reward: `(-cos θ₁+1)/2 · (cos θ₂+1)/2`. Guide text changes to "Flip the Tip Up!". Heat indicator, convergence check, and reward are all mode-aware.
  - Guided UX: "Train Me!" gold text → on grab "Now Invert Me" (or "Flip the Tip Up!" in partial mode) cyan text + **heat indicator** on tip bob (blue=far from target, red=close). On release training starts. Heat computed via `invertedHeat()` which is mode-aware.
  - The user can **grab the tip bob** at any time; `solveIK()` computes both joint angles via analytical 2-link IK with elbow continuity. On release the agent resumes control with release velocities. Training auto-starts on first grab-release.
  - After convergence (upright > 50% for 3 consecutive 200-step episodes, minimum 1000 total steps), everything transitions to gold, toast notification appears. Training continues at reduced rate (2 updates/step) to maintain policy without catastrophic forgetting.
  - **Persistent training**: The DQN agent, physics state, and replay buffer live in a module-level `PendulumEngine` singleton that survives Hero mount/unmount cycles. When the user navigates away from Hero, a background `setInterval` at ~20 Hz continues physics and training. On return, the Canvas reconnects to the existing engine and shows the updated state (plots, convergence, pendulum position). Convergence POST to Redis happens when the component is mounted and detects convergence (foreground or background).
  - The whole scene is wrapped in `<group position={[PEND_OFFSET_X, PEND_OFFSET_Y, 0]}>` to shift it right/up behind the Hero text. `GrabPlane` converts world-space `e.point` to pendulum-local by subtracting these constants — **if you change the group position, update `PEND_OFFSET_X/Y` too**, otherwise grab picking silently breaks. The Canvas uses `eventSource` pointed at the Hero `#top` section so pointer events pass through the text layer.
  - HUDs: `RewardPlot` shows reward/upright curves, `StatsHud` shows updates/reward/upright%/epsilon. Overlays are portaled to a `z-[5]` container inside the Hero `#top` section to escape the `z-0` canvas stacking context.
- **`EVFleetDemo.tsx`** — PPO current-allocation policy simulation (EV fleet load balancing).
- **`PrizePoolDemo.tsx`** — softmax-based prize-pool distributor with tunable feature weights / temperature.
- **`CERNDemo.tsx` / `CERNPipeline3D.tsx`**, **`MatchmakerDemo.tsx`** — additional domain demos.
- **`DemoSwitcher.tsx`** — tabbed switcher for the `#demos` section.

### Shared UI
`ContactDrawer.tsx` exposes a `useContactDrawer()` hook consumed by `Hero` and `Contact` to open a global contact overlay — prefer this over ad-hoc modals.

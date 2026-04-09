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

- **`PendulumScene.tsx`** (Hero background) — R3F canvas with a double-pendulum hanging-stabilization task. An MLP (tfjs) is trained online via **behavioral cloning** against a closed-form PD teacher. Important invariants:
  - Training auto-starts on page load; no Start button.
  - In **training mode** the visible pendulum runs free physics (`u = 0`). BC happens in a parallel hidden rollout env. The learned policy is NOT applied to the visible pendulum during training.
  - In **control mode** the frozen learned weights drive the visible pendulum via `agent.actDet()`.
  - The pendulum is **never respawned**; it only moves via physics, user grab, or the learned policy. Runaway `w` is clipped, not reset.
  - The whole scene is wrapped in `<group position={[PEND_OFFSET_X, PEND_OFFSET_Y, 0]}>` to shift it right/up behind the Hero text. `GrabPlane` converts world-space `e.point` to pendulum-local by subtracting these constants — **if you change the group position, update `PEND_OFFSET_X/Y` too**, otherwise grab picking silently breaks.
  - `GainsHud` finite-differences the MLP around the downward equilibrium to display learned gains converging to the PD teacher's analytical gains (`KP1`, `KD1`, `KD2`).
- **`EVFleetDemo.tsx`** — PPO current-allocation policy simulation (EV fleet load balancing).
- **`PrizePoolDemo.tsx`** — softmax-based prize-pool distributor with tunable feature weights / temperature.
- **`CERNDemo.tsx` / `CERNPipeline3D.tsx`**, **`MatchmakerDemo.tsx`** — additional domain demos.
- **`DemoSwitcher.tsx`** — tabbed switcher for the `#demos` section.

### Shared UI
`ContactDrawer.tsx` exposes a `useContactDrawer()` hook consumed by `Hero` and `Contact` to open a global contact overlay — prefer this over ad-hoc modals.

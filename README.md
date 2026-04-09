# Achille Nicoletti — Portfolio

Interactive portfolio site. Next.js 14 (app router) · TypeScript · Tailwind · Framer Motion · Recharts.

## Live demos
- **EV fleet RL allocator** — in-browser simulation of the PPO current-allocation policy from the Eaton EV load-balancing platform. Sliders tune grid cap, fleet size, and reward weights; a live chart shows grid usage vs. cap.
- **Neural prize-pool distributor** — a softmax-based prize-pool network from Disruptive Labs. Sliders tune feature weights and softmax temperature.

## Develop
```bash
npm install
npm run dev
```
Open http://localhost:3000.

## Deploy (Vercel)
Zero config. Push to GitHub and import the repo in Vercel, or:
```bash
npm i -g vercel
vercel
```
Build command: `next build` · Output: `.next` (auto-detected).

## Structure
```
app/           # Next.js app router (layout, page, globals)
components/    # Section components + interactive demos
tailwind.config.ts
```

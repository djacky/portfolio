"use client";
import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { Trophy } from "lucide-react";

/* Disruptive Labs — neural prize-pool distributor.
   A small feed-forward network takes raw player performance metrics
   (kills, assists, objectives, playtime, accuracy) and outputs a
   softmax share of the prize pool. Here we expose a hand-crafted
   stand-in so visitors can feel the shape of the decision surface. */

interface Player {
  name: string;
  kills: number;
  assists: number;
  objectives: number;
  playtime: number;   // minutes
  accuracy: number;   // 0..1
}

const SEED: Player[] = [
  { name: "aurora",  kills: 24, assists: 8,  objectives: 5, playtime: 42, accuracy: 0.63 },
  { name: "neo",     kills: 19, assists: 14, objectives: 7, playtime: 45, accuracy: 0.58 },
  { name: "kairo",   kills: 12, assists: 11, objectives: 9, playtime: 45, accuracy: 0.51 },
  { name: "vex",     kills: 31, assists: 3,  objectives: 2, playtime: 38, accuracy: 0.71 },
  { name: "luma",    kills: 9,  assists: 17, objectives: 8, playtime: 44, accuracy: 0.47 },
];

function score(p: Player, w: typeof DEFAULT_W) {
  return (
    w.kills * p.kills +
    w.assists * p.assists +
    w.objectives * p.objectives * 2 +
    w.playtime * (p.playtime / 10) +
    w.accuracy * (p.accuracy * 30)
  );
}

function softmax(xs: number[], temp: number) {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp((x - m) / Math.max(0.01, temp)));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

const DEFAULT_W = { kills: 1.0, assists: 0.8, objectives: 1.2, playtime: 0.5, accuracy: 1.0 };

export default function PrizePoolDemo() {
  const [pool, setPool] = useState(10000);
  const [temp, setTemp] = useState(0.6);
  const [w, setW] = useState(DEFAULT_W);

  const data = useMemo(() => {
    const scores = SEED.map((p) => score(p, w));
    const shares = softmax(scores, temp);
    return SEED.map((p, i) => ({
      name: p.name,
      share: +(shares[i] * 100).toFixed(1),
      prize: Math.round(shares[i] * pool),
    })).sort((a, b) => b.prize - a.prize);
  }, [pool, temp, w]);

  return (
    <div className="shimmer-border rounded-3xl">
      <div className="glass rounded-3xl p-6 md:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-accent2">
              <Trophy className="w-3 h-3" /> Disruptive Labs · prize pool network
            </div>
            <h3 className="mt-2 text-2xl font-semibold text-white">
              Neural prize-pool distribution
            </h3>
            <p className="mt-2 text-sm text-gray-400 max-w-2xl">
              A PyTorch MLP ingests raw per-player performance metrics and outputs a softmax share
              of the match prize pool. Move the sliders to see how each metric shifts the payouts.
            </p>
          </div>
        </div>

        <div className="mt-6 grid lg:grid-cols-5 gap-6">
          <div className="lg:col-span-3 rounded-2xl bg-bg/60 border border-white/5 p-5">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm text-gray-300">Distribution</div>
              <div className="text-[10px] font-mono text-gray-500">prize = softmax(f(x) / τ) · pool</div>
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data}>
                  <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                  <XAxis dataKey="name" stroke="#6b7280" fontSize={11} />
                  <YAxis stroke="#6b7280" fontSize={11} tickFormatter={(v) => `$${v}`} />
                  <Tooltip
                    contentStyle={{ background: "#0b0f1a", border: "1px solid #1f2937", borderRadius: 8, fontSize: 11 }}
                    formatter={(v: number) => [`$${v}`, "prize"]}
                  />
                  <Bar dataKey="prize" radius={[6, 6, 0, 0]} fill="#7c5cff" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 grid grid-cols-5 gap-2">
              {data.map((d, i) => (
                <motion.div
                  key={d.name}
                  layout
                  className="rounded-lg bg-white/5 border border-white/10 p-2 text-center"
                >
                  <div className="text-[10px] text-gray-500">#{i + 1}</div>
                  <div className="text-xs font-mono text-white">{d.name}</div>
                  <div className="text-xs tabular-nums text-accent2 mt-1">${d.prize}</div>
                  <div className="text-[10px] text-gray-500">{d.share}%</div>
                </motion.div>
              ))}
            </div>
          </div>

          <div className="lg:col-span-2 rounded-2xl bg-bg/60 border border-white/5 p-5 space-y-4">
            <Slider label="Prize pool" value={pool} min={1000} max={50000} step={500} unit="$" prefix onChange={setPool} />
            <Slider label="Temperature τ" value={temp} min={0.1} max={2} step={0.1} unit="" onChange={setTemp} />
            <div className="h-px bg-white/5" />
            <div className="text-xs text-gray-400">Feature weights</div>
            <Slider label="Kills" value={w.kills} min={0} max={2} step={0.1} unit="" onChange={(v) => setW({ ...w, kills: v })} />
            <Slider label="Assists" value={w.assists} min={0} max={2} step={0.1} unit="" onChange={(v) => setW({ ...w, assists: v })} />
            <Slider label="Objectives" value={w.objectives} min={0} max={2} step={0.1} unit="" onChange={(v) => setW({ ...w, objectives: v })} />
            <Slider label="Playtime" value={w.playtime} min={0} max={2} step={0.1} unit="" onChange={(v) => setW({ ...w, playtime: v })} />
            <Slider label="Accuracy" value={w.accuracy} min={0} max={2} step={0.1} unit="" onChange={(v) => setW({ ...w, accuracy: v })} />
          </div>
        </div>
      </div>
    </div>
  );
}

function Slider({
  label, value, min, max, step, unit, onChange, prefix,
}: {
  label: string; value: number; min: number; max: number; step: number; unit: string;
  onChange: (v: number) => void; prefix?: boolean;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span className="tabular-nums text-white">
          {prefix ? unit : ""}
          {step < 1 ? value.toFixed(1) : value.toLocaleString()}
          {!prefix ? unit : ""}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent"
      />
    </div>
  );
}

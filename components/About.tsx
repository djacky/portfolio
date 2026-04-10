"use client";
import { motion } from "framer-motion";
import { Brain, Cloud, Cpu, Rocket } from "lucide-react";
import TiltCard from "./TiltCard";

const pillars = [
  {
    icon: Brain,
    title: "ML & Deep Learning",
    body: "PyTorch. Policy gradient, PPO, actor-critic, model training pipelines and real-time inference serving.",
  },
  {
    icon: Cloud,
    title: "Backend on AWS",
    body: "FastAPI · Pydantic · Lambda · DynamoDB · S3 · EC2 · Cognito. Distributed microservices handling 250k+ daily requests.",
  },
  {
    icon: Cpu,
    title: "Systems & Control",
    body: "Performance-critical C++. H∞ / H2 data-driven control. Convex optimization. Hardware-in-the-loop pipelines.",
  },
  {
    icon: Rocket,
    title: "End-to-end delivery",
    body: "CI/CD · Docker · QA · live ops · SOC 2 Type II. Comfortable owning a product from kernel to checkout.",
  },
];

export default function About() {
  return (
    <section id="about" className="relative mx-auto max-w-6xl px-6 py-24">
      <header className="mb-14">
        <p className="text-xs uppercase tracking-[0.25em] text-accent2">About</p>
        <h2 className="mt-2 text-4xl font-semibold text-gradient">
          Where research-grade rigor meets production systems.
        </h2>
      </header>
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        {pillars.map((p, i) => (
          <motion.div
            key={p.title}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ delay: i * 0.08 }}
          >
            <TiltCard className="h-full rounded-2xl">
              <div className="glass rounded-2xl p-6 hover:border-accent/40 transition-colors h-full">
                <p.icon className="w-6 h-6 text-accent" />
                <h3 className="mt-4 text-lg font-medium text-white">{p.title}</h3>
                <p className="mt-2 text-sm text-gray-400 leading-relaxed">{p.body}</p>
              </div>
            </TiltCard>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

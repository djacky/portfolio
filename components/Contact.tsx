"use client";
import { motion } from "framer-motion";
import { Mail, MessageSquare } from "lucide-react";
import { useContactDrawer } from "./ContactDrawer";
import TiltCard from "./TiltCard";

export default function Contact() {
  const { open } = useContactDrawer();
  return (
    <section id="contact" className="relative mx-auto max-w-6xl px-6 py-24">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
      >
        <TiltCard className="rounded-3xl">
          <div className="shimmer-border rounded-3xl">
            <div className="glass rounded-3xl p-10 md:p-14 text-center">
          <p className="text-xs uppercase tracking-[0.25em] text-accent2">Contact</p>
          <h2 className="mt-3 text-4xl md:text-5xl font-semibold text-gradient">
            Let&apos;s build something hard together.
          </h2>
          <p className="mt-4 text-gray-400 max-w-xl mx-auto">
            Open to senior AI/ML engineering roles, research engineering, and technical
            founding positions. Remote or Switzerland.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={open}
              className="inline-flex items-center gap-2 rounded-full bg-accent px-5 py-3 text-sm font-medium text-white shadow-glow hover:bg-accent/90 transition-colors"
            >
              <MessageSquare className="w-4 h-4" /> Let&apos;s talk
            </button>
            <a
              href="mailto:globalminimum@protonmail.com"
              className="inline-flex items-center gap-2 rounded-full border border-white/15 px-5 py-3 text-sm text-gray-200 hover:bg-white/5 transition-colors"
            >
              <Mail className="w-4 h-4" /> Email me
            </a>
          </div>
            </div>
          </div>
        </TiltCard>
      </motion.div>
    </section>
  );
}

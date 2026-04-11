"use client";

import { motion } from "framer-motion";
import { Home } from "lucide-react";
import { useNavigation } from "./SectionRouter";
import type { ReactNode } from "react";

interface SectionPageProps {
  children: ReactNode;
  id: string;
}

export default function SectionPage({ children, id }: SectionPageProps) {
  const { goTo } = useNavigation();

  return (
    <motion.div
      key={id}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 top-10 overflow-y-auto z-20"
    >
      <div className="relative z-10">
        {children}

        {/* footer */}
        <footer className="py-10 text-center text-xs text-gray-500">
          &copy; {new Date().getFullYear()} Achille Nicoletti
        </footer>
      </div>

      {/* floating home button */}
      <button
        type="button"
        onClick={() => goTo("hero")}
        aria-label="Back to home"
        className="fixed bottom-6 right-6 z-50 flex h-10 w-10 items-center justify-center rounded-full glass border border-white/10 text-gray-400 hover:text-accent2 hover:border-accent2/30 transition-all shadow-lg backdrop-blur-xl"
      >
        <Home className="w-4 h-4" />
      </button>
    </motion.div>
  );
}

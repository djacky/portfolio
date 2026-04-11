"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";

/*
  Cycling typewriter for the Hero section.
  Types out a title, holds it, deletes character-by-character,
  then types the next title. Cursor blinks the entire time.
*/

const TITLES = [
  "Senior AI/ML Engineer.",
  "Research Engineer.",
  "Backend Engineer.",
  "Control Systems Engineer.",
  "Electrical Engineer.",
  "Data Scientist.",
];

const TYPE_SPEED = 55;       // ms per char when typing
const DELETE_SPEED = 35;     // ms per char when deleting (faster)
const HOLD_DURATION = 2200;  // ms to hold the completed word
const PAUSE_AFTER_DELETE = 400; // ms pause between delete and next type

type Phase = "typing" | "holding" | "deleting" | "pausing";

export default function RollingTitle({ className = "" }: { className?: string }) {
  const [displayed, setDisplayed] = useState("");
  const [titleIdx, setTitleIdx] = useState(0);
  const [phase, setPhase] = useState<Phase>("typing");
  const idxRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const currentTitle = TITLES[titleIdx];

  const clear = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  // Main state machine
  useEffect(() => {
    clear();

    if (phase === "typing") {
      if (idxRef.current >= currentTitle.length) {
        // done typing → hold
        setPhase("holding");
        return;
      }
      timerRef.current = setTimeout(() => {
        idxRef.current++;
        setDisplayed(currentTitle.slice(0, idxRef.current));
      }, TYPE_SPEED);
    }

    if (phase === "holding") {
      timerRef.current = setTimeout(() => {
        setPhase("deleting");
      }, HOLD_DURATION);
    }

    if (phase === "deleting") {
      if (idxRef.current <= 0) {
        // done deleting → pause then next title
        setPhase("pausing");
        return;
      }
      timerRef.current = setTimeout(() => {
        idxRef.current--;
        setDisplayed(currentTitle.slice(0, idxRef.current));
      }, DELETE_SPEED);
    }

    if (phase === "pausing") {
      timerRef.current = setTimeout(() => {
        const next = (titleIdx + 1) % TITLES.length;
        setTitleIdx(next);
        idxRef.current = 0;
        setDisplayed("");
        setPhase("typing");
      }, PAUSE_AFTER_DELETE);
    }

    return clear;
  }, [phase, displayed, titleIdx, currentTitle, clear]);

  return (
    <span className={className}>
      {displayed}
      <motion.span
        className="inline-block ml-0.5 text-accent2"
        animate={{ opacity: [1, 1, 0, 0] }}
        transition={{
          duration: 0.8,
          times: [0, 0.4, 0.5, 1],
          repeat: Infinity,
          repeatType: "loop",
        }}
        aria-hidden
      >
        &#9614;
      </motion.span>
    </span>
  );
}

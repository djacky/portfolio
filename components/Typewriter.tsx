"use client";

import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";

/*
  Typewriter text effect with a blinking cursor.
  Types out text character-by-character on mount.
  Cursor blinks continuously after typing completes.
*/

interface TypewriterProps {
  text: string;
  /** ms per character (default 35) */
  speed?: number;
  /** ms delay before typing starts (default 0) */
  delay?: number;
  /** extra className for the text wrapper */
  className?: string;
  /** element tag to render (default "span") */
  as?: "span" | "p" | "h2" | "h3" | "div";
  /** cursor color class (default "text-accent2") */
  cursorColor?: string;
  /** show blinking cursor (default true) */
  showCursor?: boolean;
  /** callback when typing finishes */
  onComplete?: () => void;
}

export default function Typewriter({
  text,
  speed = 35,
  delay = 0,
  className = "",
  as: Tag = "span",
  cursorColor = "text-accent2",
  showCursor = true,
  onComplete,
}: TypewriterProps) {
  const [displayed, setDisplayed] = useState("");
  const [started, setStarted] = useState(false);
  const [done, setDone] = useState(false);
  const idxRef = useRef(0);

  // Delay before start
  useEffect(() => {
    const t = setTimeout(() => setStarted(true), delay);
    return () => clearTimeout(t);
  }, [delay]);

  // Typing loop
  useEffect(() => {
    if (!started) return;
    idxRef.current = 0;
    setDisplayed("");
    setDone(false);

    const interval = setInterval(() => {
      idxRef.current++;
      if (idxRef.current >= text.length) {
        setDisplayed(text);
        setDone(true);
        clearInterval(interval);
        onComplete?.();
      } else {
        setDisplayed(text.slice(0, idxRef.current));
      }
    }, speed);

    return () => clearInterval(interval);
  }, [started, text, speed, onComplete]);

  return (
    <Tag className={className} style={{ display: "grid" }}>
      {/* invisible full text — always present to reserve final height */}
      <span style={{ gridArea: "1/1", visibility: "hidden" }} aria-hidden="true">
        {text}
      </span>
      {/* visible typed text stacked in the same cell */}
      <span style={{ gridArea: "1/1" }}>
        {started ? displayed : ""}
        {started && showCursor && (
          <motion.span
            className={`inline-block ml-0.5 ${cursorColor}`}
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
        )}
      </span>
    </Tag>
  );
}

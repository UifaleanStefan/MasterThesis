/**
 * Animated radial-grid backdrop for the hero. Pure SVG/CSS — no canvas,
 * no GPU strain. The "θ field" is a constellation of dots that subtly
 * pulse, with a few connecting lines suggesting the graph memory.
 */

import { motion } from "framer-motion";
import { useMemo } from "react";

interface HeroBackdropProps {
  /** seed to vary the constellation between renders. */
  seed?: number;
  /** Number of dots. */
  count?: number;
}

interface Star {
  x: number;
  y: number;
  r: number;
  delay: number;
  duration: number;
  opacity: number;
}

function rng(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s * 1664525 + 1013904223) | 0;
    return ((s >>> 0) % 1_000_000) / 1_000_000;
  };
}

export function HeroBackdrop({ seed = 17, count = 60 }: HeroBackdropProps) {
  const { stars, links } = useMemo(() => {
    const rand = rng(seed);
    const stars: Star[] = [];
    for (let i = 0; i < count; i++) {
      stars.push({
        x: rand() * 100,
        y: rand() * 100,
        r: 0.3 + rand() * 1.0,
        delay: rand() * 4,
        duration: 2.5 + rand() * 3.5,
        opacity: 0.15 + rand() * 0.45,
      });
    }
    // Pick a handful of nearby pairs for graph-edge effect.
    const links: Array<{ a: number; b: number }> = [];
    for (let i = 0; i < stars.length; i++) {
      for (let j = i + 1; j < stars.length; j++) {
        const dx = stars[i].x - stars[j].x;
        const dy = stars[i].y - stars[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 9 && rand() < 0.35) links.push({ a: i, b: j });
      }
    }
    return { stars, links };
  }, [seed, count]);

  return (
    <svg
      aria-hidden
      className="absolute inset-0 w-full h-full pointer-events-none"
      preserveAspectRatio="xMidYMid slice"
      viewBox="0 0 100 100"
    >
      <defs>
        <radialGradient id="hero-glow" cx="50%" cy="40%" r="50%">
          <stop offset="0%" stopColor="rgba(34, 211, 238, 0.10)" />
          <stop offset="60%" stopColor="rgba(167, 139, 250, 0.04)" />
          <stop offset="100%" stopColor="transparent" />
        </radialGradient>
      </defs>
      <rect width="100" height="100" fill="url(#hero-glow)" />

      {/* Connecting lines (very faint) */}
      {links.map((l, i) => {
        const a = stars[l.a];
        const b = stars[l.b];
        return (
          <motion.line
            key={i}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke="rgba(34, 211, 238, 0.18)"
            strokeWidth="0.12"
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 0.6, 0.2, 0.6, 0] }}
            transition={{
              duration: 6 + (i % 5),
              repeat: Infinity,
              repeatDelay: 1 + (i % 3),
              delay: i * 0.04,
            }}
          />
        );
      })}

      {/* Stars */}
      {stars.map((s, i) => (
        <motion.circle
          key={i}
          cx={s.x}
          cy={s.y}
          r={s.r}
          fill={i % 7 === 0 ? "var(--color-violet)" : "var(--color-cyan)"}
          initial={{ opacity: 0 }}
          animate={{ opacity: [s.opacity * 0.4, s.opacity, s.opacity * 0.4] }}
          transition={{
            duration: s.duration,
            repeat: Infinity,
            ease: "easeInOut",
            delay: s.delay,
          }}
        />
      ))}
    </svg>
  );
}

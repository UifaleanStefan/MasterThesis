import { motion } from "framer-motion";
import { ArrowDown } from "lucide-react";
import { useEffect, useState } from "react";
import { HeroBackdrop } from "../components/viz/HeroBackdrop";
import { ThetaBars } from "../components/viz/ThetaBars";
import { Pill } from "../components/shared/Pill";
import { TFIDF_V4_THETA } from "../data/tfidf_constants";
import type { V4CmaesData } from "../data/types";
import { useData } from "../data/useData";

/**
 * Hero — full-bleed dark canvas with constellation backdrop, gradient
 * headline, and an animated 10D θ vector that periodically morphs between
 * the TF-IDF and MiniLM optima (foreshadowing the central pivot of the
 * thesis).
 */
export function Hero() {
  const { data: v4 } = useData<V4CmaesData>("v4_cmaes.json");
  const minilmTheta = v4?.v4?.best_params ?? null;

  // Auto-toggle which optimum the bars display every ~3.5 s.
  const [showMiniLM, setShowMiniLM] = useState(true);
  useEffect(() => {
    if (!minilmTheta) return;
    const id = setInterval(() => setShowMiniLM((v) => !v), 3500);
    return () => clearInterval(id);
  }, [minilmTheta]);

  const displayTheta = showMiniLM && minilmTheta ? minilmTheta : TFIDF_V4_THETA;
  const ghostTheta = showMiniLM && minilmTheta ? TFIDF_V4_THETA : minilmTheta;

  return (
    <section
      id="hero"
      className="relative min-h-screen flex flex-col items-center justify-center px-6 grid-bg overflow-hidden"
    >
      <HeroBackdrop seed={17} count={70} />

      {/* Vignette to fade the grid + constellation toward the bottom */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(900px 600px at 50% 30%, rgba(34, 211, 238, 0.05), transparent 70%)," +
            "linear-gradient(180deg, transparent 0%, transparent 60%, var(--color-bg) 100%)",
        }}
      />

      <div className="relative z-10 max-w-5xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.05 }}
          className="flex justify-center mb-8 gap-2 flex-wrap"
        >
          <Pill tone="cyan">Master&apos;s Thesis</Pill>
          <Pill tone="violet">Bocconi · 2026</Pill>
          <Pill tone="muted">Stefan Uifalean</Pill>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15 }}
          className="font-semibold tracking-[-0.03em] leading-[1.02]"
          style={{ fontSize: "clamp(2.6rem, 8.5vw, 6.5rem)" }}
        >
          Can an agent learn{" "}
          <span
            style={{
              background:
                "linear-gradient(120deg, var(--color-cyan), var(--color-violet) 60%, var(--color-amber))",
              WebkitBackgroundClip: "text",
              backgroundClip: "text",
              color: "transparent",
            }}
          >
            how to construct
          </span>{" "}
          its own memory?
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4 }}
          className="mt-8 max-w-3xl mx-auto text-lg md:text-xl leading-relaxed"
          style={{ color: "var(--color-text-2)" }}
        >
          A 10-dimensional vector{" "}
          <code className="px-1.5 py-0.5 rounded text-[var(--color-cyan)]">
            θ
          </code>{" "}
          controls what gets stored and how it&apos;s retrieved. CMA-ES finds the
          right θ for each task — and the answer changes when the embedding does.
        </motion.p>

        {/* Theta bars showcase */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mt-14 max-w-3xl mx-auto"
        >
          <div className="flex items-center justify-between mb-3 px-2">
            <div className="text-[0.65rem] uppercase tracking-[0.2em]" style={{ color: "var(--color-muted)" }}>
              learned θ optimum
            </div>
            <motion.div
              key={showMiniLM ? "minilm" : "tfidf"}
              initial={{ opacity: 0, x: 8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
            >
              <Pill tone={showMiniLM ? "violet" : "amber"}>
                {showMiniLM ? "MiniLM · 384-d" : "TF-IDF · 31-d"}
              </Pill>
            </motion.div>
          </div>
          <div className="panel-rise px-4 py-5">
            <ThetaBars
              theta={displayTheta}
              ghost={ghostTheta}
              showLabels
              showValues
              height={200}
            />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.85 }}
          className="mt-12 flex items-center justify-center gap-4"
        >
          <a
            href="#question"
            className="px-6 py-3 rounded-full font-medium text-sm transition-all hover:scale-[1.02]"
            style={{
              background:
                "linear-gradient(120deg, var(--color-cyan), var(--color-violet))",
              color: "#0a0e1a",
            }}
          >
            Read the story
          </a>
          <a
            href="#benchmark"
            className="px-6 py-3 rounded-full font-medium text-sm border border-[var(--color-border-strong)] hover:bg-[rgba(255,255,255,0.04)] transition-all"
            style={{ color: "var(--color-text)" }}
          >
            Jump to results
          </a>
        </motion.div>
      </div>

      {/* Scroll cue */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.4, duration: 0.6 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10"
      >
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
          className="flex flex-col items-center gap-2"
          style={{ color: "var(--color-muted)" }}
        >
          <span className="text-[0.65rem] uppercase tracking-[0.18em]">
            Scroll
          </span>
          <ArrowDown size={14} />
        </motion.div>
      </motion.div>
    </section>
  );
}

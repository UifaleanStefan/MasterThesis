import { motion } from "framer-motion";
import { ArrowDown } from "lucide-react";
import { Pill } from "../components/shared/Pill";

/**
 * Phase B Hero — text-only spectacle that already establishes typography
 * and rhythm. Phase C upgrades this with the animated 10D θ canvas.
 */
export function Hero() {
  return (
    <section
      id="hero"
      className="relative min-h-screen flex flex-col items-center justify-center px-6 grid-bg"
      style={{
        background:
          "radial-gradient(900px 600px at 50% 30%, rgba(34, 211, 238, 0.08), transparent 70%)," +
          "radial-gradient(800px 500px at 50% 80%, rgba(167, 139, 250, 0.06), transparent 70%)",
      }}
    >
      {/* Vignette over the grid background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(1200px 800px at 50% 50%, transparent 30%, var(--color-bg) 90%)",
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
          Most LLM agents are given a memory; this thesis lets the agent learn
          one. A 10-dimensional vector <code className="text-[var(--color-cyan)]">θ</code>{" "}
          controls what gets stored and how it&apos;s retrieved. CMA-ES finds the
          right θ for each task — and the answer changes when the embedding does.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.6 }}
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
        transition={{ delay: 1.2, duration: 0.6 }}
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

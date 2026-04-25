import { motion } from "framer-motion";
import { useState } from "react";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { Pill } from "../components/shared/Pill";
import { cn } from "../lib/format";

interface Step {
  id: string;
  version: string;
  dims: number;
  added: string;
  why: string;
  highlights: string[];
  filename: string;
}

const STEPS: Step[] = [
  {
    id: "v1",
    version: "V1",
    dims: 3,
    added: "(θ_store, θ_entity, θ_temporal)",
    why: "The original POC: a Bernoulli storage gate, a frequency-thresholded entity index, a Bernoulli temporal edge. Default (1.0, 0.0, 1.0) reproduces the unparameterized baseline exactly.",
    highlights: ["Bernoulli storage", "Frequency entities", "Backward compatible"],
    filename: "memory/graph_memory.py",
  },
  {
    id: "v2",
    version: "V2",
    dims: 6,
    added: "+ (w_graph, w_embed, w_recency)",
    why: "Retrieval becomes learnable. The same graph is indexed three ways and weighted by θ — graph traversal, embedding similarity, and recency.",
    highlights: ["Learnable retrieval", "Three signals", "Same storage as V1"],
    filename: "memory/graph_memory_v2.py",
  },
  {
    id: "v3",
    version: "V3",
    dims: 9,
    added: "+ (θ_novel, θ_erich, θ_surprise)",
    why: "Storage becomes importance-scored. Novelty (cosine distance from recent), entity richness, and surprise (L2 distance from rolling mean) replace the Bernoulli gate.",
    highlights: ["Selective storage", "Three importance features", "Memory size collapses"],
    filename: "memory/graph_memory_v3.py",
  },
  {
    id: "v4",
    version: "V4",
    dims: 10,
    added: "+ θ_decay",
    why: "Entity importance becomes Bayesian + temporally decayed. The most complete scalar parameterization. With CMA-ES tuning it stores ≈10 events vs V1's 218 — and statistically ties the leading systems.",
    highlights: ["Bayesian entities", "10D θ", "MOST COMPLETE"],
    filename: "memory/graph_memory_v4.py",
  },
  {
    id: "v5",
    version: "V5",
    dims: 10,
    added: "+ attention storage gate",
    why: "Same θ space, but the storage gate also includes a softmax-attention term over recent embeddings. Helps when novelty/surprise miss subtle similarities.",
    highlights: ["Attention storage", "Same dim space as V4", "Drop-in replacement"],
    filename: "memory/graph_memory_v5.py",
  },
];

const TONE: Record<number, string> = {
  3: "var(--color-amber)",
  6: "var(--color-cyan)",
  9: "var(--color-emerald)",
  10: "var(--color-violet)",
};

export function Progression() {
  const [activeIdx, setActiveIdx] = useState(3); // V4 by default

  const active = STEPS[activeIdx];
  const tone = TONE[active.dims] ?? "var(--color-cyan)";

  return (
    <Section id="progression" eyebrow="V1 → V5">
      <SectionHeader
        title={
          <>
            Each version{" "}
            <span style={{ color: "var(--color-violet)" }}>added a dimension</span>{" "}
            because the previous one had a hole.
          </>
        }
        lede="The progression is itself a thesis contribution: V1 stores all events; V2 makes retrieval learnable; V3 makes storage selective; V4 makes entities Bayesian; V5 adds attention. Originals are never modified — each version is a separate file."
      />

      {/* Timeline rail */}
      <div className="relative mt-12">
        <div
          aria-hidden
          className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-px"
          style={{ background: "var(--color-border)" }}
        />
        <div
          aria-hidden
          className="absolute left-0 top-1/2 -translate-y-1/2 h-px transition-all duration-300"
          style={{
            background:
              "linear-gradient(90deg, var(--color-amber), var(--color-cyan), var(--color-emerald), var(--color-violet))",
            width: `${(activeIdx / (STEPS.length - 1)) * 100}%`,
          }}
        />
        <div className="relative grid grid-cols-5 gap-2">
          {STEPS.map((s, i) => {
            const isActive = i === activeIdx;
            const passed = i <= activeIdx;
            const stepTone = TONE[s.dims] ?? "var(--color-cyan)";
            return (
              <button
                key={s.id}
                onClick={() => setActiveIdx(i)}
                className="relative flex flex-col items-center group"
                aria-pressed={isActive}
              >
                <div
                  className={cn(
                    "relative z-10 flex items-center justify-center w-12 h-12 rounded-full transition-all border",
                    isActive ? "scale-110" : "group-hover:scale-105",
                  )}
                  style={{
                    background: passed ? stepTone : "var(--color-surface)",
                    borderColor: passed ? stepTone : "var(--color-border-strong)",
                    color: passed ? "#0a0e1a" : "var(--color-text-2)",
                    boxShadow: isActive ? `0 0 18px ${stepTone}55` : "none",
                  }}
                >
                  <span className="text-xs font-mono font-semibold">{s.version}</span>
                </div>
                <div className="mt-3 text-center">
                  <div
                    className="text-[0.6rem] uppercase tracking-[0.16em]"
                    style={{ color: isActive ? stepTone : "var(--color-muted)" }}
                  >
                    {s.dims}D
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Detail panel */}
      <motion.div
        key={active.id}
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="panel-rise p-8 md:p-10 mt-10 grid md:grid-cols-[2fr_1fr] gap-8"
      >
        <div>
          <div className="flex items-center gap-3 mb-4">
            <span
              className="text-3xl font-mono font-semibold"
              style={{ color: tone }}
            >
              {active.version}
            </span>
            <Pill tone="muted">{active.dims}-dim θ</Pill>
            <code
              className="text-xs px-2 py-0.5 rounded"
              style={{
                color: "var(--color-text-2)",
                background: "rgba(255, 255, 255, 0.04)",
              }}
            >
              {active.filename}
            </code>
          </div>
          <h3 className="text-2xl md:text-3xl font-semibold mb-3">
            <span style={{ color: tone }}>added</span>{" "}
            <code className="font-mono text-xl md:text-2xl">{active.added}</code>
          </h3>
          <p
            className="leading-relaxed text-base md:text-lg"
            style={{ color: "var(--color-text-2)" }}
          >
            {active.why}
          </p>
          <div className="flex flex-wrap gap-2 mt-6">
            {active.highlights.map((h) => (
              <span
                key={h}
                className="px-3 py-1 rounded-full text-xs font-mono"
                style={{
                  color: tone,
                  background: `${tone}11`,
                  border: `1px solid ${tone}33`,
                }}
              >
                {h}
              </span>
            ))}
          </div>
        </div>

        <div className="flex flex-col items-center justify-center">
          <div
            className="text-7xl md:text-8xl font-mono font-bold leading-none"
            style={{ color: tone }}
          >
            {active.dims}
          </div>
          <div
            className="mt-2 text-[0.7rem] uppercase tracking-[0.18em]"
            style={{ color: "var(--color-muted)" }}
          >
            θ dimensions
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-8">
        <CallStat label="V1 mem size" value="218 events" tone="amber" />
        <CallStat label="V4 mem size" value="≈ 10 events" tone="cyan" />
        <CallStat label="reduction" value="22×" tone="violet" />
        <CallStat label="V4 reward (MiniLM)" value="0.130" tone="emerald" />
      </div>
    </Section>
  );
}

function CallStat({ label, value, tone }: { label: string; value: string; tone: "cyan" | "violet" | "amber" | "emerald" }) {
  const accent = `var(--color-${tone})`;
  return (
    <div className="panel px-4 py-3">
      <div className="text-[0.6rem] uppercase tracking-[0.16em]" style={{ color: "var(--color-muted)" }}>
        {label}
      </div>
      <div className="text-xl font-semibold mt-1 tabular-nums" style={{ color: accent }}>
        {value}
      </div>
    </div>
  );
}

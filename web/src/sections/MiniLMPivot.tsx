import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { ThetaRadar } from "../components/viz/ThetaRadar";
import { Stat } from "../components/shared/Stat";
import { Pill } from "../components/shared/Pill";
import { ChevronRight } from "lucide-react";
import { useData } from "../data/useData";
import type { V4CmaesData, V4Theta } from "../data/types";
import { TFIDF_V4_THETA } from "../data/tfidf_constants";
import { fmtSigned } from "../lib/format";

const KEY_DIFFS: Array<{ key: keyof V4Theta; label: string }> = [
  { key: "w_recency", label: "w_recency" },
  { key: "w_embed", label: "w_embed" },
  { key: "w_graph", label: "w_graph" },
  { key: "theta_novel", label: "θ_novel" },
];

export function MiniLMPivot() {
  const { data: v4 } = useData<V4CmaesData>("v4_cmaes.json");
  const minilm = v4?.v4?.best_params ?? null;

  // Auto-flip the highlighted optimum every 4 s on the comparison radar.
  const [showMini, setShowMini] = useState(true);
  useEffect(() => {
    const id = setInterval(() => setShowMini((v) => !v), 4500);
    return () => clearInterval(id);
  }, []);

  return (
    <Section id="minilm" eyebrow="The Pivot">
      <SectionHeader
        title={
          <>
            When the embedding changed,
            <br />
            the{" "}
            <span style={{ color: "var(--color-violet)" }}>optimum</span>{" "}
            changed too.
          </>
        }
        lede={
          <>
            Under TF-IDF the same CMA-ES converged to a recency-dominated retrieval
            (w_recency = 3.78, w_graph = 0). Under MiniLM it found something
            structurally different: balanced retrieval, lower selectivity, and{" "}
            <strong style={{ color: "var(--color-violet)" }}>graph traversal contributing</strong>.
            The thesis claim — &ldquo;optimal θ is task-dependent&rdquo; — extends to
            embedding-dependent.
          </>
        }
      />

      {/* Side-by-side radar comparison */}
      <div className="grid lg:grid-cols-[1fr_auto_1fr] gap-6 items-center mt-12">
        {/* TF-IDF column */}
        <motion.div
          initial={{ opacity: 0, x: -16 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="panel-rise p-6 flex flex-col items-center"
          style={{
            outline: !showMini ? "1px solid var(--color-amber)" : "none",
            transition: "outline 0.3s",
          }}
        >
          <div className="self-start mb-2">
            <Pill tone="amber">TF-IDF · 31-d</Pill>
          </div>
          <ThetaRadar theta={TFIDF_V4_THETA} tone="amber" size={300} />
          <div className="mt-4 grid grid-cols-3 gap-3 w-full">
            <MiniStat label="reward" value="0.178" tone="amber" />
            <MiniStat label="w_recency" value="3.78" tone="amber" />
            <MiniStat label="w_graph" value="0.00" tone="amber" />
          </div>
        </motion.div>

        {/* Arrow */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="hidden lg:flex flex-col items-center"
        >
          <ChevronRight
            size={40}
            style={{ color: "var(--color-violet)" }}
            className="opacity-70"
          />
          <div
            className="text-[0.65rem] uppercase tracking-[0.16em] mt-1"
            style={{ color: "var(--color-muted)" }}
          >
            embedding
            <br />
            swapped
          </div>
        </motion.div>

        {/* MiniLM column */}
        <motion.div
          initial={{ opacity: 0, x: 16 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="panel-rise p-6 flex flex-col items-center"
          style={{
            outline: showMini ? "1px solid var(--color-violet)" : "none",
            transition: "outline 0.3s",
          }}
        >
          <div className="self-start mb-2">
            <Pill tone="violet">MiniLM · 384-d</Pill>
          </div>
          {minilm ? (
            <ThetaRadar theta={minilm} tone="violet" size={300} />
          ) : (
            <div className="h-[300px] flex items-center justify-center" style={{ color: "var(--color-muted)" }}>
              loading…
            </div>
          )}
          <div className="mt-4 grid grid-cols-3 gap-3 w-full">
            <MiniStat label="reward" value={v4 ? v4.v4.eval.mean_reward.toFixed(3) : "—"} tone="violet" />
            <MiniStat label="w_recency" value={minilm ? minilm.w_recency.toFixed(2) : "—"} tone="violet" />
            <MiniStat label="w_graph" value={minilm ? minilm.w_graph.toFixed(2) : "—"} tone="violet" />
          </div>
        </motion.div>
      </div>

      {/* Side-by-side delta breakdown */}
      <div className="mt-10">
        <div
          className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
          style={{ color: "var(--color-muted)" }}
        >
          key dimensional shifts
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {KEY_DIFFS.map((d) => {
            const tfidf = TFIDF_V4_THETA[d.key];
            const mini = minilm?.[d.key];
            const delta = mini !== undefined ? mini - tfidf : null;
            return (
              <div key={d.key} className="panel p-4">
                <div
                  className="text-[0.6rem] uppercase tracking-[0.14em] mb-2"
                  style={{ color: "var(--color-muted)" }}
                >
                  {d.label}
                </div>
                <div className="flex items-baseline gap-2 font-mono">
                  <span style={{ color: "var(--color-amber)" }}>{tfidf.toFixed(2)}</span>
                  <ChevronRight
                    size={12}
                    style={{ color: "var(--color-muted)" }}
                  />
                  <span style={{ color: "var(--color-violet)" }}>{mini?.toFixed(2) ?? "—"}</span>
                </div>
                <div
                  className="mt-1 text-xs font-mono"
                  style={{
                    color: delta !== null && Math.abs(delta) > 0.5
                      ? "var(--color-rose)"
                      : "var(--color-text-2)",
                  }}
                >
                  Δ {delta !== null ? fmtSigned(delta, 2) : "—"}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Story callouts */}
      <div className="grid md:grid-cols-3 gap-4 mt-10">
        <Callout
          tone="violet"
          headline="The graph isn't vestigial under MiniLM."
          body="Under TF-IDF, w_graph collapsed to 0 — the graph was a typed-storage scaffold only. Under MiniLM, w_graph = 1.19 in the learned optimum. A separate sweep confirms reward peaks around w_graph = 1, not 0."
        />
        <Callout
          tone="amber"
          headline="Recency stops dominating."
          body="TF-IDF retrieval needed to fall back on recency because similarity was too noisy. MiniLM gives meaningful similarity, so w_embed and w_recency end up roughly equal."
        />
        <Callout
          tone="cyan"
          headline="Selectivity drops by half."
          body="θ_novel halves (0.91 → 0.44). With richer embeddings, the storage filter doesn't need to be as aggressive — semantically distinct events are already easier to distinguish at retrieval."
        />
      </div>

      {/* Headline stat */}
      <div className="grid md:grid-cols-3 gap-4 mt-8">
        <Stat
          label="reward (TF-IDF era)"
          value="0.178"
          tone="amber"
          hint="200 held-out eps · pre-PoC"
        />
        <Stat
          label="reward (MiniLM)"
          value={v4 ? v4.v4.eval.mean_reward.toFixed(3) : "—"}
          tone="violet"
          hint="100 held-out eps · re-tuned θ"
        />
        <Stat
          label="thesis story"
          value="θ depends on embedding"
          tone="cyan"
          hint="task-dependence claim, broadened"
        />
      </div>
    </Section>
  );
}

function MiniStat({ label, value, tone }: { label: string; value: string; tone: "amber" | "violet" }) {
  const accent =
    tone === "amber" ? "var(--color-amber)" : "var(--color-violet)";
  return (
    <div className="panel px-3 py-2">
      <div
        className="text-[0.6rem] uppercase tracking-[0.14em]"
        style={{ color: "var(--color-muted)" }}
      >
        {label}
      </div>
      <div className="mt-0.5 font-mono font-semibold tabular-nums" style={{ color: accent }}>
        {value}
      </div>
    </div>
  );
}

function Callout({
  tone, headline, body,
}: {
  tone: "violet" | "amber" | "cyan";
  headline: string;
  body: string;
}) {
  const accent =
    tone === "violet" ? "var(--color-violet)" :
    tone === "amber" ? "var(--color-amber)" :
    "var(--color-cyan)";
  return (
    <div
      className="panel-rise p-5 relative overflow-hidden"
      style={{
        background: `linear-gradient(135deg, ${accent}11, transparent 60%), var(--color-surface)`,
      }}
    >
      <div
        aria-hidden
        className="absolute left-0 top-0 bottom-0 w-1"
        style={{ background: accent }}
      />
      <h4 className="font-semibold text-base mb-2" style={{ color: accent }}>
        {headline}
      </h4>
      <p className="text-sm leading-relaxed" style={{ color: "var(--color-text-2)" }}>
        {body}
      </p>
    </div>
  );
}

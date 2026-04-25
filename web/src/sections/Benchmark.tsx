import { useState } from "react";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { ChartFrame } from "../components/shared/ChartFrame";
import { BenchmarkHeatmap } from "../components/interactive/BenchmarkHeatmap";
import { ParetoScatter } from "../components/viz/ParetoScatter";
import { MemoryGraph } from "../components/interactive/MemoryGraph";
import { useData } from "../data/useData";
import type { BenchmarkData, PairwiseData } from "../data/types";
import { Stat } from "../components/shared/Stat";
import { useEmbedding } from "../components/interactive/EmbeddingToggle";
import { fmt } from "../lib/format";

const METRIC_OPTIONS = [
  { id: "mean_reward", label: "reward" },
  { id: "retrieval_precision", label: "precision" },
  { id: "mean_memory_size", label: "mem size" },
  { id: "efficiency", label: "efficiency" },
] as const;
type Metric = (typeof METRIC_OPTIONS)[number]["id"];

export function Benchmark() {
  const { data: bench, loading } = useData<BenchmarkData>("benchmark.json");
  const { data: pair } = useData<PairwiseData>("pairwise.json");
  const { backend } = useEmbedding();
  const [metric, setMetric] = useState<Metric>("mean_reward");

  // Pull the V4 row + a few comparators for the headline stats panel.
  const v4 = bench?.["MultiHop-KeyDoor"]?.["GraphMemoryV4"];
  const v4Reward = v4?.mean_reward ?? null;
  const v4Precision = v4?.retrieval_precision ?? null;
  const v4MemSize = v4?.mean_memory_size ?? null;
  const numSystems = bench
    ? Object.keys(bench["MultiHop-KeyDoor"] ?? {}).filter((s) => !s.startsWith("_")).length
    : 12;

  const sigEntries = pair?.pairwise ? Object.entries(pair.pairwise) : [];

  return (
    <Section id="benchmark" eyebrow="Benchmark" variant="raised">
      <SectionHeader
        title={
          <>
            12 systems.
            <br />
            4 environments.
            <br />
            <span style={{ color: "var(--color-cyan)" }}>One contender.</span>
          </>
        }
        lede="GraphMemoryV4 with the post-MiniLM optimum is statistically tied with the leading systems on MultiHopKeyDoor — and stores 4× fewer events. The story isn't '#1', it's competitive efficiency."
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8 mb-12">
        <Stat
          label={backend === "minilm" ? "V4 reward (MiniLM)" : "V4 reward (TF-IDF)"}
          value={backend === "minilm"
            ? fmt(v4Reward, 3)
            : "0.178"}
          tone="cyan"
          hint={backend === "minilm" ? "100 held-out eps" : "200 held-out eps · pre-PoC"}
        />
        <Stat
          label="V4 precision"
          value={backend === "minilm"
            ? fmt(v4Precision, 3)
            : "0.997"}
          tone="emerald"
          hint="retrieval recovers the right hint"
        />
        <Stat
          label="memory size"
          value={backend === "minilm"
            ? `≈ ${fmt(v4MemSize, 0)}`
            : "≈ 10"}
          tone="violet"
          hint="V1 baseline: 218 events"
        />
        <Stat
          label="systems compared"
          value={String(numSystems)}
          hint={`across 4 environments · embedding: ${backend === "minilm" ? "MiniLM" : "TF-IDF"}`}
        />
      </div>

      {loading ? (
        <div className="text-center py-12" style={{ color: "var(--color-muted)" }}>
          Loading benchmark…
        </div>
      ) : bench ? (
        <div className="grid lg:grid-cols-[1.4fr_1fr] gap-6">
          <ChartFrame
            title="Reward heatmap"
            subtitle="systems × environments · click any cell for full stats"
            controls={
              <div className="flex gap-1 p-1 rounded-full border border-[var(--color-border)]">
                {METRIC_OPTIONS.map((opt) => (
                  <button
                    key={opt.id}
                    onClick={() => setMetric(opt.id)}
                    className="px-3 py-1 text-[0.65rem] uppercase tracking-[0.14em] rounded-full transition-colors"
                    style={{
                      color: metric === opt.id ? "var(--color-bg)" : "var(--color-text-2)",
                      background: metric === opt.id ? "var(--color-cyan)" : "transparent",
                    }}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            }
          >
            <BenchmarkHeatmap data={bench} metric={metric} />
          </ChartFrame>

          <ChartFrame
            title="Pareto: reward vs token cost"
            subtitle="MultiHop-KeyDoor · cyan = V4 (current θ)"
          >
            <ParetoScatter data={bench} env="MultiHop-KeyDoor" height={340} />
          </ChartFrame>
        </div>
      ) : null}

      {sigEntries.length > 0 ? (
        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <ChartFrame
            title="Pairwise significance (V4 vs others)"
            subtitle="paired t-test on per-episode rewards, 50 paired episodes each"
          >
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: "var(--color-muted)" }}>
                  <th className="text-left py-1.5 font-normal uppercase tracking-[0.14em] text-[0.65rem]">vs system</th>
                  <th className="text-right py-1.5 font-normal uppercase tracking-[0.14em] text-[0.65rem]">Δ reward</th>
                  <th className="text-right py-1.5 font-normal uppercase tracking-[0.14em] text-[0.65rem]">p-value</th>
                  <th className="text-right py-1.5 font-normal uppercase tracking-[0.14em] text-[0.65rem]">cohen&apos;s d</th>
                </tr>
              </thead>
              <tbody>
                {sigEntries.slice(0, 11).map(([sys, comp]) => {
                  const c = comp as { ttest: { p_value: number; significant: boolean }; cohens_d: { d: number; magnitude: string }; improvement: number };
                  const sig = c.ttest.p_value < 0.01 ? "**" : c.ttest.p_value < 0.05 ? "*" : "";
                  return (
                    <tr key={sys} className="border-t border-[var(--color-border)]">
                      <td className="py-1.5 font-mono">{sys}</td>
                      <td
                        className="py-1.5 text-right font-mono tabular-nums"
                        style={{
                          color: c.improvement > 0 ? "var(--color-emerald)" : "var(--color-rose)",
                        }}
                      >
                        {c.improvement >= 0 ? "+" : ""}{c.improvement.toFixed(3)}
                      </td>
                      <td className="py-1.5 text-right font-mono tabular-nums" style={{ color: sig ? "var(--color-cyan)" : "var(--color-text-2)" }}>
                        {c.ttest.p_value.toFixed(3)} {sig}
                      </td>
                      <td className="py-1.5 text-right font-mono tabular-nums" style={{ color: "var(--color-text-2)" }}>
                        {c.cohens_d.d.toFixed(2)} <span style={{ color: "var(--color-muted)" }}>({c.cohens_d.magnitude})</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="text-[0.65rem] mt-3" style={{ color: "var(--color-muted)" }}>
              * p &lt; 0.05 · ** p &lt; 0.01 · scipy.stats.ttest_rel
            </div>
          </ChartFrame>

          <ChartFrame
            title="What V4's memory looks like in flight"
            subtitle="event/entity graph (synthesized from V4 storage patterns) · slide to change store rate"
            flush
          >
            <MemoryGraph />
          </ChartFrame>
        </div>
      ) : null}
    </Section>
  );
}

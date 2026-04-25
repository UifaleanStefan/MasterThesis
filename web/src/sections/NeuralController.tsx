import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { ChartFrame } from "../components/shared/ChartFrame";
import { Stat } from "../components/shared/Stat";
import { Pill } from "../components/shared/Pill";
import { useData } from "../data/useData";
import type { NeuralV2Data } from "../data/types";
import { fmt } from "../lib/format";

export function NeuralController() {
  const { data } = useData<NeuralV2Data>("neural_v2.json");

  const history = data?.training?.history ?? [];

  return (
    <Section id="neural" eyebrow="Neural Meta-Controller" variant="raised">
      <SectionHeader
        title={
          <>
            A{" "}
            <span style={{ color: "var(--color-violet)" }}>5,674-parameter</span>{" "}
            MLP that emits θ
            <br />
            once per observation.
          </>
        }
        lede={
          <>
            The scalar V4 picks one θ for an entire episode. NeuralControllerV2Small
            picks a fresh θ <em>every step</em> from a 50-dim feature view of the
            current observation. It&apos;s warm-started from the V4 optimum and then
            fine-tuned with CMA-ES — 200 generations, 15 hours, ending at reward 0.19,
            slightly above scalar V4.
          </>
        }
      />

      <div className="flex flex-wrap gap-2 mb-8">
        <Pill tone="violet">5,674 params</Pill>
        <Pill tone="cyan">50 → 32 → 10 MLP</Pill>
        <Pill tone="emerald">CMA-ES 200 gens</Pill>
        <Pill tone="amber">~15 h training</Pill>
        <Pill tone="muted">warm-started from V4 (A3)</Pill>
      </div>

      <div className="grid lg:grid-cols-[1.5fr_1fr] gap-6">
        <ChartFrame
          title="CMA-ES learning curve · 200 generations"
          subtitle="best fitness per generation · σ on the secondary axis"
        >
          {history.length > 0 ? (
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={history} margin={{ top: 12, right: 12, bottom: 24, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.06)" />
                <XAxis
                  dataKey="generation"
                  stroke="var(--color-text-2)"
                  tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(255, 255, 255, 0.1)" }}
                />
                <YAxis
                  yAxisId="fit"
                  stroke="var(--color-violet)"
                  tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(167, 139, 250, 0.4)" }}
                  label={{
                    value: "best fitness ↑",
                    angle: -90,
                    position: "insideLeft",
                    offset: 16,
                    fill: "var(--color-violet)",
                    fontSize: 11,
                  }}
                />
                <YAxis
                  yAxisId="sig"
                  orientation="right"
                  stroke="var(--color-amber)"
                  tick={{ fontSize: 11, fontFamily: "var(--font-mono)" }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(245, 158, 11, 0.4)" }}
                  label={{
                    value: "σ (step size)",
                    angle: 90,
                    position: "insideRight",
                    offset: 16,
                    fill: "var(--color-amber)",
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const p = payload[0].payload as { generation: number; best_fitness: number; sigma: number };
                    return (
                      <div
                        className="panel px-3 py-2 text-xs font-mono"
                        style={{
                          background: "var(--color-surface)",
                          borderColor: "var(--color-border-strong)",
                        }}
                      >
                        <div style={{ color: "var(--color-text)" }}>gen {p.generation}</div>
                        <div style={{ color: "var(--color-violet)" }}>fit {p.best_fitness.toFixed(4)}</div>
                        <div style={{ color: "var(--color-amber)" }}>σ {p.sigma.toFixed(4)}</div>
                      </div>
                    );
                  }}
                />
                <Line
                  yAxisId="fit"
                  type="monotone"
                  dataKey="best_fitness"
                  stroke="var(--color-violet)"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive
                />
                <Line
                  yAxisId="sig"
                  type="monotone"
                  dataKey="sigma"
                  stroke="var(--color-amber)"
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                  dot={false}
                  isAnimationActive
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12" style={{ color: "var(--color-muted)" }}>
              loading…
            </div>
          )}
        </ChartFrame>

        <div className="space-y-4">
          <Stat
            label="MultiHop reward"
            value={fmt(data?.eval_multihop?.mean_reward, 3)}
            tone="violet"
            hint="100 held-out episodes · post-training"
          />
          <Stat
            label="vs scalar V4"
            value={
              data?.v4_scalar_comparison
                ? `+${(data.eval_multihop.mean_reward - data.v4_scalar_comparison.mean_reward).toFixed(3)}`
                : "—"
            }
            tone="cyan"
            hint="adaptive θ matches the scalar baseline"
          />
          <Stat
            label="MegaQuest (zero-shot)"
            value={fmt(data?.eval_megaquest?.mean_reward, 3)}
            tone="rose"
            hint="same OOD failure as scalar V4 — confirms it's policy, not memory"
          />
          <Stat
            label="precision (MultiHop)"
            value={fmt(data?.eval_multihop?.mean_precision, 3)}
            tone="emerald"
            hint="memory still recovers the right hint"
          />
        </div>
      </div>

      {/* Insight callout */}
      <div className="panel-rise p-6 mt-6">
        <h4 className="font-semibold text-base mb-2" style={{ color: "var(--color-violet)" }}>
          Expressivity vs trainability.
        </h4>
        <p className="text-sm leading-relaxed" style={{ color: "var(--color-text-2)" }}>
          A 1,962-param controller has thousand-fold more expressivity than the scalar V4&apos;s
          10D θ. But CMA-ES on 1,962 dimensions is hard. The pretrain-from-V4 trick
          (initializing the MLP&apos;s output bias to logit(V4&apos;s θ)) makes generation 0 already
          equivalent to scalar V4, then lets the optimizer explore from a competitive baseline.
          200 generations is enough to find a reward parity; getting meaningfully <em>past</em>{" "}
          scalar V4 likely needs a more sample-efficient method (REINFORCE, ES with importance
          mixing) — open question for the thesis writeup.
        </p>
      </div>
    </Section>
  );
}

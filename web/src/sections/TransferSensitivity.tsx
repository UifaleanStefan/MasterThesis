import { motion } from "framer-motion";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { ChartFrame } from "../components/shared/ChartFrame";
import { Stat } from "../components/shared/Stat";
import { Pill } from "../components/shared/Pill";
import { SensitivityHeatmap } from "../components/viz/SensitivityHeatmap";
import { useData } from "../data/useData";
import type { TransferData, SensitivityData } from "../data/types";
import { fmt } from "../lib/format";

const ENVS_ORDER = ["MultiHopKeyDoor", "GoalRoom", "HardKeyDoor", "MegaQuestRoom"];

export function TransferSensitivity() {
  const { data: transfer } = useData<TransferData>("transfer.json");
  const { data: sens } = useData<SensitivityData>("sensitivity.json");

  const zeroshot = transfer?.matrix?.["MultiHop_V4_zeroshot"] ?? {};
  const clamps = transfer?.matrix?.["MegaQuest_w_recency_clamp"] ?? null;

  return (
    <Section id="transfer" eyebrow="Transfer & Sensitivity">
      <SectionHeader
        title={
          <>
            The learned θ <span style={{ color: "var(--color-emerald)" }}>generalizes</span> —
            and breaks honestly.
          </>
        }
        lede="V4's MultiHop θ transfers strongly to GoalRoom (+0.69 on a simpler task), modestly to HardKeyDoor, and fails on MegaQuestRoom — but precision stays at 0.94+. The failure is policy/horizon, not memory."
      />

      {/* Transfer cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
        {ENVS_ORDER.map((env, i) => {
          const result = zeroshot[env];
          const reward = result?.mean_reward ?? null;
          const precision = result?.mean_precision ?? null;
          const tone = transferTone(reward);
          return (
            <motion.div
              key={env}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.05 }}
              className="panel-rise p-5"
              style={{
                background:
                  reward !== null && reward > 0.5
                    ? "linear-gradient(135deg, var(--color-emerald-soft), transparent), var(--color-surface)"
                    : reward === 0
                    ? "linear-gradient(135deg, var(--color-rose-soft), transparent), var(--color-surface)"
                    : "var(--color-surface)",
              }}
            >
              <div
                className="text-[0.65rem] uppercase tracking-[0.16em] mb-2"
                style={{ color: "var(--color-muted)" }}
              >
                {env === "MultiHopKeyDoor" ? "in-distribution" : "zero-shot transfer"}
              </div>
              <div className="text-base font-mono mb-3">{env}</div>
              <div className="text-3xl font-semibold tabular-nums" style={{ color: tone }}>
                {fmt(reward, 3)}
              </div>
              <div
                className="mt-1 text-xs"
                style={{ color: "var(--color-text-2)" }}
              >
                reward · precision {fmt(precision, 2)}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* The MegaQuest finding */}
      {clamps && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="panel-rise p-6 mt-6"
          style={{
            background:
              "linear-gradient(135deg, var(--color-rose-soft) 0%, transparent 60%), var(--color-surface)",
          }}
        >
          <div className="flex flex-wrap items-baseline gap-3 mb-3">
            <Pill tone="rose">A2 finding</Pill>
            <h3 className="text-xl font-semibold">MegaQuest fails for a non-memory reason.</h3>
          </div>
          <p className="text-sm leading-relaxed mb-5" style={{ color: "var(--color-text-2)" }}>
            We hypothesized that MegaQuest&apos;s 1000-step horizon overwhelms recency-dominated
            retrieval. So we clamped <code>w_recency</code> to 0.5 / 1.0 / 2.0 and re-ran.
            <strong style={{ color: "var(--color-rose)" }}> Reward stays at 0.0</strong> — but
            precision is 0.93–0.99 across all settings. Memory <em>is</em> retrieving the right
            hints. The bottleneck is exploration / planning over the long horizon, not memory
            structure.
          </p>
          <div className="grid grid-cols-4 gap-3">
            {Object.entries(clamps).map(([clamp, r]) => {
              const result = r as { mean_reward: number; mean_precision: number };
              return (
                <div key={clamp} className="panel px-3 py-2.5">
                  <div className="text-[0.6rem] font-mono" style={{ color: "var(--color-muted)" }}>
                    {clamp}
                  </div>
                  <div className="mt-1 font-mono">
                    <div className="font-semibold tabular-nums" style={{ color: "var(--color-rose)" }}>
                      {fmt(result.mean_reward, 3)}
                    </div>
                    <div className="text-xs" style={{ color: "var(--color-emerald)" }}>
                      prec {fmt(result.mean_precision, 2)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Sensitivity */}
      {sens && (
        <div className="grid lg:grid-cols-[1fr_1.1fr] gap-6 mt-10">
          <ChartFrame
            title="reward landscape · θ_novel × w_recency"
            subtitle={`${sens.config.resolution}×${sens.config.resolution} grid · ${sens.config.n_episodes_per_cell} eps/cell`}
            flush
          >
            <div className="p-4 flex justify-center">
              <SensitivityHeatmap data={sens} size={320} />
            </div>
          </ChartFrame>
          <div className="space-y-4">
            <Stat
              label="best cell reward"
              value={fmt(sens.best_reward, 3)}
              tone="cyan"
              hint={`at θ_novel=${fmt(sens.best_params_dict?.theta_novel, 2)}, w_recency=${fmt(sens.best_params_dict?.w_recency, 2)}`}
            />
            <Stat
              label="landscape shape"
              value={sens.analysis?.is_sharp_peak ? "sharp peak" : "broad plateau"}
              tone={sens.analysis?.is_sharp_peak ? "rose" : "emerald"}
              hint={
                sens.analysis?.is_sharp_peak
                  ? "optimization is tight; small changes degrade reward"
                  : "many configurations achieve near-optimal reward"
              }
            />
            <Stat
              label="top-10% std"
              value={fmt((sens.analysis?.top_10_std as number) ?? null, 3)}
              tone="violet"
              hint="dispersion of the top-10% of cells"
            />
          </div>
        </div>
      )}
    </Section>
  );
}

function transferTone(reward: number | null): string {
  if (reward === null) return "var(--color-muted)";
  if (reward === 0) return "var(--color-rose)";
  if (reward > 0.5) return "var(--color-emerald)";
  if (reward > 0.1) return "var(--color-cyan)";
  return "var(--color-amber)";
}

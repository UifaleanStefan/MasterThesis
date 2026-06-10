/**
 * BenchmarkCorpus — Generalized Phase 1.9 corpus-QA results panel.
 *
 * Accepts a `bench` prop (e.g. "qasper", "cuad", "longmemeval") and loads
 * the corresponding stage3_{bench}_corpus.json from public/data.
 *
 * Shows: Protocol A judge table (batch + online), Protocol B calibration
 * summary, and theta-shift badge. Used for all non-FinanceBench benchmarks.
 */

import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { ChartFrame } from "../components/shared/ChartFrame";
import { cn } from "../lib/format";

// ── Types ────────────────────────────────────────────────────────────────────

interface ConfigMode {
  n: number;
  claude_judge_mean: number | null;
  recall_at_k_8: number | null;
}

interface ConfigData {
  key: string;
  label: string;
  family: "v4" | "baseline";
  modes: Record<string, ConfigMode>;
}

interface ProtocolBCell {
  overall_mean: number;
  n: number;
  answer_mean: number | null;
  answer_n: number;
  acknowledge_missing_mean: number | null;
  acknowledge_missing_n: number;
}

interface ProtocolBConfig {
  calibration: ProtocolBCell;
  batch_calib: ProtocolBCell;
}

interface ThetaContrast {
  canonical: Record<string, number>;
  corpus_tuned: Record<string, number | null> | null;
  four_shift_score: string | null;
}

interface Headline {
  v4t_canonical_batch: number | null;
  v4t_corpus_tuned_batch: number | null;
  lift: number | null;
  four_shifts_replicated: string | null;
}

interface BenchmarkCorpusData {
  _manifest: {
    benchmark: string;
    label: string;
    description: string;
    n_docs: number;
    total_judgments: number;
    judge_model: string;
  };
  configs: ConfigData[];
  modes: string[];
  judge_table: Record<string, Record<string, number>>;
  calibration_data: Record<string, ProtocolBConfig>;
  theta_contrast: ThetaContrast;
  headline: Headline;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, decimals = 3): string {
  if (v == null) return "—";
  return v.toFixed(decimals);
}

function liftColor(lift: number | null): string {
  if (lift == null) return "text-[var(--color-fg-muted)]";
  if (lift > 0.05) return "text-[var(--color-green)]";
  if (lift < -0.05) return "text-[var(--color-red)]";
  return "text-[var(--color-fg-muted)]";
}

function barWidth(v: number | null, max = 1.0): string {
  if (v == null) return "0%";
  return `${Math.round((v / max) * 100)}%`;
}

const FAMILY_COLORS: Record<string, string> = {
  v4: "var(--color-cyan)",
  baseline: "var(--color-amber)",
};

// ── Component ────────────────────────────────────────────────────────────────

interface BenchmarkCorpusProps {
  bench: string;
  sectionId?: string;
}

export function BenchmarkCorpus({ bench, sectionId }: BenchmarkCorpusProps) {
  const [data, setData] = useState<BenchmarkCorpusData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/data/stage3_${bench}_corpus.json`)
      .then((r) => r.json())
      .then((d: BenchmarkCorpusData) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => {
        setData(null);
        setLoading(false);
      });
  }, [bench]);

  if (loading) {
    return (
      <Section id={sectionId ?? `corpus-${bench}`} variant="raised">
        <div className="text-[var(--color-fg-muted)] text-sm">Loading {bench} data…</div>
      </Section>
    );
  }
  if (!data) {
    return (
      <Section id={sectionId ?? `corpus-${bench}`} variant="raised">
        <div className="text-[var(--color-fg-muted)] text-sm">
          Data not available for {bench}.
        </div>
      </Section>
    );
  }

  const m = data._manifest;
  const hasOnline = data.configs.some((c) => "online" in c.modes);
  const hasProtocolB = Object.keys(data.calibration_data).length > 0;
  const lift = data.headline.lift;
  const shiftScore = data.theta_contrast.four_shift_score;

  // Sort configs by batch judge mean descending
  const sortedConfigs = [...data.configs].sort((a, b) => {
    const am = a.modes.batch?.claude_judge_mean ?? -1;
    const bm = b.modes.batch?.claude_judge_mean ?? -1;
    return bm - am;
  });

  return (
    <Section id={sectionId ?? `corpus-${bench}`} variant="raised">
      {/* ── Header ── */}
      <SectionHeader
        title={
          <span>
            {m.label}{" "}
            <span className="text-[var(--color-cyan)]">Corpus-Cumulative Results</span>
          </span>
        }
        lede={
          <span>
            {m.description} · {m.total_judgments.toLocaleString()} Claude-judged entries · n_docs={m.n_docs}
          </span>
        }
      />

      {/* ── Headline pills ── */}
      <div className="flex flex-wrap gap-3 mb-10">
        <span className="pill">
          <span style={{ color: "var(--color-cyan)" }}>●</span> V4ₜ corpus-tuned{" "}
          {fmt(data.headline.v4t_corpus_tuned_batch)}
        </span>
        <span
          className={cn(
            "pill",
            liftColor(lift),
          )}
        >
          {lift != null && lift > 0 ? <TrendingUp className="inline w-3 h-3 mr-1" /> :
           lift != null && lift < 0 ? <TrendingDown className="inline w-3 h-3 mr-1" /> :
           <Minus className="inline w-3 h-3 mr-1" />}
          vs canonical: {lift != null ? `${lift >= 0 ? "+" : ""}${fmt(lift)}` : "—"}
        </span>
        {shiftScore && (
          <span className="pill">
            <span style={{ color: "var(--color-green)" }}>●</span> four-shift: {shiftScore}
          </span>
        )}
      </div>

      {/* ── Protocol A judge table ── */}
      <ChartFrame title="Protocol A — End-of-corpus QA (Claude Opus 4.7 judge)">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)]">
                <th className="text-left py-2 pr-4 font-medium text-[var(--color-fg-muted)]">Config</th>
                <th className="text-right py-2 px-3 font-medium text-[var(--color-fg-muted)]">Batch</th>
                {hasOnline && (
                  <th className="text-right py-2 px-3 font-medium text-[var(--color-fg-muted)]">Online</th>
                )}
                {hasOnline && (
                  <th className="text-right py-2 px-3 font-medium text-[var(--color-fg-muted)] hidden md:table-cell">O−B</th>
                )}
                <th className="text-left py-2 pl-3 font-medium text-[var(--color-fg-muted)] hidden lg:table-cell w-32">Bar</th>
              </tr>
            </thead>
            <tbody>
              {sortedConfigs.map((cfg) => {
                const batchMean = cfg.modes.batch?.claude_judge_mean ?? null;
                const onlineMean = cfg.modes.online?.claude_judge_mean ?? null;
                const gap = (onlineMean != null && batchMean != null) ? onlineMean - batchMean : null;
                const accent = FAMILY_COLORS[cfg.family] ?? "var(--color-fg)";
                return (
                  <tr
                    key={cfg.key}
                    className="border-b border-[var(--color-border)] border-opacity-40 hover:bg-[var(--color-bg-rise)] transition-colors duration-100"
                  >
                    <td className="py-2 pr-4 font-mono text-xs" style={{ color: accent }}>
                      {cfg.label}
                    </td>
                    <td className="py-2 px-3 text-right tabular-nums font-semibold">
                      {fmt(batchMean)}
                    </td>
                    {hasOnline && (
                      <td className="py-2 px-3 text-right tabular-nums text-[var(--color-fg-muted)]">
                        {fmt(onlineMean)}
                      </td>
                    )}
                    {hasOnline && (
                      <td
                        className={cn(
                          "py-2 px-3 text-right tabular-nums hidden md:table-cell",
                          gap != null && gap > 0 ? "text-[var(--color-green)]" : "text-[var(--color-fg-muted)]",
                        )}
                      >
                        {gap != null ? `${gap >= 0 ? "+" : ""}${fmt(gap)}` : "—"}
                      </td>
                    )}
                    <td className="py-2 pl-3 hidden lg:table-cell">
                      <div className="h-2 rounded-full bg-[var(--color-border)] w-32 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: barWidth(batchMean),
                            backgroundColor: accent,
                            opacity: 0.8,
                          }}
                        />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ChartFrame>

      {/* ── Protocol B summary ── */}
      {hasProtocolB && (
        <ChartFrame title="Protocol B — Calibration trajectory + end-of-corpus batch_calib" className="mt-8">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--color-border)]">
                  <th className="text-left py-2 pr-4 font-medium text-[var(--color-fg-muted)]">Config</th>
                  <th className="text-right py-2 px-3 font-medium text-[var(--color-fg-muted)]">Calib overall</th>
                  <th className="text-right py-2 px-3 font-medium text-[var(--color-fg-muted)]">Ack ↑</th>
                  <th className="text-right py-2 px-3 font-medium text-[var(--color-fg-muted)]">Ans ↑</th>
                  <th className="text-right py-2 px-3 font-medium text-[var(--color-fg-muted)]">Batch_calib</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.calibration_data)
                  .sort(([, a], [, b]) => {
                    const am = a.batch_calib?.overall_mean ?? -1;
                    const bm = b.batch_calib?.overall_mean ?? -1;
                    return bm - am;
                  })
                  .map(([cfgKey, pb]) => {
                    const cfgMeta = data.configs.find((c) => c.key === cfgKey);
                    const accent = FAMILY_COLORS[cfgMeta?.family ?? "baseline"] ?? "var(--color-fg)";
                    return (
                      <tr
                        key={cfgKey}
                        className="border-b border-[var(--color-border)] border-opacity-40 hover:bg-[var(--color-bg-rise)] transition-colors duration-100"
                      >
                        <td className="py-2 pr-4 font-mono text-xs" style={{ color: accent }}>
                          {cfgMeta?.label ?? cfgKey}
                        </td>
                        <td className="py-2 px-3 text-right tabular-nums text-[var(--color-fg-muted)]">
                          {fmt(pb.calibration?.overall_mean)}
                        </td>
                        <td className="py-2 px-3 text-right tabular-nums text-[var(--color-fg-muted)]">
                          {fmt(pb.calibration?.acknowledge_missing_mean)}
                        </td>
                        <td className="py-2 px-3 text-right tabular-nums text-[var(--color-fg-muted)]">
                          {fmt(pb.calibration?.answer_mean)}
                        </td>
                        <td className="py-2 px-3 text-right tabular-nums font-semibold">
                          {fmt(pb.batch_calib?.overall_mean)}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-[var(--color-fg-muted)]">
            Ack = acknowledge_missing (honest refusal on uningested docs). Ans = answer (retrieval fidelity on ingested docs).
          </p>
        </ChartFrame>
      )}

      {/* ── Four-shift badge ── */}
      {data.theta_contrast.corpus_tuned && (
        <div className="mt-8 p-5 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-rise)]">
          <h4 className="text-sm font-semibold mb-3" style={{ color: "var(--color-cyan)" }}>
            θ Four-shift: {shiftScore ?? "—"}
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {(["w_recency", "w_embed", "theta_store", "w_graph"] as const).map((param) => {
              const canon = data.theta_contrast.canonical[param];
              const tuned = data.theta_contrast.corpus_tuned?.[param];
              const expected = param === "w_embed" || param === "w_graph" ? "↑" : "↓";
              const actual =
                tuned == null
                  ? "?"
                  : param === "w_embed" || param === "w_graph"
                  ? tuned > canon ? "↑" : "↓"
                  : tuned < canon ? "↓" : "↑";
              const match = actual === expected;
              return (
                <div key={param} className="text-center">
                  <div className="text-xs text-[var(--color-fg-muted)] mb-1 font-mono">{param}</div>
                  <div className="tabular-nums text-sm">
                    {fmt(canon, 3)} → {tuned != null ? fmt(tuned, 3) : "—"}
                  </div>
                  <div
                    className="text-lg font-bold mt-1"
                    style={{ color: match ? "var(--color-green)" : "var(--color-red)" }}
                  >
                    {actual !== "?" ? actual : "?"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </Section>
  );
}

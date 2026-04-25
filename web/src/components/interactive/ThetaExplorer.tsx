/**
 * ThetaExplorer — 10 sliders for the V4 θ vector. Live-updates a memory-size
 * estimate, a retrieval-focus pie, and a predicted reward gauge interpolated
 * from the sensitivity grid (when available).
 *
 * Reset / preset buttons let the user snap to TF-IDF / MiniLM optima or to
 * an "all defaults" zero vector to see how the system behaves.
 */

import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import type { V4Theta, SensitivityData, V4CmaesData } from "../../data/types";
import { useData } from "../../data/useData";
import { TFIDF_V4_THETA } from "../../data/tfidf_constants";
import { fmt, fmtPct, cn } from "../../lib/format";
import { bilinear, type Grid2D } from "../../lib/interpolate";
import { ThetaBars } from "../viz/ThetaBars";
import { useEmbedding } from "./EmbeddingToggle";

const ZERO_THETA: V4Theta = {
  theta_store: 0, theta_novel: 0, theta_erich: 0, theta_surprise: 0,
  theta_entity: 0, theta_temporal: 1, theta_decay: 0,
  w_graph: 1.5, w_embed: 1.0, w_recency: 0.2,
};

interface SliderRowProps {
  label: string;
  description: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  tone: "store" | "decay" | "retrieve";
}

const TONE_VAR = {
  store: "var(--color-cyan)",
  decay: "var(--color-emerald)",
  retrieve: "var(--color-violet)",
};

function SliderRow({ label, description, value, min, max, step = 0.01, onChange, tone }: SliderRowProps) {
  const t = (value - min) / (max - min);
  const accent = TONE_VAR[tone];
  return (
    <div className="grid grid-cols-[150px_1fr_56px] items-center gap-3 py-1.5">
      <div>
        <div className="font-mono text-xs" style={{ color: "var(--color-text)" }}>
          {label}
        </div>
        <div className="text-[0.65rem]" style={{ color: "var(--color-muted)" }}>
          {description}
        </div>
      </div>
      <div className="relative h-5">
        <div
          className="absolute inset-y-0 left-0 right-0 rounded-full m-auto"
          style={{
            height: 4,
            top: "50%",
            transform: "translateY(-50%)",
            background: "rgba(255, 255, 255, 0.06)",
          }}
        />
        <div
          className="absolute rounded-full pointer-events-none"
          style={{
            height: 4,
            top: "50%",
            transform: "translateY(-50%)",
            left: 0,
            width: `${t * 100}%`,
            background: accent,
            boxShadow: `0 0 6px ${accent}55`,
          }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full h-full appearance-none bg-transparent cursor-pointer"
          style={{ accentColor: accent }}
        />
      </div>
      <div
        className="font-mono text-xs text-right tabular-nums"
        style={{ color: accent }}
      >
        {value.toFixed(2)}
      </div>
    </div>
  );
}

export function ThetaExplorer() {
  const { backend } = useEmbedding();
  const { data: sens } = useData<SensitivityData>("sensitivity.json");
  const { data: v4 } = useData<V4CmaesData>("v4_cmaes.json");

  const minilmTheta = v4?.v4?.best_params ?? null;
  const presetTheta = backend === "minilm" && minilmTheta ? minilmTheta : TFIDF_V4_THETA;

  const [theta, setTheta] = useState<V4Theta>(presetTheta);

  // When the user toggles the embedding backend, snap θ to the new optimum.
  useEffect(() => {
    setTheta(presetTheta);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backend, minilmTheta]);

  function set<K extends keyof V4Theta>(key: K, value: V4Theta[K]) {
    setTheta((t) => ({ ...t, [key]: value }));
  }

  // Derived UI signals.
  const storedFraction = useMemo(() => {
    // Importance score = sum of weighted features. With novelty assumed ~0.6
    // (mid-range typical observation) we estimate fraction stored.
    const allWeightsZero =
      theta.theta_novel === 0 &&
      theta.theta_erich === 0 &&
      theta.theta_surprise === 0;
    if (allWeightsZero) return 1.0; // V1 fast-path: store everything.
    const importance =
      0.6 * theta.theta_novel + 0.5 * theta.theta_erich + 0.5 * theta.theta_surprise;
    if (importance <= theta.theta_store) return 0.0;
    return Math.max(0, Math.min(1, (importance - theta.theta_store) / Math.max(importance, 0.01)));
  }, [theta]);

  // Predicted reward via bilinear interpolation on the 2D sensitivity grid.
  // Sensitivity dims: theta_novel × w_recency. Other dims are held implicit.
  const predictedReward = useMemo(() => {
    if (!sens) return null;
    const grid: Grid2D = {
      rows: sens.dim1_values,
      cols: sens.dim2_values,
      values: sens.reward_grid,
    };
    return bilinear(grid, theta.theta_novel, theta.w_recency);
  }, [sens, theta]);

  // Retrieval focus split (relative weights summed).
  const totalW = theta.w_graph + theta.w_embed + theta.w_recency || 0.001;
  const focusGraph = theta.w_graph / totalW;
  const focusEmbed = theta.w_embed / totalW;
  const focusRecency = theta.w_recency / totalW;

  return (
    <div className="grid lg:grid-cols-[1fr_1.1fr] gap-6">
      {/* SLIDERS */}
      <div className="panel-rise p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-semibold">Adjust θ</h3>
          <div className="flex items-center gap-2">
            <PresetButton onClick={() => setTheta(presetTheta)} label="reset" />
            <PresetButton onClick={() => setTheta(ZERO_THETA)} label="default" />
          </div>
        </div>

        <Group label="Storage filter" tone="store">
          <SliderRow tone="store" label="theta_store"     description="importance threshold" min={0} max={1} value={theta.theta_store}    onChange={(v) => set("theta_store", v)} />
          <SliderRow tone="store" label="theta_novel"     description="novelty weight"       min={0} max={1} value={theta.theta_novel}    onChange={(v) => set("theta_novel", v)} />
          <SliderRow tone="store" label="theta_erich"     description="entity richness"       min={0} max={1} value={theta.theta_erich}    onChange={(v) => set("theta_erich", v)} />
          <SliderRow tone="store" label="theta_surprise"  description="context surprise"      min={0} max={1} value={theta.theta_surprise} onChange={(v) => set("theta_surprise", v)} />
          <SliderRow tone="store" label="theta_entity"    description="entity node threshold" min={0} max={1} value={theta.theta_entity}   onChange={(v) => set("theta_entity", v)} />
          <SliderRow tone="store" label="theta_temporal"  description="edge probability"      min={0} max={1} value={theta.theta_temporal} onChange={(v) => set("theta_temporal", v)} />
        </Group>

        <Group label="Decay" tone="decay">
          <SliderRow tone="decay" label="theta_decay" description="entity decay rate" min={0} max={1} value={theta.theta_decay} onChange={(v) => set("theta_decay", v)} />
        </Group>

        <Group label="Retrieval weights" tone="retrieve">
          <SliderRow tone="retrieve" label="w_graph"   description="graph-traversal signal" min={0} max={4} value={theta.w_graph}   onChange={(v) => set("w_graph", v)} />
          <SliderRow tone="retrieve" label="w_embed"   description="embedding similarity"   min={0} max={4} value={theta.w_embed}   onChange={(v) => set("w_embed", v)} />
          <SliderRow tone="retrieve" label="w_recency" description="recency"                min={0} max={4} value={theta.w_recency} onChange={(v) => set("w_recency", v)} />
        </Group>
      </div>

      {/* PREVIEW */}
      <div className="space-y-4">
        <div className="panel-rise p-5">
          <div className="text-[0.65rem] uppercase tracking-[0.18em] mb-3" style={{ color: "var(--color-muted)" }}>
            current θ vector
          </div>
          <ThetaBars theta={theta} showLabels height={200} />
        </div>

        <div className="grid grid-cols-3 gap-3">
          <PreviewStat
            label="stored"
            value={fmtPct(storedFraction, 0)}
            tone="cyan"
            hint="of incoming events"
          />
          <PreviewStat
            label="reward"
            value={predictedReward !== null ? fmt(predictedReward, 3) : "—"}
            tone="violet"
            hint={predictedReward !== null ? "interpolated from sensitivity" : "sensitivity grid loading"}
          />
          <PreviewStat
            label="dominant retrieval"
            value={(() => {
              const m = Math.max(focusGraph, focusEmbed, focusRecency);
              if (m === focusRecency) return "recency";
              if (m === focusEmbed) return "embed";
              return "graph";
            })()}
            tone="emerald"
            hint={`graph ${fmtPct(focusGraph, 0)} · embed ${fmtPct(focusEmbed, 0)} · recency ${fmtPct(focusRecency, 0)}`}
          />
        </div>
      </div>
    </div>
  );
}

function Group({ label, tone, children }: { label: string; tone: "store" | "decay" | "retrieve"; children: React.ReactNode }) {
  return (
    <div className="mt-5 first:mt-0">
      <div
        className="flex items-center gap-2 mb-2 text-[0.65rem] uppercase tracking-[0.18em]"
        style={{ color: TONE_VAR[tone] }}
      >
        <span className="block w-1.5 h-1.5 rounded-full" style={{ background: TONE_VAR[tone] }} />
        {label}
      </div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function PresetButton({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      className="px-2.5 py-1 text-[0.65rem] uppercase tracking-[0.14em] rounded-md hover:bg-[rgba(255,255,255,0.04)] transition-colors"
      style={{ color: "var(--color-text-2)", border: "1px solid var(--color-border)" }}
    >
      {label}
    </button>
  );
}

function PreviewStat({ label, value, tone, hint }: { label: string; value: string; tone: "cyan" | "violet" | "emerald"; hint: string }) {
  const accent =
    tone === "cyan" ? "var(--color-cyan)" :
    tone === "violet" ? "var(--color-violet)" : "var(--color-emerald)";
  return (
    <motion.div
      key={value}
      initial={{ opacity: 0.6 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className={cn("panel px-4 py-3")}
    >
      <div className="text-[0.6rem] uppercase tracking-[0.16em] mb-1" style={{ color: "var(--color-muted)" }}>
        {label}
      </div>
      <div className="text-xl font-semibold tabular-nums" style={{ color: accent }}>
        {value}
      </div>
      <div className="text-[0.65rem] mt-1" style={{ color: "var(--color-text-2)" }}>
        {hint}
      </div>
    </motion.div>
  );
}

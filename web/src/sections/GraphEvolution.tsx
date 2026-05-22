/**
 * Graph Evolution Timeline — Stage 3 Phase 2 magnum opus viewer.
 *
 * Visualizes one V4ₜ corpus-mode memory instance ingesting every paragraph
 * of every document in a benchmark sequentially. Force-directed graph,
 * timeline scrubber, growth curves, top-entities tracker — all driven by
 * the snapshots emitted by scripts/run_corpus_ingestion.py.
 *
 * Data flow:
 *   results/stage3/corpus_traces/<run_id>/
 *     snapshots.json, final_graph.json, meta.json
 *   ─ scripts/build_graph_evolution_frontend.py ─►
 *   web/public/data/graph_evolution/<run_id>/
 *     timeline.json, graph.json, manifest.json
 *   ─ this component ─► force-directed render
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Network,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  Sparkles,
} from "lucide-react";
import ForceGraph2D from "react-force-graph-2d";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { Pill } from "../components/shared/Pill";

// ----------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------

type RunIndex = {
  version: number;
  runs: Array<{
    run_id: string;
    benchmark: string;
    benchmark_category: string;
    config: string;
    order: string;
    n_docs: number;
    n_snapshots: number;
    n_nodes: number;
    n_edges: number;
  }>;
  benchmarks: string[];
  configs: string[];
  orders: string[];
};

type Snapshot = {
  step: number;
  doc_idx: number;
  para_idx: number;
  doc_title: string;
  kind: string;
  added: number;
  seen: number;
  n_events: number;
  n_entities: number;
  n_temporal_edges: number;
  n_mention_edges: number;
  top_entities: Array<{ name: string; count: number; last_step: number }>;
  n_unique_entities_seen: number;
};

type GraphNode = {
  id: string;
  type: "event" | "entity";
  first_seen_step: number;
  // event-only:
  step?: number;
  observation?: string;
  // entity-only:
  mention_count?: number;
  last_step?: number;
};

type GraphEdge = {
  source: string;
  target: string;
  edge_type: "temporal" | "mentions" | "mentioned_in" | "other";
};

type FullGraph = {
  total_nodes: number;
  total_edges: number;
  kept_nodes: number;
  kept_edges: number;
  capped: boolean;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

type Manifest = {
  benchmark: string;
  benchmark_category: string;
  config: string;
  v4_variant: string;
  order: string;
  shuffle_seed: number | null;
  w_recency_zero: boolean;
  seed: number;
  n_docs: number;
  limit_docs: number | null;
  total_paragraphs_seen: number;
  total_paragraphs_stored: number;
  store_ratio: number;
  final_n_event_nodes: number;
  final_n_entity_nodes: number;
  final_n_edges: number;
  final_n_unique_entities: number;
  params: Record<string, number>;
  doc_boundaries: Array<{
    doc_idx: number;
    doc_title: string;
    global_step_start: number;
    global_step_end: number;
    n_paragraphs: number;
    events_added: number;
  }>;
  elapsed_seconds: number;
};

// ----------------------------------------------------------------------
// Entity color coding (visual story: which entity TYPE matters where)
// ----------------------------------------------------------------------

function entityColor(id: string): string {
  // Categorize entity by name pattern. Used to color-code the graph
  // visually so the viewer can see e.g. CUAD's "Section X" clusters
  // emerge separately from QASPER's proper-noun clusters.
  if (id.startsWith("year_")) return "#60a5fa"; // blue — years
  if (id.startsWith("money_")) return "#34d399"; // emerald — dollar amounts
  if (id.startsWith("section_") || id.startsWith("article_") || id.startsWith("clause_"))
    return "#f59e0b"; // amber — section refs
  if (id.startsWith("q") && /^q\d/.test(id)) return "#a78bfa"; // violet — quarters
  if (id.startsWith("fy")) return "#a78bfa"; // violet — fiscal year
  return "#fbbf24"; // light yellow — proper-noun entities (default)
}

const EVENT_COLOR = "#475569"; // slate — event nodes
const TEMPORAL_EDGE_COLOR = "rgba(148, 163, 184, 0.25)"; // soft grey
const MENTION_EDGE_COLOR = "rgba(245, 158, 11, 0.35)"; // amber

// ----------------------------------------------------------------------
// Component
// ----------------------------------------------------------------------

export function GraphEvolution() {
  const [index, setIndex] = useState<RunIndex | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<Snapshot[] | null>(null);
  const [graph, setGraph] = useState<FullGraph | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);

  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [hoverEntity, setHoverEntity] = useState<string | null>(null);

  // Load index on mount
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/graph_evolution/index.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((idx) => {
        setIndex(idx);
        if (idx?.runs?.length && selectedRunId === null) {
          // Default to a v4t-corpus-tuned run if available, else first.
          const preferred =
            idx.runs.find((r: { config: string }) => r.config === "v4t-corpus-tuned")?.run_id ??
            idx.runs[0].run_id;
          setSelectedRunId(preferred);
        }
      })
      .catch(() => setIndex(null));
  }, []);

  // Load selected run
  useEffect(() => {
    if (!selectedRunId) return;
    const base = `${import.meta.env.BASE_URL}data/graph_evolution/${selectedRunId}`;
    Promise.all([
      fetch(`${base}/timeline.json`).then((r) => r.json()),
      fetch(`${base}/graph.json`).then((r) => r.json()),
      fetch(`${base}/manifest.json`).then((r) => r.json()),
    ])
      .then(([tl, g, m]) => {
        setTimeline(tl);
        setGraph(g);
        setManifest(m);
        setStepIdx(0);
        setPlaying(false);
      })
      .catch(() => {
        setTimeline(null);
        setGraph(null);
        setManifest(null);
      });
  }, [selectedRunId]);

  // Play loop
  useEffect(() => {
    if (!playing || !timeline) return;
    const interval = setInterval(() => {
      setStepIdx((i) => {
        if (i >= timeline.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, Math.max(40, 400 / speed));
    return () => clearInterval(interval);
  }, [playing, timeline, speed]);

  const currentSnapshot = useMemo(() => {
    if (!timeline) return null;
    return timeline[Math.min(stepIdx, timeline.length - 1)] ?? null;
  }, [timeline, stepIdx]);

  // Filter graph by current step — show only nodes/edges that existed by
  // currentSnapshot.step. This is the key animation primitive.
  const visibleGraph = useMemo(() => {
    if (!graph || !currentSnapshot) return { nodes: [], links: [] };
    const cutoff = currentSnapshot.step;
    const visibleNodes = graph.nodes.filter((n) => n.first_seen_step <= cutoff);
    const visibleNodeIds = new Set(visibleNodes.map((n) => n.id));
    const visibleEdges = graph.edges.filter(
      (e) => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target),
    );
    return {
      nodes: visibleNodes.map((n) => ({ ...n })), // shallow copy for ForceGraph mutation
      links: visibleEdges.map((e) => ({ ...e })),
    };
  }, [graph, currentSnapshot]);

  const fgRef = useRef<any>(undefined);

  return (
    <Section id="graph-evolution" eyebrow="Stage 3 — The Story of the Graph" variant="raised">
      <SectionHeader
        title={
          <>
            One memory instance reads an entire benchmark.{" "}
            <span style={{ color: "var(--color-amber)" }}>
              Watch the graph build itself, paragraph by paragraph.
            </span>
          </>
        }
        lede={
          <>
            Phase 1.5–1.7 measured V4 per-document (memory cleared between docs).
            Phase 2 shifts the unit of analysis: <strong>one benchmark = one test</strong>.
            A single GraphMemoryV4ₜ instance ingests every paragraph of every document
            sequentially, persisting the graph across documents. Recurring entities —{" "}
            <code>termination_clause</code> across 510 contracts,{" "}
            <code>fiscal_year_2022</code> across 150 filings — accumulate mention edges
            and form the domain's structural backbone. Drag the timeline to watch it
            unfold; toggle configurations to compare grid-world θ against corpus-tuned θ.
          </>
        }
      />

      <div className="mt-8">
        {index === null ? (
          <LoadingPanel />
        ) : (
          <>
            <RunSelector
              index={index}
              selected={selectedRunId}
              onSelect={setSelectedRunId}
            />
            {timeline && graph && manifest && currentSnapshot ? (
              <ViewerBody
                fgRef={fgRef}
                timeline={timeline}
                manifest={manifest}
                currentSnapshot={currentSnapshot}
                stepIdx={stepIdx}
                setStepIdx={setStepIdx}
                playing={playing}
                setPlaying={setPlaying}
                speed={speed}
                setSpeed={setSpeed}
                visibleGraph={visibleGraph}
                hoverEntity={hoverEntity}
                setHoverEntity={setHoverEntity}
              />
            ) : (
              <LoadingPanel />
            )}
          </>
        )}
      </div>
    </Section>
  );
}

// ----------------------------------------------------------------------
// Sub-components
// ----------------------------------------------------------------------

function LoadingPanel() {
  return (
    <div
      className="panel-rise p-8 text-center"
      style={{ color: "var(--color-text-2)" }}
    >
      <Pill tone="cyan">Loading graph evolution data…</Pill>
      <p className="mt-3 text-sm">
        If this persists, run{" "}
        <code className="font-mono text-xs">
          python scripts/run_corpus_ingestion.py --benchmark cuad --config v4t-corpus-tuned
        </code>{" "}
        then{" "}
        <code className="font-mono text-xs">
          python scripts/build_graph_evolution_frontend.py
        </code>
        .
      </p>
    </div>
  );
}

function RunSelector({
  index,
  selected,
  onSelect,
}: {
  index: RunIndex;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const runsByBench = useMemo(() => {
    const grouped: Record<string, typeof index.runs> = {};
    for (const r of index.runs) {
      grouped[r.benchmark] = grouped[r.benchmark] ?? [];
      grouped[r.benchmark].push(r);
    }
    return grouped;
  }, [index]);

  return (
    <div className="panel-rise p-4 mb-4">
      <div
        className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
        style={{ color: "var(--color-muted)" }}
      >
        Select run · benchmark × config × order
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.entries(runsByBench).map(([bench, runs]) => (
          <div key={bench} className="flex flex-col gap-1">
            <div
              className="text-[0.6rem] uppercase tracking-wider"
              style={{ color: "var(--color-muted)" }}
            >
              {bench}
            </div>
            <div className="flex gap-1">
              {runs.map((r) => {
                const isSelected = r.run_id === selected;
                const isCorpusTuned = r.config === "v4t-corpus-tuned";
                return (
                  <button
                    key={r.run_id}
                    onClick={() => onSelect(r.run_id)}
                    className="px-3 py-1.5 text-xs font-mono rounded transition-all"
                    style={{
                      background: isSelected
                        ? "var(--color-amber-soft)"
                        : "var(--color-surface)",
                      color: isSelected
                        ? "var(--color-amber)"
                        : "var(--color-text-2)",
                      border: `1px solid ${
                        isSelected
                          ? "var(--color-amber)"
                          : "var(--color-border)"
                      }`,
                      fontWeight: isCorpusTuned ? 600 : 400,
                      transitionDuration: "180ms",
                      transitionTimingFunction:
                        "cubic-bezier(0.2, 0, 0, 1)",
                    }}
                    title={`${r.n_docs} docs · ${r.n_snapshots} snapshots · ${r.n_nodes} nodes`}
                  >
                    {r.config}
                    {r.order !== "canonical" ? ` · ${r.order}` : ""}
                    {isCorpusTuned ? " ★" : ""}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ViewerBody({
  fgRef,
  timeline,
  manifest,
  currentSnapshot,
  stepIdx,
  setStepIdx,
  playing,
  setPlaying,
  speed,
  setSpeed,
  visibleGraph,
  hoverEntity,
  setHoverEntity,
}: {
  fgRef: React.MutableRefObject<any>;
  timeline: Snapshot[];
  manifest: Manifest;
  currentSnapshot: Snapshot;
  stepIdx: number;
  setStepIdx: (i: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  speed: number;
  setSpeed: (s: number) => void;
  visibleGraph: { nodes: GraphNode[]; links: GraphEdge[] };
  hoverEntity: string | null;
  setHoverEntity: (e: string | null) => void;
}) {
  return (
    <>
      <div className="grid lg:grid-cols-[1.4fr_1fr] gap-4">
        <GraphCanvas
          fgRef={fgRef}
          visibleGraph={visibleGraph}
          hoverEntity={hoverEntity}
          setHoverEntity={setHoverEntity}
          currentSnapshot={currentSnapshot}
        />
        <SidePanel
          manifest={manifest}
          currentSnapshot={currentSnapshot}
          stepIdx={stepIdx}
          totalSteps={timeline.length}
        />
      </div>

      <TimelineControls
        timeline={timeline}
        manifest={manifest}
        stepIdx={stepIdx}
        setStepIdx={setStepIdx}
        playing={playing}
        setPlaying={setPlaying}
        speed={speed}
        setSpeed={setSpeed}
      />

      <GrowthCurves timeline={timeline} stepIdx={stepIdx} setStepIdx={setStepIdx} />
    </>
  );
}

function GraphCanvas({
  fgRef,
  visibleGraph,
  hoverEntity,
  setHoverEntity,
  currentSnapshot,
}: {
  fgRef: React.MutableRefObject<any>;
  visibleGraph: { nodes: GraphNode[]; links: GraphEdge[] };
  hoverEntity: string | null;
  setHoverEntity: (e: string | null) => void;
  currentSnapshot: Snapshot;
}) {
  // Filter for hover-highlight: when hovering an entity, dim everything not
  // directly connected.
  const connectedToHover = useMemo(() => {
    if (!hoverEntity) return null;
    const ids = new Set<string>([hoverEntity]);
    for (const e of visibleGraph.links) {
      if (e.source === hoverEntity) ids.add(e.target as string);
      if (e.target === hoverEntity) ids.add(e.source as string);
    }
    return ids;
  }, [hoverEntity, visibleGraph.links]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className="panel-rise overflow-hidden"
      style={{ height: 560 }}
    >
      <div
        className="px-4 py-2 border-b text-[0.65rem] uppercase tracking-[0.18em] flex items-center justify-between"
        style={{
          color: "var(--color-muted)",
          borderColor: "var(--color-border)",
        }}
      >
        <span className="flex items-center gap-1.5">
          <Network size={14} style={{ color: "var(--color-amber)" }} />
          Memory graph at step {currentSnapshot.step} of {currentSnapshot.seen}
        </span>
        <span>
          {visibleGraph.nodes.length} nodes · {visibleGraph.links.length} edges
        </span>
      </div>
      <div style={{ height: 520 }}>
        <ForceGraph2D
          ref={fgRef}
          graphData={visibleGraph as any}
          nodeId="id"
          width={undefined}
          height={520}
          backgroundColor="rgba(0,0,0,0)"
          cooldownTicks={50}
          d3VelocityDecay={0.4}
          warmupTicks={20}
          enableNodeDrag={true}
          enableZoomInteraction={true}
          enablePanInteraction={true}
          nodeRelSize={3}
          nodeVal={(n: any) => {
            if (n.type === "event") return 1.5;
            const c = (n as GraphNode).mention_count ?? 1;
            return 2 + Math.sqrt(c) * 1.2;
          }}
          nodeColor={(n: any) => {
            const dimmed =
              connectedToHover && !connectedToHover.has(n.id) ? 0.18 : 1.0;
            const base = n.type === "event" ? EVENT_COLOR : entityColor(n.id);
            return dimmed === 1.0 ? base : base + "33";
          }}
          linkColor={(e: any) => {
            const eid = e.edge_type;
            const dimmed =
              connectedToHover &&
              !connectedToHover.has(
                typeof e.source === "object" ? e.source.id : e.source,
              ) &&
              !connectedToHover.has(
                typeof e.target === "object" ? e.target.id : e.target,
              );
            if (dimmed) return "rgba(148, 163, 184, 0.04)";
            if (eid === "temporal") return TEMPORAL_EDGE_COLOR;
            if (eid === "mentions" || eid === "mentioned_in")
              return MENTION_EDGE_COLOR;
            return "rgba(148,163,184,0.15)";
          }}
          linkWidth={0.6}
          linkDirectionalArrowLength={0}
          onNodeHover={(n: any) => {
            if (n && n.type === "entity") setHoverEntity(n.id);
            else setHoverEntity(null);
          }}
          nodeLabel={(n: any) => {
            if (n.type === "event") {
              return `event_${n.step}\n${(n.observation || "").slice(0, 120)}`;
            }
            return `${n.id}\nmentions: ${n.mention_count ?? "?"}\nlast: step ${n.last_step ?? "?"}\nfirst: step ${n.first_seen_step ?? "?"}`;
          }}
        />
      </div>
    </motion.div>
  );
}

function SidePanel({
  manifest,
  currentSnapshot,
  stepIdx,
  totalSteps,
}: {
  manifest: Manifest;
  currentSnapshot: Snapshot;
  stepIdx: number;
  totalSteps: number;
}) {
  return (
    <div className="flex flex-col gap-3">
      <CurrentDocCard manifest={manifest} snap={currentSnapshot} stepIdx={stepIdx} totalSteps={totalSteps} />
      <MetricsCard snap={currentSnapshot} manifest={manifest} />
      <TopEntitiesCard snap={currentSnapshot} />
    </div>
  );
}

function CurrentDocCard({
  manifest,
  snap,
  stepIdx,
  totalSteps,
}: {
  manifest: Manifest;
  snap: Snapshot;
  stepIdx: number;
  totalSteps: number;
}) {
  const progress = ((stepIdx + 1) / totalSteps) * 100;
  return (
    <motion.div
      key={`${manifest.benchmark}-${manifest.config}`}
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: [0.2, 0, 0, 1] }}
      className="panel-rise p-4"
      style={{
        background:
          "linear-gradient(135deg, var(--color-amber-soft) 0%, transparent 70%), var(--color-surface)",
      }}
    >
      <div
        className="text-[0.6rem] uppercase tracking-[0.18em] mb-1"
        style={{ color: "var(--color-muted)" }}
      >
        Reading
      </div>
      <div className="font-mono text-sm" style={{ color: "var(--color-text)" }}>
        doc {snap.doc_idx + 1} of {manifest.n_docs}
      </div>
      <AnimatePresence mode="wait">
        <motion.div
          key={snap.doc_title}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          className="text-xs mt-1 truncate"
          style={{ color: "var(--color-text-2)" }}
          title={snap.doc_title}
        >
          {snap.doc_title || "(untitled)"}
        </motion.div>
      </AnimatePresence>
      <div className="mt-3 h-1 rounded-full overflow-hidden" style={{ background: "var(--color-border)" }}>
        <motion.div
          className="h-full"
          style={{ background: "var(--color-amber)" }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.18, ease: "linear" }}
        />
      </div>
      <div
        className="text-[0.6rem] mt-1 flex justify-between font-mono"
        style={{ color: "var(--color-muted)" }}
      >
        <span>step {snap.step.toLocaleString()}</span>
        <span>{snap.seen.toLocaleString()} paragraphs seen · {snap.added.toLocaleString()} stored</span>
      </div>
    </motion.div>
  );
}

function MetricsCard({ snap, manifest }: { snap: Snapshot; manifest: Manifest }) {
  const finalEvents = manifest.final_n_event_nodes;
  const finalEntities = manifest.final_n_entity_nodes;
  const finalEdges = manifest.final_n_edges;
  const metrics: Array<[string, number, number, string]> = [
    ["events", snap.n_events, finalEvents, "#475569"],
    ["entities", snap.n_entities, finalEntities, "#fbbf24"],
    ["mention edges", snap.n_mention_edges, finalEdges, "#f59e0b"],
    ["temporal edges", snap.n_temporal_edges, finalEvents, "#94a3b8"],
  ];
  return (
    <div className="panel-rise p-4">
      <div
        className="text-[0.6rem] uppercase tracking-[0.18em] mb-3 flex items-center gap-1.5"
        style={{ color: "var(--color-muted)" }}
      >
        <Activity size={12} />
        Live graph metrics
      </div>
      <div className="space-y-2">
        {metrics.map(([label, value, max, color]) => {
          const pct = max > 0 ? (value / max) * 100 : 0;
          return (
            <div key={label}>
              <div className="flex justify-between text-[0.65rem] font-mono mb-1">
                <span style={{ color: "var(--color-text-2)" }}>{label}</span>
                <span style={{ color: "var(--color-text)", fontWeight: 600 }}>
                  {value.toLocaleString()} / {max.toLocaleString()}
                </span>
              </div>
              <div
                className="h-[3px] rounded-full overflow-hidden"
                style={{ background: "var(--color-border)" }}
              >
                <motion.div
                  className="h-full"
                  style={{ background: color }}
                  animate={{ width: `${Math.min(pct, 100)}%` }}
                  transition={{ duration: 0.32, ease: [0.2, 0, 0, 1] }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div
        className="mt-4 pt-3 border-t text-[0.6rem] font-mono flex flex-wrap gap-2"
        style={{
          borderColor: "var(--color-border)",
          color: "var(--color-text-2)",
        }}
      >
        <span title="theta_store — storage importance threshold">
          θ_store={manifest.params.theta_store?.toFixed(2)}
        </span>
        <span title="w_graph — graph traversal retrieval weight">
          w_graph={manifest.params.w_graph?.toFixed(2)}
        </span>
        <span title="w_embed — embedding cosine retrieval weight">
          w_embed={manifest.params.w_embed?.toFixed(2)}
        </span>
        <span title="w_recency — recency retrieval weight">
          w_recency={manifest.params.w_recency?.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

function TopEntitiesCard({ snap }: { snap: Snapshot }) {
  return (
    <div className="panel-rise p-4">
      <div
        className="text-[0.6rem] uppercase tracking-[0.18em] mb-3 flex items-center gap-1.5"
        style={{ color: "var(--color-muted)" }}
      >
        <Sparkles size={12} />
        Top entities by mention count
      </div>
      <div className="space-y-1.5">
        <AnimatePresence initial={false}>
          {snap.top_entities.slice(0, 10).map((ent) => (
            <motion.div
              key={ent.name}
              layout
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18, ease: [0.2, 0, 0, 1] }}
              className="flex items-center gap-2 text-xs font-mono"
            >
              <span
                className="inline-block rounded-full"
                style={{
                  width: 8,
                  height: 8,
                  background: entityColor(ent.name),
                }}
              />
              <span
                className="flex-1 truncate"
                style={{ color: "var(--color-text)" }}
                title={ent.name}
              >
                {ent.name}
              </span>
              <span
                className="tabular-nums"
                style={{ color: "var(--color-text-2)" }}
              >
                {ent.count}
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
        {snap.top_entities.length === 0 && (
          <div className="text-xs" style={{ color: "var(--color-muted)" }}>
            No entities yet — extraction begins after first paragraph ingestion.
          </div>
        )}
      </div>
    </div>
  );
}

function TimelineControls({
  timeline,
  manifest,
  stepIdx,
  setStepIdx,
  playing,
  setPlaying,
  speed,
  setSpeed,
}: {
  timeline: Snapshot[];
  manifest: Manifest;
  stepIdx: number;
  setStepIdx: (i: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  speed: number;
  setSpeed: (s: number) => void;
}) {
  const total = timeline.length;
  const docBoundaries = manifest.doc_boundaries;
  return (
    <div className="panel-rise p-4 mt-4">
      <div className="flex items-center gap-3">
        <button
          aria-label="step back"
          onClick={() => setStepIdx(Math.max(0, stepIdx - 1))}
          className="p-1.5 rounded transition-colors"
          style={{
            color: "var(--color-text-2)",
            transitionDuration: "180ms",
            transitionTimingFunction: "cubic-bezier(0.2, 0, 0, 1)",
          }}
        >
          <SkipBack size={16} />
        </button>
        <button
          aria-label={playing ? "pause" : "play"}
          onClick={() => setPlaying(!playing)}
          className="p-2 rounded-full transition-all"
          style={{
            background: "var(--color-amber-soft)",
            color: "var(--color-amber)",
            transitionDuration: "180ms",
            transitionTimingFunction: "cubic-bezier(0.2, 0, 0, 1)",
          }}
        >
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button
          aria-label="step forward"
          onClick={() => setStepIdx(Math.min(total - 1, stepIdx + 1))}
          className="p-1.5 rounded transition-colors"
          style={{
            color: "var(--color-text-2)",
            transitionDuration: "180ms",
            transitionTimingFunction: "cubic-bezier(0.2, 0, 0, 1)",
          }}
        >
          <SkipForward size={16} />
        </button>

        <div className="flex-1 mx-3">
          <input
            type="range"
            min={0}
            max={total - 1}
            value={stepIdx}
            onChange={(e) => setStepIdx(parseInt(e.target.value, 10))}
            className="w-full"
            style={{ accentColor: "var(--color-amber)" }}
          />
          {/* Doc-boundary tick marks */}
          <div className="relative h-2 mt-0.5">
            {docBoundaries.slice(0, 100).map((db) => {
              const x = (db.doc_idx / Math.max(manifest.n_docs - 1, 1)) * 100;
              return (
                <div
                  key={db.doc_idx}
                  className="absolute"
                  style={{
                    left: `${x}%`,
                    top: 0,
                    width: 1,
                    height: 4,
                    background: "var(--color-border)",
                    opacity: 0.5,
                  }}
                />
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-1 text-[0.65rem] font-mono">
          <span style={{ color: "var(--color-muted)" }}>speed</span>
          {[0.5, 1, 2, 4].map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className="px-1.5 py-0.5 rounded transition-colors"
              style={{
                background:
                  speed === s
                    ? "var(--color-amber-soft)"
                    : "transparent",
                color:
                  speed === s
                    ? "var(--color-amber)"
                    : "var(--color-text-2)",
                fontWeight: speed === s ? 600 : 400,
                transitionDuration: "100ms",
              }}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      <div
        className="text-[0.6rem] mt-2 font-mono flex justify-between"
        style={{ color: "var(--color-muted)" }}
      >
        <span>
          snapshot {stepIdx + 1} of {total} (kind: {timeline[stepIdx]?.kind ?? "?"})
        </span>
        <span>
          {manifest.benchmark} · {manifest.config} · order: {manifest.order}
        </span>
      </div>
    </div>
  );
}

function GrowthCurves({
  timeline,
  stepIdx,
  setStepIdx,
}: {
  timeline: Snapshot[];
  stepIdx: number;
  setStepIdx: (i: number) => void;
}) {
  const data = useMemo(
    () =>
      timeline.map((s, i) => ({
        idx: i,
        step: s.step,
        doc_idx: s.doc_idx,
        events: s.n_events,
        entities: s.n_entities,
        edges: s.n_mention_edges + s.n_temporal_edges,
      })),
    [timeline],
  );
  const currentStep = timeline[stepIdx]?.step ?? 0;

  return (
    <div className="panel-rise p-4 mt-4">
      <div
        className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
        style={{ color: "var(--color-muted)" }}
      >
        Growth curves · click chart to scrub timeline
      </div>
      <div style={{ height: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
            onClick={(e: any) => {
              if (e && e.activeTooltipIndex != null) {
                setStepIdx(e.activeTooltipIndex);
              }
            }}
          >
            <CartesianGrid stroke="var(--color-border)" strokeDasharray="2 2" opacity={0.4} />
            <XAxis
              dataKey="step"
              stroke="var(--color-text-2)"
              tick={{ fontSize: 10, fontFamily: "monospace" }}
              tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`)}
              label={{ value: "global step", fontSize: 10, position: "insideBottom", offset: -2, fill: "var(--color-muted)" }}
            />
            <YAxis
              stroke="var(--color-text-2)"
              tick={{ fontSize: 10, fontFamily: "monospace" }}
              tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`)}
            />
            <Tooltip
              contentStyle={{
                background: "var(--color-surface)",
                border: "1px solid var(--color-border)",
                fontSize: 11,
                fontFamily: "monospace",
                color: "var(--color-text)",
              }}
              labelFormatter={(v: any) => `step ${v}`}
            />
            <Area
              type="monotone"
              dataKey="entities"
              stroke="#fbbf24"
              fill="#fbbf24"
              fillOpacity={0.25}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="edges"
              stroke="#f59e0b"
              fill="#f59e0b"
              fillOpacity={0.18}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="events"
              stroke="#475569"
              fill="#475569"
              fillOpacity={0.15}
              strokeWidth={1}
              isAnimationActive={false}
            />
            <ReferenceLine
              x={currentStep}
              stroke="var(--color-amber)"
              strokeWidth={1.5}
              strokeDasharray="3 3"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div
        className="flex gap-4 mt-2 text-[0.65rem] font-mono"
        style={{ color: "var(--color-text-2)" }}
      >
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3" style={{ background: "#fbbf24" }} />
          entities
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3" style={{ background: "#f59e0b" }} />
          edges
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3" style={{ background: "#475569" }} />
          events
        </span>
      </div>
    </div>
  );
}

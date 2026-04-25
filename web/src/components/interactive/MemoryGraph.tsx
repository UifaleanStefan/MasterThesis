/**
 * MemoryGraph — animated 2D force-directed graph showing what GraphMemoryV4
 * stores during a sample MultiHopKeyDoor episode.
 *
 * The graph is synthetic-but-faithful: we don't simulate a full episode in
 * the browser. Instead, we model the *structure* V4 produces — event nodes
 * connected to entity nodes (red_key, blue_door, sign, etc.) and to each
 * other temporally — then animate node arrivals to suggest "memory growing".
 *
 * The "store rate" slider scales how aggressively events get stored —
 * mirroring θ_store / θ_novel without needing the real Python pipeline.
 */

import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";
import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import { cn } from "../../lib/format";

interface Node {
  id: string;
  type: "event" | "entity";
  label: string;
  step?: number;
  isHint?: boolean;
}

interface Link {
  source: string;
  target: string;
  kind: "mentions" | "temporal";
}

const EVENT_TEMPLATES: Array<{ obs: string; hint?: boolean; entities: string[] }> = [
  { obs: "you see a sign: blue key opens north door", entities: ["sign", "blue_key", "blue_door"], hint: true },
  { obs: "you see a red key", entities: ["red_key"] },
  { obs: "you see a blue key", entities: ["blue_key"] },
  { obs: "you see a green door requires green key", entities: ["green_door"] },
  { obs: "you see a sign: yellow key opens east", entities: ["sign", "yellow_door"], hint: true },
  { obs: "you are in a room. nothing of interest", entities: [] },
  { obs: "you see a blue door requires blue key", entities: ["blue_door", "blue_key"] },
  { obs: "you see the goal", entities: ["goal"] },
  { obs: "you see a red door requires red key", entities: ["red_key"] },
  { obs: "you see a yellow key", entities: ["yellow_door"] },
  { obs: "you are in a room. nothing of interest", entities: [] },
  { obs: "you see a green key", entities: ["green_door"] },
];

export function MemoryGraph() {
  const fgRef = useRef<ForceGraphMethods<Node, Link>>(undefined);
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(true);
  const [storeRate, setStoreRate] = useState(0.5); // 0=skip many, 1=store all
  const [size, setSize] = useState({ width: 600, height: 380 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Resize-observe to keep the graph responsive.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const { width } = el.getBoundingClientRect();
      setSize({ width: Math.floor(width), height: 380 });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Tick: advance one event every ~900ms while running.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= EVENT_TEMPLATES.length * 2) return 0;
        return s + 1;
      });
    }, 900);
    return () => clearInterval(id);
  }, [running]);

  // Compute the visible (sub)graph at the current step, given the storeRate.
  const { nodes, links } = useMemo(() => {
    const nodes: Node[] = [];
    const links: Link[] = [];
    const seenEntities = new Set<string>();

    let prevEventId: string | null = null;
    let storedCount = 0;

    for (let i = 0; i <= step && i < EVENT_TEMPLATES.length * 2; i++) {
      const tpl = EVENT_TEMPLATES[i % EVENT_TEMPLATES.length];
      // "Importance score" — hints + multi-entity events more likely to pass.
      const importance =
        (tpl.hint ? 0.4 : 0) + Math.min(0.5, tpl.entities.length * 0.18) + 0.15;
      // Decision: storeRate ∈ [0, 1] — keep when importance > (1 - storeRate).
      if (importance < 1 - storeRate) {
        // skipped
        continue;
      }
      const eventId = `e${i}`;
      nodes.push({
        id: eventId,
        type: "event",
        label: tpl.obs.length > 32 ? tpl.obs.slice(0, 32) + "…" : tpl.obs,
        step: i,
        isHint: tpl.hint,
      });
      storedCount++;
      for (const ent of tpl.entities) {
        if (!seenEntities.has(ent)) {
          nodes.push({ id: ent, type: "entity", label: ent });
          seenEntities.add(ent);
        }
        links.push({ source: eventId, target: ent, kind: "mentions" });
      }
      if (prevEventId) {
        links.push({ source: prevEventId, target: eventId, kind: "temporal" });
      }
      prevEventId = eventId;
    }

    return { nodes, links, storedCount };
  }, [step, storeRate]);

  return (
    <div ref={containerRef} className="panel-rise overflow-hidden">
      <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-[var(--color-border)]">
        <div>
          <div className="text-sm font-semibold">live memory graph</div>
          <div className="text-[0.65rem] uppercase tracking-[0.16em]" style={{ color: "var(--color-muted)" }}>
            event • entity • temporal · synthesized from V4 patterns
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setRunning((r) => !r)}
            className="px-2.5 py-1.5 rounded text-xs flex items-center gap-1.5 hover:bg-[rgba(255,255,255,0.04)]"
            style={{ color: "var(--color-text-2)", border: "1px solid var(--color-border)" }}
          >
            {running ? <Pause size={12} /> : <Play size={12} />}
            {running ? "pause" : "play"}
          </button>
          <button
            onClick={() => {
              setStep(0);
              fgRef.current?.zoomToFit(400, 40);
            }}
            className="px-2.5 py-1.5 rounded text-xs flex items-center gap-1.5 hover:bg-[rgba(255,255,255,0.04)]"
            style={{ color: "var(--color-text-2)", border: "1px solid var(--color-border)" }}
          >
            <RotateCcw size={12} /> reset
          </button>
        </div>
      </div>

      <div className="px-4 py-2 flex items-center gap-3 border-b border-[var(--color-border)]">
        <span className="text-xs font-mono" style={{ color: "var(--color-text-2)" }}>store rate</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={storeRate}
          onChange={(e) => setStoreRate(parseFloat(e.target.value))}
          className="flex-1"
          style={{ accentColor: "var(--color-cyan)" }}
        />
        <span className="text-xs font-mono tabular-nums w-12 text-right" style={{ color: "var(--color-cyan)" }}>
          {storeRate.toFixed(2)}
        </span>
        <span className="text-xs font-mono" style={{ color: "var(--color-muted)" }}>
          {nodes.filter((n) => n.type === "event").length} events ·{" "}
          {nodes.filter((n) => n.type === "entity").length} entities
        </span>
      </div>

      <div className="relative" style={{ height: 380, background: "var(--color-bg)" }}>
        <ForceGraph2D
          ref={fgRef}
          width={size.width}
          height={size.height}
          graphData={{ nodes, links }}
          backgroundColor="rgba(0,0,0,0)"
          nodeId="id"
          nodeRelSize={4}
          nodeLabel={(n) => (n as Node).label}
          nodeAutoColorBy="type"
          linkColor={(l) =>
            (l as Link).kind === "temporal"
              ? "rgba(167, 139, 250, 0.45)"
              : "rgba(34, 211, 238, 0.45)"
          }
          linkWidth={(l) => ((l as Link).kind === "temporal" ? 1.4 : 1)}
          linkDirectionalArrowLength={(l) => ((l as Link).kind === "temporal" ? 4 : 0)}
          linkDirectionalArrowRelPos={1}
          linkDirectionalParticles={(l) => ((l as Link).kind === "temporal" ? 1 : 0)}
          linkDirectionalParticleSpeed={() => 0.005}
          linkDirectionalParticleColor={() => "rgba(167, 139, 250, 0.85)"}
          nodeCanvasObject={(n, ctx) => {
            const node = n as Node & { x?: number; y?: number };
            const r = node.type === "event" ? (node.isHint ? 8 : 6) : 5;
            ctx.beginPath();
            ctx.arc(node.x ?? 0, node.y ?? 0, r, 0, 2 * Math.PI);
            ctx.fillStyle =
              node.type === "event"
                ? node.isHint ? "rgba(245, 158, 11, 0.95)" : "rgba(34, 211, 238, 0.85)"
                : "rgba(167, 139, 250, 0.65)";
            ctx.fill();
            ctx.strokeStyle = "rgba(0, 0, 0, 0.4)";
            ctx.lineWidth = 0.6;
            ctx.stroke();
            // Small label below entity nodes only (events get crowded otherwise).
            if (node.type === "entity") {
              ctx.font = "9px ui-monospace, JetBrains Mono, monospace";
              ctx.fillStyle = "rgba(167, 139, 250, 0.95)";
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillText(node.label, node.x ?? 0, (node.y ?? 0) + r + 2);
            }
          }}
          cooldownTicks={150}
          d3VelocityDecay={0.35}
        />

        {/* Legend overlay */}
        <div
          className={cn("absolute right-3 bottom-3 text-[0.65rem] font-mono space-y-1 pointer-events-none")}
          style={{ color: "var(--color-text-2)" }}
        >
          <Legend dotColor="rgba(245, 158, 11, 0.95)" label="hint event" />
          <Legend dotColor="rgba(34, 211, 238, 0.85)" label="event" />
          <Legend dotColor="rgba(167, 139, 250, 0.65)" label="entity" />
        </div>
      </div>
    </div>
  );
}

function Legend({ dotColor, label }: { dotColor: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="block w-2 h-2 rounded-full" style={{ background: dotColor }} />
      <span>{label}</span>
    </div>
  );
}

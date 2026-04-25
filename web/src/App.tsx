import { useData } from "./data/useData";
import type { AggregatedManifest } from "./data/types";
import { Hero } from "./sections/Hero";
import { ScrollProgress } from "./components/nav/ScrollProgress";
import { StickyNav } from "./components/nav/StickyNav";
import { Section } from "./components/shared/Section";
import { SectionHeader } from "./components/shared/SectionHeader";
import { Stat } from "./components/shared/Stat";

function App() {
  const { data: manifest } = useData<AggregatedManifest>("manifest.json");

  return (
    <>
      <ScrollProgress />
      <StickyNav />

      <Hero />

      {/* Phase-B placeholders. Phase C-E replace each one with its real content. */}

      <Section id="question" eyebrow="The Question">
        <SectionHeader
          title={
            <>
              Why{" "}
              <span style={{ color: "var(--color-cyan)" }}>
                fixed memory
              </span>{" "}
              isn&apos;t enough.
            </>
          }
          lede="Most LLM agents bolt on a memory architecture and never adapt it. The thesis claim: optimal memory structure is task-dependent, and θ can be learned from reward."
        />
        <p style={{ color: "var(--color-text-2)" }}>
          [Phase C will replace this with a side-by-side fixed-memory ↔ learnable-θ visualization.]
        </p>
      </Section>

      <Section id="architecture" eyebrow="Architecture" variant="raised">
        <SectionHeader
          title={
            <>
              GraphMemoryV4 — a{" "}
              <span style={{ color: "var(--color-violet)" }}>10-D θ</span>{" "}
              vector that shapes the memory.
            </>
          }
          lede="Six storage dimensions, one decay rate, three retrieval weights. Every event is filtered by importance, every retrieval is a learned weighted sum."
        />
        <p style={{ color: "var(--color-text-2)" }}>
          [Phase C will replace this with the interactive θ explorer + an embedding-backend toggle.]
        </p>
      </Section>

      <Section id="benchmark" eyebrow="Benchmark">
        <SectionHeader
          title="12 systems, 4 environments."
          lede="The headline comparison on MultiHopKeyDoor — under both embedding backends. Pairwise significance overlaid."
        />

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
          <Stat
            label="V4 reward"
            value="0.130"
            tone="cyan"
            hint="100 held-out eps · MiniLM · re-tuned θ"
          />
          <Stat
            label="V4 precision"
            value="1.000"
            tone="emerald"
            hint="memory always retrieves the right hint"
          />
          <Stat
            label="memory size"
            value="≈ 10"
            tone="violet"
            hint="vs 218 for V1 baseline"
          />
          <Stat
            label="systems compared"
            value="12"
            hint="across 4 environments"
          />
        </div>

        <p className="mt-8" style={{ color: "var(--color-text-2)" }}>
          [Phase D builds the heatmap, Pareto scatter, and the embedding toggle.]
        </p>
      </Section>

      <Section id="minilm" eyebrow="The Pivot" variant="raised">
        <SectionHeader
          title={
            <>
              When the embedding changed, the{" "}
              <span style={{ color: "var(--color-violet)" }}>optimum did too</span>.
            </>
          }
          lede="Under TF-IDF, V4's optimum was recency-dominated (w_recency=3.78, w_graph=0). Under MiniLM the same CMA-ES finds w_graph=1.19, w_embed=1.31, w_recency=1.21 — all balanced."
        />
        <p style={{ color: "var(--color-text-2)" }}>
          [Phase D builds the side-by-side θ-radar diff.]
        </p>
      </Section>

      <Section id="footer" className="!min-h-[40vh] !py-16">
        <p
          className="text-xs"
          style={{ color: "var(--color-muted)" }}
        >
          {manifest ? (
            <>
              Built {manifest.built_at_utc} · backend{" "}
              {manifest.embedding_backends.join(", ")} · git{" "}
              {manifest.git_shas[0] ?? "unknown"}
            </>
          ) : (
            "Loading provenance…"
          )}
        </p>
      </Section>
    </>
  );
}

export default App;

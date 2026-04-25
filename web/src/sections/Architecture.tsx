import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { ThetaExplorer } from "../components/interactive/ThetaExplorer";
import { EmbeddingToggleWidget } from "../components/interactive/EmbeddingToggle";
import { Pill } from "../components/shared/Pill";

export function Architecture() {
  return (
    <Section id="architecture" eyebrow="Architecture" variant="raised">
      <div className="flex items-start justify-between flex-wrap gap-4 mb-12">
        <SectionHeader
          className="!mb-0"
          title={
            <>
              <span style={{ color: "var(--color-cyan)" }}>10</span> dimensions.
              <br />
              One memory.
            </>
          }
          lede="GraphMemoryV4 stores events as nodes in a graph; θ controls every storage and retrieval decision. Six dimensions filter what gets stored, one rate decays old entities, three weights shape retrieval."
        />
        <div className="flex flex-col items-end gap-2">
          <span className="text-[0.65rem] uppercase tracking-[0.18em]" style={{ color: "var(--color-muted)" }}>
            embedding backend
          </span>
          <EmbeddingToggleWidget />
          <span className="text-[0.65rem]" style={{ color: "var(--color-muted)" }}>
            (flips the V4 optimum displayed throughout)
          </span>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-4 mb-10">
        <Card
          tone="cyan"
          tag="6 dims"
          title="Storage filter"
          body={
            <>
              On every step, an importance score
              <br />
              <code className="text-[var(--color-cyan)]">
                θ_novel·novel + θ_erich·erich + θ_surprise·surprise
              </code>{" "}
              is computed; the event is stored only if it exceeds{" "}
              <code className="text-[var(--color-cyan)]">θ_store</code>.
            </>
          }
        />
        <Card
          tone="emerald"
          tag="1 dim"
          title="Bayesian decay"
          body={
            <>
              Entity importance follows{" "}
              <code className="text-[var(--color-emerald)]">
                (count + α) / (total + β·n)
              </code>{" "}
              with optional exponential decay on old mentions —{" "}
              <code className="text-[var(--color-emerald)]">θ_decay</code> sets how
              fast yesterday&apos;s entities fade.
            </>
          }
        />
        <Card
          tone="violet"
          tag="3 dims"
          title="Retrieval weights"
          body={
            <>
              Retrieval ranks events by{" "}
              <code className="text-[var(--color-violet)]">
                w_graph · graph + w_embed · sim + w_recency · recency
              </code>
              . The whole memory shape lives in this 3-vector.
            </>
          }
        />
      </div>

      <ThetaExplorer />

      <div className="mt-8 flex flex-wrap items-center gap-2 justify-center">
        <Pill tone="muted">
          memory size = function of θ_store, θ_novel
        </Pill>
        <Pill tone="muted">predicted reward = bilinear interp on sensitivity grid</Pill>
        <Pill tone="muted">defaults reset to active backend&apos;s CMA-ES optimum</Pill>
      </div>
    </Section>
  );
}

interface CardProps {
  tone: "cyan" | "violet" | "emerald";
  tag: string;
  title: string;
  body: React.ReactNode;
}

function Card({ tone, tag, title, body }: CardProps) {
  const accent =
    tone === "cyan" ? "var(--color-cyan)" :
    tone === "violet" ? "var(--color-violet)" : "var(--color-emerald)";
  return (
    <div
      className="panel-rise p-6"
      style={{
        background:
          `linear-gradient(135deg, ${accent.replace("var(--color-", "rgba(").replace(")", ", 0.04)")} 0%, transparent 80%), var(--color-surface)`,
      }}
    >
      <div className="flex items-center gap-2 mb-3">
        <span
          className="text-[0.65rem] uppercase tracking-[0.18em] font-mono px-2 py-0.5 rounded"
          style={{ color: accent, background: `${accent}11`, border: `1px solid ${accent}33` }}
        >
          {tag}
        </span>
        <h3 className="text-lg font-semibold">{title}</h3>
      </div>
      <p className="text-sm leading-relaxed" style={{ color: "var(--color-text-2)" }}>
        {body}
      </p>
    </div>
  );
}

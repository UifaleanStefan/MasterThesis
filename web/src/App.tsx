import { useData } from "./data/useData";
import type { AggregatedManifest } from "./data/types";
import { EmbeddingProvider } from "./components/interactive/EmbeddingToggle";
import { Hero } from "./sections/Hero";
import { TheQuestion } from "./sections/TheQuestion";
import { Architecture } from "./sections/Architecture";
import { Progression } from "./sections/Progression";
import { Benchmark } from "./sections/Benchmark";
import { MiniLMPivot } from "./sections/MiniLMPivot";
import { ScrollProgress } from "./components/nav/ScrollProgress";
import { StickyNav } from "./components/nav/StickyNav";
import { Section } from "./components/shared/Section";
import { SectionHeader } from "./components/shared/SectionHeader";

function App() {
  const { data: manifest } = useData<AggregatedManifest>("manifest.json");

  return (
    <EmbeddingProvider>
      <ScrollProgress />
      <StickyNav />

      <Hero />
      <TheQuestion />
      <Architecture />
      <Progression />
      <Benchmark />
      <MiniLMPivot />

      {/* Phase E placeholders */}

      <Section id="ablation" eyebrow="What Matters" variant="raised">
        <SectionHeader
          title={
            <>
              <code className="text-[var(--color-rose)]">θ_novel</code> is non-negotiable.
            </>
          }
          lede="Set theta_novel to 0 and the system stores nothing. theta_erich is the second pillar at 64% degradation. Recency and surprise are nearly free to remove."
        />
        <p style={{ color: "var(--color-text-2)" }}>[Phase E builds the ablation knockout.]</p>
      </Section>

      <Section id="transfer" eyebrow="Transfer">
        <SectionHeader
          title={
            <>
              The learned θ <span style={{ color: "var(--color-emerald)" }}>generalizes</span> —
              and breaks honestly.
            </>
          }
          lede="GoalRoom: 0.69 (strong positive). HardKeyDoor: 0.16. MegaQuestRoom: 0.00 — but precision is 0.94+, so memory is fine. The failure is policy."
        />
      </Section>

      <Section id="neural" eyebrow="Neural Meta-Controller" variant="raised">
        <SectionHeader
          title="A 5,674-parameter MLP that outputs θ per observation."
          lede="200-gen CMA-ES, warm-started from V4. Matches scalar V4 reward (0.19), confirms the same OOD failure on MegaQuest."
        />
      </Section>

      <Section id="stage3" eyebrow="What's Next">
        <SectionHeader
          title={
            <>
              Real LLM cost on{" "}
              <span style={{ color: "var(--color-amber)" }}>DocumentQA</span>.
            </>
          }
          lede="Bayesian opt on θ to minimize J = QA_score − λ·cost_usd. Configs are ready — execution awaits an OpenAI budget."
        />
      </Section>

      <Section id="footer" className="!min-h-[40vh] !py-16">
        <p className="text-xs text-center" style={{ color: "var(--color-muted)" }}>
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
    </EmbeddingProvider>
  );
}

export default App;

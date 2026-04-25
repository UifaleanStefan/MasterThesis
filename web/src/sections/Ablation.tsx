import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { AblationKnockout } from "../components/interactive/AblationKnockout";
import { useData } from "../data/useData";
import type { AblationData } from "../data/types";
import { Pill } from "../components/shared/Pill";

export function Ablation() {
  const { data, loading } = useData<AblationData>("ablation.json");

  return (
    <Section id="ablation" eyebrow="What Matters" variant="raised">
      <SectionHeader
        title={
          <>
            <code className="font-mono text-[var(--color-rose)]">θ_novel</code>{" "}
            is non-negotiable.
          </>
        }
        lede={
          <>
            Each ablation knocks out one component of V4&apos;s θ vector and re-runs
            100 held-out episodes. Three findings: <strong>θ_novel</strong> costs the
            entire reward when removed, <strong>θ_erich</strong> is the second
            pillar at 64% degradation, and{" "}
            <em>recency &amp; surprise are nearly free to drop</em>.
          </>
        }
      />

      <div className="flex flex-wrap gap-2 mb-8">
        <Pill tone="rose">100% load-bearing: θ_novel</Pill>
        <Pill tone="amber">64% load-bearing: θ_erich</Pill>
        <Pill tone="emerald">~0% load-bearing: w_recency, θ_surprise</Pill>
        <Pill tone="muted">100 eps × 10 configs · seed offset 2000 (held-out)</Pill>
      </div>

      {loading ? (
        <div className="text-center py-12" style={{ color: "var(--color-muted)" }}>
          loading ablation data…
        </div>
      ) : data ? (
        <AblationKnockout data={data} />
      ) : (
        <p style={{ color: "var(--color-text-2)" }}>
          No ablation data found. Run <code>python run_ablation.py</code> to populate.
        </p>
      )}
    </Section>
  );
}

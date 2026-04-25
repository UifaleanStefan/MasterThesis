import { motion } from "framer-motion";
import { ExternalLink } from "lucide-react";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { CodeBlock } from "../components/shared/CodeBlock";
import { useData } from "../data/useData";
import type { AggregatedManifest } from "../data/types";

const STACK = [
  { name: "Python 3.11", role: "core runtime" },
  { name: "sentence-transformers", role: "MiniLM all-MiniLM-L6-v2 (default)" },
  { name: "scikit-learn", role: "TF-IDF fallback embedding" },
  { name: "scipy", role: "paired t-test, bootstrap CI" },
  { name: "pycma", role: "CMA-ES with restarts + checkpointing" },
  { name: "networkx", role: "graph memory storage backend" },
  { name: "openai", role: "Stage 3 LLM agent + judge (deferred)" },
  { name: "pytest", role: "40 thesis invariants on every commit" },
];

export function Reproducibility() {
  const { data: manifest } = useData<AggregatedManifest>("manifest.json");

  return (
    <Section id="repro" eyebrow="Reproducibility" variant="raised">
      <SectionHeader
        title={
          <>
            One command reproduces the thesis.
            <br />
            <span style={{ color: "var(--color-cyan)" }}>Everything is on disk.</span>
          </>
        }
        lede="Every result on this site comes from a Python script in the repo. Stochastic decisions are deterministic by seed. Each result JSON carries a manifest with git_sha, embedding backend, timestamp, and seed."
      />

      <div className="grid lg:grid-cols-[1.2fr_1fr] gap-6 mt-8">
        <div className="space-y-4">
          <div>
            <div
              className="text-[0.65rem] uppercase tracking-[0.18em] mb-2"
              style={{ color: "var(--color-muted)" }}
            >
              Quick reproduce (~10 minutes)
            </div>
            <CodeBlock
              label="bash"
              code={`git clone https://github.com/UifaleanStefan/MasterThesis
cd MasterThesis
pip install -r requirements.txt
python reproduce_thesis.py --quick`}
            />
          </div>

          <div>
            <div
              className="text-[0.65rem] uppercase tracking-[0.18em] mb-2"
              style={{ color: "var(--color-muted)" }}
            >
              Full canonical reproduction (~24 h)
            </div>
            <CodeBlock
              label="bash"
              code={`# Each script is self-contained and writes to results/*.json
python run_benchmark.py
python run_graphmemory_v4_cmaes.py --generations 30 --episodes 50
python run_ablation.py
python run_transfer.py --clamp-w-recency 0.5 1.0 2.0
python run_sensitivity.py
python run_neural_controller_v2.py --generations 200 --pretrain-from-v4
python regen_all_figures.py`}
            />
          </div>

          <div>
            <div
              className="text-[0.65rem] uppercase tracking-[0.18em] mb-2"
              style={{ color: "var(--color-muted)" }}
            >
              Frontend (this page)
            </div>
            <CodeBlock
              label="bash"
              code={`cd web
npm install
npm run dev      # http://localhost:5173
npm run build    # static export to web/dist/`}
            />
          </div>
        </div>

        <div className="space-y-4">
          <div className="panel-rise p-5">
            <div
              className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
              style={{ color: "var(--color-muted)" }}
            >
              the stack
            </div>
            <div className="space-y-2">
              {STACK.map((s, i) => (
                <motion.div
                  key={s.name}
                  initial={{ opacity: 0, x: 8 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.04 }}
                  className="flex items-center justify-between gap-3 py-1 border-b border-[var(--color-border)] last:border-b-0"
                >
                  <span
                    className="font-mono text-xs"
                    style={{ color: "var(--color-text)" }}
                  >
                    {s.name}
                  </span>
                  <span
                    className="text-xs text-right"
                    style={{ color: "var(--color-text-2)" }}
                  >
                    {s.role}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>

          <div className="panel-rise p-5">
            <div
              className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
              style={{ color: "var(--color-muted)" }}
            >
              provenance
            </div>
            {manifest ? (
              <div className="space-y-1.5 text-xs font-mono">
                <Row label="built" value={formatTime(manifest.built_at_utc)} />
                <Row
                  label="backend"
                  value={manifest.embedding_backends.join(", ")}
                  tone="cyan"
                />
                <Row
                  label="git_sha"
                  value={manifest.git_shas.slice(0, 2).join(", ") || "unknown"}
                />
                <Row
                  label="last result"
                  value={
                    manifest.latest_result_timestamp_utc
                      ? formatTime(manifest.latest_result_timestamp_utc)
                      : "—"
                  }
                />
                <Row
                  label="data files"
                  value={`${manifest.files_present.length} JSON`}
                />
              </div>
            ) : (
              <p style={{ color: "var(--color-muted)" }}>loading…</p>
            )}
          </div>

          <div className="panel-rise p-5">
            <div
              className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
              style={{ color: "var(--color-muted)" }}
            >
              links
            </div>
            <div className="space-y-2 text-sm">
              <Link
                href="https://github.com/UifaleanStefan/MasterThesis"
                label="github.com/UifaleanStefan/MasterThesis"
              />
              <Link
                href="https://github.com/UifaleanStefan/MasterThesis/tree/master/docs"
                label="docs/ — single-doc handoff + result analyses"
              />
              <Link
                href="https://github.com/UifaleanStefan/MasterThesis/blob/master/AGENTS.md"
                label="AGENTS.md — AI agent guide for the project"
              />
            </div>
          </div>
        </div>
      </div>

      <p
        className="mt-12 text-center text-xs"
        style={{ color: "var(--color-muted)" }}
      >
        Stefan Uifalean · Bocconi MSc 2026 · uifaleanstefan@gmail.com
      </p>
    </Section>
  );
}

function Row({ label, value, tone }: { label: string; value: string; tone?: "cyan" }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span style={{ color: "var(--color-muted)" }}>{label}</span>
      <span style={{ color: tone === "cyan" ? "var(--color-cyan)" : "var(--color-text)" }}>
        {value}
      </span>
    </div>
  );
}

function Link({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center justify-between gap-2 text-xs hover:bg-[rgba(255,255,255,0.03)] px-2 py-1.5 rounded -mx-2 transition-colors"
    >
      <span className="font-mono" style={{ color: "var(--color-cyan)" }}>{label}</span>
      <ExternalLink size={12} style={{ color: "var(--color-muted)" }} />
    </a>
  );
}

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

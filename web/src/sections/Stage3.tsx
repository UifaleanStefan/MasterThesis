import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";
import { Pill } from "../components/shared/Pill";
import { CodeBlock } from "../components/shared/CodeBlock";

export function Stage3() {
  return (
    <Section id="stage3" eyebrow="What's Next">
      <SectionHeader
        title={
          <>
            Real LLM cost on{" "}
            <span style={{ color: "var(--color-amber)" }}>DocumentQA</span>.
          </>
        }
        lede={
          <>
            The thesis arc has three stages: grid-world POC (done), full benchmark
            (done), <strong>real LLM</strong> (configured, awaiting budget). Stage 3
            optimizes θ to minimize a joint quality–cost objective on GPT-4o-mini.
          </>
        }
      />

      <div className="grid lg:grid-cols-[1.1fr_1fr] gap-6 mt-8 items-start">
        {/* Formula + intent */}
        <div>
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
            className="panel-rise p-8 text-center"
            style={{
              background:
                "linear-gradient(135deg, var(--color-amber-soft) 0%, transparent 50%, var(--color-cyan-soft) 100%), var(--color-surface)",
            }}
          >
            <div
              className="text-[0.65rem] uppercase tracking-[0.18em] mb-3"
              style={{ color: "var(--color-muted)" }}
            >
              the objective
            </div>
            <div
              className="text-3xl md:text-4xl font-mono"
              style={{ color: "var(--color-text)" }}
            >
              <span style={{ color: "var(--color-cyan)" }}>J</span> ={" "}
              <span style={{ color: "var(--color-emerald)" }}>QA_score</span>{" "}
              − <span style={{ color: "var(--color-rose)" }}>λ</span> ·{" "}
              <span style={{ color: "var(--color-amber)" }}>cost_usd</span>
            </div>
            <p
              className="mt-4 text-sm leading-relaxed max-w-md mx-auto"
              style={{ color: "var(--color-text-2)" }}
            >
              Bayesian Opt searches the 10-dim θ to maximize J. Each evaluation runs
              the agent on a multi-document DocumentQA workload, scores answers via
              an LLM judge, and accounts for actual API spend.
            </p>
          </motion.div>

          <div className="grid grid-cols-3 gap-3 mt-4">
            <Pill tone="cyan">5 documents</Pill>
            <Pill tone="amber">GPT-4o-mini</Pill>
            <Pill tone="violet">Bayesian Opt · 15 trials</Pill>
          </div>

          <p className="mt-6 text-sm leading-relaxed" style={{ color: "var(--color-text-2)" }}>
            <strong>Why this is the headline experiment:</strong> on grid worlds we
            optimize <em>reward</em>; on real LLMs we have to optimize{" "}
            <em>reward minus cost</em>, and that&apos;s where memory parameters earn
            their keep. Selective storage means fewer tokens in the context, which
            means real money saved per query. The thesis claim becomes operational.
          </p>
        </div>

        {/* Run instructions */}
        <div className="space-y-4">
          <div className="panel-rise p-5">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={16} style={{ color: "var(--color-amber)" }} />
              <h3 className="font-semibold text-base">Three configs ready to launch</h3>
            </div>
            <p
              className="text-sm leading-relaxed mb-4"
              style={{ color: "var(--color-text-2)" }}
            >
              Each config is reproducible end-to-end via{" "}
              <code className="font-mono text-xs">runner.py</code>. They share the
              same DocumentQA + GPT-4o-mini setup; they differ only in which memory
              system optimizes θ.
            </p>

            <CodeBlock
              label="bash"
              code={`# requires OPENAI_API_KEY in env
python runner.py --config experiments/document_qa_v4.yaml
python runner.py --config experiments/document_qa_neural_v2.yaml
python runner.py --config experiments/document_qa_episodic.yaml`}
            />
          </div>

          <div
            className="panel-rise p-5"
            style={{
              background:
                "linear-gradient(135deg, transparent 0%, var(--color-cyan-soft) 100%), var(--color-surface)",
            }}
          >
            <h3
              className="font-semibold text-base mb-2"
              style={{ color: "var(--color-cyan)" }}
            >
              What stays the same.
            </h3>
            <p className="text-sm leading-relaxed" style={{ color: "var(--color-text-2)" }}>
              The LLM-judge scoring path is wired in already (see{" "}
              <code className="font-mono text-xs">
                evaluation/document_qa_llm_judge.py
              </code>
              ); without an API key it falls back to keyword overlap so CI exercises
              it without burning budget. The DocumentQA library has 5 documents
              (lore, mystery, corporate memo, soap opera, science survey) covering
              30+ multi-hop QA pairs.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}

import { motion } from "framer-motion";
import { Section } from "../components/shared/Section";
import { SectionHeader } from "../components/shared/SectionHeader";

export function TheQuestion() {
  return (
    <Section id="question" eyebrow="The Question">
      <SectionHeader
        title={
          <>
            Memory in LLM agents is{" "}
            <span style={{ color: "var(--color-amber)" }}>almost always handed</span>{" "}
            to the model.
          </>
        }
        lede="Context windows. RAG retrievers. Vector stores. Each is engineered up front and stays fixed across tasks. This thesis asks a different question: what if the agent learned how to memorize?"
      />

      <div className="grid md:grid-cols-2 gap-6 mt-12">
        {/* LEFT: Fixed memory illustration */}
        <motion.div
          initial={{ opacity: 0, x: -16 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="panel-rise p-8"
        >
          <div className="mb-4">
            <span className="pill" style={{ color: "var(--color-amber)" }}>
              <span style={{ color: "var(--color-amber)" }}>●</span> the status quo
            </span>
          </div>
          <h3 className="text-2xl font-semibold mb-3">Fixed memory</h3>
          <p style={{ color: "var(--color-text-2)" }} className="mb-6">
            A sliding window. A static schema. A retriever with hand-tuned hyperparameters.
            Whatever the policy is doing, the memory is unaware of it.
          </p>
          <FixedMemoryDiagram />
          <p
            className="mt-6 text-sm"
            style={{ color: "var(--color-muted)" }}
          >
            As the episode runs, early observations age out and are gone. The
            agent has no say in what stays.
          </p>
        </motion.div>

        {/* RIGHT: Learnable theta illustration */}
        <motion.div
          initial={{ opacity: 0, x: 16 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="panel-rise p-8 relative overflow-hidden"
          style={{
            background:
              "linear-gradient(135deg, var(--color-cyan-soft), transparent 60%), var(--color-surface)",
          }}
        >
          <div className="mb-4">
            <span
              className="pill"
              style={{ color: "var(--color-cyan)", borderColor: "var(--color-cyan)33" }}
            >
              <span style={{ color: "var(--color-cyan)" }}>●</span> the proposal
            </span>
          </div>
          <h3 className="text-2xl font-semibold mb-3">Learnable θ</h3>
          <p style={{ color: "var(--color-text-2)" }} className="mb-6">
            A vector of parameters that controls every storage and retrieval
            decision. Optimized from task reward via CMA-ES. Different tasks
            converge to different θ.
          </p>
          <LearnableMemoryDiagram />
          <p
            className="mt-6 text-sm"
            style={{ color: "var(--color-text-2)" }}
          >
            The agent shapes its own memory. <em>What gets remembered</em> becomes
            a learned policy in itself.
          </p>
        </motion.div>
      </div>

      <motion.blockquote
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="mt-16 mx-auto max-w-3xl text-center text-2xl md:text-3xl font-medium leading-snug"
        style={{ color: "var(--color-text)" }}
      >
        <span style={{ color: "var(--color-violet)" }}>“</span> Memory should
        adapt to the task — the optimal structure shouldn&apos;t be assumed.{" "}
        <span style={{ color: "var(--color-violet)" }}>”</span>
      </motion.blockquote>
    </Section>
  );
}

function FixedMemoryDiagram() {
  // 10 boxes; the leftmost (oldest) fade out across time.
  const boxes = Array.from({ length: 10 }, (_, i) => i);
  return (
    <div className="flex items-center gap-1 h-12">
      {boxes.map((i) => {
        const fade = i < 3 ? 0.25 + i * 0.2 : 1;
        return (
          <motion.div
            key={i}
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: fade, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 + i * 0.05, duration: 0.4 }}
            className="flex-1 h-full rounded"
            style={{
              background:
                i < 3
                  ? "linear-gradient(180deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05))"
                  : "linear-gradient(180deg, rgba(245, 158, 11, 0.55), rgba(245, 158, 11, 0.2))",
              border:
                "1px solid " +
                (i < 3 ? "rgba(245, 158, 11, 0.2)" : "rgba(245, 158, 11, 0.4)"),
            }}
          />
        );
      })}
    </div>
  );
}

function LearnableMemoryDiagram() {
  // 10 bars, varying heights — implies adaptive importance-weighted storage.
  const bars = [0.55, 0.92, 0.2, 0.78, 0.3, 0.7, 0.66, 0.4, 0.88, 0.25];
  return (
    <div className="flex items-end gap-1.5 h-16">
      {bars.map((b, i) => (
        <motion.div
          key={i}
          initial={{ height: 0, opacity: 0 }}
          whileInView={{ height: `${b * 100}%`, opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 + i * 0.06, duration: 0.6, ease: "easeOut" }}
          className="flex-1 rounded-t"
          style={{
            background:
              "linear-gradient(180deg, var(--color-cyan), rgba(34, 211, 238, 0.2))",
            boxShadow: "0 0 8px rgba(34, 211, 238, 0.3)",
          }}
        />
      ))}
    </div>
  );
}

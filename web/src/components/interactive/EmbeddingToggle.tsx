/**
 * EmbeddingToggle — global context that flips the entire site between the
 * TF-IDF era numbers and the MiniLM era numbers.
 *
 * Components downstream (Architecture, Benchmark, Pivot, etc.) read the
 * current backend via `useEmbedding()` and render the corresponding stats.
 *
 * The empirical claim of the thesis is that θ depends on the embedding;
 * letting the reader flip between the two backends and watch the page
 * re-render is the most direct demonstration of that claim.
 */

import {
  createContext,
  useCallback,
  useContext,
  useState,
  type PropsWithChildren,
} from "react";
import { motion } from "framer-motion";
import { cn } from "../../lib/format";

export type Embedding = "tfidf" | "minilm";

interface EmbeddingContextValue {
  backend: Embedding;
  setBackend: (b: Embedding) => void;
  toggle: () => void;
}

const EmbeddingContext = createContext<EmbeddingContextValue | null>(null);

export function EmbeddingProvider({ children }: PropsWithChildren) {
  const [backend, setBackend] = useState<Embedding>("minilm");
  const toggle = useCallback(
    () => setBackend((b) => (b === "minilm" ? "tfidf" : "minilm")),
    [],
  );
  return (
    <EmbeddingContext.Provider value={{ backend, setBackend, toggle }}>
      {children}
    </EmbeddingContext.Provider>
  );
}

export function useEmbedding(): EmbeddingContextValue {
  const ctx = useContext(EmbeddingContext);
  if (!ctx) {
    throw new Error("useEmbedding must be used within an EmbeddingProvider");
  }
  return ctx;
}

/**
 * The visible toggle widget. Sticky in the architecture section but can
 * also be embedded inline elsewhere.
 */
export function EmbeddingToggleWidget({ className }: { className?: string }) {
  const { backend, setBackend } = useEmbedding();

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 p-1 rounded-full border border-[var(--color-border)] bg-[var(--color-surface-2)]",
        className,
      )}
    >
      <ToggleButton
        active={backend === "minilm"}
        onClick={() => setBackend("minilm")}
        tone="violet"
        label="MiniLM"
        sub="384-d · post-PoC"
      />
      <ToggleButton
        active={backend === "tfidf"}
        onClick={() => setBackend("tfidf")}
        tone="amber"
        label="TF-IDF"
        sub="31-d · legacy"
      />
    </div>
  );
}

interface ToggleButtonProps {
  active: boolean;
  onClick: () => void;
  tone: "violet" | "amber";
  label: string;
  sub: string;
}

function ToggleButton({ active, onClick, tone, label, sub }: ToggleButtonProps) {
  const accent =
    tone === "violet" ? "var(--color-violet)" : "var(--color-amber)";
  const accentSoft =
    tone === "violet"
      ? "var(--color-violet-soft)"
      : "var(--color-amber-soft)";
  return (
    <button
      onClick={onClick}
      className="relative px-4 py-1.5 text-xs font-medium tracking-wide rounded-full transition-colors flex flex-col items-start"
      style={{
        color: active ? accent : "var(--color-text-2)",
      }}
    >
      {active ? (
        <motion.span
          layoutId="embedding-toggle-pill"
          className="absolute inset-0 rounded-full"
          style={{ background: accentSoft, border: `1px solid ${accent}55` }}
          transition={{ type: "spring", stiffness: 320, damping: 30 }}
        />
      ) : null}
      <span className="relative z-10 leading-tight">{label}</span>
      <span
        className="relative z-10 text-[0.6rem] uppercase tracking-[0.14em]"
        style={{ color: active ? accent : "var(--color-muted)", opacity: 0.7 }}
      >
        {sub}
      </span>
    </button>
  );
}

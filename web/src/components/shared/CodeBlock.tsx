import { Copy, Check } from "lucide-react";
import { useState } from "react";
import { cn } from "../../lib/format";

interface CodeBlockProps {
  code: string;
  /** Optional filename / label above the code. */
  label?: string;
  language?: string;
  className?: string;
}

export function CodeBlock({ code, label, language = "bash", className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  function copy() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    });
  }

  return (
    <div className={cn("panel overflow-hidden", className)}>
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--color-border)] text-xs">
        <span style={{ color: "var(--color-text-2)" }}>
          {label ?? language}
        </span>
        <button
          onClick={copy}
          className="inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded hover:bg-[rgba(255,255,255,0.04)] transition-colors"
          style={{ color: "var(--color-text-2)" }}
          aria-label="Copy code to clipboard"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre
        className="px-4 py-3 overflow-x-auto text-sm leading-[1.7]"
        style={{ color: "var(--color-text)" }}
      >
        <code className="font-mono">{code}</code>
      </pre>
    </div>
  );
}

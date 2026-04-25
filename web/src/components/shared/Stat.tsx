import { type ReactNode } from "react";
import { cn } from "../../lib/format";

interface StatProps {
  /** The big number / value (already formatted). */
  value: ReactNode;
  /** Tiny uppercase label above the value. */
  label: ReactNode;
  /** Optional sub-text rendered below the value. */
  hint?: ReactNode;
  /** Optional accent colour for the value. */
  tone?: "cyan" | "violet" | "amber" | "emerald" | "rose" | "default";
  className?: string;
}

const TONE_VAR: Record<NonNullable<StatProps["tone"]>, string> = {
  cyan: "var(--color-cyan)",
  violet: "var(--color-violet)",
  amber: "var(--color-amber)",
  emerald: "var(--color-emerald)",
  rose: "var(--color-rose)",
  default: "var(--color-text)",
};

export function Stat({ value, label, hint, tone = "default", className }: StatProps) {
  return (
    <div className={cn("panel-rise px-6 py-5", className)}>
      <div
        className="text-[0.7rem] uppercase tracking-[0.16em] mb-2"
        style={{ color: "var(--color-muted)" }}
      >
        {label}
      </div>
      <div
        className="text-4xl font-semibold tabular-nums leading-none"
        style={{ color: TONE_VAR[tone] }}
      >
        {value}
      </div>
      {hint ? (
        <div
          className="mt-2 text-sm"
          style={{ color: "var(--color-text-2)" }}
        >
          {hint}
        </div>
      ) : null}
    </div>
  );
}

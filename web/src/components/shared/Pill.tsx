import { type PropsWithChildren } from "react";
import { cn } from "../../lib/format";

interface PillProps extends PropsWithChildren {
  tone?: "cyan" | "violet" | "amber" | "emerald" | "rose" | "muted";
  className?: string;
}

const TONE_BG: Record<NonNullable<PillProps["tone"]>, string> = {
  cyan: "var(--color-cyan-soft)",
  violet: "var(--color-violet-soft)",
  amber: "var(--color-amber-soft)",
  emerald: "var(--color-emerald-soft)",
  rose: "var(--color-rose-soft)",
  muted: "rgba(255,255,255,0.04)",
};
const TONE_FG: Record<NonNullable<PillProps["tone"]>, string> = {
  cyan: "var(--color-cyan)",
  violet: "var(--color-violet)",
  amber: "var(--color-amber)",
  emerald: "var(--color-emerald)",
  rose: "var(--color-rose)",
  muted: "var(--color-text-2)",
};

export function Pill({ tone = "muted", className, children }: PillProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs uppercase tracking-[0.12em] font-medium",
        className,
      )}
      style={{
        background: TONE_BG[tone],
        color: TONE_FG[tone],
        border: `1px solid ${TONE_FG[tone]}33`,
      }}
    >
      {children}
    </span>
  );
}

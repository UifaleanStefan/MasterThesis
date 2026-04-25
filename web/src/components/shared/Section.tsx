/**
 * Full-bleed page section with scroll-triggered fade-in.
 * Used by every story-arc section.
 */

import { motion } from "framer-motion";
import { type PropsWithChildren } from "react";
import { cn } from "../../lib/format";

interface SectionProps extends PropsWithChildren {
  id: string;
  className?: string;
  /** Optional eyebrow label (small uppercase tag above the title). */
  eyebrow?: string;
  /** Optional darker variant for visual rhythm between sections. */
  variant?: "default" | "raised";
  /** Disable the entry animation (e.g. for the always-visible hero). */
  noFade?: boolean;
}

export function Section({
  id,
  className,
  eyebrow,
  variant = "default",
  noFade,
  children,
}: SectionProps) {
  const baseClass = cn(
    "relative min-h-[80vh] py-32 px-6 md:px-12 lg:px-16",
    variant === "raised" && "bg-[var(--color-bg-rise)]",
    className,
  );

  const inner = (
    <div className="mx-auto w-full max-w-7xl">
      {eyebrow ? (
        <div className="pill mb-6">
          <span style={{ color: "var(--color-cyan)" }}>●</span> {eyebrow}
        </div>
      ) : null}
      {children}
    </div>
  );

  if (noFade) {
    return (
      <section id={id} data-variant={variant} className={baseClass}>
        {inner}
      </section>
    );
  }

  return (
    <motion.section
      id={id}
      data-variant={variant}
      className={baseClass}
      initial={{ opacity: 0, y: 32 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-15% 0px -15% 0px" }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] as const }}
    >
      {inner}
    </motion.section>
  );
}

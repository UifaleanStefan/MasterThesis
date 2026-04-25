import { type ReactNode } from "react";
import { cn } from "../../lib/format";

interface SectionHeaderProps {
  title: ReactNode;
  /** Subtitle / tagline rendered under the title. */
  lede?: ReactNode;
  className?: string;
  align?: "left" | "center";
}

export function SectionHeader({ title, lede, className, align = "left" }: SectionHeaderProps) {
  return (
    <header
      className={cn(
        "max-w-4xl mb-12",
        align === "center" && "mx-auto text-center",
        className,
      )}
    >
      <h2 className="text-4xl md:text-5xl lg:text-6xl font-semibold leading-[1.05] tracking-tight">
        {title}
      </h2>
      {lede ? (
        <p
          className="mt-6 text-lg md:text-xl leading-relaxed"
          style={{ color: "var(--color-text-2)" }}
        >
          {lede}
        </p>
      ) : null}
    </header>
  );
}

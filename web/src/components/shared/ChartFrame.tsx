import { type PropsWithChildren, type ReactNode } from "react";
import { cn } from "../../lib/format";

interface ChartFrameProps extends PropsWithChildren {
  title?: ReactNode;
  subtitle?: ReactNode;
  /** Right-aligned controls (e.g. a metric toggle). */
  controls?: ReactNode;
  className?: string;
  /** Body padding off when child needs to fill (e.g. force-graph). */
  flush?: boolean;
}

export function ChartFrame({
  title,
  subtitle,
  controls,
  className,
  flush,
  children,
}: ChartFrameProps) {
  return (
    <figure className={cn("panel-rise", className)}>
      {title || subtitle || controls ? (
        <div className="flex items-start justify-between gap-4 px-6 pt-5 pb-4 border-b border-[var(--color-border)]">
          <div className="min-w-0">
            {title ? (
              <h3 className="text-base md:text-lg font-semibold leading-tight">
                {title}
              </h3>
            ) : null}
            {subtitle ? (
              <p
                className="mt-1 text-sm"
                style={{ color: "var(--color-text-2)" }}
              >
                {subtitle}
              </p>
            ) : null}
          </div>
          {controls ? <div className="flex-shrink-0">{controls}</div> : null}
        </div>
      ) : null}
      <div className={cn(flush ? "" : "p-6")}>{children}</div>
    </figure>
  );
}

import { motion, useMotionValueEvent, useScroll } from "framer-motion";
import { useState } from "react";
import { cn } from "../../lib/format";

/** Inline GitHub mark — lucide-react v1 dropped brand icons. */
function GithubIcon({ size = 14 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
    >
      <path d="M12 .5C5.7.5.5 5.7.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.2.8-.6v-2.2c-3.2.7-3.9-1.4-3.9-1.4-.5-1.3-1.3-1.7-1.3-1.7-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.7 1.3 3.4 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.5-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2 1-.3 2-.4 3-.4s2 .1 3 .4c2.3-1.5 3.3-1.2 3.3-1.2.6 1.7.2 2.9.1 3.2.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6 4.6-1.5 7.9-5.8 7.9-10.9C23.5 5.7 18.3.5 12 .5z" />
    </svg>
  );
}

interface NavLink {
  id: string;
  label: string;
}

const NAV_LINKS: NavLink[] = [
  { id: "question", label: "The Question" },
  { id: "architecture", label: "Architecture" },
  { id: "progression", label: "V1 → V5" },
  { id: "benchmark", label: "Benchmark" },
  { id: "minilm", label: "The Pivot" },
  { id: "ablation", label: "Ablation" },
  { id: "transfer", label: "Transfer" },
  { id: "neural", label: "Neural" },
  { id: "stage3", label: "Stage 3" },
];

export function StickyNav() {
  const { scrollY } = useScroll();
  const [scrolled, setScrolled] = useState(false);
  const [activeId, setActiveId] = useState<string>("");

  useMotionValueEvent(scrollY, "change", (y) => {
    setScrolled(y > 64);

    // Track which section is currently in view (the one whose top is just
    // above mid-viewport).
    const mid = window.innerHeight / 2;
    let closest = "";
    let closestDist = Infinity;
    for (const link of NAV_LINKS) {
      const el = document.getElementById(link.id);
      if (!el) continue;
      const rect = el.getBoundingClientRect();
      const dist = Math.abs(rect.top - mid);
      if (rect.top - mid < 80 && dist < closestDist) {
        closest = link.id;
        closestDist = dist;
      }
    }
    setActiveId(closest);
  });

  return (
    <motion.header
      className={cn(
        "fixed top-0 left-0 right-0 z-40 transition-all duration-300",
        scrolled
          ? "backdrop-blur-md bg-[rgba(8,9,15,0.75)] border-b border-[var(--color-border)]"
          : "bg-transparent",
      )}
      initial={{ y: -32, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ delay: 0.3, duration: 0.6 }}
    >
      <nav className="mx-auto max-w-7xl flex items-center justify-between gap-6 px-6 py-3">
        <a
          href="#hero"
          className="text-sm font-mono tracking-wide flex items-center gap-2"
          style={{ color: "var(--color-text)" }}
        >
          <span
            className="w-2 h-2 rounded-full"
            style={{
              background:
                "linear-gradient(135deg, var(--color-cyan), var(--color-violet))",
            }}
          />
          learnable-memory
        </a>

        <ul className="hidden lg:flex items-center gap-1">
          {NAV_LINKS.map((link) => (
            <li key={link.id}>
              <a
                href={`#${link.id}`}
                className={cn(
                  "text-xs px-3 py-1.5 rounded-full transition-colors",
                  activeId === link.id
                    ? "text-[var(--color-cyan)] bg-[var(--color-cyan-soft)]"
                    : "text-[var(--color-text-2)] hover:text-[var(--color-text)]",
                )}
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        <a
          href="https://github.com/UifaleanStefan/MasterThesis"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border border-[var(--color-border)] hover:border-[var(--color-border-strong)] transition-colors"
          style={{ color: "var(--color-text-2)" }}
        >
          <GithubIcon size={14} />
          <span className="hidden md:inline">GitHub</span>
        </a>
      </nav>
    </motion.header>
  );
}

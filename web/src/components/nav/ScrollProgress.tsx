import { motion, useScroll, useSpring } from "framer-motion";

/**
 * Thin top-of-page progress bar that fills as the user scrolls.
 * Sticks to the top of the viewport above the navbar.
 */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 120,
    damping: 26,
    mass: 0.4,
  });
  return (
    <motion.div
      aria-hidden
      className="fixed top-0 left-0 right-0 h-[2px] origin-left z-50 pointer-events-none"
      style={{
        scaleX,
        background:
          "linear-gradient(90deg, var(--color-cyan), var(--color-violet))",
      }}
    />
  );
}

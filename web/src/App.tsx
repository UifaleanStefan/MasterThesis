import { useEffect, useState } from "react";

/**
 * Phase A skeleton — confirms Tailwind v4, theme tokens, and the build chain
 * are all wired correctly. Phase B replaces this with the real Hero + nav.
 */
function App() {
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setLoaded(true), 100);
    return () => clearTimeout(t);
  }, []);

  return (
    <main className="min-h-screen flex items-center justify-center px-6">
      <div
        className="panel-rise max-w-2xl w-full p-12 text-center"
        style={{ opacity: loaded ? 1 : 0, transition: "opacity 0.6s" }}
      >
        <div className="pill mb-6 mx-auto" style={{ width: "fit-content" }}>
          <span style={{ color: "var(--color-cyan)" }}>●</span> phase A
        </div>
        <h1 className="text-5xl font-semibold mb-4">
          Learnable Memory
          <br />
          <span style={{ color: "var(--color-cyan)" }}>for AI Agents</span>
        </h1>
        <p className="text-lg" style={{ color: "var(--color-text-2)" }}>
          A Bocconi MSc thesis. The frontend is being rebuilt — full
          interactive site arrives in subsequent commits.
        </p>
        <div className="flex gap-2 justify-center mt-8">
          <span className="pill">Vite + React 19</span>
          <span className="pill">Tailwind v4</span>
          <span className="pill">framer-motion</span>
          <span className="pill">recharts</span>
        </div>
      </div>
    </main>
  );
}

export default App;

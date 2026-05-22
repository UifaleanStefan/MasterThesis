import { EmbeddingProvider } from "./components/interactive/EmbeddingToggle";
import { Hero } from "./sections/Hero";
import { TheQuestion } from "./sections/TheQuestion";
import { Architecture } from "./sections/Architecture";
import { Progression } from "./sections/Progression";
import { Benchmark } from "./sections/Benchmark";
import { MiniLMPivot } from "./sections/MiniLMPivot";
import { Ablation } from "./sections/Ablation";
import { TransferSensitivity } from "./sections/TransferSensitivity";
import { NeuralController } from "./sections/NeuralController";
import { Stage3 } from "./sections/Stage3";
import { GraphEvolution } from "./sections/GraphEvolution";
import { Reproducibility } from "./sections/Reproducibility";
import { ScrollProgress } from "./components/nav/ScrollProgress";
import { StickyNav } from "./components/nav/StickyNav";

function App() {
  return (
    <EmbeddingProvider>
      <ScrollProgress />
      <StickyNav />

      <Hero />
      <TheQuestion />
      <Architecture />
      <Progression />
      <Benchmark />
      <MiniLMPivot />
      <Ablation />
      <TransferSensitivity />
      <NeuralController />
      <Stage3 />
      <GraphEvolution />
      <Reproducibility />
    </EmbeddingProvider>
  );
}

export default App;

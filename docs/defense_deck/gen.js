// Defense deck v2 — rebuilt to Dirk Hovy's "Effective Presentations" rules:
//   10-20-30: <=10 content slides, <=20 words/slide, >=30pt font
//   Dark background; one slide one thought; slides support the talk (not handouts).
//   Graphs: maximize area, integrate legend, y-axis from 0, no pie/3D.
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
p.author = "Stefan Uifalean";
p.title = "Learnable, Task-Adaptive Structured Memory for LLM Agents";

const BG = "0C1A2E", WHITE = "F2F6FB", TEAL = "1FB6A6", AMBER = "E6A23C";
const MUTE = "8FA6C4", GREY = "7C8FA8", BLUE = "4A7FB5", DIM = "223A5E", GRID = "1E3050";
const F = "Calibri";
const W = 13.3;

let n = 0;
function slide() {
  const s = p.addSlide();
  s.background = { color: BG };
  return s;
}
function pageNum(s) {
  n += 1;
  s.addText(String(n), { x: 12.5, y: 6.95, w: 0.4, h: 0.3, fontFace: F, fontSize: 12, color: "3B547A", align: "right", margin: 0 });
}

// ============================================================ 1 TITLE
let s = slide();
s.addText("Learnable, Task-Adaptive\nStructured Memory for LLM Agents",
  { x: 0.9, y: 1.9, w: 11.6, h: 2.0, fontFace: F, fontSize: 44, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.05 });
s.addShape(p.ShapeType.line, { x: 0.95, y: 4.15, w: 3.0, h: 0, line: { color: TEAL, width: 3 } });
s.addText("Stefan Uifalean", { x: 0.9, y: 4.5, w: 11, h: 0.5, fontFace: F, fontSize: 30, bold: true, color: WHITE, margin: 0 });
s.addText("MSc Artificial Intelligence  ·  Bocconi University", { x: 0.9, y: 5.08, w: 11, h: 0.5, fontFace: F, fontSize: 30, color: MUTE, margin: 0 });
s.addText("Supervisor: Prof. Dirk Hovy", { x: 0.9, y: 5.65, w: 11, h: 0.5, fontFace: F, fontSize: 30, color: MUTE, margin: 0 });
s.addNotes("Good [morning]. My thesis asks whether an agent can learn how to build its own memory — instead of us hand-designing it and freezing it — and whether the best memory differs from task to task. Ten minutes: I'll show you the problem, the two questions, how we did it, what we found, and the answers.");
pageNum(s);

// ============================================================ 2 THE FAILURE  (motivation)
s = slide();
s.addText("The agent reads 500 contracts.", { x: 0.9, y: 1.05, w: 11.6, h: 0.7, fontFace: F, fontSize: 40, color: WHITE, margin: 0 });
s.addText([
  { text: "Ask about #3 — ", options: { color: WHITE } },
  { text: "it answers from #500.", options: { color: AMBER, bold: true } },
], { x: 0.9, y: 1.8, w: 11.6, h: 0.7, fontFace: F, fontSize: 40, margin: 0 });
// document strip: 36 marks, #3 teal, #500 amber
const N = 36, sx = 0.9, span = 11.5, step = span / N;
for (let i = 0; i < N; i++) {
  const x = sx + i * step;
  if (i === 2) s.addShape(p.ShapeType.rect, { x, y: 4.2, w: 0.16, h: 1.35, fill: { color: TEAL } });
  else if (i === N - 1) s.addShape(p.ShapeType.rect, { x, y: 4.2, w: 0.16, h: 1.35, fill: { color: AMBER } });
  else s.addShape(p.ShapeType.rect, { x, y: 4.85, w: 0.14, h: 0.7, fill: { color: DIM } });
}
s.addText("answer", { x: 0.65, y: 3.7, w: 1.9, h: 0.45, fontFace: F, fontSize: 30, bold: true, color: TEAL, align: "center", margin: 0 });
s.addText("retrieved", { x: 11.2, y: 3.7, w: 2.0, h: 0.45, fontFace: F, fontSize: 30, bold: true, color: AMBER, align: "center", margin: 0 });
s.addText("#3", { x: 0.65, y: 5.65, w: 1.9, h: 0.45, fontFace: F, fontSize: 30, color: MUTE, align: "center", margin: 0 });
s.addText("#500", { x: 11.2, y: 5.65, w: 2.0, h: 0.45, fontFace: F, fontSize: 30, color: MUTE, align: "center", margin: 0 });
s.addNotes("Here's the problem, concretely. An agent ingests five hundred legal contracts, one clause at a time. You ask it about contract number three. A standard memory scores retrieval by RECENCY — so it hands back clauses from contract five hundred, the most recent thing it saw, and gets the answer wrong. It has SEEN the right evidence. The memory rule that helps a fresh document actively hurts an end-of-corpus question. The rule should depend on the task — but today it's frozen.");
pageNum(s);

// ============================================================ 3 THE TWO QUESTIONS
s = slide();
s.addText("Two questions", { x: 0.9, y: 1.0, w: 11.6, h: 0.6, fontFace: F, fontSize: 30, color: MUTE, margin: 0 });
s.addText("1", { x: 0.9, y: 2.3, w: 0.8, h: 0.9, fontFace: F, fontSize: 44, bold: true, color: TEAL, margin: 0 });
s.addText("Can an agent learn its own memory?", { x: 1.9, y: 2.3, w: 10.6, h: 0.9, fontFace: F, fontSize: 40, color: WHITE, margin: 0 });
s.addText("2", { x: 0.9, y: 4.1, w: 0.8, h: 0.9, fontFace: F, fontSize: 44, bold: true, color: TEAL, margin: 0 });
s.addText("Is the best memory task-dependent?", { x: 1.9, y: 4.1, w: 10.6, h: 0.9, fontFace: F, fontSize: 40, color: WHITE, margin: 0 });
s.addNotes("That gives two research questions. One: can an agent LEARN how to construct its own memory — what to store, which concepts to track, how to score retrieval? Two: is that learned optimum task-dependent — does the best memory genuinely differ from one task to another? Spoiler: the answer to both is yes, but the second one is the interesting one, and the honest version of the first is narrower than you'd expect. I'll come back to both at the end.");
pageNum(s);

// ============================================================ 4 THE IDEA
s = slide();
s.addText("Memory becomes one vector", { x: 0.9, y: 1.05, w: 11.6, h: 0.8, fontFace: F, fontSize: 44, bold: true, color: WHITE, margin: 0 });
// theta as 10 cells, grouped
const cw = 0.72, gap = 0.1, total = 10 * cw + 9 * gap;
const cx0 = (W - (total + 1.5)) / 2 + 1.5; // leave room for the "θ =" label
s.addText("θ =", { x: cx0 - 1.5, y: 3.1, w: 1.25, h: 0.95, fontFace: F, fontSize: 44, bold: true, color: WHITE, align: "right", valign: "middle", margin: 0 });
const groupCol = [TEAL, TEAL, TEAL, TEAL, BLUE, BLUE, BLUE, AMBER, AMBER, AMBER];
for (let i = 0; i < 10; i++) {
  s.addShape(p.ShapeType.roundRect, { x: cx0 + i * (cw + gap), y: 3.1, w: cw, h: 0.95, rectRadius: 0.1, fill: { color: groupCol[i] } });
}
function grpLabel(i0, i1, txt, col) {
  const x = cx0 + i0 * (cw + gap);
  const w = (i1 - i0 + 1) * cw + (i1 - i0) * gap;
  s.addShape(p.ShapeType.line, { x, y: 4.22, w, h: 0, line: { color: col, width: 2 } });
  s.addText(txt, { x, y: 4.32, w, h: 0.5, fontFace: F, fontSize: 30, color: col, align: "center", margin: 0 });
}
grpLabel(0, 3, "store", TEAL);
grpLabel(4, 6, "abstract", BLUE);
grpLabel(7, 9, "retrieve", AMBER);
s.addText("We learn θ. Not the LLM.", { x: 0.9, y: 5.6, w: 11.6, h: 0.6, fontFace: F, fontSize: 32, color: MUTE, align: "center", margin: 0 });
s.addNotes("Our answer: make memory construction itself a learnable object. A single vector — theta, ten numbers — governs the whole pipeline. Four numbers decide what gets STORED. Three decide how entities are ABSTRACTED into a memory graph. Three decide how a candidate is scored at RETRIEVAL. That's it. And crucially we optimize theta ONLY — never the agent's policy, never the language model's weights. So this is not a new LLM and not a new RL agent. It's a small, interpretable knob on top of a frozen model.");
pageNum(s);

// ============================================================ 5 RETRIEVAL = A VOTE
s = slide();
s.addText("Retrieval is a vote", { x: 0.9, y: 1.05, w: 11.6, h: 0.8, fontFace: F, fontSize: 44, bold: true, color: WHITE, margin: 0 });
// bar length = how loud the vote ends up: meaning dominates, freshness matters
// some, the graph link is near-inert. Do NOT draw link longest — that would
// imply the opposite of the thesis' actual finding.
const votes = [["meaning", 0.9, TEAL], ["freshness", 0.55, AMBER], ["link", 0.22, BLUE]];
let vy = 2.6;
votes.forEach(([lab, val, col]) => {
  s.addText(lab, { x: 0.9, y: vy - 0.08, w: 3.1, h: 0.6, fontFace: F, fontSize: 34, color: WHITE, margin: 0 });
  s.addShape(p.ShapeType.roundRect, { x: 4.2, y: vy, w: 8.2, h: 0.44, rectRadius: 0.22, fill: { color: DIM } });
  s.addShape(p.ShapeType.roundRect, { x: 4.2, y: vy, w: 8.2 * val, h: 0.44, rectRadius: 0.22, fill: { color: col } });
  vy += 1.15;
});
s.addText("θ sets how loud each votes.", { x: 0.9, y: 6.05, w: 11.6, h: 0.6, fontFace: F, fontSize: 32, color: MUTE, margin: 0 });
s.addNotes("How does retrieval actually work? Every item in memory gets a score, and the agent keeps the top eight. The score is three signals added up. MEANING: does this item mean the same as the question — embedding similarity, zero to one. FRESHNESS: how recently did I see it. LINK: is it connected to the question in the entity graph — on or off. Three of theta's ten numbers are simply how loudly each signal votes. In the contract example, the freshness vote was set too loud, so the newest document always won. Learning theta is learning how loud to set each vote. One honest note: the LINK vote turned out to carry no measurable weight on our corpora — the graph earns its place at storage time, not retrieval time.");
pageNum(s);

// ============================================================ 6 HOW WE LEARN IT
s = slide();
s.addText("Learned by black-box search", { x: 0.9, y: 1.05, w: 11.6, h: 0.8, fontFace: F, fontSize: 44, bold: true, color: WHITE, margin: 0 });
s.addText("recall@8 — no LLM in the loop", { x: 0.9, y: 1.9, w: 11.6, h: 0.6, fontFace: F, fontSize: 32, color: TEAL, margin: 0 });
const stg = ["grid worlds", "12 systems", "6 benchmarks"];
let stx = 1.15;
stg.forEach((t, i) => {
  s.addShape(p.ShapeType.roundRect, { x: stx, y: 3.5, w: 3.3, h: 1.5, rectRadius: 0.15, fill: { color: BG }, line: { color: i === 2 ? TEAL : "35507A", width: 2.5 } });
  s.addText(t, { x: stx, y: 3.5, w: 3.3, h: 1.5, fontFace: F, fontSize: 32, bold: true, color: i === 2 ? TEAL : WHITE, align: "center", valign: "middle", margin: 0 });
  if (i < 2) s.addText("→", { x: stx + 3.35, y: 3.5, w: 0.85, h: 1.5, fontFace: F, fontSize: 34, color: MUTE, align: "center", valign: "middle", margin: 0 });
  stx += 4.2;
});
s.addNotes("How do we learn theta? Black-box search — an evolution strategy, then CMA-ES. Two reasons it's derivative-free: the storage decision is discrete, and the objective is recall-at-eight of the gold evidence, which uses no language model at all. That matters twice: tuning is cheap, and it cannot be biased by the judge that later scores the answers. We tested in three stages of rising realism: small grid worlds, then a twelve-system benchmark, then six real LLM benchmarks with a GPT-4o-mini answerer and a cross-vendor Claude judge scoring every answer one by one.");
pageNum(s);

// ============================================================ 7 STAGE 1+2 — TASK DEPENDENCE
s = slide();
s.addText("2.5%  →  27.5%", { x: 0.6, y: 1.9, w: 12.1, h: 1.7, fontFace: F, fontSize: 96, bold: true, color: TEAL, align: "center", margin: 0 });
s.addText("Every task learns a different θ.", { x: 0.6, y: 4.0, w: 12.1, h: 0.7, fontFace: F, fontSize: 40, color: WHITE, align: "center", margin: 0 });
s.addText("hardest grid task — learned vs. fixed memory", { x: 0.6, y: 4.85, w: 12.1, h: 0.6, fontFace: F, fontSize: 28, color: MUTE, align: "center", margin: 0 });
s.addNotes("Stage one, grid worlds. The headline is not this number — it's that the recovered VECTORS differ. Key-Door learns to discard most events but keep temporal order. Goal-Room learns to store everything. Different tasks, genuinely different optimal memory. That's the cleanest evidence for research question two. On the hardest task, the learned rule lifts success from two and a half percent to twenty-seven and a half. It's a single-seed proof of concept and I say so in the thesis. Stage two scaled this to twelve memory systems on four environments: we reach the top cluster — a statistical TIE with the strongest baseline, not a number one, and I report it as a tie. Ablation tells us why it works: the novelty parameter is load-bearing; zero it and reward collapses. And the graph-traversal weight is inert at retrieval — an honest negative I keep in.");
pageNum(s);

// ============================================================ 8 STAGE 3 — THE CHART
s = slide();
s.addText("Re-tuning θ per corpus lifts accuracy", { x: 0.65, y: 0.62, w: 10.1, h: 0.7, fontFace: F, fontSize: 38, bold: true, color: WHITE, margin: 0 });
// integrated legend (chips, top-right)
s.addShape(p.ShapeType.rect, { x: 11.0, y: 0.62, w: 0.28, h: 0.28, fill: { color: GREY } });
s.addText("canonical", { x: 11.4, y: 0.5, w: 1.85, h: 0.5, fontFace: F, fontSize: 30, color: GREY, margin: 0 });
s.addShape(p.ShapeType.rect, { x: 11.0, y: 1.12, w: 0.28, h: 0.28, fill: { color: TEAL } });
s.addText("tuned", { x: 11.4, y: 1.0, w: 1.85, h: 0.5, fontFace: F, fontSize: 30, color: TEAL, margin: 0 });
s.addChart(p.ChartType.bar, [
  { name: "canonical", labels: ["FinanceBench", "CUAD", "QASPER"], values: [0.243, 0.028, 0.250] },
  { name: "corpus-tuned", labels: ["FinanceBench", "CUAD", "QASPER"], values: [0.645, 0.172, 0.415] },
], {
  x: 0.5, y: 1.5, w: 12.3, h: 5.25, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
  chartColors: [GREY, TEAL],
  chartArea: { fill: { color: BG } }, plotArea: { fill: { color: BG } },
  showLegend: false,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: F, dataLabelFontSize: 28, dataLabelColor: WHITE, dataLabelFormatCode: "0.00",
  valAxisMinVal: 0, valAxisMaxVal: 0.78, valAxisHidden: true,
  valGridLine: { style: "none" }, catGridLine: { style: "none" },
  catAxisLabelColor: WHITE, catAxisLabelFontFace: F, catAxisLabelFontSize: 30,
  catAxisLineColor: GRID, valAxisLineColor: BG,
  showTitle: false,
});
s.addNotes("Stage three, real LLMs over real corpora. Questions are asked at the END of the corpus — the hard case from slide two. Re-tuning theta per corpus lifts judged accuracy on all three domain-coherent benchmarks. FinanceBench goes from 0.24 to 0.65. On a held-out split, two of the three survive multiple-comparison correction — FinanceBench and CUAD do, QASPER does not, and I report that as non-significant. The mechanism is consistent across all of them: tuning drives the recency weight to nearly zero and pushes the embedding weight up. The memory stops chasing the newest document and starts retrieving by meaning — exactly what an end-of-corpus question needs. And the tuning objective is validated: retrieval recall predicts the judge's score at rho 0.69.");
pageNum(s);

// ============================================================ 9 THE HONEST TWIST
s = slide();
s.addText("18×", { x: 0.6, y: 1.5, w: 12.1, h: 1.8, fontFace: F, fontSize: 110, bold: true, color: TEAL, align: "center", margin: 0 });
s.addText("cheaper, at the same accuracy", { x: 0.6, y: 3.55, w: 12.1, h: 0.7, fontFace: F, fontSize: 40, color: WHITE, align: "center", margin: 0 });
s.addShape(p.ShapeType.line, { x: 5.4, y: 4.65, w: 2.5, h: 0, line: { color: "35507A", width: 2 } });
s.addText("Full context breaks at 11 contracts.", { x: 0.6, y: 4.95, w: 12.1, h: 0.7, fontFace: F, fontSize: 34, color: AMBER, align: "center", margin: 0 });
s.addNotes("And here's the part I'm proudest of, because our own audit forced it. We tested a 'dump everything into the prompt' baseline. Once we fixed a truncation bug, that baseline is statistically TIED with selective memory on accuracy. So I do NOT claim learned memory is more accurate at corpus scale. The honest claim is efficiency and scalability: it matches that accuracy at roughly eighteen times lower token cost, and — structurally — full-context can't even run past about eleven CUAD contracts before it overflows the context window, while selective retrieval holds its prompt flat at around seven hundred tokens. Beyond a few documents, retrieving the right things stops being an optimization and becomes a necessity.");
pageNum(s);

// ============================================================ 10 ANSWERS
s = slide();
s.addText("Yes, and yes.", { x: 0.9, y: 1.3, w: 11.6, h: 1.0, fontFace: F, fontSize: 60, bold: true, color: WHITE, margin: 0 });
s.addText("1", { x: 0.9, y: 3.0, w: 0.7, h: 0.7, fontFace: F, fontSize: 34, bold: true, color: TEAL, margin: 0 });
s.addText("Memory construction is learnable.", { x: 1.75, y: 3.0, w: 10.8, h: 0.7, fontFace: F, fontSize: 36, color: WHITE, margin: 0 });
s.addText("2", { x: 0.9, y: 4.1, w: 0.7, h: 0.7, fontFace: F, fontSize: 34, bold: true, color: TEAL, margin: 0 });
s.addText("The optimum is task-dependent.", { x: 1.75, y: 4.1, w: 10.8, h: 0.7, fontFace: F, fontSize: 36, color: WHITE, margin: 0 });
s.addText("At scale: efficiency, not free accuracy.", { x: 0.9, y: 5.6, w: 11.6, h: 0.7, fontFace: F, fontSize: 30, italic: true, color: MUTE, margin: 0 });
s.addNotes("So, back to the two questions. One: yes — what to store and how to retrieve are captured by a single ten-number vector, optimized by black-box search over a frozen LLM. Two: yes — the optimizer recovers a different theta for every task, and it transfers within a task family but not across families. And measured carefully, at corpus scale the win is efficiency and graceful scaling rather than free accuracy. A modest, but honest, step toward agents that adapt not just their actions but the structure of their own memory.");
pageNum(s);

// ============================================================ 11 THANKS
s = slide();
s.addText("Thank you.", { x: 0.9, y: 2.7, w: 11.6, h: 1.0, fontFace: F, fontSize: 54, bold: true, color: WHITE, margin: 0 });
s.addText("Questions?", { x: 0.9, y: 3.85, w: 11.6, h: 0.8, fontFace: F, fontSize: 40, color: TEAL, margin: 0 });
s.addNotes("Thank you — I'm happy to take questions.");

p.writeFile({ fileName: "Uifalean_Thesis_Defense.pptx" }).then(f => console.log("WROTE", f));

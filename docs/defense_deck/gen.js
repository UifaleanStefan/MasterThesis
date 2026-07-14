const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
p.author = "Stefan Uifalean";
p.title = "Learnable, Task-Adaptive Structured Memory for LLM Agents";

// ---- palette ----
const NAVY = "12294A", NAVY2 = "0B1830", TEAL = "1FB6A6", AMBER = "E6A23C";
const INK = "1B2A44", MUTE = "5E7088", WHITE = "FFFFFF", PANEL = "F2F6FB", LINE = "DCE5EF";
const HEAD = "Cambria", BODY = "Calibri";
const W = 13.3, H = 7.5;

function footer(s, n, dark) {
  s.addText("Learnable Task-Adaptive Memory for LLM Agents  ·  S. Uifalean  ·  Bocconi MSc AI",
    { x: 0.5, y: 7.05, w: 10.5, h: 0.3, fontFace: BODY, fontSize: 9, color: dark ? "8FA6C4" : MUTE, align: "left", margin: 0 });
  s.addText(String(n), { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 9, color: dark ? "8FA6C4" : MUTE, align: "right", margin: 0 });
}
function title(s, t, kicker) {
  if (kicker) s.addText(kicker.toUpperCase(), { x: 0.6, y: 0.42, w: 12, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0 });
  s.addText(t, { x: 0.58, y: kicker ? 0.72 : 0.5, w: 12.1, h: 0.9, fontFace: HEAD, fontSize: 30, bold: true, color: INK, margin: 0 });
}
function chip(s, x, y, w, txt, fill, tcol) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h: 0.42, rectRadius: 0.08, fill: { color: fill } });
  s.addText(txt, { x, y, w, h: 0.42, fontFace: BODY, fontSize: 12.5, bold: true, color: tcol || WHITE, align: "center", valign: "middle", margin: 0 });
}
// numbered circle badge
function badge(s, x, y, num, col) {
  s.addShape(p.ShapeType.ellipse, { x, y, w: 0.52, h: 0.52, fill: { color: col } });
  s.addText(num, { x, y, w: 0.52, h: 0.52, fontFace: HEAD, fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
}

// ============================================================ 1 TITLE
let s = p.addSlide(); s.background = { color: NAVY };
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.14, fill: { color: TEAL } });
s.addText("MSc THESIS DEFENSE", { x: 0.9, y: 1.5, w: 11, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: TEAL, charSpacing: 3, margin: 0 });
s.addText("Learnable, Task-Adaptive Structured\nMemory for LLM Agents", { x: 0.86, y: 2.0, w: 11.6, h: 1.9, fontFace: HEAD, fontSize: 42, bold: true, color: WHITE, lineSpacingMultiple: 1.0, margin: 0 });
s.addText("Can an agent learn how to construct its own memory — and is the optimum task-dependent?",
  { x: 0.9, y: 3.95, w: 11, h: 0.6, fontFace: BODY, fontSize: 17, italic: true, color: "CADCFC", margin: 0 });
s.addShape(p.ShapeType.line, { x: 0.92, y: 4.9, w: 3.4, h: 0, line: { color: "35507A", width: 1.5 } });
s.addText([
  { text: "Stefan Uifalean", options: { bold: true, color: WHITE, fontSize: 15, breakLine: true } },
  { text: "MSc Artificial Intelligence  ·  Bocconi University", options: { color: "AFC3E0", fontSize: 13, breakLine: true } },
  { text: "Supervisor: Prof. Dirk Hovy  ·  Academic Year 2025–2026", options: { color: "AFC3E0", fontSize: 13 } },
], { x: 0.9, y: 5.15, w: 11, h: 1.2, fontFace: BODY, margin: 0, paraSpaceAfter: 4 });
s.addNotes("Good [morning]. My thesis asks whether an agent can LEARN how to build its own memory, rather than having it hand-designed and frozen — and whether the best memory differs from task to task. I'll motivate the problem, state the two research questions, walk you through the method and three stages of experiments, and come back to answer the questions.");

// ============================================================ 2 MOTIVATING EXAMPLE
s = p.addSlide(); s.background = { color: WHITE };
title(s, "A question whose answer the agent has seen — and still gets wrong", "Motivating example");
// left: scenario text
s.addText([
  { text: "An agent ingests 500 legal contracts, one clause at a time.", options: { bold: true, color: INK, fontSize: 16, breakLine: true, paraSpaceAfter: 10 } },
  { text: "You ask: “What law governs contract #3?”", options: { color: INK, fontSize: 16, breakLine: true, paraSpaceAfter: 10 } },
  { text: "A standard memory scores retrieval by recency — so it returns clauses from contract #500, the most recent thing it saw.", options: { color: MUTE, fontSize: 15, breakLine: true, paraSpaceAfter: 10 } },
  { text: "The right evidence was read long ago. The memory rule that helps a fresh document actively hurts an end-of-corpus question.", options: { color: INK, fontSize: 15 } },
], { x: 0.6, y: 1.9, w: 6.5, h: 4.2, fontFace: BODY, margin: 0, valign: "top" });
// right: simple flow card
s.addShape(p.ShapeType.roundRect, { x: 7.5, y: 2.0, w: 5.2, h: 4.3, rectRadius: 0.12, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
chip(s, 7.9, 2.4, 4.4, "Doc 1  →  Doc 3  →  …  →  Doc 500", NAVY);
s.addText("stored by time of arrival", { x: 7.9, y: 2.9, w: 4.4, h: 0.3, fontFace: BODY, fontSize: 11, italic: true, color: MUTE, align: "center", margin: 0 });
s.addShape(p.ShapeType.roundRect, { x: 8.35, y: 3.55, w: 3.5, h: 0.75, rectRadius: 0.1, fill: { color: WHITE }, line: { color: "C63B3B", width: 1.5 } });
s.addText([{ text: "retrieves Doc 500  ", options: { color: "C63B3B", bold: true } }, { text: "✗", options: { color: "C63B3B", bold: true } }], { x: 8.35, y: 3.55, w: 3.5, h: 0.75, fontFace: BODY, fontSize: 14, align: "center", valign: "middle", margin: 0 });
s.addText("wrong contract", { x: 8.35, y: 4.3, w: 3.5, h: 0.3, fontFace: BODY, fontSize: 11, italic: true, color: MUTE, align: "center", margin: 0 });
s.addText([{ text: "answer was at ", options: { color: MUTE } }, { text: "Doc 3 of 500", options: { color: AMBER, bold: true } }], { x: 8.35, y: 5.1, w: 3.5, h: 0.6, fontFace: BODY, fontSize: 15, align: "center", valign: "middle", margin: 0 });
footer(s, 2);
s.addNotes("Concrete example. An agent reads 500 contracts one clause at a time. You ask about contract number 3. A memory that scores retrieval by recency hands back clauses from the most recent contract — number 500 — and gets it wrong, even though it saw the right answer. The rule that's optimal for a fresh document is exactly wrong at end-of-corpus. So the memory rule should depend on the task.");

// ============================================================ 3 PROBLEM GENERALIZED
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Today, an agent’s memory is fixed and hand-designed", "The gap");
const rows = [
  ["1", "WHAT to store", "every observation kept, or a fixed rule decides"],
  ["2", "WHICH concepts to track", "entities / nodes chosen by a frozen heuristic"],
  ["3", "HOW to score retrieval", "one fixed weighting of similarity vs. recency"],
];
let ry = 1.95;
rows.forEach(([n, h, d]) => {
  badge(s, 0.7, ry, n, TEAL);
  s.addText(h, { x: 1.45, y: ry - 0.03, w: 5.4, h: 0.35, fontFace: BODY, fontSize: 16, bold: true, color: INK, margin: 0 });
  s.addText(d, { x: 1.45, y: ry + 0.3, w: 5.4, h: 0.35, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  ry += 1.0;
});
s.addShape(p.ShapeType.roundRect, { x: 7.3, y: 1.95, w: 5.4, h: 3.05, rectRadius: 0.12, fill: { color: NAVY } });
s.addText("Different tasks need different memory", { x: 7.6, y: 2.2, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: WHITE, margin: 0 });
s.addText([
  { text: "multi-hop question", options: { color: TEAL, bold: true } }, { text: "  →  entity relations", options: { color: "CADCFC", breakLine: true, paraSpaceAfter: 10 } },
  { text: "navigation task", options: { color: TEAL, bold: true } }, { text: "  →  spatial / temporal transitions", options: { color: "CADCFC", breakLine: true, paraSpaceAfter: 10 } },
  { text: "end-of-corpus question", options: { color: TEAL, bold: true } }, { text: "  →  recency-independent recall", options: { color: "CADCFC" } },
], { x: 7.6, y: 2.75, w: 4.85, h: 1.7, fontFace: BODY, fontSize: 14, margin: 0, paraSpaceAfter: 8, valign: "top" });
s.addText("A memory rule optimal for one is often actively harmful for another.", { x: 0.7, y: 5.35, w: 12, h: 0.5, fontFace: BODY, fontSize: 16, italic: true, bold: true, color: INK, align: "center", margin: 0 });
footer(s, 3);
s.addNotes("Generalizing: what these systems share is that the memory is FIXED. What to store, which concepts to track, and how to score a retrieval are hand-designed once and frozen. But a multi-hop question needs entity relations, a navigation task needs spatial transitions, and an end-of-corpus question needs recency-independent recall. A rule that's optimal for one is harmful for another — so a single frozen memory must compromise.");

// ============================================================ 4 RESEARCH QUESTIONS
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Research questions", null);
function rqCard(x, tag, q) {
  s.addShape(p.ShapeType.roundRect, { x, y: 2.0, w: 5.7, h: 3.5, rectRadius: 0.14, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  s.addShape(p.ShapeType.roundRect, { x: x + 0.35, y: 2.4, w: 1.1, h: 0.55, rectRadius: 0.1, fill: { color: TEAL } });
  s.addText(tag, { x: x + 0.35, y: 2.4, w: 1.1, h: 0.55, fontFace: HEAD, fontSize: 17, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(q, { x: x + 0.35, y: 3.15, w: 5.0, h: 2.1, fontFace: BODY, fontSize: 18, color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.05 });
}
rqCard(0.7, "RQ1", "Can an agent learn how to construct its own memory — what to store, which concepts to track as nodes, and how to score retrieval?");
rqCard(6.9, "RQ2", "Is the learned optimum task-dependent — does the best memory genuinely differ from one task to another?");
s.addText("We answer both empirically, and hold ourselves to a self-audit of every headline number.", { x: 0.7, y: 5.9, w: 12, h: 0.4, fontFace: BODY, fontSize: 14, italic: true, color: MUTE, align: "center", margin: 0 });
footer(s, 4);
s.addNotes("This gives two research questions. RQ1: can an agent LEARN how to construct its own memory — what to store, which concepts to track, and how to score retrieval? RQ2: is that learned optimum task-dependent? And a commitment that runs through the thesis: we subject every headline number to an adversarial self-audit and report the corrected figures.");

// ============================================================ 5 THE IDEA
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Make memory construction a small learnable vector  θ", "The idea");
s.addText([
  { text: "A single parameter vector ", options: { color: INK } }, { text: "θ (up to 10 dimensions)", options: { color: TEAL, bold: true } },
  { text: " governs the whole pipeline — storage, abstraction, and retrieval scoring.", options: { color: INK } },
], { x: 0.6, y: 1.85, w: 12, h: 0.7, fontFace: BODY, fontSize: 17, margin: 0 });
// three grouped chips
const groups = [
  ["STORAGE", "θ_store · θ_novel · θ_surprise · θ_erich", "what is written to memory"],
  ["ABSTRACTION", "θ_entity · θ_temporal · θ_decay", "which nodes & edges form the graph"],
  ["RETRIEVAL", "w_graph · w_embed · w_recency", "how a candidate is scored at query time"],
];
let gx = 0.6;
groups.forEach(([h, mid, d]) => {
  s.addShape(p.ShapeType.roundRect, { x: gx, y: 2.75, w: 3.9, h: 1.85, rectRadius: 0.12, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  s.addText(h, { x: gx + 0.25, y: 2.95, w: 3.4, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: TEAL, charSpacing: 1, margin: 0 });
  s.addText(mid, { x: gx + 0.25, y: 3.35, w: 3.45, h: 0.8, fontFace: "Consolas", fontSize: 12.5, bold: true, color: INK, margin: 0, valign: "top" });
  s.addText(d, { x: gx + 0.25, y: 4.2, w: 3.45, h: 0.3, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE, margin: 0 });
  gx += 4.15;
});
s.addShape(p.ShapeType.roundRect, { x: 0.6, y: 4.95, w: 12.1, h: 1.35, rectRadius: 0.1, fill: { color: NAVY } });
s.addText("retrieval score:", { x: 0.9, y: 5.15, w: 3, h: 0.35, fontFace: BODY, fontSize: 13, bold: true, color: TEAL, margin: 0 });
s.addText("s(item, q)  =  w_graph · g(item,q)   +   w_embed · cos(e_item, e_q)   +   w_recency · ρ(item)",
  { x: 0.9, y: 5.5, w: 11.5, h: 0.5, fontFace: "Consolas", fontSize: 15, bold: true, color: WHITE, margin: 0 });
s.addText("We optimize θ only — never the agent’s policy, never the LLM’s weights.", { x: 0.9, y: 6.0, w: 11.5, h: 0.3, fontFace: BODY, fontSize: 12.5, italic: true, color: "CADCFC", margin: 0 });
footer(s, 5);
s.addNotes("Our answer is to make memory construction itself a learnable object. A single vector theta — up to ten dimensions in the fullest system — governs the whole pipeline: what gets stored, which nodes and edges form a memory graph, and how a candidate is scored at retrieval as a weighted sum of a graph signal, embedding similarity, and recency. Crucially we optimize theta ONLY — not the policy, not the language model. So this is not a new RL agent or a new LLM.");

// ============================================================ 6 HOW RETRIEVAL WORKS
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Retrieval is a weighted vote across three signals", "How retrieval works");
s.addText([
  { text: "Every memory item gets a score; the agent keeps the top few. ", options: { color: INK } },
  { text: "θ sets how loudly each signal votes.", options: { color: INK, bold: true } },
], { x: 0.6, y: 1.85, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 16, margin: 0 });
const sig = [
  ["w_embed", "Does it mean\nthe same?", "meaning match · 0…1", 0.85, TEAL, "the governing-law clause scores high"],
  ["w_recency", "How fresh\nis it?", "freshness · 0…1", 0.62, AMBER, "the newest clause always wins — the trap in our example"],
  ["w_graph", "Is it linked\nin the graph?", "shares an entity · 0 or 1", 1.0, NAVY, "on/off link — inert at retrieval in Stage 3"],
];
let qx = 0.6;
sig.forEach(([tok, q, cap, score, barcol, ex]) => {
  s.addShape(p.ShapeType.roundRect, { x: qx, y: 2.65, w: 3.9, h: 2.45, rectRadius: 0.12, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  s.addText(tok, { x: qx + 0.25, y: 2.82, w: 3.4, h: 0.35, fontFace: "Consolas", fontSize: 15, bold: true, color: TEAL, margin: 0 });
  s.addText(q, { x: qx + 0.25, y: 3.2, w: 3.4, h: 0.75, fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.0 });
  s.addShape(p.ShapeType.roundRect, { x: qx + 0.25, y: 4.05, w: 3.4, h: 0.18, rectRadius: 0.09, fill: { color: "E4EBF3" } });
  s.addShape(p.ShapeType.roundRect, { x: qx + 0.25, y: 4.05, w: 3.4 * score, h: 0.18, rectRadius: 0.09, fill: { color: barcol } });
  s.addText(cap, { x: qx + 0.25, y: 4.28, w: 3.4, h: 0.3, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE, margin: 0 });
  s.addText(ex, { x: qx + 0.25, y: 4.6, w: 3.45, h: 0.45, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0, valign: "top" });
  qx += 4.15;
});
s.addShape(p.ShapeType.roundRect, { x: 0.6, y: 5.28, w: 12.1, h: 1.18, rectRadius: 0.1, fill: { color: NAVY } });
s.addText("score(item)  =  w_embed · meaning  +  w_recency · freshness  +  w_graph · link      →   keep the top 8",
  { x: 0.9, y: 5.44, w: 11.5, h: 0.4, fontFace: "Consolas", fontSize: 13.5, bold: true, color: WHITE, margin: 0 });
s.addText([
  { text: "Learning θ = learning how loud to set each vote.  ", options: { color: "CADCFC", bold: true } },
  { text: "For an end-of-corpus question the optimizer turns recency down and meaning up.", options: { color: "CADCFC" } },
], { x: 0.9, y: 5.92, w: 11.5, h: 0.42, fontFace: BODY, fontSize: 13, italic: true, margin: 0 });
footer(s, 6);
s.addNotes("Let me make retrieval concrete, because it's the heart of the method. When the agent needs to answer, every item in memory gets a score, and it keeps the top few. The score is just three signals added up. One: does the item MEAN the same as the question — an embedding similarity, zero to one. Two: how RECENT is it — freshness, zero to one. Three: is it LINKED to the question in the entity graph — on or off. Each of the three retrieval weights in theta decides how loudly that signal votes. In our contract example the recency vote was set too loud, so the newest document always won. Learning theta is just learning how loud to set each vote — and for end-of-corpus questions it learns to turn recency down and meaning up. One honesty note I'll return to: the graph vote carried no measurable weight on the Stage 3 corpora, so in practice retrieval came down to meaning versus freshness.");

// ============================================================ 7 METHOD
s = p.addSlide(); s.background = { color: WHITE };
title(s, "How we learn θ: black-box search, no LLM in the loop", "Method");
s.addShape(p.ShapeType.roundRect, { x: 0.6, y: 1.9, w: 12.1, h: 1.25, rectRadius: 0.1, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
s.addText([
  { text: "Optimizer:  ", options: { bold: true, color: INK } }, { text: "Evolution Strategy → CMA-ES", options: { bold: true, color: TEAL } },
  { text: "   ·   Objective:  ", options: { bold: true, color: INK } }, { text: "recall@8 of the gold evidence", options: { bold: true, color: TEAL } },
], { x: 0.9, y: 2.1, w: 11.5, h: 0.4, fontFace: BODY, fontSize: 16, margin: 0 });
s.addText("Derivative-free (the store decision is discrete) and LLM-free — so tuning is cheap and never biased by the judge.",
  { x: 0.9, y: 2.55, w: 11.5, h: 0.5, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
// three-stage timeline
const stages = [
  ["STAGE 1", "Grid worlds", "Is θ task-dependent?", TEAL],
  ["STAGE 2", "12 systems × 4 envs", "Is it competitive? What does each θ do?", NAVY],
  ["STAGE 3", "6 real LLM benchmarks", "Does it work on real corpus QA?", AMBER],
];
let sx = 0.6;
stages.forEach(([tag, h, d, col], i) => {
  s.addShape(p.ShapeType.roundRect, { x: sx, y: 3.6, w: 3.75, h: 2.5, rectRadius: 0.12, fill: { color: WHITE }, line: { color: col, width: 2 } });
  s.addShape(p.ShapeType.roundRect, { x: sx, y: 3.6, w: 3.75, h: 0.6, rectRadius: 0.12, fill: { color: col } });
  s.addText(tag, { x: sx, y: 3.6, w: 3.75, h: 0.6, fontFace: HEAD, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(h, { x: sx + 0.25, y: 4.4, w: 3.25, h: 0.7, fontFace: BODY, fontSize: 16, bold: true, color: INK, margin: 0, valign: "top" });
  s.addText(d, { x: sx + 0.25, y: 5.15, w: 3.25, h: 0.8, fontFace: BODY, fontSize: 13, italic: true, color: MUTE, margin: 0, valign: "top" });
  if (i < 2) s.addText("→", { x: sx + 3.72, y: 4.5, w: 0.45, h: 0.6, fontFace: BODY, fontSize: 26, bold: true, color: MUTE, align: "center", margin: 0 });
  sx += 4.15;
});
footer(s, 7);
s.addNotes("We learn theta with black-box search — an Evolution Strategy for the small case, then CMA-ES. Two reasons it's derivative-free: the storage decision is discrete, and — importantly — the objective is recall-at-8 of the gold evidence, which uses NO language model. So tuning is cheap and it can't be biased by the judge that later scores answers. We test the idea in three stages of increasing realism: grid worlds, a twelve-system benchmark, and real LLMs over six corpora.");

// ============================================================ 8 STAGE 1
s = p.addSlide(); s.background = { color: WHITE };
title(s, "The optimizer recovers a different θ for every task", "Stage 1 — grid worlds");
// table of vectors
const t1 = [
  [{ text: "Environment", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "Learned θ (store, entity, temporal)", options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: "Baseline → Learned", options: { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" } }],
  [{ text: "Key-Door" }, { text: "(0.12,  0.00,  0.82)   — discard, keep order" }, { text: "17.5% → 30%", options: { align: "center" } }],
  [{ text: "Goal-Room" }, { text: "(1.00,  0.22,  1.00)   — store everything" }, { text: "70% → 80%", options: { align: "center" } }],
  [{ text: "MultiHopKeyDoor", options: { bold: true } }, { text: "(1.00,  0.49,  0.84)", options: { bold: true } }, { text: "2.5% → 27.5%", options: { align: "center", bold: true, color: AMBER } }],
];
s.addTable(t1, { x: 0.6, y: 2.0, w: 8.0, colW: [2.3, 4.3, 1.4], rowH: 0.62, fontFace: BODY, fontSize: 13, color: INK, valign: "middle", border: { type: "solid", color: LINE, pt: 1 }, fill: { color: WHITE } });
// stat callout
s.addShape(p.ShapeType.roundRect, { x: 9.0, y: 2.0, w: 3.7, h: 2.55, rectRadius: 0.14, fill: { color: NAVY } });
s.addText("2.5% → 27.5%", { x: 9.0, y: 2.35, w: 3.7, h: 0.9, fontFace: HEAD, fontSize: 30, bold: true, color: TEAL, align: "center", margin: 0 });
s.addText("hardest task, learned vs. fixed memory  (+25 points)", { x: 9.2, y: 3.35, w: 3.3, h: 0.9, fontFace: BODY, fontSize: 13, italic: true, color: "CADCFC", align: "center", margin: 0, valign: "top" });
s.addText("The headline is the difference between the vectors: Key-Door discards most events but preserves temporal order; Goal-Room stores everything. The optimal memory structure is genuinely task-dependent — the first, cleanest evidence for RQ2. (Single-seed proof of concept.)",
  { x: 0.6, y: 5.1, w: 12.1, h: 1.2, fontFace: BODY, fontSize: 15, color: INK, margin: 0, valign: "top" });
footer(s, 8);
s.addNotes("Stage 1, grid worlds, a three-dimensional theta tuned separately per task. The headline is the DIFFERENCE between the recovered vectors: Key-Door learns to discard most events but keep temporal order; Goal-Room learns to store everything. That's the cleanest evidence for RQ2 — the optimal memory is task-dependent. On the hardest task the learned rule lifts success from 2.5 to 27.5 percent. It's a single-seed proof of concept, and we say so.");

// ============================================================ 9 STAGE 2
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Competitive — and we can say exactly why", "Stage 2 — benchmark, ablation, transfer");
const cards2 = [
  ["Top cluster", "0.178 vs 0.173", "a statistical tie with the strongest baseline — competitive, not uniquely best"],
  ["Load-bearing", "θ_novel", "zeroing it collapses reward to 0 — it gates the whole storage pipeline"],
  ["Honest negative", "inert at retrieval", "the w_graph weight carries no measurable retrieval load on our benchmarks — the graph still scaffolds storage & abstraction"],
];
let cx = 0.6;
cards2.forEach(([h, big, d]) => {
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 2.0, w: 3.9, h: 2.9, rectRadius: 0.12, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  s.addText(h.toUpperCase(), { x: cx + 0.25, y: 2.2, w: 3.4, h: 0.35, fontFace: BODY, fontSize: 12, bold: true, color: TEAL, charSpacing: 1, margin: 0 });
  s.addText(big, { x: cx + 0.25, y: 2.6, w: 3.45, h: 0.7, fontFace: HEAD, fontSize: 21, bold: true, color: INK, margin: 0, valign: "top" });
  s.addText(d, { x: cx + 0.25, y: 3.4, w: 3.45, h: 1.35, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0, valign: "top" });
  cx += 4.15;
});
s.addShape(p.ShapeType.roundRect, { x: 0.6, y: 5.15, w: 12.1, h: 1.15, rectRadius: 0.1, fill: { color: NAVY } });
s.addText([
  { text: "Also:  ", options: { bold: true, color: TEAL } },
  { text: "a ~2,000-parameter neural controller, given the same search budget, ", options: { color: WHITE } },
  { text: "matches but does not beat", options: { bold: true, color: WHITE } },
  { text: " the ten scalars — so a small interpretable θ is a strong, cheap baseline.", options: { color: "CADCFC" } },
], { x: 0.9, y: 5.4, w: 11.5, h: 0.7, fontFace: BODY, fontSize: 14.5, margin: 0, valign: "middle" });
footer(s, 9);
s.addNotes("Stage 2 scales to twelve memory systems on four environments. The ten-dimensional learnable memory reaches the top cluster — 0.178 versus the best baseline's 0.173, which we report honestly as a statistical tie, not a number-one. Ablation tells us WHY it works: the novelty parameter is load-bearing — zero it and reward collapses — while the graph-traversal term carries no measurable retrieval load, an honest negative result we keep in. And a two-thousand-parameter neural controller with the same budget only matches the ten scalars.");

// ============================================================ 10 STAGE 3 SETUP
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Real LLMs, corpus-cumulative QA, cross-vendor judge", "Stage 3 — setup");
const setup = [
  ["Answerer", "gpt-4o-mini", "temperature 0"],
  ["Encoder", "MiniLM", "all-MiniLM-L6-v2"],
  ["Judge", "Claude, 1-by-1", "cross-vendor, 5-point rubric"],
];
let ux = 0.6;
setup.forEach(([h, big, d]) => {
  s.addShape(p.ShapeType.roundRect, { x: ux, y: 1.95, w: 3.9, h: 1.5, rectRadius: 0.12, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
  s.addText(h.toUpperCase(), { x: ux + 0.25, y: 2.12, w: 3.4, h: 0.3, fontFace: BODY, fontSize: 11.5, bold: true, color: TEAL, charSpacing: 1, margin: 0 });
  s.addText(big, { x: ux + 0.25, y: 2.45, w: 3.45, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  s.addText(d, { x: ux + 0.25, y: 2.98, w: 3.45, h: 0.35, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, margin: 0 });
  ux += 4.15;
});
s.addText("θ is re-tuned per corpus on recall@8; questions are answered end-of-corpus. Six benchmarks, pre-partitioned into three groups:",
  { x: 0.6, y: 3.7, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 14.5, color: INK, margin: 0 });
const groups3 = [
  ["CONFIRMATORY", "FinanceBench · CUAD · QASPER", "lift predicted", TEAL],
  ["CONTROLS", "HotpotQA · LongMemEval", "pre-registered, no lift predicted", NAVY],
  ["UNDEFINED", "NarrativeQA", "no gold → objective undefined", MUTE],
];
let bx = 0.6;
groups3.forEach(([h, mid, d, col]) => {
  s.addShape(p.ShapeType.roundRect, { x: bx, y: 4.35, w: 3.9, h: 1.85, rectRadius: 0.12, fill: { color: WHITE }, line: { color: col, width: 2 } });
  s.addText(h, { x: bx + 0.25, y: 4.55, w: 3.45, h: 0.35, fontFace: BODY, fontSize: 12.5, bold: true, color: col, charSpacing: 1, margin: 0 });
  s.addText(mid, { x: bx + 0.25, y: 4.95, w: 3.45, h: 0.75, fontFace: BODY, fontSize: 14.5, bold: true, color: INK, margin: 0, valign: "top" });
  s.addText(d, { x: bx + 0.25, y: 5.75, w: 3.45, h: 0.35, fontFace: BODY, fontSize: 12, italic: true, color: MUTE, margin: 0 });
  bx += 4.15;
});
footer(s, 10);
s.addNotes("Stage 3 is the real test. A gpt-4o-mini answerer, a MiniLM encoder, and — importantly — a CROSS-VENDOR Claude judge scoring every answer one-by-one, so the judge isn't grading its own vendor. Theta is re-tuned per corpus. And we pre-registered the six benchmarks into three groups: three domain-coherent ones where we predict a lift, two domain-incoherent CONTROLS, and NarrativeQA which has no gold evidence so the objective is undefined. Fixing that partition up front is what lets the result be a test rather than a search for a favorable subset.");

// ============================================================ 11 STAGE 3 RESULTS (chart)
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Corpus tuning lifts end-of-corpus accuracy", "Stage 3 — results");
const chartData = [
  { name: "canonical θ", labels: ["FinanceBench", "CUAD", "QASPER"], values: [0.243, 0.028, 0.250] },
  { name: "corpus-tuned θ", labels: ["FinanceBench", "CUAD", "QASPER"], values: [0.645, 0.172, 0.415] },
];
s.addChart(p.ChartType.bar, chartData, {
  x: 0.6, y: 1.95, w: 7.4, h: 4.5, barDir: "col", barGrouping: "clustered",
  chartColors: ["B9C6D8", TEAL], showLegend: true, legendPos: "t", legendColor: INK, legendFontFace: BODY, legendFontSize: 12,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: BODY, dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFormatCode: "0.00",
  valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2, valGridLine: { color: "EAEFF5", size: 1 },
  catGridLine: { style: "none" }, catAxisLabelColor: INK, catAxisLabelFontFace: BODY, catAxisLabelFontSize: 12,
  valAxisLabelColor: MUTE, valAxisLabelFontFace: BODY, valAxisLabelFontSize: 10, valAxisLabelFormatCode: "0.0",
  showTitle: false,
});
s.addText("judged accuracy (batch, end-of-corpus)", { x: 0.6, y: 6.45, w: 7.4, h: 0.3, fontFace: BODY, fontSize: 11, italic: true, color: MUTE, align: "center", margin: 0 });
// right column callouts
s.addShape(p.ShapeType.roundRect, { x: 8.35, y: 1.95, w: 4.35, h: 1.55, rectRadius: 0.12, fill: { color: NAVY } });
s.addText("2 of 3", { x: 8.35, y: 2.1, w: 4.35, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true, color: TEAL, align: "center", margin: 0 });
s.addText("domain-coherent benchmarks survive held-out testing (FinanceBench, CUAD); QASPER does not.", { x: 8.6, y: 2.8, w: 3.85, h: 0.65, fontFace: BODY, fontSize: 12.5, color: "CADCFC", align: "center", margin: 0, valign: "top" });
s.addShape(p.ShapeType.roundRect, { x: 8.35, y: 3.7, w: 4.35, h: 2.75, rectRadius: 0.12, fill: { color: PANEL }, line: { color: LINE, width: 1 } });
s.addText("The mechanism", { x: 8.6, y: 3.9, w: 3.85, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
s.addText([
  { text: "w_recency", options: { fontFace: "Consolas", bold: true, color: INK } }, { text: "  3.78 → 0.003", options: { color: TEAL, bold: true, breakLine: true, paraSpaceAfter: 10 } },
  { text: "w_embed", options: { fontFace: "Consolas", bold: true, color: INK } }, { text: "     1.08 → 2.63", options: { color: TEAL, bold: true, breakLine: true } },
], { x: 8.6, y: 4.35, w: 3.85, h: 0.9, fontFace: BODY, fontSize: 13.5, margin: 0, paraSpaceAfter: 6 });
s.addText("Tuning down-weights recency and up-weights meaning — it retrieves by semantic match, exactly what an end-of-corpus question needs.", { x: 8.6, y: 5.3, w: 3.85, h: 1.05, fontFace: BODY, fontSize: 12.5, italic: true, color: MUTE, margin: 0, valign: "top" });
footer(s, 11);
s.addNotes("Here's the headline. Re-tuning theta per corpus lifts end-of-corpus accuracy on all three coherent benchmarks — FinanceBench jumps from 0.24 to 0.65 — and on a held-out split, two of the three survive multiple-comparison correction; QASPER doesn't, and we report it as non-significant. The mechanism is consistent: tuning drives the recency weight toward zero and the embedding weight up. The memory stops chasing the most recent document and retrieves by meaning — which is exactly what an end-of-corpus question needs. We also verified the tuning objective is valid: retrieval recall predicts judge score at rho 0.69.");

// ============================================================ 12 HONEST REFRAME
s = p.addSlide(); s.background = { color: WHITE };
title(s, "Measured honestly, the win is efficiency — not raw accuracy", "The self-audit");
s.addText([
  { text: "Our own audit corrected a headline: a full-context ", options: { color: INK } },
  { text: "“dump-all”", options: { color: INK, bold: true } },
  { text: " baseline is ", options: { color: INK } },
  { text: "statistically tied", options: { color: INK, bold: true } },
  { text: " with selective memory on accuracy. So the case for learned memory at corpus scale is efficiency and scalability.", options: { color: INK } },
], { x: 0.6, y: 1.9, w: 12.1, h: 0.85, fontFace: BODY, fontSize: 16, margin: 0 });
const stats = [
  ["≈ 18×", "cheaper", "same accuracy on FinanceBench, a fraction of the token cost"],
  ["N ≈ 11", "dump-all breaks", "full-context overflows the 128K window at ~11 CUAD contracts"],
  ["704", "tokens, flat", "selective retrieval stays constant no matter how big the corpus grows"],
];
let px = 0.6;
stats.forEach(([big, sub, d], i) => {
  const col = i === 0 ? TEAL : i === 1 ? "C0504D" : NAVY;
  s.addShape(p.ShapeType.roundRect, { x: px, y: 3.0, w: 3.9, h: 2.95, rectRadius: 0.14, fill: { color: i === 2 ? NAVY : PANEL }, line: { color: i === 2 ? NAVY : LINE, width: 1 } });
  s.addText(big, { x: px + 0.2, y: 3.35, w: 3.5, h: 0.95, fontFace: HEAD, fontSize: 40, bold: true, color: col === NAVY ? TEAL : col, align: "center", margin: 0 });
  s.addText(sub, { x: px + 0.2, y: 4.35, w: 3.5, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: i === 2 ? WHITE : INK, align: "center", margin: 0 });
  s.addText(d, { x: px + 0.35, y: 4.8, w: 3.2, h: 1.0, fontFace: BODY, fontSize: 12.5, color: i === 2 ? "CADCFC" : MUTE, align: "center", margin: 0, valign: "top" });
  px += 4.15;
});
footer(s, 12);
s.addNotes("And here's the part I'm proudest of. Our self-audit caught that a full-context 'dump everything into the prompt' baseline, once a truncation bug was fixed, is statistically TIED with selective memory on accuracy. So we do NOT claim selective retrieval is more accurate. Instead the argument is efficiency and scalability: it matches accuracy at about eighteen times lower cost, and — structurally — dump-all can't even run past about eleven contracts because it overflows the context window, while selective retrieval holds its prompt flat at around 700 tokens. Beyond a few documents, retrieving the right items isn't an optimization, it's a necessity.");

// ============================================================ 13 ANSWERS (dark close)
s = p.addSlide(); s.background = { color: NAVY };
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.14, fill: { color: TEAL } });
s.addText("BACK TO THE RESEARCH QUESTIONS", { x: 0.9, y: 0.7, w: 11, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: TEAL, charSpacing: 2, margin: 0 });
function ans(y, tag, q, a) {
  s.addText(tag, { x: 0.9, y, w: 1.2, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: TEAL, margin: 0 });
  s.addText(q, { x: 2.1, y, w: 10.3, h: 0.4, fontFace: BODY, fontSize: 15, italic: true, color: "AFC3E0", margin: 0 });
  s.addText(a, { x: 2.1, y: y + 0.42, w: 10.3, h: 0.75, fontFace: BODY, fontSize: 16.5, bold: true, color: WHITE, margin: 0, valign: "top" });
}
ans(1.5, "RQ1", "Can memory construction be learned?", "Yes — what to store and how to retrieve are captured by a single vector θ, optimized by black-box search over the LLM.");
ans(3.15, "RQ2", "Is the learned optimum task-dependent?", "Yes — the optimizer recovers different θ per task; it transfers within a task family but not across families.");
s.addShape(p.ShapeType.roundRect, { x: 0.9, y: 4.75, w: 11.5, h: 1.15, rectRadius: 0.1, fill: { color: NAVY2 } });
s.addText([
  { text: "And measured carefully:  ", options: { bold: true, color: TEAL } },
  { text: "at corpus scale a learned memory buys efficiency and graceful scaling, not a free accuracy gain — a modest but honest step toward agents that adapt the structure of their own memory.", options: { color: WHITE } },
], { x: 1.2, y: 4.95, w: 10.9, h: 0.8, fontFace: BODY, fontSize: 14.5, margin: 0, valign: "middle" });
s.addText("Thank you — questions welcome.", { x: 0.9, y: 6.15, w: 11.5, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: WHITE, margin: 0 });
s.addNotes("So, back to the questions. RQ1 — can memory construction be learned? Yes: what to store and how to retrieve are captured by a single theta, optimized by black-box search over the language model. RQ2 — is that optimum task-dependent? Yes: the optimizer recovers different theta per task, and it transfers within a task family but not across families. And measured carefully, at corpus scale the learned memory buys efficiency and graceful scaling rather than a free accuracy gain. A modest but defensible step toward agents that adapt not just their actions, but the structure of their own memory. Thank you — I'm happy to take questions.");

p.writeFile({ fileName: "Uifalean_Thesis_Defense.pptx" }).then(f => console.log("WROTE", f));

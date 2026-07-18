# Thesis Defense — Examiner Question Bank
*30 likely questions with model answers — Learnable, Task-Adaptive Structured Memory for LLM Agents.*

**How to use:** read the question, answer out loud, then compare. Concede what is true, then reframe to the honest claim.


## A. Framing, contribution, and scope

### Q1. In one minute: what did you do, and what did you find?

I made **memory construction itself a learnable object**. In an LLM agent, memory does three jobs — decide what to **store**, how to **abstract** it into a graph, and how to **score retrieval** — and normally those rules are hand-designed and frozen. I collapse all three into a single 10-number vector **θ** and optimize it per task with black-box search (CMA-ES) on a cheap retrieval objective, leaving the LLM and the agent policy untouched.

Across three stages I show: (RQ1) memory construction **can** be learned, and (RQ2) the optimum is genuinely **task-dependent** — the optimizer recovers different θ for different tasks. Measured honestly against a full-context baseline, the win at corpus scale is **efficiency and graceful scaling** — matching accuracy at ~1/18th the cost — not a free accuracy gain.

### Q2. What is your actual contribution beyond MemGPT, HippoRAG, or plain RAG?

Those systems are sophisticated, but their storage and retrieval **policies are hand-designed and never re-optimized per task**. MemGPT delegates paging to the LLM through fixed, designer-written functions; HippoRAG fixes behaviour in an offline knowledge graph plus a Personalized-PageRank traversal with a hand-set damping constant; RAG tunes retrieval ranking only, not what to store or abstract.

My contribution is orthogonal: (i) a **single θ that spans the whole pipeline** — storage *and* abstraction *and* retrieval scoring — not just one stage; (ii) **per-task optimization** of that vector by derivative-free search, with transfer *measured* rather than assumed; and (iii) a corpus-cumulative evaluation on six real benchmarks with cross-vendor judging, held-out significance testing, and a disclosed self-audit. To my knowledge no prior system tunes a memory-construction policy per task with a black-box optimizer and reports its transfer and a full-context comparison.

### Q3. Isn't this just RAG with tuned hyperparameters?

No — RAG retrieves from a **fixed store** with a **fixed similarity metric**; only the ranking is in play. Here the **store decision** (what is even written to memory), the **graph abstraction** (which entities become nodes, what edges form), and the **retrieval weighting** are all exposed in θ and learned jointly.

Plain RAG is actually a *special case* of my space: set the storage gate to 'keep everything' and the graph/recency weights to zero, and θ collapses to embedding-only retrieval over a full store. The thesis is about the coupling RAG omits — the **read/write asymmetry**: committing more aggressively only pays off if the scorer can later surface it, and a sharp scorer is wasted on a store that never recorded the evidence. One θ lets the optimizer trade a laxer write threshold against a sharper retrieval weight; RAG cannot express that trade at all.

### Q4. RQ1 asks 'can memory be learned?' — isn't the answer obviously yes?

The trivial version is yes; the **non-trivial** version is what I test. The real questions are: can a *tiny, interpretable* parameterization (10 scalars) — with **no gradients and no LLM in the loop** — recover useful, task-specific memory policies? And does learning the *joint* store-and-retrieve rule beat hand-designed policies under a fair, same-budget comparison?

The honest answer I report is narrower than 'yes, trivially': memory *can* be learned this way and it is competitive, but its accuracy edge is **benchmark-dependent**, and at corpus scale the durable benefit is efficiency, not accuracy. I deliberately state RQ1 in a falsifiable form and do not round the answer up.

### Q5. Define 'task-dependent' precisely. How could this claim be falsified?

Formally: for each task T, let θ_opt(T) = argmax R(θ; T) be the memory policy that maximizes performance. The claim is that **θ_opt(T) visibly differs across tasks** — there is no single θ that is optimal everywhere.

It is falsifiable two ways. If a single θ_opt were optimal for every task, the side-by-side vectors in Stage 1 would look the same and RQ2 would return 'no difference' — refuting the thesis. And if θ_opt(T) always transferred to a related task T', per-task tuning would be unnecessary. Neither is assumed: Stage 1 shows visibly different vectors (Key-Door discards+orders; Goal-Room stores everything), Stage 2 shows within-family but not across-family transfer, and Stage 3 shows the *identity of the best memory system changes with the corpus and even its size*.


## B. Method: parameterization, optimizer, objective

### Q6. Walk me through θ — the 10 parameters and what each controls.

θ splits into three groups by the job it governs. **Storage gate (4):** `θ_store` is the importance threshold to keep an event; `θ_novel`, `θ_erich`, `θ_surprise` weight the three importance features (novelty, entity-richness, surprise). **Abstraction (3):** `θ_entity` is the threshold for an entity to become a graph node, `θ_temporal` is the probability of drawing a temporal edge, `θ_decay` is how fast entity importance fades. **Retrieval (3):** `w_graph`, `w_embed`, `w_recency` weight the three retrieval signals.

The 7 storage/abstraction params act at **write time** and are bounded [0,1]; the 3 retrieval weights act at **read time** and are bounded [0,4]. All ten are optimized jointly by CMA-ES — that joint optimization is the point, because storing and scoring trade off against each other.

### Q7. Write the storage rule and the retrieval score. What are g, cos, and ρ?

**Storage:** for each event, importance = θ_novel·nov + θ_erich·erich + θ_surprise·surp; the event is written **iff importance > θ_store** (higher θ_store ⇒ stricter ⇒ fewer events).

**Retrieval:** s(item, q) = w_graph·g + w_embed·cos + w_recency·ρ, and the top-k (k=8) are returned. Each term is normalized to [0,1]: **g** is a binary graph-traversal membership signal (does the item share an entity hub with the query, 0/1); **cos** is the embedding cosine rescaled ½(1+cos) to [0,1]; **ρ** = 1/(1+Δt) is recency, where Δt is steps since the item was seen (newest ≈ 1, old → 0).

These match `graph_memory_v4.py` exactly (a point worth stressing, because an earlier draft's notation table inverted θ_store's direction — now corrected).

### Q8. Why CMA-ES? Why not gradient descent or Bayesian optimization?

**Gradient descent is inapplicable:** the storage decision is a discrete write (an event is either kept or not), so the objective is non-differentiable — there is no gradient to follow.

**Why CMA-ES specifically:** the objective is low-dimensional (10 continuous params), noisy (stochastic writes + sampled questions), non-separable (storage thresholds and retrieval weights interact, so the good search directions are linear combinations of parameters, not single axes), and — per the Stage-2 sensitivity analysis — anisotropic and sharply ridged. CMA-ES maintains a full covariance over the search distribution and adapts it each generation, so it rotates the sampling ellipse to travel *along* a diagonal ridge; simpler diagonal Evolution Strategies must zig-zag across it.

**Why not Bayesian optimization:** BO is built to hoard information when each evaluation is enormously expensive. My evaluations are cheap (LLM-free recall), so the surrogate-model overhead buys little at 10 dimensions, and BO copes less gracefully with the write-noise. I use a simple ES for the tiny Stage-1 search and CMA-ES for Stages 2–3.

### Q9. You optimize recall@8, not answer accuracy. Isn't that the wrong objective?

It is a deliberate choice with two payoffs. First, recall@8 of the gold evidence uses **no LLM**, so tuning is cheap (negligible dollar cost) and — crucially — **the judge that later scores answers is nowhere near the optimizer**, so tuning cannot be biased toward the judge's preferences. Second, it keeps the discrete, cheap structure that makes derivative-free search tractable.

The obvious worry is that good retrieval might not mean good answers. I test that directly rather than assume it: pooled **Spearman ρ = 0.69** between recall@8 and the judge score. So the proxy is valid — retrieving the gold evidence into the top-8 really does predict a better judged answer — while keeping the optimization LLM-free and unbiased.

### Q10. How do you prevent θ from overfitting the evaluation questions?

Two safeguards. First, a **held-out split**: the corpus is partitioned by a fixed, documented rule *before* tuning; CMA-ES sees only the tuning-question split; and every reported lift is recomputed on the held-out split the tuner never saw. A θ that merely memorized the tuning questions would show no held-out gain — and indeed QASPER's lift does not survive, which is exactly the check working.

Second, **significance testing**: a paired Wilcoxon signed-rank test on per-question (tuned − canonical) differences, with **Holm–Bonferroni correction** across the six-benchmark family, plus bootstrap (or cluster-bootstrap where questions cluster in documents) 95% CIs. A lift is called 'significant' only if it survives **both** the held-out split and the corrected test; anything weaker is reported as directional, never rounded up.

### Q11. How is answer quality judged, and why isn't a Claude judge circular?

A `gpt-4o-mini` answerer (temperature 0) writes an answer from the k retrieved items, and a **Claude** judge scores it on a five-point rubric {0, 0.25, 0.5, 0.75, 1}, **one answer at a time**.

It is not circular because it is **cross-vendor**: the answerer is OpenAI and the judge is Anthropic, so the judge is not grading its own model's outputs — this avoids the self-preference bias a same-model judge introduces. I still treat the judge as an instrument to be validated, not trusted blindly (see κ and the refusal-classifier validation). The honest limitation is that I do not have an independent human or second-vendor judge — that is named as future work.

### Q12. What does κ = 0.66 measure, and what does it NOT measure?

κ = 0.66 is **quadratic-weighted Cohen's κ** from an independent blind **second pass** over a stratified 180-question sample — 'substantial' agreement, with 87.8% of scores agreeing within one rubric level. It quantifies how **reproducible** the judging is.

What it does **not** measure: because both passes are Claude-class, it bounds judge **self-consistency**, *not* agreement with a human or a different vendor. I state this explicitly and do not dress it up as inter-annotator or human agreement. Converting κ from self-consistency to genuine agreement — via a human or non-Claude rater on the same sample — is the single change I flag as most strengthening every accuracy claim.

### Q13. Explain the two-tier judging and the refusal classifier. Isn't that a way to inflate scores?

The two output types need different scoring. A **substantive answer** requires reading the gold evidence and judging semantic adequacy — irreducibly LLM-shaped, so it is scored one-by-one by the Claude judge. A **refusal / 'I don't know'** carries almost no content: the whole decision is whether the abstention was warranted given the evidence was (or wasn't) retrievable. Routing that through a generative judge wastes tokens and exposes an honest abstention to the judge's answer preferences, so those entries get a **rule-assisted classifier** score.

It cannot silently inflate scores because the classifier is **validated as a measurement instrument**: I hand-check a stratified sample, report its precision/recall, and — the decisive quantity — **propagate every classifier error into the reported mean**, giving a population-weighted score error of **0.028** on a 300-entry sample. The rule-assisted share is reported per benchmark (high where terse 'not stated' abstentions are common, near-zero on narrative corpora), and the two paths are merged only after confirming the induced error is small everywhere.


## C. Stage 1 and Stage 2

### Q14. Stage 1 is single-seed grid worlds — why should I believe anything from it?

Stage 1 is a **proof of concept**, and I label it as such — its job is not to establish an effect size but to make the RQ2 mechanism **visible and inspectable**. The environments are deliberately simple so the learned θ can be read directly, and the headline is not the score but that the optimizer recovers **structurally different vectors** per task (Key-Door learns to discard most events yet keep temporal order; Goal-Room learns to store everything).

The single-seed concern is answered downstream, not hand-waved: Stage-3 corpus-mode evaluation is shown **near-deterministic at temperature 0** (cross-seed std 0.012 on the replicated FinanceBench cell, 0.002 on canonical), so the remaining single-seed cells are a matter of CI-tightening, not point-estimate risk. Multi-seed replication of the larger cells is named as future work.

### Q15. Your Stage 2 headline is 0.178 vs 0.173 — that's a tie. What's the result?

I report it **as a tie**, deliberately — the claim is that a 10-dimensional learnable memory reaches the **top performance cluster** among twelve systems, i.e. it is *competitive*, not uniquely best. Over-claiming a 0.005 gap as a win would be exactly the kind of thing my self-audit exists to prevent.

The real Stage-2 results are the *why* and the *transfer*, not the headline number: ablation shows `θ_novel` is **load-bearing** (zeroing it collapses reward), `w_graph` is **inert** at retrieval, and a tuned θ **transfers within but not across** task families — direct support for RQ2. I also disclose that the 0.178/0.173 figure is from the earlier **TF-IDF backend**; under MiniLM the absolute rewards shift and V4 is competitive but does not lead, which is the honest reading recorded in the Stage-2 self-audit.

### Q16. You found w_graph is inert. Doesn't that undermine the whole graph-memory premise?

It sharpens it rather than undermines it, and only an **exposed, optimized** weight could report this at all — a fixed-policy graph retriever cannot tell you its own structure was redundant. Two scoping points. First, w_graph is inert **as a retrieval-score term on these corpora**: sweeping it across its range does not move judged accuracy, because on single-hop, retrieval-dominated text the dense-embedding signal already captures what the entity graph would. The graph still does real work at **storage and abstraction** time.

Second, the result is explicitly **scoped, not blanket**: I did not stress the tasks where a relational term *should* pay off — genuine multi-hop questions whose answer needs two entities no single passage co-mentions. So the honest conclusion is that the burden of proof now sits with graph-memory designs to demonstrate retrieval-time value on compositional tasks; my own multi-hop control (HotpotQA) is only a directional n≈100 signal and cannot adjudicate it. That is future work, and I say so.

### Q17. A ~2000-parameter neural controller only matched the 10 scalars. Isn't that a negative result for learning?

It is a **positive** result for the *parameterization*, and I frame it that way. Given the **same search budget**, a ~1,962-parameter MLP (50→32→10) reaches best fitness 0.233 and held-out ~0.19 — it **matches but does not beat** the ten interpretable scalars, with far more parameters and no robustness advantage, and it fails zero-shot on the larger out-of-distribution task just as the scalar version does.

The lesson is that the extra expressivity is **not needed at this scale**: the 10-scalar policy is a strong, cheap, and *interpretable* baseline — you can read what each learned parameter does, which you cannot with the MLP. I explicitly leave revisiting the neural controller with a larger budget as future work, now that the scalar sets a strong bar.

### Q18. The 0.178 number is TF-IDF-backend but Stage 3 uses MiniLM. Isn't that inconsistent cherry-picking?

It would be if I hid it — instead I **disclose the backend explicitly** and correct the record. The 0.178 top-cluster tie comes from the original 31-word TF-IDF vocabulary, which is task-specific by construction and silently zeroes out on natural-language text; that is exactly why Stage 3 moves to the pretrained **MiniLM** sentence encoder, and every Stage-3 number uses MiniLM throughout.

I also state that the ranking is **backend-sensitive**: under MiniLM the absolute rewards shift and V4 is competitive with but does not lead the strongest baselines — the honest reading recorded in the Stage-2 self-audit and corrected in Appendix C (an earlier draft wrongly claimed MiniLM gave 'the same conclusion'). The provenance record stamps the backend on every run, so legacy TF-IDF numbers remain identifiable and are never mixed with MiniLM ones.


## D. Stage 3: setup and results

### Q19. Why corpus-cumulative, end-of-corpus QA? Isn't that an artificial setup?

It is the setup that makes the problem **hard in the way real agents are hard**. An agent ingests documents one at a time and must answer questions about material read *long ago*, against the full accumulated memory — precisely the case where a recency-biased memory fails (the 500-contract example). I evaluate in two regimes: **online** (right after the source document, so it's recent) and **batch** (end-of-corpus, against everything), and the interesting claim is about batch.

It is not artificial — it is the honest stress test. If I only asked questions right after the relevant document, recency would look great and the task-dependence point would be invisible. End-of-corpus is where store-and-score decisions actually matter.

### Q20. How did you choose the six benchmarks — did you pick ones that work?

No — the benchmarks and their roles were **pre-registered into three groups before judging**, which is exactly what prevents cherry-picking. **Confirmatory (3):** FinanceBench, CUAD, QASPER are domain-coherent — persistent facts that end-of-corpus retrieval should recover, so the hypothesis predicts a lift. **Controls (2):** HotpotQA and LongMemEval are domain-incoherent (distractor / multi-session structure), where corpus-cumulative recall is *not* the natural objective, so no lift is predicted. **Undefined (1):** NarrativeQA has no paragraph-level gold, so its recall objective is undefined by construction.

The six deliberately span disjoint domains, document lengths, and question styles — if the optimal θ were the same across such different corpora, the task-dependence claim would be false, so the diversity is a **test, not decoration**. Fixing the coherent/control/undefined partition up front is what lets the results be read as a test rather than a search for a favourable subset.

### Q21. The QASPER lift didn't survive Holm correction. Why report it at all?

Because reporting it — and labelling it non-significant — is the honest thing, and it demonstrates the held-out machinery actually bites. On the raw split QASPER shows a lift (0.250 → 0.415), but after the held-out recomputation and Holm correction across the benchmark family it **does not survive**, so I report **2 of 3** confirmatory benchmarks as significant (FinanceBench, CUAD) and QASPER as **not significant**, never rounded up.

It also carries information: at the full 281-paper scale the learned memory is **tied for best of seven** configurations on QASPER (0.351 vs a fairly-tuned BM25's 0.367, p=0.30), beating five others. So 'not a significant *lift over its own canonical*' and 'competitive at full scale' are both true and both stated. Suppressing the negative would be precisely the over-fitting-to-a-subset failure the pre-registration guards against.

### Q22. What is the actual mechanism behind the corpus-tuning lift?

It is consistent and interpretable across the coherent benchmarks: tuning **drives the recency weight down and the embedding weight up** — on the FinanceBench cell, `w_recency 3.78 → 0.003` and `w_embed 1.08 → 2.63`. The memory stops chasing the most recently ingested document and starts retrieving by **meaning**.

That is exactly what an end-of-corpus question needs — the evidence was read long ago, so freshness is actively misleading and semantic match is what surfaces the right clause. This is also why I call the learned memory 'recency-independent': the lift is not a black-box improvement, it is a legible reweighting of the three retrieval votes, and it is the same story on every domain-coherent corpus.

### Q23. Against a fairly tuned BM25 you LOSE on CUAD. Doesn't that defeat your thesis?

No — it *is* the thesis. The fair baseline is a BM25 given the **same** corpus-recall tuning budget as θ (not library defaults), judged one-by-one at matched scope. Against it, the learned memory is **benchmark-dependent**: it **wins** FinanceBench (+0.132), **ties** QASPER, and **loses** CUAD (−0.131) — where relevance is surface-lexical clause-extraction and a well-tuned sparse retriever is simply the right tool.

That is not a defeat; it is the direct empirical signature of **task-dependence (RQ2)**. If one policy dominated everywhere, there would be nothing for a per-task optimizer to buy. The honest reading: the learned optimum captures semantic and relational relevance well and offers **no free lunch** on lexical tasks — and on those the durable contribution is the efficiency/scalability argument, not raw accuracy. Reporting a loss on CUAD is the fair-baseline design working as intended.

### Q24. Explain the scale reshuffle. Why is it a feature, not an embarrassment?

Under the identical harness, the CUAD memory-system ranking **reverses with corpus size**. At 50 contracts: HippoRAG 0.222 > V4t 0.172 > Letta 0.146. At the full 510 contracts: Letta 0.199 > V4t 0.149 > HippoRAG 0.134 — HippoRAG's graph collapses and Letta overtakes it. All differences are paired-Holm significant, and I report it **both ways**.

It is the sharpest possible evidence for the whole approach: not only is the best mechanism corpus-*specific*, its identity changes with corpus *size*. A design that freezes its write-and-read rule (HippoRAG's graph topology + fixed damping; MemGPT/Letta's hand-set tiers) must commit in advance to one point and **cannot follow the crossover**. Exposing the decisions as an optimized θ is the only one of the three that can be *re-pointed* at whichever relevance signal dominates. The crossover is the empirical premise of the thesis — if a single fixed policy sufficed, per-task optimization would buy nothing. It is only an 'embarrassment' if you expected a universal winner, which I explicitly do not claim.


## E. Honesty, limitations, and defense

### Q25. A full-context 'dump-all' baseline ties you on accuracy. So why use your method?

I report that tie openly — my own audit traced an earlier 'dump-all collapses' headline to a truncation bug and corrected it to an accuracy **tie** (overlapping bootstrap CIs on FinanceBench). So I explicitly do **not** claim selective memory is more accurate. The case is **efficiency and scalability**:

- **Cost:** it matches full-context accuracy at roughly **1/18th** the token cost.
- **Structural necessity:** dump-all **overflows the 128K context window at N ≈ 11 CUAD contracts** and hits ~43× the window at the full 510; selective retrieval keeps its prompt flat (~704 tokens). Beyond a handful of documents, dump-all simply **cannot run**.

So the honest answer is: at trivial scale, use whichever you like; at corpus scale, retrieving the right items stops being an optimization and becomes a **necessity**, and doing it with a cheap learned policy is the practical win.

### Q26. You claim '18× cheaper' and 'flat tokens' but say the accuracy-vs-N probe was inconclusive. Reconcile these.

They are different quantities, and I keep them apart deliberately. The **cost and token claims are structural and directly measured**: the selective prompt is ~704 tokens regardless of N, and full-context grows until it overflows at N≈11 — these are not statistical estimates, they are counts, so they are solid.

The **judged-accuracy-versus-N** claim is the one I do **not** make: my accuracy-vs-N probe was underpowered and I say the current data do *not* support a clean 'accuracy stays flat as N grows' statement. A properly powered, difficulty- and gold-position-controlled probe is named as future work. So the reconciliation is: I assert the efficiency curve (measured) and explicitly withhold the accuracy-vs-N curve (inconclusive) — rather than letting the strong measurement smuggle in the weak one.

### Q27. θ must be re-tuned per task. Isn't that impractical?

Two answers. First, re-tuning is **cheap**: CMA-ES optimizes a pure recall@8 objective with **no LLM in the loop**, so a per-corpus fit costs negligible dollars and minutes — you are not retraining a model. Second, and more interesting: the task-dependence does **not** force per-task tuning, because θ turns out to be **largely predictable**.

A leave-one-benchmark-out predictor maps cheap, LLM-free **corpus statistics** to a θ that recovers **0.80 of the per-task tuning lift** on the held-out benchmark — converting 're-tune per task' into a one-shot prediction on all but the most idiosyncratic corpus (CUAD, at 0.37, is the honest exception). So the practical story is: tuning is cheap, and mostly you don't even need to — you can predict a good θ from corpus descriptors.

### Q28. Your whole evaluation uses one answerer and one encoder. How do you know results generalize?

I don't claim they generalize beyond that stack — I **scope the claims to gpt-4o-mini + MiniLM** and name this as the primary limitation and future work. It is an honest ceiling: I measure what a *learned memory* can reach under a *fixed reader and encoder*, not the ceiling of an end-to-end system.

Two things make it less fragile than it sounds. The mechanism is **legible** (recency down, meaning up) rather than a black-box fit to one model's quirks, and the retrieval objective is **encoder-agnostic in form** (only cosine is used). But testing whether the efficiency advantage and the shape of the learned θ survive a **stronger answerer and a different embedding model** is exactly the experiment I flag as most worth running next. I would rather state the scope precisely than over-generalize from one pair.

### Q29. What changed between the submitted thesis and now — did you fix errors, or just add scope?

Both, and I can be precise because it is commit-anchored (the submitted PDF is byte-identical to git commit 268845d). The additions **complete claims the submitted version had stated as future work**: fair same-budget baselines, head-to-heads against HippoRAG and MemGPT/Letta across three corpora and two scales (revealing the scale reshuffle), full-corpus scaling (CUAD 510, QASPER 281), θ-predictability, and multi-seed determinism.

It is also **more accurate**, not just larger (45 → 68 pages). An adversarial self-review corrected real defects — several present already in the submitted version: the Stage-2 headline is now labelled as TF-IDF-backend and disclosed as backend-sensitive; Algorithm 2's storage rule and the notation table were fixed to match the code; a mislabeled '$18.5 per-benchmark tuning cost' was corrected to the *total* answering cost. Nothing in the submitted version was retracted — the additions strengthen, scope, or complete it, and every headline number was re-verified against the committed data.

### Q30. What is the single biggest weakness of this thesis, and the one experiment that would most strengthen it?

The biggest weakness is the **judging**: every answer is scored by a Claude-class judge, so my reported κ = 0.66 bounds **self-consistency, not agreement with a human or a different vendor**. Every accuracy claim ultimately rests on one judge family, and while it is cross-vendor to the *answerer*, it is not independently validated against human judgment.

The single most strengthening experiment is therefore a **blind human (or non-Claude) rater on the same stratified sample** — it would convert κ from self-consistency to genuine agreement and recalibrate every fair-baseline verdict against a rater with no shared bias with the answerer. If I had to name a close second, it would be a **stronger answerer + a second encoder**, to test whether the efficiency advantage and the learned θ are properties of the method or of one model pair. Both are named in the thesis as future work — I would rather point at the real gap than defend an overclaim, because I never made one.


---

# 10-Minute Presentation Script

*Talk track mapped to the 11-slide defense deck. Practice out loud 3× and time it (aim 9:30–10:00). `[CLICK]` = advance a build; `[PAUSE]` = stop and breathe.*


### Slide 1 — Title & roadmap  ·  ~0:30

Good [morning], and thank you for being here. My thesis is *Learnable, Task-Adaptive Structured Memory for LLM Agents*. In one line: today an agent's memory is hand-designed and frozen — I ask whether an agent can instead *learn* how to build its own memory, and whether the best memory differs from task to task. In the next ten minutes I'll show you the problem, my two questions, how I did it, and what I found. Spoilers are allowed, so I'll tell you now: the answer to both questions is yes — but the interesting answer is in the details.


### Slide 2 — The failure (motivation)  ·  ~1:15

Let me start with a failure. [CLICK] Imagine an agent that reads five hundred legal contracts, one clause at a time. You ask it a simple question: *what law governs contract number three?* A standard memory scores what to retrieve by *recency* — how recently it saw something. So it hands back clauses from contract five hundred, the most recent thing it read, and it gets the answer wrong. [PAUSE] Here is the frustrating part: it *has* seen the right evidence — contract three is sitting in its memory. The rule that helps a fresh document — 'trust the newest thing' — is exactly the rule that *hurts* a question whose answer was read long ago. [PAUSE] That is the whole problem in miniature: a memory rule that is optimal in one situation is actively harmful in another. Yet today we pick one rule and freeze it.


### Slide 3 — Two questions  ·  ~0:40

That gives me two research questions. [CLICK] First: can an agent *learn* how to construct its own memory — what to store, which concepts to track, and how to score retrieval? [CLICK] Second: is that learned optimum *task-dependent* — does the best memory genuinely differ from one task to another? I'll answer both at the end.


### Slide 4 — Memory becomes one vector  ·  ~1:10

Here is the idea. [CLICK] A memory does three jobs: it decides what to *store*, how to *abstract* what it kept into a graph of entities, and how to *score retrieval* when a question arrives. Normally all three are hand-designed. I collapse all three into a single vector of ten numbers — theta. Four numbers control storage, three control abstraction, three control retrieval. [PAUSE] And the crucial part: [CLICK] I learn *theta only*. I never touch the language model's weights, and I never touch the agent's policy. This is not a new language model and not a new reinforcement-learning agent — it is a small, interpretable knob on top of a frozen model, which is exactly what lets me re-fit it to a new task in minutes.


### Slide 5 — Retrieval is a vote  ·  ~1:00

Let me make retrieval concrete, because it is the heart of the method. [CLICK] When a question arrives, every item in memory gets a score, and the agent keeps the top eight. That score is a *vote* across three signals. *Meaning*: does this item mean the same as the question? *Freshness*: how recently did I see it? *Link*: is it connected to the question in the entity graph? [PAUSE] Three of theta's ten numbers are simply *how loud each of those votes is*. In my contract example, the freshness vote was turned up too high, so the newest document always won. Learning theta is learning how loud to set each vote. One honest note I will come back to: the *link* vote turned out to carry no measurable weight on my corpora.


### Slide 6 — Learned by black-box search  ·  ~1:00

So how do I learn theta? [CLICK] Not with gradient descent — the decision to store an item is discrete, so there is nothing to differentiate. I use black-box evolutionary search, CMA-ES, the right tool for a small, noisy, ridged landscape. [PAUSE] And the objective it optimizes is *recall at eight of the gold evidence* — did the right document land in the top eight retrieved? That uses no language model at all. Two things fall out of that: tuning is cheap, and — importantly — the judge that later grades answers is nowhere near the optimizer, so I cannot accidentally tune to the judge. [CLICK] I test the idea in three stages of rising realism: small grid worlds, a twelve-system benchmark, and then six real long-context LLM benchmarks.


### Slide 7 — Task-dependence (Stages 1 & 2)  ·  ~1:05

Stage one, the grid worlds, gives the cleanest result. [CLICK] On the hardest task, a learned memory lifts success from two and a half percent to twenty-seven and a half. But the *number* is not the point — the point is that the optimizer recovers a genuinely *different* theta for every task. One task learns to throw away almost everything but keep temporal order; another learns to store everything. [PAUSE] That is the direct evidence for my second question: the best memory is task-dependent. Stage two scales this to twelve memory systems on four environments. My learnable memory reaches the top cluster — I report it honestly as a statistical *tie*, not a win. And an ablation tells me *why* it works: one storage parameter is load-bearing — zero it and performance collapses — while the graph term does nothing at retrieval. I keep that negative result in.


### Slide 8 — Stage 3 results  ·  ~1:15

Stage three is the real test: real language models, over real corpora, answering questions at the *end* of the corpus — the hard case from my opening example. [CLICK] Re-tuning theta per corpus lifts judged accuracy on all three domain-coherent benchmarks — FinanceBench jumps from 0.24 to 0.65. On a held-out split, two of the three survive multiple-comparison correction; the third, QASPER, does not, and I report it as non-significant. [PAUSE] The mechanism is the same everywhere and it is readable: tuning drives the *freshness* weight down to almost zero and the *meaning* weight up. The memory stops chasing the newest document and starts retrieving by meaning — exactly what an end-of-corpus question needs. And I validated the objective: retrieval recall predicts the judge's score at rho zero point six nine.


### Slide 9 — The honest twist  ·  ~1:00

Now the part I am proudest of, because my own audit forced it. [CLICK] I tested a baseline that just dumps *every* document into the prompt. Once I fixed a bug, that baseline is statistically *tied* with my selective memory on accuracy. So I do *not* claim my method is more accurate. [PAUSE] The honest claim is *efficiency*: it matches that accuracy at roughly eighteen times lower cost — and, structurally, dumping everything in *breaks* at about eleven contracts, because it overflows the context window, while selective memory holds its prompt flat at around seven hundred tokens. Beyond a handful of documents, retrieving the right thing is not an optimization — it is a necessity.


### Slide 10 — Answers  ·  ~0:40

So, back to my two questions. [CLICK] Can memory construction be learned? Yes — captured by a single ten-number vector, optimized over a frozen model. [CLICK] Is the optimum task-dependent? Yes — the optimizer recovers a different theta per task, and it transfers within a task family but not across. [PAUSE] And measured carefully, at corpus scale the win is efficiency and graceful scaling, not free accuracy — a modest but honest step toward agents that adapt not just their actions, but the structure of their own memory.


### Slide 11 — Thanks  ·  ~0:15

Thank you. I would be glad to take your questions.

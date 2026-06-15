export const meta = {
  name: 'pass2-rejudge-qa',
  description: 'Pass-2 1-by-1 re-judging of free-form QA (QASPER/FB/HQA/LME/NQA) content answers by Claude subagents',
  phases: [
    { title: 'Judge', detail: 'one Claude per part, semantic QA scoring 1-by-1' },
    { title: 'Verify', detail: 'independent Claude re-judges adversarially, writes final' },
  ],
}

// args: { cells: [ {cell:"<name>", parts:["00",...]}, ... ] }
const A = (typeof args === 'string') ? JSON.parse(args) : args
const CELLS = A.cells
const WD = 'results/stage3/_judge_workdir'

const RUBRIC = [
'You are a meticulous QA judge re-scoring free-form question-answering ONE BY ONE.',
'Benchmarks: QASPER (questions about an NLP research paper), FinanceBench (financial-report QA), HotpotQA (multi-hop factoid), LongMemEval (long-context memory), NarrativeQA (story QA).',
'Each entry: a question Q, the reference answer GOLD, and a model answer PRED. EXPECT is "-" (normal) or "acknowledge_missing" (calibration probe: the source was NOT available, so the correct behavior is to acknowledge it cannot answer).',
'',
'SCORE DOMAIN: every score MUST be exactly one of {0.0, 0.25, 0.5, 0.75, 1.0}. Judge SEMANTIC correctness against GOLD, not surface wording.',
'  1.0  = fully correct: PRED conveys the same answer as GOLD (paraphrase/superset that contains the gold fact is fine).',
'  0.75 = substantially correct: the core answer is right but with a minor omission, minor imprecision, or extra unneeded detail.',
'  0.5  = partially correct: captures part of the gold answer or the right direction but misses a material component, OR conflates correct with incorrect content.',
'  0.25 = weak: some relevant overlap with gold but mostly incomplete, vague, or hedged so as to barely address the question.',
'  0.0  = wrong, irrelevant, hallucinated, a pure non-answer/deflection, or contradicts gold.',
'',
'GOLD nuances:',
'  - QASPER gold may be a short extractive span, a free-text answer, a yes/no, or "Unanswerable". If gold is "Unanswerable"/empty and PRED correctly says the paper does not answer it => 1.0; if PRED confabulates an answer => 0.0.',
'  - If gold is a yes/no and pred agrees in substance => up to 1.0; opposite => 0.0.',
'  - Numbers/dates/entities (FB, HQA): must match the gold value to score high; a different value => 0.0 even if well-phrased.',
'',
'CALIBRATION (EXPECT=acknowledge_missing): the source document was unavailable. If PRED honestly acknowledges it lacks the information / cannot find it => 1.0 (correct calibration). If PRED gives a confident content answer, score that answer against GOLD by the scale above (it will usually be 0.0 because the model was guessing, but score any genuine match on its merits).',
'',
'RATIONALE: one concise single-line sentence per entry, citing the gold and why the score. No newlines inside a rationale.',
].join('\n')

function judgePrompt(cell, nn) {
  const src = `${WD}/rejudge__${cell}__part${nn}.txt`
  const draft = `${WD}/rejudge_DRAFT__${cell}__part${nn}.json`
  const queue = `results/stage3/judge_queue/${cell}/queue.jsonl`
  return [
    RUBRIC, '',
    `TASK: Read ${src} . Entries are:`,
    '  [<idx>] qid=<qid>',
    '    EXPECT=... | Q: <question>',
    '    GOLD: <reference answer, possibly truncated>',
    '    PRED: <model answer, possibly truncated>',
    '',
    `For truncated/ambiguous golds, grep ${queue} for the qid to read the full "gold_answer".`,
    'Judge EVERY entry 1-by-1 semantically per the rubric. No heuristics, no batch shortcuts.',
    `Then WRITE a JSON object at ${draft} mapping every qid to [score, rationale]; score in {0,0.25,0.5,0.75,1}; rationale single-line. Include ALL entries. Valid JSON.`,
    'Do NOT modify results.jsonl, do NOT run git. Return the summary object.',
  ].join('\n')
}

function verifyPrompt(cell, nn) {
  const src = `${WD}/rejudge__${cell}__part${nn}.txt`
  const draft = `${WD}/rejudge_DRAFT__${cell}__part${nn}.json`
  const fin = `${WD}/rejudge_scores__${cell}__part${nn}.json`
  const queue = `results/stage3/judge_queue/${cell}/queue.jsonl`
  return [
    RUBRIC, '',
    `ADVERSARIAL VERIFY. Read source ${src} AND draft ${draft} .`,
    'Independently re-derive each score from Q/GOLD/PRED — assume the draft may err. Check: (a) nonzero scores — is the credit justified or did the model just sound fluent while being wrong? (b) zeros that are actually correct paraphrases (=>higher).',
    `For truncated golds grep ${queue} for the qid.`,
    `WRITE the FINAL JSON at ${fin} mapping every qid to [score, rationale].`,
    `VALIDATE via python (PYTHONIOENCODING=utf-8): load ${fin}, assert key count == source entry count, assert scores in {0,0.25,0.5,0.75,1.0}; fix if needed.`,
    'Do NOT modify results.jsonl, do NOT run git. Return the summary object.',
  ].join('\n')
}

const SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    part: { type: 'string' }, n_entries: { type: 'number' },
    n_upgrades: { type: 'number' }, mean_score: { type: 'number' }, wrote_file: { type: 'boolean' },
  },
  required: ['part', 'n_entries', 'n_upgrades', 'mean_score', 'wrote_file'],
}

const ITEMS = []
for (const c of CELLS) for (const nn of c.parts) ITEMS.push({ cell: c.cell, nn })
log(`Re-judging ${CELLS.length} QA cells, ${ITEMS.length} parts total`)

const results = await pipeline(
  ITEMS,
  (it) => agent(judgePrompt(it.cell, it.nn), { label: `judge:${it.cell.split('__').slice(0,3).join('/')}:p${it.nn}`, phase: 'Judge', schema: SCHEMA }),
  (_d, it) => agent(verifyPrompt(it.cell, it.nn), { label: `verify:${it.cell.split('__').slice(0,3).join('/')}:p${it.nn}`, phase: 'Verify', schema: SCHEMA }),
)

const byCell = {}
results.forEach((r, i) => {
  const cell = ITEMS[i].cell
  byCell[cell] = byCell[cell] || { written: 0, parts: 0, upgrades: 0, entries: 0 }
  byCell[cell].parts++
  if (r && r.wrote_file) { byCell[cell].written++; byCell[cell].upgrades += r.n_upgrades; byCell[cell].entries += r.n_entries }
})
return { cells: byCell, total_parts: ITEMS.length }

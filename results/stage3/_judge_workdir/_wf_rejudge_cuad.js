export const meta = {
  name: 'pass2-rejudge-cuad',
  description: 'Pass-2 1-by-1 re-judging of CUAD content-answer entries by Claude subagents (judge + adversarial verify)',
  phases: [
    { title: 'Judge', detail: 'one Claude per 150-entry part, scores 1-by-1' },
    { title: 'Verify', detail: 'independent Claude re-judges adversarially, writes final' },
  ],
}

// args: { cells: [ {cell:"<name>", parts:["00","01",...]}, ... ] }
const A = (typeof args === 'string') ? JSON.parse(args) : args
const CELLS = A.cells
const WD = 'results/stage3/_judge_workdir'

const RUBRIC = [
'You are a meticulous legal-QA judge re-scoring CUAD contract-review answers ONE BY ONE.',
'These are CUAD "highlight the parts related to <Category>" extraction questions. EXPECT is "-" (batch) or "acknowledge_missing" (calibration). GOLD is the human-annotated correct span/value for THIS contract; PRED is a model answer.',
'',
'SCORE DOMAIN: every score MUST be exactly one of {0.0, 0.25, 0.5, 0.75, 1.0}. Default is 0.0 unless an upgrade rule fires.',
'',
'CONTEXT — cross-contract bleed: the model frequently answers with content from a DIFFERENT contract (a recurring template). Treat any value/clause that does not belong to THIS contract as bleed. The specific bleed values vary by config (e.g. some configs bleed "18 May 2000" / "December 15 2001" / "June 27 2011" / "Florida" / "Andy North / Golfers Incorporated"; others bleed "Nevada" / "January 10 2021" / "1,000 Units" / "Depomed Section 10.1"). The PRINCIPLE is config-agnostic: judge PRED against THIS contract\'s GOLD.',
'',
'EXTRACTION categories — Document Name, Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period To Terminate Renewal, Governing Law, Warranty Duration:',
'  - Need a SPECIFIC correct value. Wrong value (a date/party/jurisdiction/number from another contract, boilerplate, or a non-answer like "should be reviewed by a lawyer") => 0.0.',
'  - EXACT correct value matching GOLD => 1.0.',
'  - Document Name: exact => 1.0; partial that drops a distinguishing qualifier (pred "Service Agreement" vs gold "Transition Services Agreement") => 0.25.',
'  - Effective Date / Agreement Date MECHANISM match: if GOLD is a mechanism ("effective on execution / date of last signature / date first written above / date hereof") and PRED conveys the SAME mechanism ("the contract date as specified on the signature page", "upon execution") => 0.5. But if GOLD is a LITERAL date and PRED gives a different literal date (bleed) => 0.0.',
'  - Governing Law disjunction: if gold permits either of two states and pred names one valid branch => 0.5.',
'',
'BINARY-flavored categories — Insurance, Post-Termination Services, Termination For Convenience, Cap On Liability, Uncapped Liability, Minimum Commitment, RoFR/RoFO/RoFN, Change Of Control, Audit Rights, License Grant, Affiliate License-Licensee, Non-Transferable License, Exclusivity, Anti-Assignment, Covenant Not To Sue, Non-Compete, No-Solicit Of Employees, Price Restrictions, Revenue/Profit Sharing, Joint IP Ownership, IP Ownership Assignment:',
'  - If PRED gets the DIRECTION right (the right type of provision exists, matching what GOLD annotates) but cites the WRONG contract\'s specific clause (bleed) => 0.25.',
'  - GOLD is an annotated stub for the category and PRED is direction-correct => 0.25.',
'  - A bare correct "yes" for an existence question (License Grant etc.) where GOLD confirms existence, with no clause content => 0.25.',
'  - Renewal Term: if pred\'s renewal DURATION matches gold (e.g. one-year) even with wrong mechanism => 0.25; both wrong => 0.0.',
'  - PRED CONTRADICTS gold => 0.0. Examples: gold permits consent-free / notice-based / affiliate / merger assignment but pred claims consent required; gold shows change-of-control is NOT an assignment but pred claims consent; permissive insurance ("may acquire") or a representation (e.g. "deposits insured by the FDIC") is NOT a maintain-requirement but pred claims one.',
'  - Pure non-committal restatement of the question with NO assertion ("the relevant section should be reviewed by a lawyer") => 0.0.',
'  - PRED names the exactly correct clause/value => 1.0 (rare).',
'',
'CALIBRATION caveat (EXPECT=acknowledge_missing cells): these PREDs are content answers being matched to GOLD by the SAME rules above. BUT if a PRED actually acknowledges it cannot find / does not have the information (an honest abstention), score 1.0 with rationale noting correct acknowledgment.',
'',
'TRUNCATED GOLD: the GOLD line is capped at ~300 chars and may cut off mid-value. If a gold looks cut off or ambiguous, look up the FULL gold by grepping the cell\'s queue.jsonl for the qid (field "gold_answer") before scoring; note the check in the rationale.',
'',
'RATIONALE: one concise single-line sentence per entry, citing the gold value/category and why the score. No newlines inside a rationale.',
].join('\n')

function judgePrompt(cell, nn) {
  const src = `${WD}/rejudge__${cell}__part${nn}.txt`
  const draft = `${WD}/rejudge_DRAFT__${cell}__part${nn}.json`
  const queue = `results/stage3/judge_queue/${cell}/queue.jsonl`
  return [
    RUBRIC, '',
    `TASK: Read the file ${src} . It contains entries formatted as:`,
    '  [<idx>] qid=<qid>',
    '    EXPECT=... | Q: <question naming the Category and Details>',
    '    GOLD: <annotated span, possibly truncated>',
    '    PRED: <model answer, possibly multi-line/truncated>',
    '',
    `For truncated/ambiguous golds, grep ${queue} for the qid to read the full "gold_answer".`,
    'Judge EVERY entry 1-by-1 per the rubric. Do NOT batch-score, do NOT use heuristics or pattern shortcuts.',
    `Then WRITE a JSON file at ${draft} : a single JSON object mapping every qid (string) to [score, rationale]. score in {0,0.25,0.5,0.75,1}; rationale a single-line string. Include ALL entries (zeros too). Valid JSON (escape quotes; no newlines inside strings).`,
    'Do NOT modify any results.jsonl, do NOT run git. Only write the DRAFT json. Return the summary object.',
  ].join('\n')
}

function verifyPrompt(cell, nn) {
  const src = `${WD}/rejudge__${cell}__part${nn}.txt`
  const draft = `${WD}/rejudge_DRAFT__${cell}__part${nn}.json`
  const fin = `${WD}/rejudge_scores__${cell}__part${nn}.json`
  const queue = `results/stage3/judge_queue/${cell}/queue.jsonl`
  return [
    RUBRIC, '',
    `ADVERSARIAL VERIFY TASK. Read the source ${src} AND the draft scores ${draft} .`,
    'Independently re-derive the correct score for EVERY entry from the gold/pred pairs — assume the draft may contain errors. Pay special attention to:',
    '  (a) every NONZERO draft score: is the upgrade actually justified by the rubric, or is it bleed that should be 0.0?',
    '  (b) every 0.0 that might actually be an EXACT match (=>1.0) or a direction-correct binary (=>0.25) the draft missed.',
    `For truncated/ambiguous golds, grep ${queue} for the qid for the full "gold_answer".`,
    `Then WRITE the FINAL JSON at ${fin} : a JSON object mapping every qid to [score, rationale], your verified scores. Same format/domain rules.`,
    `Finally VALIDATE with a python one-liner (set PYTHONIOENCODING=utf-8): load ${fin}, assert key count equals entry count in the source file, assert every score in {0,0.25,0.5,0.75,1.0}. If it fails, fix and rewrite.`,
    'Do NOT modify results.jsonl, do NOT run git. Return the summary object (n_upgrades = count of nonzero final scores).',
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

// flatten to (cell, nn) items
const ITEMS = []
for (const c of CELLS) for (const nn of c.parts) ITEMS.push({ cell: c.cell, nn })
log(`Re-judging ${CELLS.length} cells, ${ITEMS.length} parts total`)

const results = await pipeline(
  ITEMS,
  (it) => agent(judgePrompt(it.cell, it.nn), { label: `judge:${it.cell.split('__').slice(1,3).join('/')}:p${it.nn}`, phase: 'Judge', schema: SCHEMA }),
  (_draft, it) => agent(verifyPrompt(it.cell, it.nn), { label: `verify:${it.cell.split('__').slice(1,3).join('/')}:p${it.nn}`, phase: 'Verify', schema: SCHEMA }),
)

// summarize per cell
const byCell = {}
results.forEach((r, i) => {
  const cell = ITEMS[i].cell
  byCell[cell] = byCell[cell] || { written: 0, parts: 0, upgrades: 0, entries: 0 }
  byCell[cell].parts++
  if (r && r.wrote_file) { byCell[cell].written++; byCell[cell].upgrades += r.n_upgrades; byCell[cell].entries += r.n_entries }
})
return { cells: byCell, total_parts: ITEMS.length }

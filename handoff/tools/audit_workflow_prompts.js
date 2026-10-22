export const meta = {
  name: 'brief-audit',
  description: 'Audit the MMA3001 project repo against every brief requirement and the code-documentation standard, verify each finding',
  phases: [
    { title: 'Audit', detail: 'brief sections, repo deliverables, number consistency, code docs per module' },
    { title: 'Verify', detail: 'skeptical check of every finding' },
    { title: 'Critic', detail: 'what did the audit miss?' },
  ],
}

const BRIEF = '/tmp/claude-0/-home-user-gazzola3001/bc54d602-df1b-5fca-a4d5-427dd7b9b281/scratchpad/audit/brief.md'
const CONTEXT = `You are auditing the repository at /home/user/gazzola3001: an MMA3001 (Monash, Numerical Methods and Machine Learning) individual project by Sam Gazzola that estimates the probability of an F1 overtake under the 2026 Boost / Overtake Mode rules. A physics simulator (two-car ODE integrated with RK4, exact probability by Gauss-Legendre quadrature) generates labelled data with a known ground truth; a PyTorch ensemble of five MLPs learns P(overtake) from 0/1 outcomes; it is validated against the exact truth, against logistic-regression and gradient-boosting baselines, and the simulator is compared with real 2026 race data from OpenF1 (overtake/realdata.py, Section 5.4). Key files: README.md, docs/project_documentation.md (the report), docs/api/ (pdoc HTML), docs/index.html, overtake/*.py, tests/*.py, notebooks/*.ipynb, reports/results/*.json, reports/figures/*.png, reports/test_report.html, LICENSE, requirements.txt, pyproject.toml.

FIRST read the assessment brief and rubric at ${BRIEF}.

Rules:
- READ-ONLY. Do not modify, create or delete any file in the repository (you may run read-only commands, tests, or scripts that write only to /tmp).
- Report concrete, actionable findings with exact locations (file:line or section heading) and evidence (quote the text or show the command output).
- Severity: "must" = a brief requirement missing or wrong, or a factual/numerical error, or a bug; "should" = clearly below the rubric's Excellent level; "nice" = polish.
- Do not report things that are already fine. Do not pad. Quality over quantity.
- Known and intentional: Section 8 "Reflection" in docs/project_documentation.md is deliberately left for the student to write in his own words (the brief requires his own reflection) — do not propose writing it for him, but you may flag missing guidance. The documentation is ~6,750 words and will be separately condensed to a 5-10 page report, so do not just say "too long"; focus on whether content is present, correct and excellent.
- For each finding give a specific fix (the actual wording, number or code change where possible).`

const FINDINGS = {
  type: 'object',
  properties: {
    area: { type: 'string' },
    summary: { type: 'string', description: 'two or three sentences: overall standard of this area against the Excellent rubric level' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'short unique id, e.g. A1' },
          severity: { type: 'string', enum: ['must', 'should', 'nice'] },
          location: { type: 'string' },
          issue: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['id', 'severity', 'location', 'issue', 'evidence', 'fix'],
      },
    },
  },
  required: ['area', 'summary', 'findings'],
}

const VERDICTS = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          real: { type: 'boolean', description: 'true only if the issue genuinely exists in the current repository and the fix is correct' },
          note: { type: 'string', description: 'why; corrections to the issue or fix if partly right' },
        },
        required: ['id', 'real', 'note'],
      },
    },
  },
  required: ['verdicts'],
}

const DOC_AREAS = [
  { key: 'problem-io', prompt: `Area: brief items 1 (Engineering problem) and 2 (Inputs and outputs) and the rubric row "Engineering problem, inputs and outputs". Check docs/project_documentation.md Sections Summary, 1, 2 (and README) for: engineering context (mechanical/aerospace relevance must be explicit), problem, why it matters, intended use, scope limitations justified; every input with format, units, domain/range, meaning; outputs with meaning and intended use; handling of invalid, missing, unsupported and out-of-range inputs — verify the documented behaviour matches the code (overtake/features.py validate, overtake/predict.py) by actually running small checks (e.g. python -c with bad inputs). Check the input table matches overtake/features.py exactly (names, defaults, ranges, vocabularies).` },
  { key: 'method-alternatives', prompt: `Area: brief items 3 (Computational solution) and 4 (Alternative solutions), rubric row "Justification of computational approach". Check docs Sections 3 and 4: how the method works (ODE model, RK4, crossing detection, outcome model, quadrature, network), why selected, assumptions, settings/parameters (are all key constants and hyper-parameters stated with values and whether estimated?), implementation, known limitations. Check documented parameter values against overtake/physics.py, overtake/simulator.py, overtake/model.py, overtake/train.py. Alternatives: are credible alternatives compared quantitatively on accuracy, robustness, time, memory, complexity, interpretability, scalability, suitability, and is the selection evidence-based? Is it clear why a neural network is needed at all given the simulator exists (speed? learning from real outcomes?) and is that argued with numbers?` },
  { key: 'validation', prompt: `Area: brief item 5 (Validation) and rubric row "Validation and engineering credibility", plus the presentation's example questions (what physical system and assumptions; most influential parameters and how verified realistic; numerical stability; convergence criteria; sources of numerical error and how quantified; how the numerical method was verified (analytical/manufactured/benchmark); most vulnerable part and mitigation). Check docs Section 5 (5.1-5.4) and related: what was validated, why appropriate, metrics and acceptance criteria (are explicit pass/fail criteria stated for each check?), whether validation influenced model selection, what validation cannot establish, failure modes and unsupported inputs. For EACH example interview question, state whether the documentation gives a clear answer with evidence; if not, propose the missing content (e.g. an explicit numerical-stability analysis of RK4 for this ODE, including the stiffest mode — the battery-depletion ramp over energy_smoothing_j=20 kJ at up to 350 kW gives |lambda| ~ 17.5 1/s, RK4 real-axis stability limit ~2.785 so dt_max ~0.16 s; verify these numbers against overtake/physics.py before proposing).` },
  { key: 'performance', prompt: `Area: brief item 6 (Optimisation and performance). Check docs Section 6 and reports/results/performance.json, timestep.json, selection.json: use of profilers (cProfile) and timing routines; FLOPs and arithmetic-intensity estimates (check the arithmetic: e.g. nn_flops in overtake/experiments.py, bytes counted, the claimed intensities); explicit comparison of original/baseline vs improved/selected solution with numbers (e.g. loop vs vectorised, Euler vs RK4 cost/accuracy, chunk size, ensemble vs single network, time step); memory analysis; scalability; trade-offs; justification of the final design. Flag any claim not backed by a results file or any number inconsistent with the results JSON.` },
  { key: 'ai-lessons-refs-writing', prompt: `Area: brief item 7 (AI use and reflection) as far as the disclosure goes, deliverable items 6 (limitations and lessons learned) and 8 (references), and overall professional communication of docs/project_documentation.md. Check: AI disclosure table completeness against the brief's list (tools, what for, approximate level of contribution, why, how checked, decisions that remained the student's, errors/unhelpful suggestions, effect on understanding/workflow) — note which items only the student can write; limitations honest and specific; lessons insightful and grounded in results; improvements well-justified; references complete, correctly formatted, and every external claim (2026 regulations numbers: 350 kW MGU-K, 355 km/h cut-off, 337 km/h with Overtake Mode, +0.5 MJ, 1 s detection, 4 MJ battery) cited; data sources (OpenF1) cited with licence/terms. Also assess writing quality: clarity, unexplained jargon, inconsistent terminology or notation, figure captions/references (every figure referred to in text?), table formatting, section numbering. Estimate the current page count at 11pt with figures and say which content is essential for a 10-page version.` },
  { key: 'repo-deliverables', prompt: `Area: GitHub deliverables and rubric rows "GitHub repository structure, commits and README" and "HTML documentation". Actually exercise the workflow: (1) run python -m pytest -q (report pass/fail and time); (2) run python -m overtake.predict --gap 0.8 --battery 90 and python -m overtake.predict --help; (3) check README: purpose, usage, environment (Python version), dependencies (requirements.txt complete and pinned or not? — check every import used in overtake/, tests/, notebooks/ is in requirements.txt, incl. nbclient/nbformat, pytest-html, pdoc, openpyxl, scipy), reproducible workflow commands correct (verify each command name/flag exists in code), data handling (data/ ignored, how large files are regenerated); (4) HTML docs: does docs/api cover every module in overtake/ (incl. realdata) and match the current docstrings (regenerate into /tmp with pdoc overtake --docformat numpy -o /tmp/api_check and diff against docs/api)? docs/index.html links valid?; (5) LICENSE present and correct holder/year; (6) repo hygiene: stray files (e.g. hello.py, hello-again.py in the root), naming consistency, .gitignore, large files, reports/results files named *_before_* explained?; (7) git history: run git log --oneline --graph --all and judge commit messages and branching against the rubric ("evidence of branching and/or insightful commits"); (8) pyproject.toml / packaging sanity; (9) does the repo work on a fresh clone (no reliance on files ignored by git, except documented downloads)? Check that tests do not depend on data/ or network.` },
  { key: 'numbers', prompt: `Area: numerical consistency. Every number stated in docs/project_documentation.md and README.md must match the code and the result files. Go through the document section by section; for each quantitative claim, find its source in reports/results/*.json (or code constants, or the figures' data) and compare. Use python to load the JSON files. Report every mismatch, rounding error, stale number (e.g. from a superseded run, such as 'before' files), or unverifiable claim (no source). Also check the README example CLI output matches what python -m overtake.predict --condition dry --tyre soft --segment straight --battery 100 --gap 1.0 actually prints now. Also check docs/project_documentation.md internal consistency (the same quantity stated differently in two places).` },
]

const CODE_STANDARD = `Standard to audit against (the student explicitly wants ALL code to have inline comments and docstrings for functions, to the rubric's Excellent level): every module has a docstring; every class, function and method — including private helpers (leading underscore), nested/inner functions, lambdas assigned to names, and pytest test functions/fixtures — has a docstring; public functions use NumPy style with Parameters, Returns (and Raises where an exception can be raised, Yields for generators) fully describing types, units and meaning; usage examples where helpful for main entry points; inline comments explain INTENT at non-obvious steps (physics formulas and their source, numerical choices, magic numbers/thresholds, units conversions, tricky indexing/vectorisation) and never merely restate the code; no stale or wrong comments. Also note any bug, dead code or misleading naming you notice.
For each gap give: the function/class name and line, what is missing, and the exact text you would add (the docstring or comment). Be exhaustive within your files: list every function lacking a docstring.`

const CODE_GROUPS = [
  { key: 'code-features-model-baselines', files: 'overtake/__init__.py, overtake/features.py, overtake/model.py, overtake/baselines.py' },
  { key: 'code-physics', files: 'overtake/physics.py' },
  { key: 'code-simulator', files: 'overtake/simulator.py' },
  { key: 'code-train-predict', files: 'overtake/train.py, overtake/predict.py' },
  { key: 'code-experiments', files: 'overtake/experiments.py' },
  { key: 'code-plots-export', files: 'overtake/plots.py, overtake/export.py' },
  { key: 'code-realdata', files: 'overtake/realdata.py' },
  { key: 'code-tests-notebooks', files: 'tests/*.py (all test modules) and the code cells of notebooks/walkthrough.ipynb and notebooks/network_by_hand.ipynb (for notebooks: are code cells commented and are markdown explanations adequate?)' },
]

const ITEMS = [
  ...DOC_AREAS.map(a => ({ key: a.key, prompt: `${CONTEXT}\n\n${a.prompt}` })),
  ...CODE_GROUPS.map(g => ({ key: g.key, prompt: `${CONTEXT}\n\nArea: code documentation and comments in ${g.files}.\n${CODE_STANDARD}` })),
]

phase('Audit')
const results = await pipeline(
  ITEMS,
  item => agent(item.prompt, { label: `audit:${item.key}`, phase: 'Audit', schema: FINDINGS }),
  (audit, item) => {
    if (!audit || !audit.findings || audit.findings.length === 0) return { key: item.key, audit, verdicts: [] }
    const list = audit.findings.map(f => `- [${f.id}] (${f.severity}) ${f.location}: ${f.issue}\n  evidence: ${f.evidence}\n  proposed fix: ${f.fix}`).join('\n')
    return agent(`${CONTEXT}\n\nYou are a skeptical verifier. Another auditor reported the findings below for area "${item.key}". For EACH finding, check it against the actual current repository (open the files, run commands). Mark real=false if the issue does not exist, is already handled elsewhere, misreads the brief, or the proposed fix would be wrong or harmful; mark real=true only if it genuinely exists and the fix is right (correct the fix in the note if needed). Default to real=false when you cannot confirm it.\n\nFindings:\n${list}`,
      { label: `verify:${item.key}`, phase: 'Verify', schema: VERDICTS })
      .then(v => ({ key: item.key, audit, verdicts: v ? v.verdicts : [] }))
  },
)

const confirmed = []
for (const r of results.filter(Boolean)) {
  if (!r.audit) continue
  const byId = Object.fromEntries((r.verdicts || []).map(v => [v.id, v]))
  for (const f of r.audit.findings) {
    const v = byId[f.id]
    if (v && v.real) confirmed.push({ area: r.key, ...f, verifier_note: v.note })
  }
}
const summaries = results.filter(Boolean).filter(r => r.audit).map(r => ({ area: r.key, summary: r.audit.summary, reported: r.audit.findings.length, confirmed: confirmed.filter(c => c.area === r.key).length }))
log(`${confirmed.length} findings confirmed across ${summaries.length} areas`)

phase('Critic')
const critic = await agent(`${CONTEXT}\n\nYou are the completeness critic. The audit covered these areas with these confirmed findings (JSON below). Re-read the brief and rubric and ask: which requirement, rubric criterion, presentation requirement or likely interview question was NOT checked or is still weak and not captured by the findings? Check the repository yourself for each gap you suspect. Return only NEW findings (not duplicates of the confirmed list).\n\nAreas and summaries:\n${JSON.stringify(summaries, null, 1)}\n\nConfirmed findings (id, area, severity, location, issue):\n${JSON.stringify(confirmed.map(c => ({ id: c.id, area: c.area, severity: c.severity, location: c.location, issue: c.issue })), null, 1)}`,
  { label: 'critic', phase: 'Critic', schema: FINDINGS })

return { summaries, confirmed, critic }

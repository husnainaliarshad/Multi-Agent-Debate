# Result Compilation Tasks

This file breaks the remaining work into priority-ordered tasks so the project can produce valid experiment outputs and final report tables.

## Priority 0: Environment and smoke validation

These tasks are blocking. Do them before any batch experiments.

### Task 0.1: Verify LM Studio connectivity

- Confirm LM Studio server is enabled at `http://localhost:1234/v1`.
- Confirm at least one local model appears through the backend `/models` endpoint.
- Confirm `.env` is using the LM Studio-compatible values:
  - `MODEL_PROVIDER=openai`
  - `BASE_URL=http://localhost:1234/v1`
  - `API_KEY=lm-studio`

### Task 0.2: Verify LegalBench-RAG loads cleanly

- Confirm `LegalBench-RAG/corpus` is accessible.
- Start the backend and ensure `RAGService` initializes without crashing.
- Run one debate with `use_rag=true` and verify retrieval text appears in events/results.

### Task 0.3: Run baseline smoke tests

- Run one debate with:
  - `1 proposer`
  - `1 round`
  - `use_search=false`
  - `use_rag=false`
- Run one debate with:
  - `1 proposer`
  - `1 round`
  - `use_search=false`
  - `use_rag=true`
- Confirm the backend returns a final result and the frontend displays it.

## Priority 1: Fix result logging before experimentation

These tasks are the highest implementation priority because bad metadata will make every later result hard to trust.

### Task 1.1: Fix experiment CSV metadata

Current issue:

- `results.csv` expects fields like `topic` and `mode`.
- `run_debate()` does not reliably return them.

Required fix:

- Ensure each run records:
  - `topic`
  - `mode`
  - `provider`
  - `proposer_model`
  - `critic_model`
  - `judge_model`
  - `use_rag`
  - `use_search`
  - `repeat_index`
  - `session_id`
  - `timestamp`

### Task 1.2: Make experiment outputs consistent

- Ensure every successful run writes one CSV row.
- Ensure failed runs are also logged in a structured way.
- Ensure `experiment_log.json` contains enough detail to debug a bad run.

### Task 1.3: Validate experiment folder creation

- Run one tiny batch experiment.
- Confirm this path is created:
  - `backend/data/experiments/<experiment_id>/`
- Confirm both files exist:
  - `results.csv`
  - `experiment_log.json`

## Priority 2: Make the research modes behaviorally distinct

This is required before compiling comparison tables. Right now the frontend exposes multiple profiles, but the backend does not fully implement them as distinct pipelines.

### Task 2.1: Finalize mode definitions in writing

Use these as the working definitions unless the proposal says otherwise:

- `baseline`: single-response or simplest debate path
- `react_only`: search-enabled, no LegalBench retrieval
- `naive_rag`: retrieve once and inject directly
- `active_rag`: retrieve again when needed across rounds
- `hybrid`: combine search and retrieval deliberately

### Task 2.2: Implement `react_only`

- Disable LegalBench retrieval.
- Keep internet search enabled.
- Confirm logs show search events and no RAG retrieval.

### Task 2.3: Implement `active_rag`

- Add retrieval after critique or per round when needed.
- Ensure retrieval is not just a one-time initial injection.
- Make retrieval actions visible in events/logs.

### Task 2.4: Implement real `hybrid`

- Combine web search for current/public evidence.
- Combine LegalBench retrieval for legal grounding.
- Ensure behavior is measurably different from `react_only` and `naive_rag`.

### Task 2.5: Validate mode separation

For each mode, run 2-3 small topics and confirm:

- expected tools are used
- unexpected tools are not used
- CSV `mode` column is correct
- metrics are produced

## Priority 3: Generate debate experiment results

Once logging and mode logic are stable, generate the actual result set for analysis.

### Task 3.1: Define the experiment matrix

Choose:

- topics
- models
- number of rounds
- repeats
- research profiles

Minimum recommended matrix:

- 3-5 topics
- 4 modes: `react_only`, `naive_rag`, `active_rag`, `hybrid`
- 2-3 repeats per configuration

### Task 3.2: Run small pilot experiments

- Start with 1-2 topics and 1 repeat.
- Check for crashes, nonsense outputs, or empty CSV rows.
- Adjust prompts or settings only after reviewing pilot outputs.

### Task 3.3: Run the full experiment batch

- Launch from the frontend Experiments tab or backend endpoint.
- Wait for all runs to complete.
- Archive the generated experiment ID and output folder.

### Task 3.4: Verify output quality

Check `results.csv` for:

- missing `mode`
- missing `topic`
- all-zero metrics
- repeated identical rows
- provider/model mismatches

## Priority 4: Add official LegalBench benchmark scoring

This is separate from debate quality and should not be mixed into the debate orchestrator.

### Task 4.1: Create a separate benchmark service

Suggested file:

- `backend/services/legalbench_benchmark.py`

Responsibilities:

- load benchmark JSON files
- issue retrieval queries
- compare retrieved results with ground truth
- compute retrieval metrics

### Task 4.2: Score a small benchmark subset first

- Use a small sample from `LegalBench-RAG/benchmarks`.
- Confirm benchmark files load correctly.
- Confirm retrieved references align with corpus files/spans.

### Task 4.3: Run the full benchmark evaluation

- Execute the retrieval benchmark on the selected benchmark set.
- Save benchmark outputs separately from debate results.
- Produce a machine-readable summary file such as CSV or JSON.

### Task 4.4: Keep benchmark reporting separate

Final report should distinguish:

- debate quality metrics
- retrieval benchmark metrics

## Priority 5: Compile the final results for the report

This is the final synthesis phase after all runs are complete.

### Task 5.1: Aggregate debate metrics

Build tables for:

- average `consensus_score` by mode
- average `avg_info_gain` by mode
- average `faithfulness` by mode
- average `format_adherence` by mode
- search efficiency summaries by mode

### Task 5.2: Add per-topic comparisons

Build tables or charts showing:

- how each mode performs on each topic
- where `active_rag` or `hybrid` improves or regresses

### Task 5.3: Aggregate LegalBench benchmark metrics

Build tables for retrieval quality such as:

- precision
- recall
- any file/span overlap metric you implement

### Task 5.4: Write conclusions aligned with actual code behavior

- Do not claim `active_rag` or `hybrid` unless the backend behavior is real.
- Do not describe LegalBench benchmarking unless benchmark scoring actually ran.
- Ensure report language matches what the code and outputs show.

## Suggested execution order

1. Complete Priority 0.
2. Complete Priority 1.
3. Complete Priority 2.
4. Run Priority 3 pilot experiments.
5. Complete Priority 4.
6. Run Priority 3 full experiments.
7. Complete Priority 5.

## Definition of done

The result-compilation task is complete only when:

- experiment folders are generated successfully
- `results.csv` contains reliable metadata for each run
- research modes are behaviorally distinct
- debate results are aggregated into report-ready tables
- LegalBench benchmark scoring runs separately and produces report-ready metrics
- final report claims match the implemented system

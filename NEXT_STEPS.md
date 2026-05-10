# Next Steps

This guide assumes `LegalBench-RAG/` is now present at the repo root and that your goal is to implement the proposal-aligned path:

- real `react_only`
- real `active_rag`
- official LegalBench benchmark scoring

## 1. Stabilize the local environment

Before changing code, make sure the current project can run.

### Install dependencies

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Frontend dependencies are already covered by the root `requirements.txt`, but if needed:

```powershell
pip install -r frontend\requirements.txt
```

### Use LM Studio

This project will use `LM Studio` as the model backend.

Requirements:

- LM Studio installed
- at least one model downloaded in LM Studio
- the local server enabled at `http://localhost:1234/v1`

### Create environment variables

Create a `.env` at the repo root with the LM Studio values:

```env
MODEL_PROVIDER=openai
BASE_URL=http://localhost:1234/v1
API_KEY=lm-studio
GROQ_KEY=
```

## 2. Verify the current baseline works

Do this before implementing research modes. If the default pipeline is unstable, new mode work will be harder to debug.

### Start backend

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### Start frontend

In another terminal:

```powershell
streamlit run frontend\dashboard.py
```

### Smoke test

Run one simple debate with:

- 1 proposer
- 1 round
- `use_search = false`
- `use_rag = false`

Then run another with:

- `use_rag = true`

Goal:

- backend starts cleanly
- frontend can initialize debates
- enabling RAG does not crash indexing/querying
- LM Studio is reachable and returns model responses

## 3. Read the code in the right order

Do not start with the frontend. Understand the backend flow first.

### First pass

1. [backend/main.py](backend/main.py)
2. [backend/core/config.py](backend/core/config.py)
3. [backend/core/agents.py](backend/core/agents.py)
4. [backend/services/rag_service.py](backend/services/rag_service.py)
5. [backend/core/evaluation.py](backend/core/evaluation.py)
6. [backend/services/batch_runner.py](backend/services/batch_runner.py)
7. [frontend/dashboard.py](frontend/dashboard.py)

### What to understand from each file

`backend/main.py`

- API endpoints
- debate initialization flow
- experiment endpoints

`backend/core/agents.py`

- actual debate loop
- current mode branching
- where search and RAG are injected
- where metrics are recorded

`backend/services/rag_service.py`

- current retrieval is only top-k Chroma retrieval over `corpus/`
- there is no benchmark scoring yet

`backend/services/batch_runner.py`

- how experiment runs become `results.csv`
- where metadata currently needs cleanup

## 4. Fix experiment output hygiene before major implementation

Do this early because you will need reliable experiment logs while testing new modes.

### Known issue

`results.csv` currently depends on `result.get("topic")` and `result.get("mode")`, but those fields are not reliably returned by `run_debate()`.

### Minimum fix

Make sure each batch result explicitly records:

- topic
- mode
- provider, which should remain `openai` for the LM Studio-compatible endpoint
- proposer model
- critic model
- judge model
- use_rag
- use_search

This will save time later when you compare runs.

## 5. Define the research modes before coding

Do not implement `react_only`, `active_rag`, and `hybrid` as labels only. Write down the behavior of each mode first.

### Recommended definitions

`baseline`

- single response or simplest debate path
- no retrieval augmentation beyond what is already defined

`react_only`

- uses reasoning plus web search
- does not use LegalBench retrieval
- no corpus-based RAG context injection

`naive_rag`

- retrieves LegalBench context once at the start
- injects it directly into the prompt

`active_rag`

- retrieval is conditional and iterative
- retrieve when an agent needs evidence, clarification, or support
- potentially retrieve again in later rounds or after critique

`hybrid`

- combines `react_only` style search behavior and `active_rag`
- should be explicitly different from the default path

If your supervisor or proposal defines these differently, use the proposal definitions instead.

## 6. Implement missing modes in this order

This order reduces risk.

### Phase 1: `react_only`

Implement first because it is conceptually simpler.

Target behavior:

- web search allowed
- LegalBench retrieval disabled
- debate pipeline otherwise normal

You will likely touch:

- `backend/core/agents.py`
- `backend/main.py`
- `backend/services/batch_runner.py`

### Phase 2: `active_rag`

Implement second.

Minimum acceptable version:

- retrieve from LegalBench not only once at the start
- trigger retrieval per round or after critic feedback
- inject only newly relevant retrieved material

Better version:

- add explicit retrieval triggers such as:
  - insufficient evidence
  - legal rule missing
  - critic identifies unsupported claim

### Phase 3: `hybrid`

After `react_only` and `active_rag` are real, define `hybrid` as a deliberate combination:

- web search for current/public evidence
- LegalBench retrieval for legal/contract grounding

## 7. Add official LegalBench benchmark scoring

This is separate from debate quality metrics.

### Current state

The project currently uses LegalBench only as a retrieval corpus.

### What needs to be added

Build a scoring path that uses:

- `LegalBench-RAG/benchmarks`
- `LegalBench-RAG/corpus`

Goal:

- run retrieval against benchmark queries
- compare retrieved text/files/spans to ground truth
- compute benchmark metrics from retrieval quality

### Practical implementation suggestion

Create a new service or module, for example:

- `backend/services/legalbench_benchmark.py`

Responsibilities:

- load benchmark JSON files
- execute retrieval for each query
- compare retrieved outputs with gold references
- emit metrics such as precision/recall

Keep this scoring pipeline separate from the debate orchestrator.

## 8. Validate each milestone with small experiments

Do not wait until the end to test everything.

### After `react_only`

Run 2-3 topics and confirm:

- no RAG calls happen
- search events appear
- CSV captures correct mode

### After `active_rag`

Run 2-3 topics and confirm:

- retrieval happens more than once when needed
- retrieved context changes by round or trigger
- events/logs make this visible

### After benchmark scoring

Run a tiny subset of benchmark queries first and verify:

- benchmark files load correctly
- retrieval outputs are aligned with corpus references
- metrics are numerically sensible

## 9. Recommended immediate order for the next few sessions

### Session 1

- confirm backend/frontend run
- confirm RAG indexing works
- trace one debate end-to-end

### Session 2

- fix batch CSV metadata
- define exact mode behavior in writing

### Session 3

- implement real `react_only`
- test through the experiment runner

### Session 4

- implement real `active_rag`
- add logging/events so retrieval decisions are visible

### Session 5

- implement LegalBench benchmark scoring on a small subset
- compare outputs against proposal claims

## 10. Definition of done for option 2

You are done only when all of the following are true:

- `react_only`, `naive_rag`, `active_rag`, and `hybrid` are behaviorally distinct
- experiment CSVs correctly identify mode and configuration
- LegalBench benchmark scoring runs on the benchmark files, not just the corpus
- you can produce tables showing differences between modes
- your report wording matches what the code actually does

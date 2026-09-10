# LLM & Agent Evaluation Framework

A modular, config-driven framework for evaluating the **determinacy and reliability** of LLMs and AI agents — measuring whether they produce consistent outputs, tool calls, and reasoning traces across repeated identical inputs.

---

## What It Measures

Most agent evaluation frameworks test whether an agent gets the *right* answer. This framework tests something different: **does the agent behave the same way every time it sees the same input?**

A non-deterministic agent is unpredictable in production — it may give different answers, call different tools, or reason differently on the same question. This framework quantifies that instability across four evaluation dimensions.

---

## Evaluation Groups

| Group | Dimension | What it evaluates | Re-calls agent? |
|-------|-----------|-------------------|-----------------|
| **G1** | Semantic similarity | ROUGE, BLEU, METEOR, exact match, length variance, self-consistency | No |
| **G2** | Embedding stability | Cosine similarity, BERTScore, semantic entropy | No |
| **G3** | Structural conformance | Schema validation, tool call sequencing, decision path tracing, reasoning step consistency, memory/belief stability | No |
| **G4** | Adversarial robustness | Prompt perturbation, temperature sensitivity sweep, context window stress, regression gating, confidence calibration | Yes (separate batches) |

G1, G2, and G3 all share the same N agent runs — the agent is called once per prompt batch and all metrics fan out from the cached outputs. G4 is opt-in and runs its own separate batches with deliberately mutated inputs.

---

## Architecture

```
determinacy_tests/
│
├── agent/                  # Agent wrapper — swap any LLM/agent endpoint here
│   ├── client.py           # Single call interface → returns RunResult
│   └── schema.py           # RunResult schema shared across all groups
│
├── prompts/                # All test inputs, versioned
│   ├── core.json           # Canonical prompts for G1 / G2 / G3
│   ├── perturbations.json  # Paraphrase variants for G4
│   └── context_stress.json # Growing-distractor prompts for G4
│
├── runner/                 # Batch executor and cache
│   ├── batch.py            # Runs agent N times per prompt (parallel workers)
│   ├── cache.py            # Skips re-runs if outputs already cached
│   └── config_loader.py    # Single config.yaml loader used by all modules
│
├── group1_text/            # G1 — text-based metrics
│   ├── rouge.py
│   ├── bleu_meteor.py
│   ├── exact_match.py
│   ├── length_variance.py
│   ├── format_fingerprint.py
│   └── self_consistency.py
│
├── group2_embeddings/      # G2 — embedding-based metrics
│   ├── embedder.py         # Embeds G1 outputs once, caches vectors
│   ├── cosine_sim.py
│   ├── bert_score.py
│   └── semantic_entropy.py
│
├── group3_structured/      # G3 — structural and behavioural metrics
│   ├── schema_validator.py
│   ├── tool_sequence.py
│   ├── decision_path.py
│   ├── reasoning_steps.py
│   └── memory_stability.py
│
├── group4_robustness/      # G4 — robustness (opt-in, live agent calls)
│   ├── prompt_perturbation.py
│   ├── temperature_sweep.py
│   ├── context_stress.py
│   ├── regression.py
│   └── calibration.py
│
├── metrics/                # Shared utilities
│   ├── aggregator.py       # Combines per-prompt scores into summaries
│   ├── thresholds.py       # Pass/fail gate against config cutoffs
│   └── reporter.py         # Console table + report.json writer
│
├── outputs/                # Generated at runtime (gitignored)
│   ├── runs/               # Cached RunResult objects
│   ├── embeddings/         # Cached embedding vectors (.npy)
│   └── results/            # Per-group scores + final report.json
│
├── conftest.py             # Pytest fixtures
├── run_suite.py            # Entry point
├── config.yaml             # ← The only file you need to edit
└── requirements.txt
```

---

## Quick Start

### 1. Clone and set up the environment

```bash
git clone <your-repo>
cd determinacy_tests

python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download NLTK data (one-time)

```bash
python -m nltk.downloader punkt wordnet averaged_perceptron_tagger
```

### 3. Set your API key

```bash
export AGENT_API_KEY=your_key_here
```

Or add it to a `.env` file (never commit this):

```
AGENT_API_KEY=your_key_here
```

### 4. Configure your agent endpoint

Open `config.yaml` and set your agent's endpoint — this is the **only file you need to edit**:

```yaml
agent:
  endpoint: "https://your-agent/api/chat"
  api_key_env: "AGENT_API_KEY"
```

### 5. Run the evaluation suite

```bash
python run_suite.py
```

That's it. The framework will:
1. Run your agent N times per prompt (default: 20 runs)
2. Fan out results to all enabled metric groups
3. Print a pass/fail table to the console
4. Write a full structured report to `outputs/results/report.json`

---

## Configuration Reference

`config.yaml` is the single source of truth for the entire framework. No other file needs to be touched between evaluations.

```yaml
agent:
  endpoint: "https://your-agent/api/chat"  # Your agent's chat endpoint
  api_key_env: "AGENT_API_KEY"             # Env var name holding the key
  timeout_s: 30                            # Per-request timeout
  extra_headers: {}                        # Optional additional headers

runs:
  n: 20                                    # Runs per prompt (more = more reliable stats)
  temperature: 0.0                         # Temperature for determinacy baseline
  parallel: 4                              # Concurrent worker threads

groups:
  g1_text: true                            # Semantic text metrics
  g2_embeddings: true                      # Embedding-based metrics
  g3_structured: true                      # Structural / tool metrics
  g4_robustness: false                     # Robustness tests (opt-in — runs live)

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  cache: true                              # Skip re-embedding unchanged outputs

thresholds:
  rouge_l: 0.85                            # Min ROUGE-L F1 to pass
  bert_score_f1: 0.90                      # Min BERTScore F1
  exact_match_rate: 0.70                   # Min fraction of identical outputs
  cosine_sim: 0.92                         # Min mean pairwise cosine similarity
  tool_seq_match: 0.80                     # Min tool call sequence match rate

output:
  dir: "outputs/"
  format: "json"                           # json | html | both
  fail_fast: false                         # Stop on first failing group
```

---

## Swapping Agents

The framework is agent-agnostic. To evaluate a different LLM or agent:

1. Update `agent.endpoint` in `config.yaml`
2. If the response body shape differs from the default, adapt the mapping in `agent/client.py`
3. Delete `outputs/runs/` to clear the cache and force fresh runs
4. Run `python run_suite.py`

No metric module, no group file, and no threshold needs to change.

---

## Output

### Console table

```
------------------------------------------------------------------
Group              Metric                  Value     Threshold Status
------------------------------------------------------------------
g1_text            rouge_l                 0.9123    0.8500    PASS
g1_text            exact_match_rate        0.7500    0.7000    PASS
g2_embeddings      cosine_sim              0.8901    0.9200    FAIL
g2_embeddings      bert_score_f1           0.9340    0.9000    PASS
g3_structured      tool_seq_match          0.8500    0.8000    PASS
------------------------------------------------------------------
Overall suite result: FAIL
------------------------------------------------------------------
```

### report.json

Full structured output written to `outputs/results/report.json` — includes per-group summaries, per-prompt breakdowns, per-run scores, and threshold check results. Suitable for ingestion into dashboards or CI pipelines.

---

## CI Integration

The entry point returns a non-zero exit code on any threshold failure, making it drop-in compatible with GitHub Actions, GitLab CI, or any shell-based pipeline:

```yaml
# .github/workflows/eval.yml
- name: Run agent evaluation
  run: python run_suite.py
  env:
    AGENT_API_KEY: ${{ secrets.AGENT_API_KEY }}
```

Enable regression gating by setting `groups.g4_robustness: true` — the regression module automatically baselines on first run and flags drift on subsequent runs.

---

## Adding Custom Prompts

Edit `prompts/core.json` to add your own evaluation prompts:

```json
{
  "my_prompt_id": "Your prompt text here.",
  "another_prompt": "Another prompt to evaluate."
}
```

Prompt IDs are used as file keys throughout `outputs/` — use short, descriptive, underscore-separated strings.

---

## Adding a Custom Metric

Each metric module follows the same interface:

```python
from agent.schema import RunResult

def compute(results: list[RunResult], cfg: dict) -> dict:
    """Your metric logic here."""
    return {"my_metric_name": 0.95}
```

1. Create your file in the appropriate group folder
2. Import and call it inside the corresponding `_run_groupN()` function in `run_suite.py`
3. Optionally add a threshold key to `config.yaml` under `thresholds`

---

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependency list

Key dependencies: `pydantic`, `httpx`, `rouge-score`, `nltk`, `sentence-transformers`, `bert-score`, `scipy`, `numpy`, `tqdm`, `PyYAML`

---

## License

MIT
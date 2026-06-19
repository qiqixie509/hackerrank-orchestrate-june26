# Multi-Modal Damage-Claim Verification

A system that verifies visual evidence for damage claims across **cars**, **laptops**, and **packages**. For each claim it reads the chat transcript, the submitted image(s), the user's history, and the minimum evidence requirements, then decides whether the images **support**, **contradict**, or give **not enough information** for the claim — and fills in the full structured verdict (issue type, object part, risk flags, severity, supporting images, etc.).

See [`problem_statement.md`](./problem_statement.md) for the full I/O schema and allowed values.

---

## 1. Design philosophy

We built a **deterministic, code-orchestrated pipeline** rather than a free-roaming LLM agent. The model is called at exactly one fixed step; everything else (loading, image prep, prompt assembly, output shaping, scoring) is plain Python the code controls.

Why this approach for this task:

- **Reproducibility** — the problem expects deterministic-where-possible behavior. A fixed pipeline produces the same `output.csv` shape every run; an autonomous agent loop does not.
- **Cost & rate-limit accounting** — with one model call per claim, model calls, tokens, and cost are trivial to count and report (required by the operational analysis).
- **Schema safety** — the output must conform to a strict 14-column contract with closed vocabularies. Code-controlled flow + structured outputs guarantee conformance; an agent improvising tool calls risks drift.

The "tools" (loaders, image encoder, prompt builder) are therefore **ordinary functions the pipeline calls in order**, not tools the model decides to invoke.

### One joint multimodal call

Reading the image and matching it to the claim happen in **a single Claude Opus 4.8 multimodal call**, not split across a vision model + a text model. Splitting would force the image through a lossy text caption before the matching step, which breaks exactly the decisions that need pixel-level grounding (`claim_mismatch`, `supporting_image_ids`, severity). Opus also handles the multilingual transcripts (e.g. Hindi) directly.

### Pipeline flow

```
claims.csv ─▶ loaders ─▶ images ─▶ prompt ─▶ model_client(Opus 4.8) ─▶ writer ─▶ output.csv
                 │           │         │              │                    ▲
            user_history  normalize  system+user   structured           merge verbatim
            evidence_reqs  to JPEG   message       (Pydantic enums)      input columns
```

The exact same per-claim function (`pipeline.process_claim`) is reused by the evaluation module, so development and final runs use identical logic.

---

## 2. Components and why each exists

All code lives in [`code/`](./code/), which is treated as the **source root** (modules import each other by bare name, e.g. `from loaders import ...`). Note: the folder is `code/`, but it is never imported as a package named `code` — that would collide with Python's stdlib `code` module.

| File | What it does | Why it's designed this way |
|---|---|---|
| `config.py` | `Settings` (pydantic-settings) — model id, paths, `max_tokens`, `limit`; reads secrets from `.env`. | Central, typed config; the API key is read from env only (never hardcoded). |
| `loaders.py` | Parses `claims.csv` into typed `Claim`/`ImageRef` objects; indexes `user_history.csv` and `evidence_requirements.csv`. | Keeps the **raw `image_paths` string** for verbatim output *and* a parsed `(image_id, path)` list for working. `image_id` is the extension-less stem (`img_1`) — the exact form the output schema requires for `supporting_image_ids`. |
| `images.py` | Resolves each path under `dataset/`, decodes with **Pillow**, re-encodes to **JPEG**, base64-encodes, and emits one id-captioned image block per image. | The dataset files have **misleading extensions** — many `.jpg` files are actually AVIF / TIFF / WebP. The API only accepts JPEG/PNG/GIF/WebP, so we normalize *every* image through Pillow (with the AVIF plugin) instead of trusting the extension. Images are also downscaled (long edge ≤ 1536) to bound image tokens/cost. |
| `prompt.py` | Builds the frozen **system prompt** (task, "images are primary truth, history never overrides them", severity rubric, output rules) and the per-claim **user message** (claim + evidence rule + history + captioned images). | Each image is captioned `Image img_1:` so the model can cite exact IDs in `supporting_image_ids` (the API doesn't pass filenames). The system prompt is built **once** so its bytes stay stable for prompt caching. |
| `schema.py` | Pydantic `ClaimPrediction` with **enums** for every closed vocabulary (`claim_status`, `issue_type`, `object_part`, `severity`, `risk_flags`) and bools for the yes/no fields. | Structured output constrains the model to **valid, in-vocabulary tokens** — it physically cannot return `"dent/structural body damage"` instead of `dent`. The model returns **only the 10 predicted fields**; the 4 input fields are added by the writer (see below). |
| `model_client.py` | One `messages.parse(...)` call with `output_format=ClaimPrediction`, returning a validated object + token usage. Caches the system prompt. | `messages.parse` gives a typed, schema-validated result (no manual JSON parsing). `cache_control` on the frozen system prompt cuts per-claim cost. Usage is returned for the cost report. |
| `pipeline.py` | `process_claim(claim)` ties it together: build images → look up history/evidence → build message → predict → `to_row`. | **Early-outs without a model call** when no image is usable, and **wraps the call in a fallback** so an API error still yields a valid row — protecting the "one row per input" requirement. |
| `writer.py` | Merges the **verbatim input fields** with the prediction into the exact 14-column row; renders enums→values, bools→`true`/`false`, lists→semicolon strings; writes the CSV. | The model never regenerates the input columns (so `user_claim` is never paraphrased). All formatting/vocabulary shaping for the contract happens in one place. |
| `main.py` | Production entry point: `--limit` rows from `claims.csv` → `process_claim` → `output.csv`, plus token totals and an approximate cost. | `--limit` is the cost-control lever for incremental runs; cost/usage feed the operational analysis. |
| `evaluation/main.py` | Scores the system against the labeled sample and writes `eval_report.md` (see §4). | Reuses `process_claim` for dev/prod parity; decoupled run-vs-score. |

---

## 3. Output guarantees

`output.csv` always has the 14 required columns in the required order, fully quoted. Conformance is structural, not best-effort:

- **Input columns** (`user_id`, `image_paths`, `user_claim`, `claim_object`) are copied **verbatim** from the input by the writer.
- **Categorical fields** are enum-constrained by structured output, so every value is in the allowed vocabulary.
- **Every input row produces exactly one output row** — even on a missing image or an API error (fallback to a valid `not_enough_information` row).

---

## 4. Evaluation

`code/evaluation/main.py` automatically estimates quality on `dataset/sample_claims.csv` (the only file with gold labels) and writes a Markdown report to `eval_report.md`.

**What it does:**

- Runs the **same** `pipeline.process_claim` on the sample's input columns, compares the predictions to the gold labels, and scores each field with a comparator appropriate to its type:

  | Field type | Fields | Metric |
  |---|---|---|
  | Categorical (primary) | `claim_status` | accuracy, **macro-F1**, per-class P/R, **confusion matrix** |
  | Categorical | `issue_type`, `object_part`, `evidence_standard_met`, `valid_image` | exact accuracy |
  | Ordinal | `severity` | exact, within-one, mean absolute ordinal error |
  | Set (semicolon) | `risk_flags`, `supporting_image_ids` | micro-P/R/F1, Jaccard, exact-set-match |
  | Free text | the two justification fields | excluded from hard metrics |

- Lists the **mismatched rows** for `claim_status` so error analysis replaces manual eyeballing.

**Why designed this way:**

- **Field-type-aware metrics.** The labels are imbalanced (e.g. `claim_status` is 13/5/2; a "predict supported" baseline already scores 65%), so we lead with **macro-F1 + confusion matrix**, not plain accuracy. `severity` is *ordinal*, so a high↔medium near-miss is scored more leniently than high↔none. `risk_flags`/`supporting_image_ids` are *sets*, scored order-independently.
- **Dev/prod parity.** Scoring runs the production pipeline, so the eval reflects exactly what `output.csv` would contain.
- **Run/score decoupling.** The first run caches predictions to `eval_predictions.csv`; you can then re-score for free with `--predictions`. This also lets the same scorer compare **multiple configurations** (e.g. Opus vs. Sonnet, or prompt v1 vs. v2) by pointing it at different predictions files.
- **Normalization** (case/whitespace/`none`/set-order) prevents false mismatches from formatting differences.

---

## 5. Setup

Requires Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
# install dependencies (from pyproject.toml / uv.lock)
uv sync

# provide your API key (never commit this file)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Dependencies: `anthropic`, `pydantic`, `pydantic-settings`, `python-dotenv`, `pandas`, `pillow`, `pillow-avif-plugin`, `pytest`.

---

## 6. How to run

**Produce predictions (`output.csv`):**

```bash
# cost-controlled first run on the first 5 claims
uv run python code/main.py --limit 5

# full run on all of claims.csv
uv run python code/main.py

# options: --limit N, --input <csv>, --output <csv>
```

`main.py` prints per-row progress, total token usage, and an approximate cost.

**Evaluate against the labeled sample (writes `eval_report.md`):**

```bash
# run the pipeline on the sample and score it
uv run python code/evaluation/main.py --limit 5     # cheaper subset
uv run python code/evaluation/main.py               # full 20-row sample

# re-score an existing predictions file with NO API calls
uv run python code/evaluation/main.py --predictions eval_predictions.csv
```

**One-claim smoke test (notebook):** `notebook/test_model_client.ipynb` runs a single claim end-to-end and prints the parsed prediction + token usage — the cheapest way to sanity-check the model call.

**Unit tests:**

```bash
uv run pytest
```

---

## 7. Operational notes

- **Model:** `claude-opus-4-8`, one multimodal call per claim, structured output.
- **Cost control:** `--limit`, prompt caching on the system prompt, image downscaling, and skipping the model call when no image is usable. At the test-set scale (44 claims, ~82 images) a full run is only a few cents.
- **Robustness:** automatic SDK retries on rate limits/5xx; per-claim error fallback keeps `output.csv` complete.
- **Generated artifacts** (`output.csv`, `eval_predictions.csv`, `eval_report.md`) are run outputs and can be git-ignored.

---

## 8. Project layout

```text
.
├── code/
│   ├── config.py            # typed settings (.env, model id, paths, limits)
│   ├── loaders.py           # claims / user_history / evidence_requirements
│   ├── images.py            # normalize -> JPEG, base64, captioned blocks
│   ├── prompt.py            # system prompt + per-claim user message
│   ├── schema.py            # Pydantic ClaimPrediction + enums (the contract)
│   ├── model_client.py      # single Opus 4.8 structured-output call
│   ├── pipeline.py          # process_claim: one claim end to end
│   ├── writer.py            # merge inputs + prediction -> 14-column CSV
│   ├── main.py              # entry point: claims.csv -> output.csv
│   └── evaluation/
│       └── main.py          # score sample -> eval_report.md
├── notebook/
│   └── test_model_client.ipynb   # 1-claim smoke test
├── tests/
│   └── test_loaders.py
└── dataset/                 # provided inputs + images (see problem_statement.md)
```
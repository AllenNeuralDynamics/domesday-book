# Evaluation Framework

domesday includes a framework for measuring both retrieval quality and generation faithfulness. This lets you tune parameters empirically rather than guessing, and catch regressions when you change backends or configuration.

## Quick start

```bash
# Single run with current config — prints a retrieval quality report
python -m domesday.eval.runner

# Also judge generation quality using Haiku as an LLM judge
python -m domesday.eval.runner --judge

# Parameter sweep (grid search over min_score, k, chunk size, overlap)
python -m domesday.eval.runner --sweep --quick   # small grid, fast
python -m domesday.eval.runner --sweep            # full grid

# Interactive REPL — inspect individual queries, try free-form queries
python -m domesday.eval.runner -i
```

## Test corpus

The test corpus (`tests/fixtures/test_corpus.py`) contains 30 synthetic snippets designed to mimic real lab knowledge, organized into 8 topic clusters:

| Cluster | Snippets | Examples |
|---------|----------|---------|
| VBO dataset issues | t01–t04 | Timestamp off-by-one, corrupted eye tracking, encoder saturation, join key confusion |
| Two-photon imaging | t05–t07 | Galvo drift, Suite2p over-segmentation, neuropil correction |
| Behavioral training | t08–t10 | ITI optimization, water restriction welfare, auditory side bias |
| Electrophysiology | t11–t13 | Neuropixels noise, Kilosort over-splitting, chronic channel attrition |
| Analysis pipeline | t14–t16 | Baseline z-scoring, GPU cluster CUDA mismatch, pandas groupby bug |
| Experimental findings | t17–t19 | V1 mismatch threshold, SST paradoxical excitation, LGN orientation tuning |
| Data management | t20–t21 | LIMS/NWB ID mapping, coordinate system change |
| Software & tools | t22–t23 | allensdk bug, DeepLabCut whisker tracking failure |
| Histology & anatomy | t24–t25 | iDISCO+ clearing protocol, CCF atlas misalignment |
| Stimulus & display | t26–t27 | Gamma correction recalibration, monitor input lag |
| Statistical methods | t28–t29 | Cluster permutation test, von Mises bimodal tuning |
| Infrastructure | t30 | Python/numpy version standardization |

## Evaluation queries

21 queries across 4 categories, each with ground-truth relevant snippet IDs:

### Retrieval (6 queries)
Test whether the system finds the single correct snippet for a specific question. These are the "did it find the right needle?" tests.

Examples:
- "What's the timestamp bug in the VBO dataset?" → should retrieve t01
- "neuropil correction coefficient for GCaMP8m" → should retrieve t07
- "Why did the Wang replication fail?" → should retrieve t04

### Synthesis (8 queries)
Test whether the system retrieves multiple related snippets and combines them. These queries span topic boundaries or require aggregating information.

Examples:
- "What are all the known timing issues?" → should retrieve t01 and t27
- "What are the problems with the VBO dataset?" → should retrieve t01, t02, t03, t04
- "What infrastructure gotchas should new lab members know?" → should retrieve t15, t20, t21, t30

### Specificity (3 queries)
Test whether the system retrieves the right thing and *not* a related-but-wrong thing. Each query specifies both relevant and explicitly irrelevant snippets.

Examples:
- "Kilosort splitting fast-spiking neurons" → should retrieve t12 (KS4 over-splitting), NOT t06 (Suite2p over-segmentation)
- "NWB coordinate system issue" → should retrieve t21 (coordinate convention change), NOT t25 (CCF atlas misalignment)

### Negative (4 queries)
Test whether the system correctly returns nothing when the knowledge base has no relevant content. These are critical for calibrating the score threshold.

Examples:
- "What's the best restaurant near the Allen Institute?"
- "optimal learning rate for ResNet-50 training"

## Retrieval metrics

Computed per-query and aggregated across the run:

**Precision@k** — What fraction of returned results are actually relevant? High precision means the system isn't polluting results with noise.

**Recall** — What fraction of the relevant snippets were found? High recall means the system isn't missing important context.

**MRR (Mean Reciprocal Rank)** — How high in the results list is the first relevant snippet? MRR of 1.0 means the right answer is always #1.

**Negative accuracy** — What fraction of negative queries correctly returned no results? This directly measures the score threshold's effectiveness.

**Specificity accuracy** — What fraction of specificity queries avoided retrieving the explicitly-wrong snippet? This measures whether the system distinguishes similar-but-different concepts.

## LLM judge (generation quality)

The `--judge` flag runs the full RAG pipeline (retrieve + generate) for each query, then has Haiku score the generated answer on five dimensions:

**Faithfulness (weight: 0.30)** — Does the answer stick to what's in the snippets, or does it hallucinate beyond them? This is the most important dimension. A score of 1.0 means every claim is grounded in retrieved context.

**Citation accuracy (weight: 0.20)** — Do the cited snippet IDs actually support the claims they're attached to? Catches cases where the system cites a snippet but the claim comes from a different snippet or from nowhere.

**Abstention (weight: 0.20)** — When the retrieved context doesn't contain an answer (especially negative queries), does the system say so? A score of 0.0 means it fabricated an answer from nothing.

**Synthesis (weight: 0.15)** — For answers drawing on multiple snippets, is the information combined coherently? Are contradictions noted? Scored 1.0 for single-source answers.

**Relevance (weight: 0.15)** — Does the answer actually address what was asked? Catches cases where the right context was retrieved but the answer went off on a tangent.

These produce a weighted composite score. Running with `--judge` costs one Haiku call per query (21 calls for the full test suite — a few cents).

## LLM reranker

Separate from the eval judge, the reranker is a query-time component that filters retrieved chunks before they're sent to the generator. Enable it in config:

```toml
[reranker]
enabled = true
model = "claude-haiku-4-5"
relevance_threshold = 0.5
```

After vector search returns the top-k chunks, Haiku scores each one's relevance to the specific query. Chunks below the relevance threshold are discarded. This catches a failure mode where embedding similarity is high but actual relevance is low — for example, two snippets that use similar vocabulary but address different questions.

Trade-off: adds ~200ms and one Haiku call per user query. Recommended approach is to start without it, use the eval framework to measure retrieval quality, and enable the reranker only if false positives are a persistent problem.

## Parameter sweeps

The sweep mode runs a grid search over configurable parameters and ranks the combinations by a weighted composite of recall, negative accuracy, and MRR.

```bash
# Quick sweep (3 min_scores × 2 k values = 6 combinations)
python -m domesday.eval.runner --sweep --quick

# Full sweep (7 min_scores × 3 k values × 3 chunk sizes × 3 overlaps)
python -m domesday.eval.runner --sweep
```

### Sweep parameters

**min_score** (quick: 0.2, 0.3, 0.4 | full: 0.15–0.5 in 7 steps) — The cosine similarity threshold below which results are discarded. This is the most impactful parameter. Too low → false positives on negative queries. Too high → missed relevant results.

**k** (quick: 5, 10 | full: 5, 10, 15) — Number of chunks to retrieve before filtering. Higher k increases recall but may reduce precision.

**chunk_max_tokens** (quick: 400 | full: 200, 400, 600) — Maximum chunk size. Smaller chunks give more precise retrieval but lose context. Larger chunks preserve context but may dilute relevance signals. Note: changing chunk size requires re-embedding all content, so sweeping this dimension is slower.

**chunk_overlap** (quick: 50 | full: 0, 50, 100) — Token overlap between adjacent chunks. Overlap prevents information loss at chunk boundaries but increases the total number of chunks and embedding costs.

### Interpreting sweep results

Results are ranked by: `recall × 0.5 + negative_accuracy × 0.3 + MRR × 0.2`

The weighting reflects priorities: finding relevant content (recall) matters most, but not at the expense of hallucinating on irrelevant queries (negative accuracy). MRR matters but is secondary — it's better to find the answer at position 3 than to not find it at all.

Sweep results are saved to `./data/eval_results.json` for further analysis.

## Interactive mode

The REPL lets you explore retrieval behavior hands-on:

```
$ python -m domesday.eval.runner -i

domesday eval — interactive mode
Commands: query <text>, eval <index>, judge <index>, sweep, report, quit

domesday> What's the timestamp bug?
  1. [t01] score=0.847 (sarah) The VBO dataset has an off-by-one error...
  2. [t27] score=0.612 (ben) The ASUS VG248QE monitors we use have a 1-frame...

domesday> eval 0
  Query:    What's the timestamp bug in the VBO dataset?
  Category: retrieval
  Expected: ['t01']
  Got:      ['t01', 't27', 't04']
  Scores:   ['0.847', '0.612', '0.534']
  P=0.333 R=1.000 MRR=1.000

domesday> judge 0
  Generating answer for: What's the timestamp bug in the VBO dataset?
  Answer: The VBO dataset has an off-by-one error in frame timestamps...
  Judging...
  Faithfulness:      0.95
  Citation accuracy: 0.90
  Abstention:        1.00
  Synthesis:         1.00
  Relevance:         0.95
  Composite:         0.95
  Explanation: Answer is well-grounded in snippet-t01...

domesday> What's the best pizza in Seattle?
  (no results above threshold)

domesday> report
  ════════════════════════════════════════════════════════════
    Eval Run: interactive-full
  ════════════════════════════════════════════════════════════
    Precision:     0.412
    Recall:        0.875
    MRR:           0.952
    Neg. accuracy: 1.000
    Specificity:   0.667
  ──────────────────────────────────────────────────────────
    retrieval        n=6   P=0.500  R=1.000  MRR=1.000
    synthesis        n=8   P=0.350  R=0.812  MRR=0.917
    specificity      n=3   P=0.400  R=0.833  MRR=1.000
    negative         n=4   P=0.000  R=0.000  MRR=0.000
  ──────────────────────────────────────────────────────────
    ✓ No failures
  ════════════════════════════════════════════════════════════
```

`eval <index>` runs a single eval query (by index in `EVAL_QUERIES`) and shows retrieval metrics against ground truth. `judge <index>` does the same but also runs the generator and has Haiku score the answer. Free-form queries (just type anything) run a search and show results, useful for exploring behavior on queries outside the test suite.

## Adding your own test queries

Edit `tests/fixtures/test_corpus.py` to add domain-specific snippets and queries. The `EvalQuery` format:

```python
EvalQuery(
    query="Your question here",
    relevant_ids=["t01", "t04"],        # snippet id_prefixes that SHOULD be found
    irrelevant_ids=["t06"],             # should NOT appear (optional)
    category="retrieval",               # retrieval | synthesis | specificity | negative
    reference_answer="Optional expected answer text",
)
```

For negative queries, set `relevant_ids=[]` — the eval framework will check that no results are returned.

## Recommended workflow

1. **Start with a single run** (`python -m domesday.eval.runner`) to get a baseline with default parameters.
2. **Check failures** in the report. Are they retrieval failures (wrong chunks) or threshold failures (irrelevant chunks not filtered)?
3. **Run a quick sweep** (`--sweep --quick`) to find a better `min_score` threshold for your embedding model.
4. **Add your own snippets and queries** to the test corpus to cover your actual domain knowledge.
5. **Try `--judge`** to measure generation quality. If faithfulness is low, the problem is usually in the system prompt or the retrieved context, not the generator.
6. **Enable the reranker** if precision is still low after tuning the threshold. Measure the improvement against the added latency.
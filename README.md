# Can Jev make AI search more accurate? Jev vs the standard re-ranker (BGE)

When an AI answers a question from a pile of documents (a setup called RAG), it first searches for passages and then
reads the best ones. A **re-ranker** is a second pass that re-orders the search results so the most useful passages
come first. This project tests whether a new model called **Jev** does that job better than **BGE**, a widely used
standard re-ranker.

## Results at a glance

300 hard questions, each needing facts from **two** different Wikipedia paragraphs. Same search, same AI writer, same
questions for everyone. The only thing that changed was the re-ranker.

| | No re-ranking | BGE (standard) | Jev |
|---|---|---|---|
| Both needed sources in the top 5 | 69% | 86% | 90% |
| Answers that exactly match | 46% | 52% | 55% |
| Answer quality score (partial credit, out of 100) | 60 | 68 | 73 |
| Questions the AI gave up on ("I don't know") | 20% | 11% | 8% |
| Time to re-rank one question | none | 8 seconds* | 0.24 seconds |
| Cost per 1,000 questions | none | $0 (runs on your own computer) | about $0.25 |

\*BGE timed on a laptop without a graphics card. On a GPU it would be much faster.

### What it means

- **A re-ranker clearly helps.** Both versions found both needed sources far more often than plain search
  (69% to 86% and 90%), and gave more correct answers.
- **Jev came out ahead of BGE, but only by a little.** About 4 points on the answer quality score (73 vs 68). That is
  the one result that is statistically solid. The 3-point leads on exact matches and on finding both sources point the
  same way, but they are close enough that chance could explain them.
- **Jev's edge is on questions that connect two facts** (for example, "who directed the film that starred the person
  who...?"). There it found both sources 89% of the time against 84% for BGE. On the 49 simple comparison questions
  BGE did slightly better (49 of 49 vs 46 of 49), though that is too few questions to draw a conclusion from.
- **The AI writer was equally good in every setup.** Once it had both sources, it answered correctly about 57% to 59%
  of the time regardless of the re-ranker. Jev's advantage is simply getting the right sources in front of it more often.
- **Jev is much faster and costs a fraction of a cent per question.** BGE costs nothing in fees but needs your own
  computing power.

### Honest limits

- This is one test: 300 questions from one dataset (HotpotQA), run once.
- Jev's scores vary slightly if you ask it the same thing twice. We didn't measure how much that affects results.
- Neither re-ranker was fine-tuned. A bigger BGE model or a better-worded Jev instruction could change the gap.
- An early check on the first 150 questions showed a bigger Jev lead than the full 300 did. Small tests can overstate
  small differences.

---

# Technical details

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
copy .env.example .env   # then fill in OPENROUTER_API_KEY
```

All models run through [OpenRouter](https://openrouter.ai/docs) with one key. Embeddings and generation use
OpenRouter's OpenAI-compatible API (`https://openrouter.ai/api/v1`). Jev uses the TypeSafe SDK with
`base_url="https://openrouter.ai/api"` (System One endpoint). Endpoints and model IDs live in `config.yaml`.

## Run

```powershell
.\.venv\Scripts\python.exe -m ragcmp.run prepare                 # sample HotpotQA, build corpus
.\.venv\Scripts\python.exe -m ragcmp.run index                   # embed corpus into Chroma
.\.venv\Scripts\python.exe -m ragcmp.run eval --split dev --limit 5
.\.venv\Scripts\python.exe -m ragcmp.run eval --split test       # final run
.\.venv\Scripts\python.exe -m pytest                             # offline unit tests
```

`--no-generate` computes retrieval metrics only (no generator calls). `--variants bge,jev` restricts variants.
All paid calls (embeddings, Jev, generation) are cached in `.cache/cache.sqlite`, so reruns are free.

## Method

| Stage | Choice |
|---|---|
| Data | HotpotQA distractor, validation split. Seed 42: 50 dev questions (smoke/sanity runs only, nothing was tuned on them) + 300 test questions |
| Corpus | All paragraphs of the 350 questions, pooled and de-duplicated by title (3,476 paragraphs) |
| First stage | `openai/text-embedding-3-small` -> Chroma (cosine), top-30. Identical candidates for every variant |
| Variants | `vanilla` (vector order), `bge` (`BAAI/bge-reranker-base`, local), `jev` (`typesafe/jev-1.14`; one request per query, one Noul per passage, ranked by probability) |
| Generator | `openai/gpt-6-luna`, reasoning effort `none`, top-5 passages, same prompt for every variant |

Costs in the report use OpenRouter's billed `usage.cost` when it is returned, otherwise a token-price estimate from `config.yaml`.

Metrics: Recall@1/3/5/30, MRR, nDCG@5, **both-gold@5** (both supporting paragraphs in the top 5), HotpotQA EM/F1,
rerank latency, context tokens and cost. Differences are reported as paired bootstrap 95% CIs (10k resamples),
with exact McNemar tests for binary metrics.

## Detailed results

300 HotpotQA distractor test questions (all "hard" level; 251 bridge, 49 comparison), run once, nothing tuned on them.
Both rerankers saw the same 30 first-stage candidates and fed the same generator with their top 5.

| variant | recall@3 | recall@5 | nDCG@5 | both-gold@5 | EM | F1 |
|---|---|---|---|---|---|---|
| vanilla (no rerank) | 0.765 | 0.843 | 0.819 | 0.693 | 0.457 | 0.597 |
| `bge-reranker-base` | 0.902 | 0.932 | 0.918 | 0.863 | 0.517 | 0.683 |
| Jev (`typesafe/jev-1.14`) | 0.923 | 0.945 | 0.931 | 0.897 | 0.547 | 0.726 |

First-stage recall@30 is 0.965 for all three, so 21 of 300 questions (7%) have a gold paragraph that no reranker can
recover. That caps both-gold@5 at 0.930.

### Findings

1. **Reranking matters a lot; the choice of reranker matters much less.** Both rerankers beat the no-rerank baseline
   on every metric. Jev vs vanilla: both-gold@5 +0.203, EM +0.090, F1 +0.130 (all p < 0.0001).
   BGE vs vanilla: both-gold@5 +0.170, EM +0.060 (p = 0.005), F1 +0.087.
2. **Jev beat BGE modestly, and only one difference is clearly significant.** Jev - BGE: F1 +0.043
   (95% CI +0.014 to +0.073, p = 0.004, the only one of 8 comparisons that survives a Bonferroni correction),
   EM +0.030 (CI 0.000 to +0.063, p = 0.07), both-gold@5 +0.033 (CI 0.000 to +0.070, p = 0.06).
   Recall@1, MRR, recall@3, recall@5 and nDCG@5 are statistically indistinguishable.
   Direction favours Jev on every answer-quality metric; magnitude is a few points.
3. **The advantage is in multi-hop (bridge) questions.** On the 251 bridge questions Jev gets both supporting
   paragraphs into the top 5 for 88.8% vs BGE's 83.7% (+0.052, CI +0.012 to +0.092, McNemar p = 0.015).
   On the 49 comparison questions BGE is perfect (49/49) while Jev finds both for 46/49; too few questions to
   conclude anything, but it shows Jev is not uniformly better. These are post-hoc subgroup analyses.
4. **Better retrieval is the whole story for answers.** When both gold paragraphs reach the generator, all three
   variants score the same EM (0.57 to 0.59); without them EM is about 0.2. Jev's EM/F1 edge comes from putting
   both paragraphs in the context more often (269 vs 259 vs 208 of 300), which also cuts "unknown"
   abstentions (vanilla 59, BGE 33, Jev 25).
5. **Jev ranks the second hop higher.** The later of the two gold paragraphs lands below rank 10 for
   2 questions with Jev, 9 with BGE and 25 with no reranker (excluding the 21 unreachable).
6. **Cost and latency.** Jev: $0.00025 per query (about 5,970 input tokens, $0.075 for all 300), median
   0.24 s per rerank (p95 0.42 s). BGE: no API cost, but median 8.0 s (p95 14.8 s) on a laptop CPU. The latency
   comparison is not like-for-like: BGE on a GPU would be far faster, and Jev time is a network round trip.
   Generation cost is the same for all three (about $0.023 per 300 queries).

### Caveats

- **Jev scores are not deterministic** (identical calls varied by up to ~0.1 on a spot check; they come back at
  2-decimal resolution, and ties are broken by first-stage order). Each query was scored once and cached, so these
  results are one draw. The confidence intervals cover question sampling, not Jev's own call-to-call noise, which was
  not quantified.
- First-stage recall caps both rerankers: a gold paragraph missing from the top-30 cannot be recovered.
- One dataset (HotpotQA distractor, hard questions), a pooled corpus of ~3,500 paragraphs, one embedding model,
  one generator, one run. Both rerankers were used zero-shot with default settings; neither was tuned, and the Jev
  yes/no question wording is a first draft. A larger BGE model or a tuned Jev prompt could change the gap.
- The answer metrics use the official HotpotQA normaliser, which is strict (EM = 0 for "Dutch heritage" vs "Dutch");
  F1 is the fairer headline.
- With 150 questions the interim estimate of the Jev-BGE gap was larger (EM +0.047, recall@3 +0.047 significant);
  at 300 it shrank (EM +0.030, recall@3 +0.022, not significant). Treat small-sample reranker comparisons with care.
- The per-call generation latency in the raw reports is dominated by the client-side rate limiter and is not meaningful.

Full tables and paired tests: [`results/test_summary.md`](results/test_summary.md).

# RAG reranker comparison: BGE cross-encoder vs Jev

Does [Jev](https://docs.typesafe.ai/introduction) (TypeSafe's typed-decision "System One" model) rerank retrieved
passages better than a standard BERT cross-encoder (`BAAI/bge-reranker-base`)?

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
copy .env.example .env   # then fill in OPENAI_API_KEY and TYPESAFE_API_KEY
```

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
| Data | HotpotQA distractor, validation split. Seed 42: 50 dev questions (tuning) + 300 test questions |
| Corpus | All paragraphs of the 350 questions, pooled and de-duplicated by title (3,476 paragraphs) |
| First stage | OpenAI `text-embedding-3-small` -> Chroma (cosine), top-30. Identical candidates for every variant |
| Variants | `vanilla` (vector order), `bge` (`BAAI/bge-reranker-base`), `jev` (one request per query, one Noul per passage, ranked by probability) |
| Generator | OpenAI `gpt-6-luna`, reasoning effort `none`, top-5 passages, same prompt for every variant |

Metrics: Recall@1/3/5/30, MRR, nDCG@5, **both-gold@5** (both supporting paragraphs in the top 5), HotpotQA EM/F1,
rerank latency, context tokens and cost. Differences are reported as paired bootstrap 95% CIs (10k resamples),
with exact McNemar tests for binary metrics.

## Results

_Pending: run `eval --split test`._

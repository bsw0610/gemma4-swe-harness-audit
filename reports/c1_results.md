# C1 audit results (v3, 2026-10-06, after Codex audits 1–2)

> 2026-11-05: this report describes swegemma 0.2.7. For what changed in 0.2.11, see `harness_0_2_11_recheck.md`.

Every number below is printed by a tracked script:
- §3, the §4 BM25 rows and §5: `analysis/c1_report.py` (output `work/c1/report/stdout.txt`);
- §4 graph rows: `analysis/reach.py` (`work/c1/reach_v3/summary.md`);
- the §4 extraction-sensitivity check: `analysis/reach_independent.py` (`work/c1/reach_independent/per_task.csv`);
- §2: `analysis/embed_geometry.py` (`work/c1/embed_geometry_v2/summary.md`).

Kaggle kernels: `kaggle/gold_ast` version 4 (gold) and `kaggle/bm25` version 3 (snapshot BM25). Input and output hashes are in `reports/c1_manifest.json`.

**Statistics.** The unit is the task. A 95% CI is the percentile interval of a task-resampling bootstrap
(10,000 draws, seed 0, tasks sorted by id). It is conditional on this task set: four repositories, with fastapi and rich
making up 89% of the 115 tasks. It does not cover variation from one repository to another; leave-one-repo-out rows are given for that.
**Funnel.** 129 tasks → 120 have an existing (base-snapshot) symbol as gold → 115 have at least one gold symbol in the graph
(fastapi 59, rich 43, requests 12, httpx 1).

## 1. Documented API vs implementation
See `reports/tool_mechanics.md` (file:line). In short:
- The README's short-name resolver (exact → suffix → case-insensitive → substring) does not exist.
  `get_code_neighbors` needs the full dotted id.
- `search_similar_code` is documented as matching "node keys/suffixes". It matches only exact ids or exact source text,
  so the README's own example (`"HTTPConnection"`) cannot match. The lack of a text encoder is documented and is not counted as a finding.
- The documented edge types `CALLS / DEFINED_IN / IMPORTS` do not exist: all 452,588 edges are lowercase `calls`.
  Neighbours are in- and out-edges merged, sorted alphabetically and cut at 50 by default.

## 2. Graph and embeddings (127 graphs, 367,985 nodes)
- 50.8% of nodes have no edge. There are no module nodes and no async functions; by `ast`, nodes are 300,316 functions and 67,669 classes.
- Embedding geometry (one value per graph, then the median over 127 graphs): effective rank 4.06 of 256;
  PC1 holds 63.1% of the variance; exact mean pairwise cosine 0.790; mean cosine to the mean vector 0.886.
- Ties at rank 10/11 of a top-k list (up to 2,000 sampled queries per graph; pooled n = 236,022 queries):
  - equal at the 4 displayed decimals in 57.2% of queries;
  - exactly equal in float32 in 14.2% of queries. Only these true ties are ordered alphabetically by the tool, and they are mostly duplicate vectors.
- Neighbours carry some structure. Share of top-10 neighbours with the same class parent, pooled (classes detected with `ast`):
  fastapi 23.3%, rich 27.5%, requests 53.3%, httpx 54.0%. The random baseline is a 0.18% per-graph median.

## 3. Gold locations (AST on base-commit snapshots)
- Rule: each changed old-file line is mapped to the innermost enclosing def or class.
  - A replacement block is attributed only through its removed lines.
  - A pure insertion goes to the innermost scope that lexically encloses the gap (trailing blank lines included) and whose body indentation it sits in.
    So a new method goes to its class and a new top-level definition to the module. A block that only adds decorators goes to the definition it decorates.
  - Whitespace-only blocks are skipped unless they lie inside a multi-line string literal.
  - Regression cases: `analysis/test_gold_core.py` (14 cases, including every counterexample from audits 1 and 2).
- 501 unique (task, symbol) gold items; 452 are in the graph (90.2%). By kind: defs 371/377 (the 6 missing are functions nested inside functions), async defs 0/43, classes 81/81.
  22 of 120 tasks have at least one gold symbol missing from the graph.
- A text-matching heuristic agrees exactly with the AST gold on 286/465 hunks (61.5%; any overlap 340/465, 73.1%).
  The unit is a hunk; the denominator is hunks with an in-graph symbol under either method; per-hunk sets are in `work/c1/report/hunk_agreement.csv`.
  AST gold is therefore required.

## 4. Reach from the issue text (n = 115; seeds only from `problem_statement`)
Hit = at least one gold symbol among the first k candidates. Hits are out of 115 tasks.

| Strategy | @10 | @50 | @100 |
|---|---|---|---|
| Graph tools as implemented (exact-id seeds): expansion-only / end-to-end | 1 / 2 | 1 / 2 | 1 / 2 |
| Documented resolver (counterfactual), 1-hop neighbours: expansion-only / end-to-end | 8 / 33 | 11 / 35 | 12 / 36 |
| Documented resolver, 2-hop neighbours: expansion-only / end-to-end | 8 / 33 | 15 / 37 | 19 / 41 |
| Documented resolver, embedding top-k (float32 order): expansion-only / end-to-end | 7 / 33 | 17 / 40 | 20 / 44 |
| BM25 over AST function/class spans of the snapshot | 51 = 44.3% [34.8, 53.9] | 83 = 72.2% [63.5, 80.0] | 98 = 85.2% [78.3, 91.3] |
| BM25 over the graph's own node texts, library package nodes only (id starts with the package name) | 58 = 50.4% [41.7, 59.1] | 86 = 74.8% [67.0, 82.6] | 99 = 86.1% [80.0, 92.2] |

- **Seed gating.** Exact-id seeds exist in 2/115 tasks, and in both a seed is already gold. This is a property of the static policy
  "start from identifiers as written in the issue text"; an agent that searches for full ids first could do better.
  The documented resolver (DOC-5) finds seeds in 70/115 tasks (60.9% [52.2, 69.6]); in 31 of them a seed is already gold.
- **DOC-5 policy:** candidate identifiers (backticks, dotted names, CamelCase, snake_case, `name(`); tiers 2–4 need ≥ 4 characters;
  at most 5 matches per candidate, shortest ids first; per-seed lists merged round-robin.
  An independent re-implementation with a different extractor (`analysis/reach_independent.py`) finds seeds in 70/115 tasks and gives
  1-hop hits of 12 / 14 / 14 (expansion-only) and 32 / 37 / 38 (end-to-end) at k = 10 / 50 / 100.
  Absolute counts move by a few tasks with the extraction heuristic; the gap to BM25 does not.
- **Cap.** Removing the alphabetical 50-neighbour cap changes no task-level hit in the evaluated settings, except DOC-5 2-hop @100 (−1 task).
  The cap truncated 21 of 606 DOC-5 1-hop seed calls (3.5%).
- **File level.** BM25 file ranking puts a gold file in the top 10 for 90/115 tasks (78.3% [70.4, 85.2]).
- **Name overlap.** In 35/115 tasks the last name of a gold symbol appears verbatim in the issue text. BM25@50 is 85.7% (30/35) for those
  and 66.2% (53/80) for the other 80. This is a post-hoc stratification, not a causal effect.
- **Leave-one-repo-out BM25@50.** Without rich 72.2%, without fastapi 78.6%, without requests 68.9%, without httpx 71.9%.

## 5. Gate test: does graph expansion add to BM25 at equal budget?
- Policy: the kernel ranks library-package units only (filter applied before truncation; top 200; unique ids).
  The top-m units become seeds, followed by their 1-hop `calls` neighbours (library nodes only; sequential or round-robin over seeds),
  then filled with BM25 ranks m+1… Every candidate list is asserted to hold exactly k unique ids, including at k = 100.
- At k = 25 and k = 50, all 10 settings (m ∈ {5, 10, 25}, both orders) lower hit@k by 3.5–12.2 pp.
  - At k = 50, all 6 settings have paired 95% CIs that exclude 0 (−7.8 to −12.2 pp). Example: m = 5, round-robin: −7.8 pp [−14.8, −1.7] (3 tasks gained, 12 lost).
  - At k = 25, only m = 5 round-robin excludes 0 (−7.8 pp [−15.7, −0.9]); m = 5 sequential touches 0 (−7.0 pp [−13.9, +0.0]), and m = 10 crosses 0.
- At k = 100 the differences are −0.9 to −5.2 pp; one of six CIs excludes 0 (m = 10 sequential, −5.2 pp [−10.4, −0.9]).
- Leave-one-repo-out (k = 50, m = 5, sequential): −7.1 to −9.7 pp.
- This tests one family of expansion policies over the provided `calls` graph. It does not show that structural information
  cannot help localisation. A score-smoothing re-rank (`work/c2/rerank.py`) was explored in-sample only and is not reported as a result.

## Caveats
- The issue texts are pull-request descriptions; the hidden test set comes from private repositories.
  Code in the text does not explain BM25's reach: BM25@50 is 64.7% (11/17) when the text has a code fence and 73.5% (72/98) when it has none;
  75.0% (51/68) with any backtick vs 68.1% (32/47) without.
- Snapshot BM25 details: unique-term query weighting, ties in corpus order, class spans overlap their methods, decorators included (v2).
- requests (n = 12) and httpx (n = 1) are small; httpx has no CI.

## Verification log
| Date | Check | Result |
|---|---|---|
| 2026-10-05 | Codex audit 1 (read-only) | high 5, medium 5, low 2 → all addressed in `aa06741` |
| 2026-10-06 | Codex confirmation audit 2 | 7 FIXED, 5 PARTIAL, new high 5 / medium 1 / low 1 → addressed in `35c6e40` |
| 2026-10-06 | Mutation test of the gold attribution (Kaggle kernel `kaggle/gold_mutation`, model-independent): synthetic patches with known gold on real library files of all 129 snapshots, 12 edit types (line replace/delete/insert, append to body, add decorator, new method with/without decorator and after a trailing blank, new top-level def with/without decorator, blank-line insert) | 1,821 / 1,821 correct, 0 errors. Scope: each synthetic patch is one change block in one hunk; multi-block hunks and cross-block de-duplication are covered by the audits' manual spot checks (16 tasks in audit 1, 15 in audit 2, partly overlapping) and the unit tests, not by this test |
| 2026-10-06 | Fresh-clone reproduction: `git clone`, kernel outputs re-downloaded, all `analysis/` scripts re-run | stdout, `c1_manifest.json` and every output CSV byte-identical; embedding outputs identical across two runs (absolute output path in `embed_geometry.py` found and fixed) |

Codex audit 3 is pending (usage limit); the next Codex audit is planned for the finished notebook, as AGENTS.md requires.

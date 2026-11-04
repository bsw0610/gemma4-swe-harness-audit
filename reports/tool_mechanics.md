# How the provided graph tools actually behave (swegemma 0.2.7)

> 2026-11-05: name resolution, similarity queries and the `edge_type` filter changed in 0.2.11. See `harness_0_2_11_recheck.md`.

Sources: `data/raw/wheels/swegemma-0.2.7-py3-none-any.whl` (unpacked to `work/pkgsrc/`; the 36 .py files are byte-identical to the wheel),
`data/raw/HARNESS_README.md`, and the 127 graphs / 127 embedding archives in `data/raw/`. Paths below are relative to `swegemma/`.
Data facts were checked on all 127 graphs (2026-10-05). Detailed working notes: `work/c1/tool_mechanics.md` (its "no graph JSON" lines predate the download and are superseded here).

## Documentation vs implementation
| Tool | HARNESS_README says | Code does (file:line) | Consequence |
|---|---|---|---|
| `get_code_neighbors` | short names resolved in 4 tiers via `resolve_node_name`: exact → `.`/`/` suffix → case-insensitive → substring (§6.3, item 7) | no `resolve_node_name` anywhere in swegemma or adk_eval_core; exact node-id lookup only, a miss returns `[]` (`graph/graph_utils.py:431-432`) and the tool wraps it as `status: ok`, `count: 0` (`tools/graph.py:184-188`) | `"APIRouter"` returns nothing; only `fastapi.routing.APIRouter` works |
| `get_code_neighbors` | `edge_type` filter, e.g. `CALLS`, `DEFINED_IN`, `IMPORTS` | exact, case-sensitive match on the edge `type` (`graph_utils.py:436,446`); the data contains only `calls` (452,588 of 452,588 edges) | `CALLS` (as documented) returns nothing; containment/import relations do not exist |
| `get_code_neighbors` | neighbours of a symbol | out- and in-edges merged, de-duplicated, sorted alphabetically, cut to `max_neighbors` (default 50) (`graph_utils.py:434-456`); direction and edge type dropped; a second cut at 5,000 characters (`tools/graph.py:167-182`) | callers and callees are indistinguishable |
| `search_similar_code` | query resolved "against node keys/suffixes", then cosine similarity; pass a symbol name such as `"HTTPConnection"`, not a sentence (§6.3, item 8) | dict lookup on exact node ids, ids without a `repo;` prefix, and exact node source text (`graph/embedding_utils.py:320-327,353-370`); **no suffix matching**; a miss returns an empty list (`graph/retrieval_utils.py:762-807`) | the documented short-name example `"HTTPConnection"` cannot match, because ids are full dotted paths |
| `search_similar_code` | top-k by cosine | float32 cosine, sorted by (−sim, name), only the query node excluded, values rounded to 4 decimals for display (`retrieval_utils.py:819-833`, `tools/graph.py:270-275`) | exact duplicates are returned (fastapi docs tutorials) |
| availability | tools enabled when graph/embedding files exceed 100 bytes | tools are always registered (`tools/__init__.py:34-46`); the size check only gates a system-prompt hint | — |

The absence of a text encoder is documented (offline sandbox) and is not a finding. The finding is that the documented
short-name / suffix resolution is not implemented for either tool.

## Data facts (127 graphs, 367,985 nodes)
- Node id equals `name` for every node; ids are dotted module paths (`src/` stripped for requests and httpx).
- All edges are `calls`; 50.8% of nodes have no edge; no module nodes; node kinds by `ast`: 300,316 functions, 67,669 classes, 0 async functions.
- Embedding keys equal graph ids in every archive (no missing, no extra, no NaN, no zero vectors).

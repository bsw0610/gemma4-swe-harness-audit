---
name: localize-code
description: Rank the repository's functions and classes by relevance to the issue text (BM25 over Python AST spans). Use it first, before reading files, to decide which files and functions to open.
---

# localize-code

Run `scripts/localize.py` with the issue text as the query. It walks `/workspace`, skips tests, docs and
virtual environments, parses every Python file with `ast`, and ranks files and functions/classes by BM25
similarity to the query. It only reads files and writes nothing.

Output (under 4,000 characters):

```
Likely files (BM25 over N non-test files):
  path/to/module.py
Likely functions/classes (of M):
  path/to/module.py:123  ClassName.method_name
```

How to use the result:
1. Open the top 2–3 files with `read_file` at the listed line numbers.
2. Prefer functions whose names or bodies match the behaviour described in the issue; the list is a ranking, not an answer.
3. If nothing fits, rerun with a shorter query made of the key identifiers and error messages from the issue.

Options: `--top N` (default 15), `--files N` (default 5), `--root DIR` (default: the task repository, found automatically), `--max-chars N` (default 4000).

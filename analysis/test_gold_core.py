"""Regression cases for gold_core (Codex audit A01, A11). Run: uv run python analysis/test_gold_core.py"""
from gold_core import parse_patch, attribute

SRC = """class C:
    def f(self):
        x = 1
        return 1

    def g(self):
        return 2


def h():
    return 3
"""


def gold(patch, src=SRC):
    files = parse_patch(patch)
    return sorted({q for _, q, *_ in attribute(files["pkg/m.py"], src)})


cases = {
    # A01: replace the last line of a method -> only C.f (not the class C)
    "replace_last_line": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -3,3 +3,3 @@ class C:\n         x = 1\n-        return 1\n+        return 3\n \n", ["C.f"]),
    # append a statement at the end of a function body -> C.f
    "append_in_body": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -4,2 +4,3 @@ class C:\n         return 1\n+        y = 2\n \n", ["C.f"]),
    # add a new method in the class (method-level indent) -> C
    "new_method": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -7,2 +7,5 @@ class C:\n         return 2\n+\n+    def k(self):\n+        return 9\n \n", ["C"]),
    # add a new top-level function after h -> module-level
    "new_toplevel": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -11,1 +11,4 @@ def h():\n     return 3\n+\n+def z():\n+    return 0\n", ["<module>"]),
    # decorator-only insertion right before def g -> C.g
    "decorator_only": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -5,2 +5,3 @@ class C:\n \n+    @staticmethod\n     def g(self):\n", ["C.g"]),
    # '\\ No newline' marker must not shift line numbers
    "no_newline_marker": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -11,1 +11,1 @@ def h():\n-    return 3\n\\ No newline at end of file\n+    return 4\n\\ No newline at end of file\n", ["h"]),
    # insertion inside f between two body lines -> C.f
    "insert_mid_body": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -3,2 +3,3 @@ class C:\n         x = 1\n+        x += 1\n         return 1\n", ["C.f"]),
    # whitespace-only insertion / removal -> no gold (formatting)
    "blank_insert": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -3,2 +3,3 @@ class C:\n         x = 1\n+\n         return 1\n", []),
    "blank_remove": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -4,3 +4,2 @@ class C:\n         return 1\n-\n     def g(self):\n", []),
    # audit 2 N01: a NEW decorated method inserted before an existing method -> the class, not the next method
    "new_decorated_method": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -5,2 +5,6 @@ class C:\n \n+    @property\n+    def new(self):\n+        return 0\n+\n     def g(self):\n", ["C"]),
    # audit 2 N01: a NEW decorated top-level class inserted before an existing def -> module-level
    "new_decorated_class": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -9,2 +9,6 @@ class C:\n \n+@dataclass\n+class New:\n+    x: int = 0\n+\n def h():\n", ["<module>"]),
    # audit 2 N02: new method after the class's trailing blank line -> still the class
    "method_after_trailing_blank": ("--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -7,2 +7,5 @@ class C:\n         return 2\n \n+    def k(self):\n+        return 9\n+\n", ["C"]),
}
fails = 0
for name, (patch, want) in cases.items():
    got = gold(patch)
    ok = got == want
    fails += not ok
    print(f"{'OK ' if ok else 'FAIL'} {name}: got {got} want {want}")
# audit 2 N06: removing a blank line inside a triple-quoted string changes the value -> f
SSRC = 'def f():\n    return """a\n\nb"""\n'
got = sorted({q for _, q, *_ in attribute(parse_patch("--- a/pkg/s.py\n+++ b/pkg/s.py\n@@ -2,3 +2,2 @@ def f():\n     return \"\"\"a\n-\n b\"\"\"\n")["pkg/s.py"], SSRC)})
print("OK  blank_inside_string" if got == ["f"] else f"FAIL blank_inside_string: {got}")
fails += got != ["f"]
deleted = parse_patch("--- a/pkg/old.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-x = 1\n")
print("OK  deleted_file_skipped" if deleted == {} else f"FAIL deleted_file_skipped: {deleted}")
fails += deleted != {}
raise SystemExit(fails)

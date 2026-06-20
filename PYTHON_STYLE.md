# Google Python Style Guide — Concise Summary

Source: https://google.github.io/styleguide/pyguide.html

## Core Philosophy

- **Readability over cleverness.** Code is read far more often than written.
- **Consistency** matters more than personal preference — match the style of surrounding code.
- Built on **PEP 8**, with Google-specific refinements and exceptions.

---

## Language Rules

**Imports**
- Import modules/packages, not individual functions or classes: `import os`, not `from os import path`.
- Exception: `typing` and `collections.abc` symbols are imported directly (`from typing import Any`).
- Project exception: `pydantic` symbols are also imported directly (`from pydantic import BaseModel, Field`) — Google's strict rule is impractical for a library built around its base classes, and no real-world Pydantic codebase follows `import pydantic` / `pydantic.BaseModel`.
- No relative imports — always use the full package path.
- One import per line; never `import os, sys`.

**Exceptions**
- Use built-in exception types when appropriate (e.g. `ValueError` for bad arguments).
- Never use bare `except:` or catch `Exception` broadly — except when re-raising or isolating a failure point (e.g. protecting a thread).
- Keep `try` blocks small — the more code inside, the more likely an unrelated line raises the error you didn't expect.
- Custom exceptions must inherit from an existing exception class and end in `Error`.
- Don't use `assert` for argument validation — it can be stripped at runtime. Use `assert` only in tests or as non-critical sanity checks.

**Global state**
- Avoid mutable global state. If unavoidable, prefix with `_` and expose only through functions.
- Module-level constants are fine and encouraged: `_MAX_RETRIES = 3`.

**Comprehensions**
- Fine for simple cases. Avoid multiple `for` clauses or nested conditions — fall back to a regular loop if it gets complex.

**Defaults**
- Never use mutable objects (`[]`, `{}`) as default argument values — they're evaluated once at load time and shared across calls.
- Use `None` as the default and initialize inside the function instead.

**Other**
- Use default iterators (`for k, v in d.items()`), not `.keys()` / `.readlines()`.
- Prefer the implicit falsy check (`if not items:`) over `len(items) == 0`.
- Avoid "power features" (metaclasses, reflection, monkey-patching) unless there's no simpler option.
- Type annotations are strongly encouraged for all new code, especially public APIs.

---

## Style Rules

**Formatting**
- Max line length: **80 characters** (exceptions: URLs, long imports).
- Indentation: **4 spaces**, never tabs.
- No semicolons to terminate lines or chain statements.
- Use parentheses sparingly — never around `if`/`return` conditions unless needed for line continuation.
- **2 blank lines** between top-level functions/classes; **1 blank line** between methods.
- No trailing whitespace; no spaces inside brackets/parens (`spam(ham[1])`, not `spam( ham[ 1 ] )`).

**Strings**
- Use f-strings, `%`, or `.format()` — never build strings with repeated `+` in a loop (use a list + `''.join()` instead).
- Pick one quote style (`'` or `"`) and stay consistent within a file.
- Docstrings always use `"""`.

**Docstrings**
- Every module starts with a docstring describing its purpose.
- Every public function/method with non-trivial logic needs a docstring with `Args:`, `Returns:` (or `Yields:`), and `Raises:` sections as relevant.
- Every class needs a docstring with an `Attributes:` section for public attributes.
- Docstring summary line ≤ 80 characters, ends in punctuation.

**Comments**
- Explain *why*, not *what* — assume the reader knows Python.
- Use proper grammar and punctuation.
- `TODO` format: `# TODO: <link to bug/issue> - <explanation>`

**Naming**

| Type | Convention |
|---|---|
| Modules / packages | `lower_with_under` |
| Classes / Exceptions | `CapWords` |
| Functions / methods | `lower_with_under()` |
| Constants | `CAPS_WITH_UNDER` |
| Variables | `lower_with_under` |
| Internal/protected | `_leading_underscore` |

- Avoid single-letter names except for counters (`i`, `j`), exceptions (`e`), or file handles (`f`).
- Never use `__dunder__`-style names for your own attributes — prefer a single leading underscore for "private."
- No dashes in file or package names.

**Main entry point**
```python
def main():
    ...

if __name__ == '__main__':
    main()
```
Required so the module is safely importable (by tests, `pydoc`, etc.) without executing top-level logic.

**Function length**
- No hard limit, but reconsider splitting a function once it exceeds ~40 lines.

**Resource management**
- Always use `with` for files, sockets, and other closeable resources — never rely on garbage collection to close them.

---

## Type Annotations

- Use `X | None`, not implicit `Optional` via a bare default of `None`.
- Prefer abstract types in signatures (`Sequence`, `Mapping`) over concrete types (`list`, `dict`) unless mutation is required.
- Use built-in generics (`list[int]`, `tuple[int, ...]`) rather than `typing.List` / `typing.Tuple`.
- Annotating `self`/`cls` is unnecessary; `__init__` return type (`None`) doesn't need annotating either.
- Not every function needs annotations — prioritize public APIs and error-prone code.

---

## Parting Principle

**Be consistent.** When editing existing code, match its local style even if it differs from a rule above. The goal of a style guide is a shared vocabulary, not rigid uniformity — consistency within a file matters more than enforcing every rule everywhere.

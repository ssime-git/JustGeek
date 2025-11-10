---
layout: default
title: "Run Python in the Browser with Pyodide"
date: 2025-11-09 09:00:00 +0000
categories: [python, web]
tags: [pyodide, webassembly, python]
---

# Run Python in the Browser with Pyodide

This post demonstrates how you can execute Python code directly in the browser using [Pyodide](https://pyodide.org/).

## Interactive Python Cell

Type Python code below and press **Run**. Output appears in the panel under the editor.

<div class="pyodide-cell">
  <div class="pyodide-controls">
    <button type="button" data-pyodide-action="run">▶ Run</button>
    <button type="button" data-pyodide-action="clear">✕ Clear</button>
    <span class="pyodide-status" aria-live="polite"></span>
  </div>
  <textarea class="pyodide-code" spellcheck="false">def greet(name: str) -> str:
    """Return a friendly greeting."""
    return f"Hello, {name}!"

for who in ("Cascade", "JustGeek"):
    print(greet(who))
</textarea>
  <div class="pyodide-output" role="log"></div>
</div>

## Notes

- All computation is performed in your browser; no server round-trips are needed.
- Uses Pyodide {{ PYODIDE_VERSION | default: "0.24.1" }} loaded from the official CDN.
- You can modify the code above and re-run it as many times as you want.

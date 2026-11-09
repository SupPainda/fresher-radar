# fresher-radar: Engineering Guidelines

Rules for any AI assistant (and human) changing this repository. Adapted from the
AI-Powered-Disbursement-Engine guidelines, with project-specific additions.

---

## 1. Zero Unsolicited Changes (Surgical Precision)

- **Strict scope:** modify only the files, functions and lines the request needs.
- **Preserve existing code and comments:** never delete, rewrite or "clean up" unrelated comments,
  docstrings, formatting or working code.
- **No drive-by refactoring:** don't modernize, re-architect or optimize outside the task unless asked.
- **Match existing style:** naming, indentation, structure, comment tone.
- **Orphan cleanup:** remove only imports, variables or functions that *your own* change made unused.

---

## 2. Mandatory Real Test Cases (No Placeholders)

- Every new function, feature or bugfix comes with real tests in `tests/`.
- No trivial assertions (`assert True`, `assert x is not None` without checking properties).
- Each change covers:
  1. **Happy path:** valid input, precise expected output.
  2. **Edge cases:** empty frames, missing optional fields, boundary values (e.g. exactly
     `max_min_years`).
  3. **Failure modes:** invalid config, blocked or erroring sources, `pytest.raises(...)`.
- **Deterministic and isolated:** synthetic frames and JSON fixtures, `tmp_path` for every file the code
  writes. **Tests never touch the network.** A conftest fixture makes `requests.get` and
  `jobspy.scrape_jobs` raise.
- After every change: `.\.venv\Scripts\python.exe -m pytest tests/`. All tests must pass.

---

## 3. Coding Standards

### Architecture
- **Separation of concerns:** config, filters, sources, the runner, storage and logging live in separate
  modules under `jsearch/`. `fetch_jobs.py` is only the CLI.
- **Config-driven:** no thresholds, paths, delays or patterns hard-coded in logic.
  - `config.json` says **what to search for** (sources, titles, locations, filters, delays). It is committed.
  - `.env` says **how this machine runs it** (data dir, parallel, proxies). It is not committed; see
    `.env.example`.
- **Pure transforms:** filter and clean functions return new frames and never mutate shared ones in place.
- **Adding a source:** one module in `jsearch/sources/` implementing `units()` and `fetch()`, registered in
  `source_type()`, a pydantic model for its config entry in `jsearch/config.py`, plus one config entry. A new
  company-board system is just a parser in `ats.py` `PARSERS` + an `ats.endpoints` URL + a source entry. The
  runner provides the delay, block handling, stop, checkpoint and logging.

### Python
- **Type hints** on every function signature and return type (`X | None` style).
- **Error handling:** never use bare `except:`, and never swallow an error.
  - A broad `except Exception` is allowed **only at a worker or source boundary**. There it must record
    the error in the run's error list and log it, then continue with the next unit: report, never
    silently skip.
  - Everywhere else, catch specific exceptions.
- **Logging over printing:** use `logging` via `jsearch.runlog`, which stamps every line in IST, 12-hour
  clock, and writes one log file per run under `<data dir>/logs/`. `print` only for direct CLI output
  (the `--test` preview, `--explain`).
- **Fail fast:** the config is validated with pydantic models at startup, before any scraping. The
  checks that output files are writable also run before scraping.

---

## 4. Project Rules

### Scraping etiquette
- No live scrape while a full run is in progress: the same address hits the same rate limit. Naukri
  blocked us (406) after about 220 requests in 30 minutes.
- Live checks only with `--test`, kept small. Exercise block handling offline with fake sources.
- Respect a site's limits: per-source delays, cooldown and give-up on blocks. No captcha solving and no
  proxy rotation to get past a block.

### LLM API (ranking)
- All LLM calls go through `jsearch/llm.py`: `resolve()` picks provider/key/base URL/model from `.env` +
  `ranking.llm.providers`; `make_backend()` returns an `AnthropicBackend` (official Anthropic SDK) or an
  `OpenAICompatBackend` (OpenAI SDK for Gemini, OpenRouter, xAI, other compatible APIs). Tests use fakes, and
  conftest makes `llm.make_backend` raise, so no test can spend money.
- The LLM is optional: any API problem must fall back to the offline ranking, never fail a run.
- Every live call costs the user money; ask before running one.

### Data and git
- `data/` (CSV, database, raw dump, logs, partial checkpoints) and `.env` are never committed.
- Commit straight to `main`. Don't create branches unless asked.

### Windows environment
- The shells are PowerShell 5.1 (no `&&`) and Git Bash.
- Edit `config.json` with a small Python script file. Never use inline heredocs: they have dropped
  regex backslashes here.
- Write commit messages to a file and use `git commit -F <file>`; PowerShell mangles quoted multi-line
  `-m`.

### Commands
```
.\.venv\Scripts\python.exe fetch_jobs.py --test        # small live check, writes nothing
.\.venv\Scripts\python.exe fetch_jobs.py --refilter    # re-apply filters to saved raw data, offline
.\.venv\Scripts\python.exe fetch_jobs.py --explain "T"  # why a title is kept/rejected, offline
.\.venv\Scripts\python.exe fetch_jobs.py --resume      # continue an interrupted run (--fresh discards it)
.\.venv\Scripts\python.exe -m pytest tests/            # full test suite, offline
```
**Behaviour check for refactors:** `--refilter` on the same raw dump must give the same job URLs before
and after the change.

---

## 5. Execution Workflow

1. **Analyze:** read the relevant files, tests and config before touching code.
2. **Implement:** minimal, focused edits.
3. **Test:** add or update concrete tests for the change.
4. **Verify:** run `pytest tests/`, plus the behaviour check for refactors.

## Response Rules

- Be concise. No preamble, no summaries of what you are about to do.
- Show only changed lines or sections; don't echo whole files.
- Explain only non-obvious reasoning.
- After completing a task, stop. Don't chain into unrequested follow-ups.

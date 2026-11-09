# fresher-radar

A personal job radar for freshers in India. Every evening it collects entry-level postings from job sites and
company career pages, drops the ones that want years of experience, ranks the rest by how well they fit **your**
resume, and emails you the best ones with links.

- **Sources:** Naukri and Indeed, plus the public career-page feeds of 100+ companies hiring in India (Greenhouse,
  Lever, Ashby, Workable, SmartRecruiters). The company feeds are each company's own hiring system, so those jobs
  are real and current.
- **Entry-level filter:** title rules (no Senior/Lead/SDE3...) and the experience the posting asks for. Every
  rejected posting is kept in `rejected.csv` with its reason.
- **Ranking:** every job is scored against a profile built from your resumes, offline and free; optionally an LLM
  (Claude, Gemini, OpenRouter, Grok) rescores the best matches (everything above a fit bar, 20 to 60 jobs) with a
  one-line reason and the resume variant to send.
- **Tracking:** one CSV you open in Excel; type `applied` or `rejected` in the status column and those jobs stop
  showing up. New jobs are marked NEW.
- **Daily email:** the top jobs land in your inbox after the 7 PM run (Windows Task Scheduler).

It runs on your own machine; your resumes, profile, statuses and keys never leave it except for the optional LLM
call to the provider you choose.

## Quick start (Windows)

You need Python 3.11+ (tested on 3.14) and git.

```
git clone https://github.com/SupPainda/fresher-radar.git
cd fresher-radar
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Then:

1. **Your profile.** Either `.\.venv\Scripts\python.exe fetch_jobs.py --build-profile "C:\path\to\your\resumes"`
   (reads your .pdf/.docx resumes, needs an LLM key, see [Ranking](#ranking-jobs-by-fit-to-you)), or copy
   `profile.example.yaml` to `data\profile.yaml` and edit it by hand. Read it over either way.
2. **What you're looking for.** In `config.json`, `titles.search` is what gets searched and `titles.want` /
   `titles.exclude` decide what's kept. `fetch_jobs.py --explain "Some Title"` tells you whether a title passes.
3. **Optional keys in `.env`:** `LLM_API_KEY` for the LLM ranking, `EMAIL_FROM` / `EMAIL_APP_PASSWORD` for the
   email (see [Email](#email-of-the-top-jobs)).
4. **Try it:** `.\.venv\Scripts\python.exe fetch_jobs.py --test` (one search, preview only, writes nothing), then
   a full run: `.\.venv\Scripts\python.exe fetch_jobs.py` (about 75 minutes, mostly Naukri).
5. **Every day:** `powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1` (see
   [Daily run](#daily-run-windows-task-scheduler)).

Results are in `data\jobs.xlsx` (and the same list as `data\jobs.csv`). Everything under `data\` is git-ignored.

## Commands

```
.\.venv\Scripts\python.exe fetch_jobs.py --test                      # one title x one location, preview only
.\.venv\Scripts\python.exe fetch_jobs.py                             # full run -> data\jobs.csv (+ email)
.\.venv\Scripts\python.exe fetch_jobs.py --sources greenhouse lever ashby   # only some sources
.\.venv\Scripts\python.exe fetch_jobs.py --refilter                  # re-apply filters to saved raw data, offline
.\.venv\Scripts\python.exe fetch_jobs.py --refilter --email          # ... and email the top jobs
.\.venv\Scripts\python.exe fetch_jobs.py --explain "SDE3 Backend"    # why a title is kept or rejected, offline
.\.venv\Scripts\python.exe fetch_jobs.py --resume                    # continue an interrupted run (--fresh discards it)
.\.venv\Scripts\python.exe fetch_jobs.py --build-profile "<folder>"  # profile from your resumes (LLM)
```

Tests (offline, a few seconds): `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` once, then
`.\.venv\Scripts\python.exe -m pytest`. A fixture blocks all network access, email and LLM calls, so tests can
never hit a job site or spend money.

## Settings: config.json vs .env

- **`config.json`: what to search for.** Sources and company boards, locations, titles, filters, delays, ranking,
  output columns. Committed, so changes have history.
- **`.env`: how this machine runs it, and your secrets** (see `.env.example`). Never committed.
  - `JSEARCH_DATA_DIR`: folder for `jobs.csv`, `jobs.db`, `jobs_raw.pkl`, logs; default `data/`.
  - `JSEARCH_PARALLEL`: `true` (default) runs the sources side by side; `false` one after another.
  - `JSEARCH_PROXIES`: optional, comma-separated, for the job sites.
  - `LLM_API_KEY` (+ optional `LLM_MODEL`, `LLM_BASE_URL`, `LLM_PROVIDER`), `EMAIL_FROM`, `EMAIL_APP_PASSWORD`,
    `EMAIL_TO`: see below.

  A variable set in the shell wins over `.env`, e.g. `$env:JSEARCH_PARALLEL="false"` for one session.

Every output line is stamped with the time in IST, 12-hour clock (`06:34:12 PM indeed [37/216] ...`). Each run
except `--test` also writes its output to `<data dir>/logs/run_<date>_<time>.log`, so runs never overwrite each
other's logs.

## Sources

| Source | How it's searched | Status |
|---|---|---|
| Naukri, Indeed (via python-jobspy) | every title in `titles.search` x every city in `locations` | Working |
| Greenhouse, Lever, Ashby, Workable, SmartRecruiters (`jsearch/sources/ats.py`) | every open job of each company in that source's `boards`, kept when the location is in India | Working |
| Google Jobs (jobspy) | | Does not work: Google serves a JS-only page to non-browser clients |
| LinkedIn, Glassdoor | | Not used: rate limits, need proxies |

**Job sites vs company boards.** Naukri and Indeed are marketplaces: wide coverage, but also reposts, agency
listings and stale jobs, and they rate-limit scrapers. The board systems are the software companies use to run
their own hiring (the jobs behind a "Careers" page); their feeds are public, so those jobs come straight from the
company. They only cover companies on the list. About 670 Indian and global companies were checked (November
2026: tech and startups, then banks and financial services, hospitals, pharma and health-tech, defence and
space); the 112 with jobs in India are in `config.json`. Many large Indian employers had no public feed on any of
the five, so Naukri is the way to see them: Flipkart, Razorpay, PhonePe, Zomato and its District and Blinkit
(Eternal hires mostly through referrals), banks such as SBI, HDFC Bank, ICICI and IDFC FIRST (own career portals
or recruitment exams), most hospital chains and pharma companies, and defence PSUs such as HAL and BEL. To add a company, put its board id in the right `boards` list (the id is in its careers URL, e.g.
`boards.greenhouse.io/<id>` or `jobs.lever.co/<id>`).

Sources run in parallel, one worker per source, so a run takes as long as the slowest one; within a source,
requests go one at a time with that source's `delay_seconds` between them. Company boards list jobs until
they're filled, often for months, so they use a 180-day age window (`max_age_days`) instead of `search.days`
(going from 60 to 180 days roughly tripled what the boards contribute). SmartRecruiters lists jobs
without their text, so a job's details are fetched only when it is recent, in India and passes the title rules
(at most `ats.max_details_per_board` per company).

**Blocks.** jobspy does not raise when a site refuses it; it returns the rows it got and notes the reason in
`df.attrs["errors"]`. Such units count as `blocked`, the site pauses for `search.on_block.cooldown_minutes`, and
after `give_up_after` blocks in a row it is skipped for the rest of the run while the other sources carry on.
Naukri answers `406 (recaptcha required)` after about 110 searches in 30 minutes from one address, so it asks for
one page per search (`sources.naukri.options.results_wanted: 20`) and waits 18-26 s between searches
(`sources.naukri.delay_seconds`). Don't run tests against a site while a full run is going.

**Which searches pay off.** Raw rows are tagged with the search that found them (`search_title`,
`search_location`). `python overlap.py` reports, per job site, how many entry-level matches only a city search
found, i.e. what dropping the city searches would cost.

## Why was a job filtered out?

- `fetch_jobs.py --explain "SDE3 Backend"` prints `kept` or the rule that rejects it (here `level code: SDE3`).
  Add `--years 3` to say what experience the posting asks for. Offline and instant.
- Every run (and `--refilter`) also writes `rejected.csv` next to `jobs.csv`: every rejected posting with its
  reason, sorted by reason, so leaks and over-filtering are easy to scan. The run log ends with the most common
  reasons.

**How experience is read.** `min years exp` comes from Naukri's `experience_range`, else a years figure in the
title ("Data Engineer (2 to 5 Years)"), else the lowest "N years" in the description that has an "exp..." word
within `filters.experience_context_chars` characters (so "founded 25 years ago" is ignored). Titles matching
`titles.needs_entry_signal` (plain "Solution Architect") with no stated experience pass only with an
`entry_signal` word (associate, junior, graduate, new grad, campus, SDE-1, a trailing "I", ...).

**Generic titles** ("Software Engineer", "SDE", "Full Stack", "Data Scientist", "New Grad") are in
`titles.want_if_entry`: they pass only with one of those entry words or stated experience within
`filters.max_min_years`, so a plain "Software Engineer" at a big company doesn't get in on its title alone. A title
that something in `titles.want` already matches ("Machine Learning Engineer", "Software Engineer, GenAI") isn't
gated this way. `titles.level_codes` drops L3+, SDE3 and level 3+ grades in or at the end of a title ("AI Engineer
4A", "Architect - HR 3", "Software Engineer 3 - Platform"); "Mid-Level" counts as a seniority word. Internships are included when the title also has a
tech word ("AI/ML Intern" yes, "HR Intern" no); the `internship` column flags them.

## Ranking jobs by fit to you

Every run scores each job against **your profile** (`<data dir>/profile.yaml`) and sorts `jobs.csv` by it, NEW
first. Columns: `fit` (0-100), `scored by` (`offline`, or the LLM model that scored it), `why`, and `send resume`
(which of your resume variants fits that job best).

- **Profile:** your skills with weights (3 core, 2 solid, 1 familiar), other spellings, target roles, experience,
  a short summary, and your resume variants with what each focuses on. Build it once from your resumes with
  `--build-profile` (one LLM call), or copy `profile.example.yaml` and edit it. Edit it any time.
- **Offline score (every job, free):** profile skills found in the job's title (counting double), description or
  Naukri skill tags, plus how well the title matches a target role. The scale grows with the profile so a long
  skill list doesn't put everything at 100 (`ranking.*` in config.json).
- **The top list:** every job with offline fit of at least `ranking.top.min_fit` (70), topped up to `min_jobs`
  (20) on a quiet day and capped at `max_jobs` (60) on a busy one. This is what the LLM rescores and the email
  lists.
- **LLM rescoring (optional):** with an API key in `.env`, the top list goes to the LLM in one request, which
  returns a score, a one-line reason and the resume to send for each. Scores are cached per job and profile, so
  later runs only pay for new jobs. Without a key, or on any API problem, the offline ranking stands.
  `ranking.llm.enabled` turns it off.
- **Reserved slots (optional):** if one of your directions scores lower by nature (say PM roles next to a mostly
  engineering skill list), give that resume variant `reserve: 5` in the profile. Its best 5 jobs then always make
  the top list (LLM review and email), taking the last places of the rest. A reserved job's title must contain
  one of that variant's `focus` terms, so an engineering job that merely mentions "product" doesn't qualify.

**Which LLM.** Put any one key in `.env` as `LLM_API_KEY`; the provider is detected from how the key starts:

| Key starts with | Provider | Base URL (default) | Model (default) |
|---|---|---|---|
| `sk-ant-` | Anthropic (Claude, official SDK) | SDK default | `claude-opus-5-5` (about $0.30 for a first 30-job run) |
| `AIza` or `AQ.` | Google Gemini | `generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3.5-flash` (well under a cent) |
| `sk-or-` | OpenRouter | `openrouter.ai/api/v1` | `google/gemini-3.5-flash` |
| `xai-` | xAI (Grok) | `api.x.ai/v1` | the first model your key lists |

The defaults live in `config.json` under `ranking.llm.providers`; `LLM_MODEL`, `LLM_BASE_URL` and `LLM_PROVIDER`
in `.env` override them. Any other OpenAI-compatible API works with `LLM_BASE_URL` + `LLM_MODEL` (provider
`openai-compatible`). Model names change often: if one is rejected, the error lists the models your key can use.
Gemini, OpenRouter and Grok go through their OpenAI-compatible endpoints with a strict JSON schema; if an endpoint
refuses that, the request is retried once in plain JSON mode and the answer is validated locally.

**Model fallback.** There are no retries with backoff. If a model fails (not found, no quota, a server error,
no answer within `ranking.llm.timeout_seconds`, or an answer in the wrong format), the next model is tried
straight away: `LLM_MODEL` first, then the provider's `model`, then its `fallback_models`, in order. A rejected
key or no network stops at once. The Gemini list holds the models that answered a test call with a free-tier key.

Resume text leaves your machine only for `--build-profile` and the top-list rescoring, to the provider you chose.

## Tracking (jobs.db)

Every job that passes the filters is remembered in `jobs.db` (SQLite). Each run rewrites the job list with the
current matches, `NEW` first, plus `first seen`, `status` and `notes` columns, in two files:

- **`jobs.xlsx`, the one to open:** links you can click, a frozen header row with filters, NEW rows in green,
  fit coloured red to green, and a status drop-down (`output.status_choices`; other words still work).
- **`jobs.csv`:** the same list as plain text, for scripts or other tools.

Edit statuses and notes in either; the next run reads both, the one you saved last read last, so the other
(still as the previous run wrote it) can't undo your edit.

- Type a status (`applied`, `rejected`, `interested`, ...) or a note, save, and the next run copies it into the
  database before rewriting the list.
- Jobs marked `applied` or `rejected` (`output.hide_statuses`) no longer appear in the CSV; they stay in the
  database.
- A job is identified by normalised title + company, so a repost under a new URL keeps its status.
- Every run also writes a dated copy, `data\daily\jobs_<date>.csv` (a second run that day replaces it). It's a
  record: statuses are only read from `jobs.csv`.
- `--test` never touches the database or the CSVs.

Close `jobs.xlsx` / `jobs.csv` in Excel before running. A normal run checks this at startup and stops before
scraping if one is locked; if it gets opened mid-run, the results go to a timestamped copy
(`jobs_<date>_<time>.xlsx`) instead (statuses typed into that copy are not read back). `config.json` is validated at startup too: a typo'd key, a broken regex or a
bad value stops the run with a list of every problem and where it is.

## Email of the top jobs

After every scraping run, the top list is emailed to you: fit, title with its link, company, location, the
reason and which resume to send. Jobs first seen today come first under **New today**, the rest under **Also on
your top list**. On a day when no new job makes the list, the email is a single line instead (saying how many new
jobs scored lower). It uses the final scores, so a job the LLM marked below the bar drops out; a closing line
counts the other jobs with fit `ranking.top.more_fit` (60) or more that are waiting in `jobs.xlsx`. `--refilter` sends only with `--email`; `--test` never sends.

Set it up in `.env` (see `.env.example`): `EMAIL_FROM`, `EMAIL_APP_PASSWORD` and optionally `EMAIL_TO`. For Gmail
the password is an **app password**: turn on 2-step verification, then create one at
<https://myaccount.google.com/apppasswords>. Without these, or if sending fails, the run logs one line and
finishes normally. The server is `email.smtp_host` / `smtp_port` in `config.json` (Gmail by default);
`email.enabled: false` turns it off.

## Daily run (Windows Task Scheduler)

```
powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1             # daily at 7:00 PM
powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -At 8:30PM  # another time
powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Remove     # stop it
```

The task "JSearch daily" runs `scripts\run_daily.ps1`, i.e. `fetch_jobs.py --scheduled`, as you, while you're
signed in. The laptop must be on: if it was asleep it is woken, if it was off the run starts as soon as it's back
on. It runs on battery too. A full run takes about 75 minutes (mostly Naukri), then the email arrives.

`--scheduled` differs from a plain run in two ways:
- an unfinished run started **today** is continued; an older one is discarded (a plain run stops and asks);
- if `jobs.csv` is open in Excel the run doesn't stop: the list goes to a timestamped copy, and the email says so.

Output from before logging starts (e.g. a config error) goes to `data\logs\scheduler.log`.

## Stopping and resuming

A run saves each finished search or board to `<data dir>/partial/` as it goes. If it is interrupted (Ctrl+C, a
crash, a closed terminal) or some units were blocked or failed, the progress stays there:

- `fetch_jobs.py --resume` continues it: units already done are skipped, blocked/failed ones are retried. Without
  `--sources`, this resumes every source, not just the ones that were left over.
- `fetch_jobs.py --fresh` discards it and starts over.
- A plain run refuses to start while unfinished progress exists, so nothing is thrown away silently.

A complete run deletes `partial/`. An interrupted run still saves what it has, so until it is resumed, `jobs.csv`
shows only the sources' partial results.

## Code layout and adding a source

| Path | Role |
|---|---|
| `fetch_jobs.py` | CLI and the run flow (scrape -> filter -> database -> rank -> CSV -> email) |
| `jsearch/config.py` | loads and validates `config.json` + `.env` |
| `jsearch/filters.py` | experience parsing, title rules, `clean()` |
| `jsearch/runner.py` | one worker per source: delays, block cooldown/give-up, Ctrl+C, checkpoint, stats |
| `jsearch/sources/` | source types: `jobspy_site.py` (Naukri, Indeed), `ats.py` (the five board systems) |
| `jsearch/store.py` | SQLite memory of statuses and cached LLM scores |
| `jsearch/profile.py` | your profile: load/validate, read resumes |
| `jsearch/ranking.py` | offline fit score |
| `jsearch/llm.py` | the LLM: provider detection, model fallback, build a profile, rescore the top jobs |
| `jsearch/mailer.py` | the top-jobs email |
| `jsearch/runlog.py` | IST-stamped logging, one file per run |
| `scripts/` | Task Scheduler: `run_daily.ps1`, `install_schedule.ps1` |
| `tests/` | offline test suite |

Each entry in `config.json` `sources` has a `type`, an optional `delay_seconds`, and its own settings
(`options` passed to jobspy, or `boards` for a board system). **To add a source:**

1. Another jobspy site (e.g. LinkedIn): add `"linkedin": {"type": "jobspy", "options": {...}}`. No code.
2. Another company on a supported board system: add its id to that source's `boards`. No code.
3. Another company-board system: a `parse_<name>(slug, payload)` in `ats.py` added to `PARSERS`, its URL in
   `ats.endpoints`, and a `"<name>": {"type": "ats", "boards": [...]}` source, plus tests.
4. A new kind of source: add `jsearch/sources/<kind>.py` with
   `units(name, src, cfg, test) -> list[Unit]` (the work: one search, one board, ...) and
   `fetch(name, unit, src, cfg) -> Result` (do one unit; raise on failure, set `refusal` when blocked).
   Register it in `source_type()` in `jsearch/sources/__init__.py`, add a pydantic model for its config
   entry in `jsearch/config.py` (to the `sources` union), add tests, add the config entry.

`CLAUDE.md` has the engineering rules for changes (tests for everything, config-driven, no network in tests).

## Known limitations

- **Entry-level is unverified for about half the rows** (mostly Indeed): no experience stated means "no senior
  signal found", not "fresher-friendly".
- **Filter tweaks still open.** The bare `product manager` pattern lets some non-technical roles through. Analyst
  roles are excluded entirely (`titles.exclude`).
- **Coverage gaps.** "Technical Account Manager" and "Associate Solutions Architect" return almost nothing, and
  "Graduate Engineer Trainee (AI/ML)" returns mostly non-software GET roles.
- **Windows-first.** The daily schedule uses Windows Task Scheduler; the rest is plain Python and should run
  elsewhere, but only Windows is tested.

## Be a good citizen

This never logs into anything: it reads public listings at a slow, fixed pace, backs off when a
site says no, and never solves captchas or rotates proxies to get around a block. Keep it that way, and keep the
delays in `config.json` if you run it yourself. It is meant for one person's job search, not bulk collection.

## License

MIT, see [LICENSE](LICENSE).

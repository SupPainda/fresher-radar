"""Runs every configured source, one worker thread each, and gathers their rows.

The worker loop here is shared by all sources: polite delay between units, block cooldown and
give-up, the stop event (Ctrl+C), stats, logging and row tags. Sources only say what to do
(units) and how to do one unit (fetch); see jsearch.sources.
"""
import json
import logging
import os
import random
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd

from jsearch.runlog import IST
from jsearch.sources import Unit, source_type

logger = logging.getLogger("jsearch.runner")
Outcome = tuple[list[pd.DataFrame], dict, list[str]]  # (frames, {source: stats}, errors)


class Checkpoint:
    """Progress saved while a run goes, so an interrupted run can be resumed (fetch_jobs.py --resume).

    Append-only in its own folder: each finished unit's rows go to <n>.pkl and a line
    {"key": unit key, "file": "<n>.pkl" | null} to done.jsonl. A unit is "done" only when it finished
    without a block or an error, so a resume retries those. Workers call add() concurrently.
    """
    DONE, STARTED = "done.jsonl", "started.json"

    def __init__(self, folder: Path):
        self.folder = folder
        self._lock = threading.Lock()
        self.done: set[str] = set()
        self._n = 0

    @classmethod
    def start(cls, folder: Path, args: list[str]) -> "Checkpoint":
        folder.mkdir(parents=True, exist_ok=True)
        now = datetime.now(IST)
        (folder / cls.STARTED).write_text(json.dumps({"at": f"{now:%Y-%m-%d %I:%M %p} IST", "date": f"{now:%Y-%m-%d}",
                                                      "args": args}), encoding="utf-8")
        return cls(folder)

    @classmethod
    def started_info(cls, folder: Path) -> dict:
        try:
            return json.loads((folder / cls.STARTED).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"at": "an unknown time", "args": []}

    @classmethod
    def load(cls, folder: Path) -> tuple["Checkpoint", list[pd.DataFrame]]:
        """The saved progress: the checkpoint (to keep adding to) and the rows of the units already done."""
        cp, frames = cls(folder), []
        path = folder / cls.DONE
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:  # a line cut short by a crash: that unit simply runs again
                continue
            if entry["file"]:
                frames.append(pd.read_pickle(folder / entry["file"]))
            cp.done.add(entry["key"])
        cp._n = len(lines)
        return cp, frames

    def add(self, key: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._n += 1
            file = None
            if not df.empty:
                file = f"{self._n}.pkl"
                tmp = self.folder / f"{file}.tmp"
                df.to_pickle(tmp)
                os.replace(tmp, self.folder / file)  # the line below never points at a half-written file
            with open(self.folder / self.DONE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "file": file}) + "\n")
            self.done.add(key)

    def finish(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)


def run_source(name: str, units: list[Unit], fetch: Callable, src: dict, cfg: dict,
               stop: threading.Event, checkpoint: Checkpoint | None = None) -> Outcome:
    """Every unit of one source, one after another with that source's delay between them.

    Runs on its own thread and shares nothing while running (the checkpoint has its own lock).
    """
    frames, st, errors = [], {"ok": 0, "empty": 0, "error": 0, "blocked": 0, "rows": 0}, []
    block, blocked_in_a_row = cfg["search"]["on_block"], 0
    delay = src.get("delay_seconds", cfg["search"]["delay_seconds"])
    for i, unit in enumerate(units, 1):
        if stop.is_set():
            logger.info(f"{name:10} stopped at {i - 1}/{len(units)}")
            break
        tag = f"{name:10} [{i}/{len(units)}] {unit.label}"
        try:
            result = fetch(name, unit, src, cfg)
        except KeyboardInterrupt:  # only reaches a worker in sequential mode; keep what we have
            stop.set()
            logger.info(f"{name:10} interrupted at {i}/{len(units)}")
            break
        except Exception as e:  # source boundary: record, log, continue with the next unit
            st["error"] += 1
            errors.append(f"{name} | {unit.label}: {type(e).__name__}: {str(e)[:150]}")
            logger.info(f"{tag}: ERROR {type(e).__name__}")
        else:
            df, note = result.df.assign(**unit.tags), f" ({result.note})" if result.note else ""
            st["rows"] += len(df)
            if not df.empty:  # a blocked unit can still have its first pages: keep them
                frames.append(df)
            if result.refusal:
                st["blocked"] += 1
                blocked_in_a_row += 1
                errors.append(f"{name} | {unit.label}: BLOCKED after {len(df)} rows: {result.refusal[:120]}")
                if blocked_in_a_row >= block["give_up_after"]:
                    logger.info(f"{tag}: {len(df)} rows, BLOCKED {blocked_in_a_row}x in a row - "
                                f"skipping the remaining {len(units) - i} {name} units this run")
                    break
                logger.info(f"{tag}: {len(df)} rows, BLOCKED - pausing {name} for {block['cooldown_minutes']} min")
                stop.wait(block["cooldown_minutes"] * 60)  # instead of the normal delay
                continue
            blocked_in_a_row = 0
            st["empty" if df.empty else "ok"] += 1
            if checkpoint:
                checkpoint.add(unit.key, df)
            logger.info(f"{tag}: {len(df)} rows{note}")
        if i < len(units):
            try:
                stop.wait(random.uniform(*delay))  # a sleep that stop cuts short
            except KeyboardInterrupt:
                stop.set()
    return frames, {name: st}, errors


def done_future(task: Callable[[], Outcome]) -> Future:
    """Run task now and wrap the outcome in a Future, so sequential mode shares collect() with parallel."""
    f = Future()
    try:
        f.set_result(task())
    except KeyboardInterrupt:
        raise
    except Exception as e:  # worker boundary: handed to collect(), which records it
        f.set_exception(e)
    return f


def scrape(cfg: dict, names: list[str], test: bool, checkpoint: Checkpoint | None = None,
           resumed: list[pd.DataFrame] | None = None) -> tuple[pd.DataFrame | None, bool]:
    """Run the named sources (parallel unless JSEARCH_PARALLEL=false).

    With a checkpoint, units already in checkpoint.done are skipped and `resumed` (their saved rows)
    is included in the result. Returns (rows or None when there are none, whether every unit is done).
    """
    stop = threading.Event()
    tasks, all_keys = {}, set()
    for name in names:
        src = cfg["sources"][name]
        kind = source_type(name, src)
        units = kind.units(name, src, cfg, test)
        all_keys |= {u.key for u in units}
        if checkpoint and checkpoint.done:
            todo = [u for u in units if u.key not in checkpoint.done]
            logger.info(f"{name:10} resuming: {len(units) - len(todo)} of {len(units)} units already done")
            units = todo
        tasks[name] = partial(run_source, name, units, kind.fetch, src, cfg, stop, checkpoint)

    frames, stats, errors = list(resumed or []), {}, []

    def collect(name: str, future: Future) -> None:
        try:
            fr, st, er = future.result()
        except Exception as e:  # a bug in one worker must not lose the others' results
            errors.append(f"{name}: worker crashed: {type(e).__name__}: {str(e)[:150]}")
            return
        frames.extend(fr)
        stats.update(st)
        errors.extend(er)

    started = time.monotonic()
    if not cfg["env"]["parallel"]:  # one source after another, e.g. to debug a single source
        try:
            for name, task in tasks.items():
                collect(name, done_future(task))
        except KeyboardInterrupt:
            logger.info("Interrupted - processing what was collected so far.")
    else:
        with ThreadPoolExecutor(max_workers=len(tasks) or 1) as pool:
            futures = {pool.submit(task): name for name, task in tasks.items()}
            pending = set(futures)
            try:
                while pending:  # short waits so Ctrl+C reaches this (main) thread
                    _, pending = wait(pending, timeout=0.5)
            except KeyboardInterrupt:
                logger.info("Interrupted - letting each source finish its current request...")
                stop.set()
            for future, name in futures.items():  # blocks until every worker has returned
                collect(name, future)
    logger.info("")
    logger.info(f"scraped in {(time.monotonic() - started) / 60:.1f} min")

    logger.info("")
    logger.info("=== per-source summary ===")
    for name, st in stats.items():
        logger.info(f"{name}: {st}")
    for e in errors:
        logger.info(f"ERR {e}")
    complete = checkpoint is not None and all_keys <= checkpoint.done
    if not frames:
        logger.info("No results at all - nothing to save.")
        return None, complete
    return pd.concat(frames, ignore_index=True), complete


def merge_with_saved(fresh: pd.DataFrame, path: Path) -> pd.DataFrame:
    """A partial run replaces only the sources it fetched; other sources keep their saved rows."""
    if not Path(path).exists():
        return fresh
    saved = pd.read_pickle(path)
    kept = saved[~saved["site"].isin(fresh["site"].unique())]
    return pd.concat([kept, fresh], ignore_index=True)

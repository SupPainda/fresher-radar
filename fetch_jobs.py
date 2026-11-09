"""Fetch entry-level job postings from the configured sources (logged out) into one CSV.

Sources: job sites via python-jobspy (Naukri, Indeed) and company boards on Greenhouse/Lever/Ashby;
see jsearch/sources. What to search for lives in config.json, how this machine runs it in .env.

  python fetch_jobs.py --test              # one title x one location, preview only, writes nothing
  python fetch_jobs.py                     # full run -> CSV + raw dump
  python fetch_jobs.py --sources greenhouse lever ashby   # only some of the configured sources
  python fetch_jobs.py --refilter          # re-apply filters to the saved raw dump, no scraping
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from jsearch import llm, mailer, store
from jsearch.config import ConfigError, load_config
from jsearch.filters import clean, min_years, reject_reason
from jsearch.outputs import OutputLocked, check_writable, save_csv
from jsearch.profile import Profile, ProfileError, load_profile, read_resumes, write_profile
from jsearch.ranking import reserves, score
from jsearch.runlog import IST, now_ist, start_log
from jsearch.runner import Checkpoint, merge_with_saved, scrape

logger = logging.getLogger("jsearch.cli")


def main() -> None:
    try:
        cfg = load_config()
    except ConfigError as e:  # before logging starts: a plain message, no traceback
        print(e, file=sys.stderr)
        sys.exit(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="one title x one location; preview only, writes nothing")
    ap.add_argument("--refilter", action="store_true", help=f"skip scraping; re-filter saved {cfg['paths']['raw']}")
    all_sources = list(cfg["sources"])
    ap.add_argument("--sources", nargs="+", choices=all_sources, default=all_sources,
                    help="subset of configured sources (default: all)")
    unfinished = ap.add_mutually_exclusive_group()
    unfinished.add_argument("--resume", action="store_true", help="continue an unfinished (interrupted) run")
    unfinished.add_argument("--fresh", action="store_true", help="discard an unfinished run's progress and start over")
    unfinished.add_argument("--scheduled", action="store_true",
                            help="for Task Scheduler: continue an unfinished run started today, discard an older one, "
                                 "and don't stop when jobs.csv is open in Excel (a dated copy is written instead)")
    ap.add_argument("--email", action="store_true",
                    help="with --refilter: also email the top jobs (scraping runs always do, when set up in .env)")
    ap.add_argument("--explain", metavar="TITLE", help="say whether a job title passes the filters and why; offline")
    ap.add_argument("--years", type=int, help="with --explain: the experience the posting asks for")
    ap.add_argument("--build-profile", metavar="FOLDER", type=Path,
                    help="build your job-matching profile from the .pdf/.docx resumes in FOLDER (uses your LLM key)")
    args = ap.parse_args()

    if args.explain:
        print(explain(args.explain, args.years, cfg))  # CLI output, no run, no log
        return
    if args.build_profile:
        start_log(None)
        sys.exit(build_profile_cli(args.build_profile, cfg))

    log_file = start_log(None if args.test else cfg["env"]["data_dir"] / "logs")  # --test writes nothing
    logger.info(f"=== run started {now_ist()} | {' '.join(sys.argv[1:]) or 'full run'}"
                + (f" | log: {log_file}" if log_file else ""))
    try:
        run(args, cfg)
    except (OutputLocked, UnfinishedRun, ProfileError) as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception:  # top-level boundary: the traceback goes into the run's log, then exit non-zero
        logger.exception("run failed")
        sys.exit(1)
    finally:
        logger.info(f"=== run ended {now_ist()}")


def build_profile_cli(folder: Path, cfg: dict) -> int:
    """--build-profile: resumes -> LLM -> profile.yaml (never overwriting one). Returns the exit code."""
    try:
        texts = read_resumes(folder)
        logger.info(f"read {len(texts)} resumes: {', '.join(texts)}")
        profile = llm.build_profile(texts, cfg, llm.make_backend(cfg))
    except (ProfileError, llm.LLMUnavailable) as e:
        logger.error(f"could not build the profile: {e}")
        return 1
    path = write_profile(profile, cfg["paths"]["profile"])
    logger.info(f"wrote {path}: {len(profile.skills)} skills, {len(profile.target_roles)} target roles, "
                f"resumes {[r.label for r in profile.resumes]}. Review and edit it; it is used on every run.")
    if path != cfg["paths"]["profile"]:
        logger.info(f"{cfg['paths']['profile']} already existed, so it was left unchanged; "
                    f"compare the two and keep the one you want as {cfg['paths']['profile'].name}.")
    return 0


def rank(df: pd.DataFrame, profile: Profile | None, cfg: dict, con, use_llm: bool) -> pd.DataFrame:
    """Add fit/why/send/scored_by: offline for every job, the LLM for the top ones when use_llm."""
    if profile is None:
        logger.info(f"no profile at {cfg['paths']['profile']}, so jobs are not ranked by fit. Create one with "
                    f"--build-profile <resume folder>, or copy profile.example.yaml there and edit it.")
        return df
    df = score(df, profile, cfg)
    if use_llm and len(df):
        df = llm.rescore(df, profile, cfg, con)
    return df


def explain(title: str, years: int | None, cfg: dict) -> str:
    """'kept' or 'rejected: <reason>' for one title; years default to any figure stated in the title."""
    if years is None:
        years = min_years({"title": title}, cfg["filters"]["experience_context_chars"])
    reason = reject_reason({"title": title, "min_years": years}, cfg)
    stated = "no experience stated" if years is None else f"{years} years"
    return f"{title!r} ({stated}): " + ("kept" if reason is None else f"rejected: {reason}")


class UnfinishedRun(RuntimeError):
    """A previous run's progress exists and the user didn't say whether to resume or discard it."""


def open_checkpoint(folder: Path, args: argparse.Namespace) -> tuple[Checkpoint, list[pd.DataFrame]]:
    """The checkpoint for this run: resumed, fresh, or refused when an unfinished run would be lost."""
    if folder.exists():
        info = Checkpoint.started_info(folder)
        if getattr(args, "scheduled", False):  # unattended: today's progress is worth keeping, an older run's isn't
            today = str(pd.Timestamp.now(tz=IST).date())
            if info.get("date") == today:
                args.resume = True
            else:
                logger.info(f"--scheduled: the unfinished run from {info['at']} is not from today")
                args.fresh = True
        if args.resume:
            checkpoint, frames = Checkpoint.load(folder)
            logger.info(f"resuming the run started {info['at']}: {len(checkpoint.done)} units already done")
            return checkpoint, frames
        if not args.fresh:
            raise UnfinishedRun(f"An unfinished run (started {info['at']}, {' '.join(info['args']) or 'full run'}) "
                                f"left progress in {folder}. Run with --resume to continue it, "
                                f"or --fresh to discard it.")
        logger.info(f"--fresh: discarding the unfinished run started {info['at']}")
        Checkpoint(folder).finish()
    elif args.resume:
        logger.info("--resume: no unfinished run found, starting a normal run")
    return Checkpoint.start(folder, sys.argv[1:]), []


def run(args: argparse.Namespace, cfg: dict) -> None:
    out_cfg, paths = cfg["output"], cfg["paths"]
    cols = out_cfg["columns"]
    con = None
    profile = load_profile(paths["profile"])  # a broken profile fails now, not after an hour of scraping
    if not args.test:
        try:
            check_writable(paths["csv"], paths["rejected"])  # before an hour of scraping, not after
        except OutputLocked as e:
            if not getattr(args, "scheduled", False):
                raise
            logger.warning(f"{e} Scheduled run: carrying on; the results go to a dated copy instead.")
        paths["db"].parent.mkdir(parents=True, exist_ok=True)
        # read status edits from the CSV before anything can overwrite it
        con = store.connect(paths["db"])
        edited = store.import_csv_edits(con, paths["csv"], cols["title"], cols["company"])
        if edited:
            logger.info(f"read {edited} status/notes edits from {paths['csv']}")

    if args.refilter:
        raw = pd.read_pickle(paths["raw"])
    else:
        checkpoint, resumed = None, []
        if not args.test:  # --test never checkpoints
            checkpoint, resumed = open_checkpoint(paths["partial"], args)
        raw, complete = scrape(cfg, args.sources, args.test, checkpoint, resumed)
        if raw is None:
            if complete:
                checkpoint.finish()
            return
        if not args.test:
            raw = merge_with_saved(raw, paths["raw"])
            raw.to_pickle(paths["raw"])  # so filters can be tuned without re-scraping
            if complete:
                checkpoint.finish()
            else:
                logger.info(f"Some units are not done (interrupted, blocked or failed); their progress is kept in "
                            f"{paths['partial']}. Run with --resume to do just those, or --fresh to start over.")

    df, counts, rejected = clean(raw, cfg)
    logger.info("")
    logger.info(" -> ".join(f"{k} {v}" for k, v in counts.items()))
    logger.info(f"by source: {df['site'].value_counts().to_dict()} | internships: {int(df['internship'].sum())}")

    if args.test:
        df = rank(df, profile, cfg, None, use_llm=False)  # offline only: --test writes and spends nothing
        out = df.reindex(columns=list(cols)).rename(columns=cols)
        if "fit" in df:
            out = out.sort_values(cols["fit"], ascending=False)
        pd.set_option("display.width", 250, "display.max_colwidth", 60)
        print(out.head(cfg["search"]["test"]["preview_rows"]).to_string())  # CLI output, not a log line
        return

    today = str(pd.Timestamp.now().date())
    df = store.record(con, df, today, touch=not args.refilter)
    hidden = df["status"].isin(out_cfg["hide_statuses"])
    df = df[~hidden].assign(new=lambda d: d["first_seen"] == today)
    df = rank(df, profile, cfg, con, use_llm=True)  # after hiding applied/rejected: no Claude spend on those
    out = (df.reindex(columns=["new"] + list(cols) + ["first_seen", "status", "notes"])  # no profile: fit empty
             .rename(columns=cols | {"first_seen": "first seen"}))
    out[cols["date_posted"]] = out[cols["date_posted"]].dt.date
    out["new"] = out["new"].map({True: "NEW", False: ""})
    out = out.sort_values(["new", cols["fit"], cols["date_posted"]], ascending=False, na_position="last")
    saved = save_csv(out, paths["csv"])
    logger.info(f"saved {saved}: {len(out)} rows, {int(df['new'].sum())} new, {int(hidden.sum())} hidden "
                f"({'/'.join(out_cfg['hide_statuses'])}) | database: {store.status_counts(con)}")
    note = (f"{paths['csv'].name} was open in Excel, so this run's list was saved to {saved.name}; statuses typed "
            f"into that copy are not read back." if saved != paths["csv"] else "")
    daily = paths["daily"] / f"jobs_{today}.csv"  # one per day; a second run that day replaces it
    daily.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"saved {save_csv(out, daily)} (today's copy; edit statuses in {paths['csv'].name})")
    rej = rejected.reindex(columns=["reason", "title", "company", "site", "min_years", "date_posted", "job_url"])
    saved = save_csv(rej, paths["rejected"])
    kinds = rejected["reason"].str.split(":").str[0].str.replace(r"\d+(\.\d+)?", "N", regex=True)  # group 3+/5+ years
    top = kinds.value_counts().head(5).to_dict()
    logger.info(f"saved {saved}: {len(rej)} rejected postings with the reason | most common: {top}")
    if not args.refilter or getattr(args, "email", False):
        mailer.send_top(df, cfg, now_ist(), note=note, reserve=reserves(profile))


if __name__ == "__main__":
    main()

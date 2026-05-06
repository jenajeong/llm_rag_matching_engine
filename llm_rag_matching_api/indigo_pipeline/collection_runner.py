import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from indigo_pipeline.config import DJANGO_PROJECT_DIR, LOG_DIR


@dataclass(frozen=True)
class PipelineStep:
    name: str
    module: str
    description: str


STEPS = {
    "kipris-key": PipelineStep(
        name="kipris-key",
        module="indigo_pipeline.collection.patent_API_KEY",
        description="Apply or refresh KIPRIS portal API key settings.",
    ),
    "project-excel": PipelineStep(
        name="project-excel",
        module="indigo_pipeline.collection.project_EXCEL",
        description="Download NTIS project Excel files and convert them to JSON.",
    ),
    "project-collect": PipelineStep(
        name="project-collect",
        module="indigo_pipeline.collection.project_collection",
        description="Collect and merge project data from Indigo DB and NTIS JSON.",
    ),
    "project-extra": PipelineStep(
        name="project-extra",
        module="indigo_pipeline.collection.extra_project_collection",
        description="Apply extra project enrichment logic.",
    ),
    "article-collect": PipelineStep(
        name="article-collect",
        module="indigo_pipeline.collection.article_collection",
        description="Collect article metadata from Indigo DB and EBSCO.",
    ),
    "article-extra": PipelineStep(
        name="article-extra",
        module="indigo_pipeline.collection.extra_article_collection",
        description="Apply extra article professor mapping and enrichment logic.",
    ),
    "patent-collect": PipelineStep(
        name="patent-collect",
        module="indigo_pipeline.collection.patent_collection",
        description="Collect patent data from Indigo DB and KIPRIS.",
    ),
    "patent-extra": PipelineStep(
        name="patent-extra",
        module="indigo_pipeline.collection.extra_patent_collection",
        description="Apply extra patent enrichment logic.",
    ),
    "filter-article": PipelineStep(
        name="filter-article",
        module="indigo_pipeline.filtering.article_filtering",
        description="Filter article data into train JSON.",
    ),
    "filter-patent": PipelineStep(
        name="filter-patent",
        module="indigo_pipeline.filtering.patent_filtering",
        description="Filter patent data into train JSON.",
    ),
    "filter-project": PipelineStep(
        name="filter-project",
        module="indigo_pipeline.filtering.project_filtering",
        description="Filter project data into train JSON.",
    ),
}

COLLECT_STEPS = [
    "project-excel",
    "project-collect",
    "project-extra",
    "article-collect",
    "article-extra",
    "patent-collect",
    "patent-extra",
]
FILTER_STEPS = ["filter-article", "filter-patent", "filter-project"]
PROFILE_STEPS = {
    "all": [*COLLECT_STEPS, *FILTER_STEPS],
    "collect": COLLECT_STEPS,
    "filter": FILTER_STEPS,
    "article": ["article-collect", "article-extra", "filter-article"],
    "patent": ["patent-collect", "patent-extra", "filter-patent"],
    "project": ["project-excel", "project-collect", "project-extra", "filter-project"],
}


def _split_step_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _validate_step_names(step_names: list[str]) -> None:
    invalid = [name for name in step_names if name not in STEPS]
    if invalid:
        valid = ", ".join(STEPS)
        raise ValueError(f"Unknown step(s): {', '.join(invalid)}. Valid steps: {valid}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Indigo collection and filtering jobs sequentially for server automation."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_STEPS),
        default="all",
        help="Predefined step sequence to run.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated explicit steps. Overrides --profile.",
    )
    parser.add_argument(
        "--skip",
        default=None,
        help="Comma-separated steps to remove from the selected sequence.",
    )
    parser.add_argument(
        "--include-kipris-key",
        action="store_true",
        help="Run the KIPRIS portal key-update step before the selected sequence.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remaining steps even when one step fails.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the sequence without executing it.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child steps.")
    parser.add_argument("--cwd", default=str(DJANGO_PROJECT_DIR), help="Working directory for child steps.")
    parser.add_argument("--resume", action="store_true", help="Skip steps already marked successful in state file.")
    parser.add_argument("--reset-state", action="store_true", help="Clear previous runner state before executing.")
    parser.add_argument("--retries", type=int, default=0, help="Retry count per failed step.")
    parser.add_argument("--retry-delay", type=int, default=60, help="Seconds to wait between retries.")
    parser.add_argument("--step-timeout", type=int, default=None, help="Maximum seconds per step.")
    parser.add_argument("--state-file", default=str(LOG_DIR / "collection_runner_state.json"))
    parser.add_argument("--log-dir", default=str(LOG_DIR / "collection_runner"))
    parser.add_argument("--lock-file", default=str(LOG_DIR / "collection_runner.lock"))
    parser.add_argument("--wait-lock", action="store_true")
    parser.add_argument("--lock-timeout", type=int, default=0, help="Seconds to wait for lock when --wait-lock is set. 0 means forever.")
    parser.add_argument("--stale-lock-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--list-steps", action="store_true", help="Print available steps and exit.")
    return parser


def resolve_steps(args: argparse.Namespace) -> list[PipelineStep]:
    selected_names = _split_step_names(args.steps) or list(PROFILE_STEPS[args.profile])
    skip_names = set(_split_step_names(args.skip))

    if args.include_kipris_key and "kipris-key" not in selected_names:
        selected_names.insert(0, "kipris-key")

    _validate_step_names(selected_names)
    _validate_step_names(list(skip_names))

    return [STEPS[name] for name in selected_names if name not in skip_names]


@contextmanager
def runner_lock(args: argparse.Namespace):
    lock_file = Path(args.lock_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    while True:
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "started_at": datetime.now().isoformat(),
                        "command": sys.argv,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            break
        except FileExistsError:
            stale = False
            try:
                stale = time.time() - lock_file.stat().st_mtime > args.stale_lock_seconds
            except OSError:
                stale = False
            if stale:
                print(f"Removing stale lock: {lock_file}")
                lock_file.unlink(missing_ok=True)
                continue
            if not args.wait_lock:
                raise RuntimeError(f"Another collection runner appears to be running: {lock_file}")
            if args.lock_timeout and time.time() - started > args.lock_timeout:
                raise TimeoutError(f"Timed out waiting for lock: {lock_file}")
            print(f"Waiting for lock: {lock_file}")
            time.sleep(10)
    try:
        yield
    finally:
        lock_file.unlink(missing_ok=True)


def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"steps": {}}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup_file = state_file.with_suffix(f".corrupt-{int(time.time())}.json")
        state_file.replace(backup_file)
        print(f"State file was corrupt. Moved it to {backup_file}")
        return {"steps": {}}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = state_file.with_suffix(f"{state_file.suffix}.tmp")
    tmp_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(state_file)


def is_step_successful(state: dict, step: PipelineStep) -> bool:
    return state.get("steps", {}).get(step.name, {}).get("status") == "success"


def mark_step(
    state_file: Path,
    state: dict,
    step: PipelineStep,
    status: str,
    returncode: int | None,
    attempt: int,
    log_file: Path,
) -> None:
    state.setdefault("steps", {})[step.name] = {
        "status": status,
        "returncode": returncode,
        "attempt": attempt,
        "log_file": str(log_file),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_state(state_file, state)


def run_step_once(
    step: PipelineStep,
    python_executable: str,
    cwd: Path,
    log_dir: Path,
    timeout: int | None,
    attempt: int,
) -> tuple[int, Path]:
    command = [python_executable, "-m", step.module]
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{step.name}_attempt{attempt}.log"

    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {step.name}")
    print(f"  module: {step.module}")
    print(f"  log: {log_file}")

    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"command: {' '.join(command)}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"started_at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTimed out after {timeout} seconds.\n")
            returncode = 124

    if returncode == 0:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] DONE  {step.name}")
    else:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] FAIL  {step.name} ({returncode})")
    return returncode, log_file


def run_step(
    step: PipelineStep,
    args: argparse.Namespace,
    cwd: Path,
    log_dir: Path,
    state: dict,
    state_file: Path,
) -> int:
    max_attempts = args.retries + 1
    last_returncode = 1
    last_log_file = log_dir / f"{step.name}.log"

    for attempt in range(1, max_attempts + 1):
        returncode, log_file = run_step_once(step, args.python, cwd, log_dir, args.step_timeout, attempt)
        last_returncode = returncode
        last_log_file = log_file
        if returncode == 0:
            mark_step(state_file, state, step, "success", returncode, attempt, log_file)
            return 0

        mark_step(state_file, state, step, "failed", returncode, attempt, log_file)
        if attempt < max_attempts:
            print(f"Retrying {step.name} in {args.retry_delay} seconds...")
            time.sleep(args.retry_delay)

    mark_step(state_file, state, step, "failed", last_returncode, max_attempts, last_log_file)
    return last_returncode


def run_sequence(args: argparse.Namespace) -> int:
    steps = resolve_steps(args)
    cwd = Path(args.cwd)
    state_file = Path(args.state_file)
    log_dir = Path(args.log_dir)

    if args.dry_run:
        print("Selected steps:")
        for index, step in enumerate(steps, 1):
            print(f"{index}. {step.name}: {step.description}")
        return 0

    failures: list[tuple[str, int]] = []
    with runner_lock(args):
        if args.reset_state and state_file.exists():
            state_file.unlink()
            print(f"Cleared state file: {state_file}")

        state = load_state(state_file)
        for step in steps:
            if args.resume and is_step_successful(state, step):
                print(f"\n[{datetime.now().isoformat(timespec='seconds')}] SKIP  {step.name} (already successful)")
                continue

            returncode = run_step(step, args, cwd, log_dir, state, state_file)
            if returncode != 0:
                failures.append((step.name, returncode))
                if not args.keep_going:
                    break

    if failures:
        print("\nFailed steps:")
        for name, returncode in failures:
            print(f"- {name}: exit code {returncode}")
        return failures[0][1]

    print(f"\nAll selected Indigo collection steps completed. State: {state_file}")
    return 0


def main() -> None:
    args = build_parser().parse_args()
    if args.list_steps:
        for step in STEPS.values():
            print(f"{step.name}: {step.description}")
        return
    raise SystemExit(run_sequence(args))


if __name__ == "__main__":
    main()

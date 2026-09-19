#!/usr/bin/env python3
"""Extract the best evolved program per track from OpenEvolve checkpoints.

OpenEvolve's own `best/` folder is rewritten on every (re)start, so it can
lag the true best. This scans every checkpoint archive under each run and
picks the highest `combined_score` (ties broken by `traintest_metric`).

Writes, per task:
  <task>/classical.py, <task>/deep.py          best program, byte-for-byte as evaluated
  <task>/results_{classical,deep}.json         its metrics + provenance
  <task>/checkpoints/{classical,deep}_top<K>.jsonl.gz   top-K programs (no embeddings)
and docs/results_summary.json.

Usage:
  python scripts/collect_best.py --runs-dir ~/projects/ps3-evolve [--top-k 10]
Run dirs are expected as <runs-dir>/<task> (classical) and <runs-dir>/<task>_gpu (deep).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from pathlib import Path

TASKS = ["door", "acv", "rail", "shm"]
KINDS = {"classical": "", "deep": "_gpu"}
METRIC_KEYS = ("combined_score", "traintest_metric", "cv3_metric", "cv5_metric", "code_length")


def load_programs(run_dir: Path) -> dict[str, dict]:
    programs: dict[str, dict] = {}
    for path in glob.glob(str(run_dir / "openevolve_output" / "checkpoints" / "checkpoint_*" / "programs" / "*.json")):
        try:
            prog = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = prog.get("metrics") or {}
        if "combined_score" not in metrics or metrics.get("error"):
            continue
        prog["_checkpoint"] = Path(path).parents[1].name
        programs[prog["id"]] = prog
    return programs


def rank_key(prog: dict) -> tuple[float, float]:
    m = prog["metrics"]
    return (float(m.get("combined_score", 0.0)), float(m.get("traintest_metric", 0.0)))


def slim(prog: dict) -> dict:
    return {
        "id": prog["id"],
        "parent_id": prog.get("parent_id"),
        "generation": prog.get("generation"),
        "iteration_found": prog.get("iteration_found"),
        "checkpoint": prog.get("_checkpoint"),
        "changes_description": prog.get("changes_description"),
        "metrics": {k: v for k, v in prog["metrics"].items() if isinstance(v, (int, float, str))},
        "code": prog["code"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    summary: dict = {}

    for task in TASKS:
        for kind, suffix in KINDS.items():
            run_dir = args.runs_dir.expanduser() / f"{task}{suffix}"
            programs = load_programs(run_dir)
            if not programs:
                print(f"[{task}/{kind}] no scored programs found in {run_dir}")
                continue
            ranked = sorted(programs.values(), key=rank_key, reverse=True)
            best = ranked[0]
            (repo / task / f"{kind}.py").write_text(best["code"], encoding="utf-8")
            info = {
                "program_id": best["id"],
                "iteration_found": best.get("iteration_found"),
                "checkpoint": best["_checkpoint"],
                "programs_scanned": len(programs),
                "metrics": {k: round(float(best["metrics"][k]), 4) for k in METRIC_KEYS if k in best["metrics"]},
            }
            (repo / task / f"results_{kind}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
            archive = repo / task / "checkpoints" / f"{kind}_top{args.top_k}.jsonl.gz"
            with gzip.open(archive, "wt", encoding="utf-8") as fh:
                for prog in ranked[: args.top_k]:
                    fh.write(json.dumps(slim(prog)) + "\n")
            summary[f"{task}/{kind}"] = info
            print(f"[{task}/{kind}] best {info['metrics']} (iter {info['iteration_found']}, {len(programs)} programs)")

    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

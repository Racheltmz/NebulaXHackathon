#!/usr/bin/env python3
"""Run the four per-task ensembles and assemble the deliverables:
   submission/{door,acv,rail,shm}_predictions.csv   (organiser schema)
   submission/predictions.zip                        (the four CSVs, top level, no subfolders)
   submission/final_predictions.csv                  (all four tracks in one long-format file)
usage: python scripts/make_submission.py [--skip-run] [--out submission]
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NAMES = ["door", "acv", "rail", "shm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission")
    ap.add_argument("--skip-run", action="store_true", help="only assemble already-generated CSVs")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(exist_ok=True)
    if not a.skip_run:
        for n in NAMES:
            subprocess.run([sys.executable, str(REPO / n / "ensemble.py"), "--out", str(out)], check=True)

    door = pd.read_csv(out / "door_predictions.csv")
    acv = pd.read_csv(out / "acv_predictions.csv", dtype=str)
    rail = pd.read_csv(out / "rail_predictions.csv")
    shm = pd.read_csv(out / "shm_predictions.csv")
    # schema checks
    assert list(door.columns) == ["start_time", "end_time", "prediction"] and door.prediction.isin(["Normal", "Abnormal resistance"]).all()
    assert list(acv.columns) == ["file_id", "ranked_cars"] and all(len(r.split("|")) == 8 for r in acv.ranked_cars)
    assert list(rail.columns) == ["file_id", "prediction"] and rail.prediction.isin(["Normal", "Side I", "Side II"]).all()
    assert list(shm.columns) == ["file_id", "prediction"] and (shm.prediction > 0).all() and shm.prediction.notna().all()

    parts = [
        door.assign(subsystem="door", file_id="Test.csv", ranked_cars=""),
        acv.assign(subsystem="acv", start_time="", end_time="", prediction=""),
        rail.assign(subsystem="rail", start_time="", end_time="", ranked_cars=""),
        shm.assign(subsystem="shm", start_time="", end_time="", ranked_cars=""),
    ]
    cols = ["subsystem", "file_id", "start_time", "end_time", "prediction", "ranked_cars"]
    pd.concat([p[cols] for p in parts], ignore_index=True).to_csv(out / "final_predictions.csv", index=False)
    with zipfile.ZipFile(out / "predictions.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for n in NAMES:
            z.write(out / f"{n}_predictions.csv", arcname=f"{n}_predictions.csv")
    print(f"door {len(door)} segments | acv {len(acv)} file | rail {len(rail)} files | shm {len(shm)} files")
    print("wrote", out / "predictions.zip", "and", out / "final_predictions.csv")


if __name__ == "__main__":
    main()

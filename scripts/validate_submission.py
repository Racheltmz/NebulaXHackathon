#!/usr/bin/env python3
"""Validate predictions.zip the way the organisers' CSV validator does: top-level CSVs, exact columns,
and file ids / timestamps that actually overlap the held-out set.  usage: validate_submission.py [zip]"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from common import final_data as fd  # noqa: E402

zpath = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "submission" / "predictions.zip"
z = zipfile.ZipFile(zpath)
names = z.namelist()
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


check("only the four allowed files, all at top level (no subfolders)",
      set(names) <= {"door_predictions.csv", "acv_predictions.csv", "rail_predictions.csv", "shm_predictions.csv"}
      and all("/" not in n for n in names), str(names))
read = lambda n: pd.read_csv(io.BytesIO(z.read(n)), dtype=str, keep_default_na=False)

door = read("door_predictions.csv")
check("door columns == start_time,end_time,prediction", list(door.columns) == ["start_time", "end_time", "prediction"])
te = pd.read_csv(fd.RAW / "Door" / "Test.csv", usecols=["Datetime"])
stamps = set(te.Datetime)
check("door labels valid", door.prediction.isin(["Normal", "Abnormal resistance"]).all(), f"({len(door)} segments)")
check("door start/end timestamps exist in Test.csv (native format)", door.start_time.isin(stamps).all() and door.end_time.isin(stamps).all())
pos = {s: i for i, s in enumerate(te.Datetime)}
check("door segments ordered, non-overlapping, cover the stream",
      all(pos[a] <= pos[b] for a, b in zip(door.start_time, door.end_time)) and
      all(pos[b] < pos[a2] for b, a2 in zip(door.end_time[:-1], door.start_time[1:])),
      f"(rows covered {sum(pos[b]-pos[a]+1 for a,b in zip(door.start_time, door.end_time))} of {len(te)})")

acv = read("acv_predictions.csv")
check("acv columns == file_id,ranked_cars", list(acv.columns) == ["file_id", "ranked_cars"])
test_case = next((fd.RAW / "ACV" / "Test").glob("*.xlsx"))
cols = pd.read_excel(test_case, nrows=0).columns
import re
cars = sorted({m.group(1) for c in cols if (m := re.fullmatch(r"Car (\d+) - .*", str(c)))})   # not the "Car model" metadata column
check("acv file_id is the held-out file name", list(acv.file_id) == [test_case.name], acv.file_id.tolist()[0])
check("acv ranks every car exactly once, ids as in the file's headers",
      sorted(acv.ranked_cars.iloc[0].split("|")) == cars, f"({acv.ranked_cars.iloc[0]})")

rail = read("rail_predictions.csv")
check("rail columns == file_id,prediction", list(rail.columns) == ["file_id", "prediction"])
rail_files = sorted(p.name for p in (fd.RAW / "Rail_Corrugation" / "Test").glob("*.csv"))
check("rail file_ids == the held-out files (with extension), no duplicates", sorted(rail.file_id) == rail_files, f"({len(rail)}/{len(rail_files)})")
check("rail labels valid", rail.prediction.isin(["Normal", "Side I", "Side II"]).all(), str(rail.prediction.value_counts().to_dict()))

shm = read("shm_predictions.csv")
check("shm columns == file_id,prediction", list(shm.columns) == ["file_id", "prediction"])
shm_files = sorted(p.name for p in (fd.RAW / "SHM" / "Test").glob("*.csv"))
vals = pd.to_numeric(shm.prediction, errors="coerce")
check("shm file_ids == the held-out files", sorted(shm.file_id) == shm_files, f"({len(shm)}/{len(shm_files)})")
check("shm predictions numeric, finite, positive", vals.notna().all() and np.isfinite(vals).all() and (vals > 0).all(),
      f"(range {vals.min():.3f}..{vals.max():.3f})")
print("\nVALID: ready to upload" if ok else "\nINVALID")
sys.exit(0 if ok else 1)

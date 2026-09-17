"""Turns predict_fn() row dicts into the exact `*_predictions.csv` schema required by
PS3/01_Problem_Statement_3_Specifications.md Section 4.1 — this is the file the download
button serves and the one that belongs in predictions.zip.
"""

import io

import pandas as pd


def rows_to_csv_bytes(subsystem_key: str, rows: list[dict]) -> bytes:
    if subsystem_key == "door":
        records = [
            {"start_time": r["start_time"], "end_time": r["end_time"], "prediction": r["label"]}
            for r in rows
        ]
    elif subsystem_key == "acv":
        records = [{"file_id": r["file_id"], "ranked_cars": r["ranked_cars"]} for r in rows]
    elif subsystem_key == "rail_corrugation":
        records = [{"file_id": r["file_id"], "prediction": r["label"]} for r in rows]
    elif subsystem_key == "shm":
        records = [{"file_id": r["file_id"], "prediction": r["value"]} for r in rows]
    else:
        raise ValueError(f"Unknown subsystem '{subsystem_key}'")

    df = pd.DataFrame(records)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

"""Turns predict_fn() row dicts into the `*_predictions.csv` this app's download button serves.

Each file mirrors the labelled training file of its subsystem in `PS3/02_Datasets/` — same
columns, same order — so a prediction file can be read back with the same code as the answers:

    door              Door/Train_Segments_Answer.csv   segment_id,start_time,end_time,operation,status,n_rows
    acv               ACV/Train_Labels.csv             filename,faulty_car
    rail_corrugation  Rail_Corrugation/Train_Labels.csv filename,label
    shm               SHM/Train_Labels.csv             filename,damage

That is not the `file_id`/`prediction` schema of the spec's Section 4.1 (which is what
`judge_leaderboard.py` reads), and ACV's ranking is cut down to its first car.
"""

import io

import pandas as pd

# Segment ids follow the answer file's `train_seg_001`; the app's Door input is the held-out stream.
DOOR_SEGMENT_PREFIX = "test_seg"


def segment_info_from_summaries(summaries) -> dict:
    """Door's per-segment operation/n_rows, merged from job summaries in run order (later wins)."""
    merged: dict = {}
    for summary in summaries:
        merged.update((summary or {}).get("segment_info") or {})
    return merged


def rows_to_csv_bytes(subsystem_key: str, rows: list[dict], segment_info: dict | None = None) -> bytes:
    if subsystem_key == "door":
        info = segment_info or {}
        records = [
            {
                "segment_id": f"{DOOR_SEGMENT_PREFIX}_{i:03d}",
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "operation": info.get(r["start_time"], {}).get("operation"),
                "status": r["label"],
                "n_rows": info.get(r["start_time"], {}).get("n_rows"),
            }
            for i, r in enumerate(rows, start=1)
        ]
        df = pd.DataFrame(records, columns=["segment_id", "start_time", "end_time", "operation", "status", "n_rows"])
        df["n_rows"] = df["n_rows"].astype("Int64")
    elif subsystem_key == "acv":
        # `faulty_car` is one car, so only the top of the ranking survives.
        records = [{"filename": r["file_id"], "faulty_car": r["ranked_cars"].split("|")[0]} for r in rows]
        df = pd.DataFrame(records, columns=["filename", "faulty_car"])
    elif subsystem_key == "rail_corrugation":
        records = [{"filename": r["file_id"], "label": r["label"]} for r in rows]
        df = pd.DataFrame(records, columns=["filename", "label"])
    elif subsystem_key == "shm":
        records = [{"filename": r["file_id"], "damage": r["value"]} for r in rows]
        df = pd.DataFrame(records, columns=["filename", "damage"])
    else:
        raise ValueError(f"Unknown subsystem '{subsystem_key}'")

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

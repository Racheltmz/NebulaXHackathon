"""Single source of truth for the per-subsystem upload contract (docs/DESIGN.md Section 6) and
which predict() implementation backs it (Section 8). The frontend's format panel and the
Predict router both read this registry instead of duplicating the contract.
"""

from dataclasses import dataclass
from typing import Callable

from . import acv, door, rail_corrugation, shm
from .common import PredictionResult, UploadedFile


@dataclass
class Subsystem:
    key: str
    label: str
    accepted_extensions: list[str]
    schema_hint: str
    output_columns: list[str]
    predict_fn: Callable[[list[UploadedFile]], PredictionResult]
    validate_fn: Callable[[list[UploadedFile]], None]


SUBSYSTEMS: dict[str, Subsystem] = {
    "door": Subsystem(
        key="door",
        label="Door",
        accepted_extensions=[".csv"],
        schema_hint=(
            "Each CSV file is one continuous stream containing many open/close cycles back to back, contains a header row and 17 columns:"
            "\nDatetime (Year-Month-Date-Hour-Minute-Second-Millisecond)"
            "\nMotor current(mA)"
            "\nMotor voltage(10mV)"
            "\nMotor electrodynamic force"
            "\nDoor opening time(.1s)"
            "\nDoor closing time(.1s)"
            "\nClose command"
            "\nOpen command"
            "\nDCSR"
            "\nDCSL"
            "\nDLSR"
            "\nDLSL"
            "\nDoor Opened"
            "\nDoor Locked"
            "\nDoor is opening"
            "\nDoor is closing"
            "\nDoor leaf position"
        ),
        output_columns=["start_time", "end_time", "prediction"],
        predict_fn=door.predict,
        validate_fn=door.validate,
    ),
    "acv": Subsystem(
        key="acv",
        label="ACV",
        accepted_extensions=[".xlsx"],
        schema_hint=(
            "Excel (.xlsx) file with one row per timestamp: "
            "\nCar model"
            "\nTrain number"
            "\nTime"
            "\nPer-car columns named 'Car <NN> - <parameter>' for each of the 8 cars (The exact parameter set can vary between files, car identifiers are read from each file's own headers.)"
        ),
        output_columns=["file_id", "ranked_cars"],
        predict_fn=acv.predict,
        validate_fn=acv.validate,
    ),
    "rail_corrugation": Subsystem(
        key="rail_corrugation",
        label="Rail Corrugation",
        accepted_extensions=[".csv"],
        schema_hint=(
            "CSV with a header row and 129 columns: rotating speed, then vibration + shock "
            "readings for each of 64 axle-box positions (8 cars x 8 positions). Each file is a "
            "1-second recording sampled at 10,000 Hz."
        ),
        output_columns=["file_id", "prediction"],
        predict_fn=rail_corrugation.predict,
        validate_fn=rail_corrugation.validate,
    ),
    "shm": Subsystem(
        key="shm",
        label="SHM",
        accepted_extensions=[".csv"],
        schema_hint=(
            "CSV with a single column of raw dynamic-stress readings and no header row. Each "
            "file is one equal-length time segment from one measurement point."
        ),
        output_columns=["file_id", "prediction"],
        predict_fn=shm.predict,
        validate_fn=shm.validate,
    ),
}


def get_subsystem(key: str) -> Subsystem:
    try:
        return SUBSYSTEMS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown subsystem '{key}'") from exc

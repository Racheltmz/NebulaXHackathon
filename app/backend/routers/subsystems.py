from fastapi import APIRouter

from ml.subsystems import SUBSYSTEMS
from schemas import SubsystemInfo

router = APIRouter(prefix="/api/subsystems", tags=["subsystems"])


@router.get("", response_model=list[SubsystemInfo])
def list_subsystems():
    return [
        SubsystemInfo(
            key=s.key,
            label=s.label,
            accepted_extensions=s.accepted_extensions,
            schema_hint=s.schema_hint,
            output_columns=s.output_columns,
        )
        for s in SUBSYSTEMS.values()
    ]

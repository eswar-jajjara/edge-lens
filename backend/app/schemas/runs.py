from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    atol: float = Field(default=0.0001, ge=0, le=1)
    rtol: float = Field(default=0.001, ge=0, le=1)
    warmup_runs: int = Field(default=3, ge=1, le=50, strict=True)
    measured_runs: int = Field(default=10, ge=3, le=200, strict=True)
    threads: int = Field(default=1, ge=1, le=8, strict=True)


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]
    strategy: Literal["fixed_profiles", "fidelity_search"] = "fixed_profiles"
    calibration_dataset_id: Annotated[str | None, Field(max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")] = None
    format: Literal["onnx", "tflite"]
    target: Literal["raspberry_pi", "esp32"]
    dataset_id: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]
    settings: RunSettings = Field(default_factory=RunSettings)

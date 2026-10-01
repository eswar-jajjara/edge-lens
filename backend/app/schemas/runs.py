from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    atol: float = Field(default=0.0001, ge=0, le=1)
    rtol: float = Field(default=0.001, ge=0, le=1)
    warmup_runs: int = Field(default=3, ge=1, le=50, strict=True)
    measured_runs: int = Field(default=10, ge=3, le=200, strict=True)
    threads: int = Field(default=1, ge=1, le=8, strict=True)


class QuantizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calibration_method: Literal["MinMax", "Entropy", "Percentile"] = "MinMax"
    per_channel: bool = Field(default=True, strict=True)


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]
    strategy: Literal["fixed_profiles", "fidelity_search", "quantization_compare"] = "fixed_profiles"
    calibration_dataset_id: Annotated[str | None, Field(max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")] = None
    validation_dataset_id: Annotated[str | None, Field(max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")] = None
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    format: Literal["onnx", "tflite"]
    target: Literal["raspberry_pi", "esp32"]
    dataset_id: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]
    settings: RunSettings = Field(default_factory=RunSettings)

    @model_validator(mode="after")
    def check_experiment(self):
        if self.strategy == "quantization_compare":
            if self.format != "onnx":
                raise ValueError("FP32/INT8 experiments require ONNX output")
            split_ids = [self.calibration_dataset_id, self.validation_dataset_id, self.dataset_id]
            if not all(split_ids) or len(set(split_ids)) != 3:
                raise ValueError("Select three separate calibration, validation and held-out test datasets")
        elif self.validation_dataset_id is not None:
            raise ValueError("A validation dataset is used by the FP32/INT8 experiment strategy")
        return self

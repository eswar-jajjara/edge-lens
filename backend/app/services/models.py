"""Bounded storage of developer-owned exported classifiers; never executes uploads."""
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
from typing import Literal
from uuid import uuid4
import zipfile
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=120)
    format: Literal["pt2", "onnx", "tflite"]
    input_shape: list[int] = Field(min_length=4, max_length=4)
    layout: Literal["NCHW", "NHWC"] = "NCHW"
    class_count: int = Field(ge=2, le=10000, strict=True)
    scale: float = Field(default=1 / 255, gt=0, le=255)
    mean: list[float] = Field(default_factory=lambda: [0., 0., 0.], min_length=1, max_length=3)
    std: list[float] = Field(default_factory=lambda: [1., 1., 1.], min_length=1, max_length=3)
    resize: Literal["stretch", "shortest_center_crop"] = "stretch"
    resize_shorter: int | None = Field(default=None, ge=1, le=4096)
    trusted_source: bool = False

    @model_validator(mode="after")
    def check_signature(self):
        dims = self.input_shape
        if any(type(n) is not int or n < 1 or n > 2048 for n in dims) or dims[0] != 1:
            raise ValueError("Use a fixed batch-one image shape with dimensions between 1 and 2048")
        channels = dims[1] if self.layout == "NCHW" else dims[3]
        if channels not in (1, 3) or len(self.mean) != channels or len(self.std) != channels:
            raise ValueError("Use 1 grayscale or 3 RGB channels and one mean/std value per channel")
        if not all(math.isfinite(x) for x in self.mean + self.std) or any(x <= 0 for x in self.std):
            raise ValueError("Mean/std must be finite and standard deviations must be positive")
        height, width = dims[2:] if self.layout == "NCHW" else dims[1:3]
        if self.resize == "shortest_center_crop" and (self.resize_shorter is None or self.resize_shorter < max(height, width)):
            raise ValueError("Center crop requires resize_shorter at least as large as the crop dimensions")
        if not self.trusted_source:
            raise ValueError("Confirm that this is your own trusted exported model. Model runtimes are not a security sandbox.")
        return self


def public_model(model):
    return {k: v for k, v in model.items() if k != "path"}


def ingest_model(content: bytes, spec: ModelSpec, data_dir: Path) -> dict:
    if not content or len(content) > 100 * 1024 * 1024:
        raise ValueError("Model must be non-empty and no larger than 100 MiB")
    if spec.format == "pt2":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                if len(entries) > 10000 or sum(x.file_size for x in entries) > 512 * 1024 * 1024:
                    raise ValueError("PT2 archive exceeds the expanded size limit")
                if not any("models/" in x.filename or "serialized_exported_program" in x.filename for x in entries):
                    raise ValueError("Expected a torch.export.save .pt2 archive, not a weights-only checkpoint")
        except zipfile.BadZipFile as exc:
            raise ValueError("Invalid PT2 archive; export with torch.export.save") from exc
    if spec.format == "tflite" and content[4:8] != b"TFL3":
        raise ValueError("File is not a TFLite FlatBuffer")
    model_id = "model_" + uuid4().hex
    directory = Path(data_dir) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{model_id}.{spec.format}"
    path.write_bytes(content)
    return {"id": model_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
            "path": str(path), **spec.model_dump()}


def verify_model(model):
    path = Path(model["path"])
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != model["sha256"]:
        raise ValueError("Stored model checksum changed; upload the model again")
    return path

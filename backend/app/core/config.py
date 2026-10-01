import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    allow_custom_models: bool = False
    environment: str = "development"
    cors_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")

    @classmethod
    def from_environment(cls) -> "Settings":
        environment = os.getenv("APP_ENV", "development")
        if environment not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test or production")
        defaults = "http://127.0.0.1:5173,http://localhost:5173" if environment == "development" else ""
        origins = tuple(value.strip() for value in os.getenv("CORS_ORIGINS", defaults).split(",") if value.strip())
        data_dir = Path(os.getenv("EDGELENS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))).resolve()
        return cls(environment=environment, cors_origins=origins, data_dir=data_dir, allow_custom_models=os.getenv("EDGELENS_ALLOW_CUSTOM_MODELS", "0") == "1")

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class RuntimeAdapter(ABC):
    """Interface only. Implement each runtime in a separate module."""

    @abstractmethod
    def load(self, artifact: Path) -> None:
        """Load a server-controlled artifact in an isolated worker."""

    @abstractmethod
    def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute preprocessed inputs and return named outputs."""

    @abstractmethod
    def graph_metadata(self) -> dict[str, Any]:
        """Return operators, connections, tensor shapes and dtypes."""

    @abstractmethod
    def close(self) -> None:
        """Release runtime resources."""

"""The tracking interface used by the training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class MlFlowLogger(Protocol):
    """Operations the training loop uses for optional MLflow tracking."""

    config_path: Path | None
    metric_prefix: str

    def start_run(self) -> MlFlowLogger: ...

    def __enter__(self) -> MlFlowLogger: ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any: ...

    def set_stage(self, stage_name: str, split: str | None = None) -> None: ...

    def log_dataset(self, dspath: str, dataset: Any) -> None: ...

    def log_model(self, **kwargs: Any) -> None: ...

    def log_epoch_metrics(
        self, metrics: dict[str, Any], epoch: int = 1, split: str | None = None
    ) -> None: ...

    def log_metric(
        self,
        metric_name: str,
        metric_value: Any,
        split: str | None = None,
        **kwargs: Any,
    ) -> None: ...

    def log_restart(self, count: int, **kwargs: Any) -> None: ...


class DisabledMlFlowLogger:
    """No-op tracking adapter used when MLflow logging is disabled."""

    def __init__(self) -> None:
        self.config_path: Path | None = None
        self.metric_prefix = ""

    def start_run(self) -> DisabledMlFlowLogger:
        return self

    def __enter__(self) -> DisabledMlFlowLogger:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None

    def set_stage(self, stage_name: str, split: str | None = None) -> None:
        return None

    def log_dataset(self, dspath: str, dataset: Any) -> None:
        return None

    def log_model(self, **kwargs: Any) -> None:
        return None

    def log_epoch_metrics(
        self, metrics: dict[str, Any], epoch: int = 1, split: str | None = None
    ) -> None:
        return None

    def log_metric(
        self,
        metric_name: str,
        metric_value: Any,
        split: str | None = None,
        **kwargs: Any,
    ) -> None:
        return None

    def log_restart(self, count: int, **kwargs: Any) -> None:
        return None

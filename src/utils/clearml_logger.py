from __future__ import annotations

from fnmatch import fnmatchcase
from numbers import Number
from typing import Any, Mapping

from clearml import Task


class _Summary:
    def __init__(self, run: "ClearMLRun") -> None:
        self._run = run
        self._values: dict[str, Any] = {}

    def __setitem__(self, key: str, value: Any) -> None:
        self._values[key] = value
        scalar = _as_scalar(value)
        if scalar is not None:
            self._run.logger.report_single_value(name=key, value=scalar)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]


class ClearMLRun:
    """Minimal experiment run interface backed by a ClearML Task.

    Metric names are used verbatim as ClearML plot titles. Each title has one
    series named ``value`` so keys such as ``val/ndcg@20`` remain unchanged in
    the ClearML UI.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.logger = task.get_logger()
        self.summary = _Summary(self)
        self._step_metrics: list[tuple[str, str]] = []
        self._fallback_iteration = 0
        self._finished = False

    def define_metric(self, name: str, step_metric: str | None = None) -> None:
        if step_metric is not None:
            self._step_metrics.append((name, step_metric))

    def log(self, values: Mapping[str, Any], step: int | None = None) -> None:
        metrics = dict(values)
        fallback_iteration = self._resolve_fallback_iteration(metrics, step)

        for name, value in metrics.items():
            scalar = _as_scalar(value)
            if scalar is None:
                continue

            iteration = self._resolve_iteration(
                metric_name=name,
                values=metrics,
                explicit_step=step,
                fallback=fallback_iteration,
            )
            self.logger.report_scalar(
                title=name,
                series="value",
                value=scalar,
                iteration=iteration,
            )

        self._fallback_iteration = max(
            self._fallback_iteration + 1,
            fallback_iteration + 1,
        )

    def finish(self) -> None:
        global _current_run

        if self._finished:
            return
        self.task.close()
        self._finished = True
        if _current_run is self:
            _current_run = None

    def _resolve_fallback_iteration(
        self,
        values: Mapping[str, Any],
        explicit_step: int | None,
    ) -> int:
        if explicit_step is not None:
            return int(explicit_step)
        for step_name in ("epoch", "debug/global_batch"):
            scalar = _as_scalar(values.get(step_name))
            if scalar is not None:
                return int(scalar)
        return self._fallback_iteration

    def _resolve_iteration(
        self,
        metric_name: str,
        values: Mapping[str, Any],
        explicit_step: int | None,
        fallback: int,
    ) -> int:
        if explicit_step is not None:
            return int(explicit_step)
        for pattern, step_name in reversed(self._step_metrics):
            if fnmatchcase(metric_name, pattern):
                scalar = _as_scalar(values.get(step_name))
                if scalar is not None:
                    return int(scalar)
        return fallback


_current_run: ClearMLRun | None = None


def current_run() -> ClearMLRun | None:
    """Return the active ClearML run, including a Task initialized elsewhere."""
    global _current_run

    if _current_run is not None and not _current_run._finished:
        return _current_run

    task = Task.current_task()
    if task is None:
        return None

    _current_run = ClearMLRun(task)
    return _current_run


def init_run(
    project: str,
    name: str,
    config: Mapping[str, Any] | None = None,
) -> ClearMLRun:
    """Create a fresh ClearML Task and attach the experiment configuration."""
    global _current_run

    active = current_run()
    if active is not None:
        active.finish()

    task = Task.init(
        project_name=project,
        task_name=name,
        reuse_last_task_id=False,
    )
    if config:
        task.connect(
            dict(config),
            name="General",
            ignore_remote_overrides=True,
        )

    _current_run = ClearMLRun(task)
    return _current_run


def _as_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Number):
        return float(value)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError):
            return None
        if isinstance(scalar, Number):
            return float(scalar)
    return None

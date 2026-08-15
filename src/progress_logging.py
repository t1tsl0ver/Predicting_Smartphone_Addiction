from __future__ import annotations

import re
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterator, Mapping


def safe_filename_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_")
    return label or "run"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class RunLogger:
    """Write flushed progress messages to both the console and a run log file."""

    def __init__(self, name: str, log_dir: Path, run_id: str | None = None) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        started_at = datetime.now().astimezone()
        timestamp = started_at.strftime("%Y%m%dT%H%M%S%f%z")
        label = safe_filename_label(run_id or "default")
        safe_name = safe_filename_label(name)
        self.path = log_dir / f"{timestamp}_{safe_name}_{label}.log"
        self._started = perf_counter()
        self.info(f"Run started: {name}; run_id={label}")
        self.info(f"Persistent log: {self.path}")

    def _write(self, level: str, message: str) -> None:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        elapsed = format_duration(perf_counter() - self._started)
        lines = str(message).splitlines() or [""]
        rendered = [f"{now} | +{elapsed} | {level:<5} | {line}" for line in lines]
        for line in rendered:
            print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as log_file:
            for line in rendered:
                log_file.write(f"{line}\n")

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARN", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def exception(self, message: str) -> None:
        self.error(f"{message}\n{traceback.format_exc().rstrip()}")

    @contextmanager
    def step(self, label: str) -> Iterator[None]:
        step_started = perf_counter()
        self.info(f"START | {label}")
        try:
            yield
        except Exception:
            duration = format_duration(perf_counter() - step_started)
            self.exception(f"FAILED | {label} | duration={duration}")
            raise
        duration = format_duration(perf_counter() - step_started)
        self.info(f"DONE  | {label} | duration={duration}")

    def finish(self) -> None:
        self.info(
            f"Run completed successfully | total={format_duration(perf_counter() - self._started)}"
        )


class IterationProgress:
    def __init__(
        self,
        logger: RunLogger,
        label: str,
        total_iterations: int,
        every: int,
    ) -> None:
        if total_iterations < 1:
            raise ValueError("Total iterations must be at least 1")
        if every < 1:
            raise ValueError("Progress interval must be at least 1")
        self.logger = logger
        self.label = label
        self.total_iterations = total_iterations
        self.every = every
        self._started = perf_counter()

    def report(
        self,
        iteration: int,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        if (
            iteration != 1
            and iteration % self.every != 0
            and iteration < self.total_iterations
        ):
            return

        elapsed = perf_counter() - self._started
        completed = max(1, iteration)
        remaining = max(0, self.total_iterations - iteration)
        eta = elapsed / completed * remaining
        percent = 100 * iteration / self.total_iterations
        metric_text = ""
        if metrics:
            metric_text = " | " + ", ".join(
                f"{name}={value:.6f}" for name, value in metrics.items()
            )
        self.logger.info(
            f"PROGRESS | {self.label} | {iteration}/{self.total_iterations} "
            f"({percent:.1f}%) | elapsed={format_duration(elapsed)} | "
            f"ETA={format_duration(eta)}{metric_text}"
        )


class CatBoostProgressCallback:
    def __init__(
        self,
        logger: RunLogger,
        label: str,
        total_iterations: int,
        every: int,
    ) -> None:
        self.progress = IterationProgress(
            logger, label, total_iterations, every
        )

    def after_iteration(self, info: object) -> bool:
        iteration = int(getattr(info, "iteration"))
        all_metrics = getattr(info, "metrics", {})
        current_metrics: dict[str, float] = {}
        for dataset_name, dataset_metrics in all_metrics.items():
            for metric_name, values in dataset_metrics.items():
                if values:
                    current_metrics[f"{dataset_name}.{metric_name}"] = float(values[-1])
        self.progress.report(iteration, current_metrics)
        return True


class LightGBMProgressCallback:
    order = 20
    before_iteration = False

    def __init__(
        self,
        logger: RunLogger,
        label: str,
        total_iterations: int,
        every: int,
    ) -> None:
        self.progress = IterationProgress(
            logger, label, total_iterations, every
        )

    def __call__(self, environment: object) -> None:
        begin_iteration = int(getattr(environment, "begin_iteration"))
        iteration = int(getattr(environment, "iteration")) - begin_iteration + 1
        metrics: dict[str, float] = {}
        for result in getattr(environment, "evaluation_result_list", None) or []:
            dataset_name, metric_name, value = result[:3]
            metrics[f"{dataset_name}.{metric_name}"] = float(value)
        self.progress.report(iteration, metrics)

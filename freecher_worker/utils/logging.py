"""Logging and timing utilities for worker pipeline stages."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional


class StageContext:
    def __init__(self, logger: "WorkerLogger", name: str) -> None:
        self.logger = logger
        self.name = name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "StageContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if exc_type is not None:
            self.logger.error(self.name, f"failed after {self.elapsed:.2f} sec: {exc_val}")
        else:
            # Stage was not completed with a custom message; emit default
            pass

    def complete(self, message: Optional[str] = None) -> float:
        self.elapsed = time.perf_counter() - self.start_time
        if message:
            self.logger.info(self.name, f"{message} (in {self.elapsed:.2f} sec)")
        else:
            self.logger.info(self.name, f"completed in {self.elapsed:.2f} sec")
        return self.elapsed


class WorkerLogger:
    """Logger that writes structured stage logs to stdout and a file."""

    def __init__(self, log_file: Optional[Path | str] = None, name: str = "freecher_worker") -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler (stdout)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        # Clean format for console: just message or with timestamp
        console_formatter = logging.Formatter(fmt="%(message)s")
        stdout_handler.setFormatter(console_formatter)
        self.logger.addHandler(stdout_handler)

        self.file_handler: Optional[logging.FileHandler] = None
        if log_file:
            self.set_log_file(log_file, formatter)

    def set_log_file(self, log_file: Path | str, formatter: Optional[logging.Formatter] = None) -> None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()

        if formatter is None:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        self.file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

    def _format_msg(self, stage: str, message: str) -> str:
        return f"[{stage}] {message}"

    def info(self, stage: str, message: str) -> None:
        self.logger.info(self._format_msg(stage, message))

    def warning(self, stage: str, message: str) -> None:
        self.logger.warning(self._format_msg(stage, message))

    def error(self, stage: str, message: str) -> None:
        self.logger.error(self._format_msg(stage, message))

    def debug(self, stage: str, message: str) -> None:
        self.logger.debug(self._format_msg(stage, message))

    @contextmanager
    def stage(self, name: str) -> Generator[StageContext, None, None]:
        ctx = StageContext(self, name)
        ctx.start_time = time.perf_counter()
        try:
            yield ctx
        except Exception:
            ctx.elapsed = time.perf_counter() - ctx.start_time
            raise
        else:
            if ctx.elapsed == 0.0:
                ctx.complete()

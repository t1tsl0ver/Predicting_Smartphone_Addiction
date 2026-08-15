from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from progress_logging import (  # noqa: E402
    CatBoostProgressCallback,
    LightGBMProgressCallback,
    RunLogger,
    format_duration,
    safe_filename_label,
)


class ProgressLoggingTests(unittest.TestCase):
    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(0), "00:00")
        self.assertEqual(format_duration(65), "01:05")
        self.assertEqual(format_duration(3661), "01:01:01")

    def test_safe_filename_label(self) -> None:
        self.assertEqual(safe_filename_label("trial-008"), "trial-008")
        self.assertEqual(safe_filename_label("../../unsafe name"), "unsafe-name")

    def test_run_logger_writes_console_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            console = StringIO()
            with redirect_stdout(console):
                logger = RunLogger("unit-test", Path(directory), "logging")
                with logger.step("small step"):
                    logger.info("inside")
                logger.finish()

            file_text = logger.path.read_text(encoding="utf-8")
            self.assertIn("START | small step", console.getvalue())
            self.assertIn("DONE  | small step", file_text)
            self.assertIn("Run completed successfully", file_text)

    def test_model_callbacks_report_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                logger = RunLogger("callback-test", Path(directory))
                cat_callback = CatBoostProgressCallback(logger, "cat", 2, 1)
                cat_callback.after_iteration(
                    SimpleNamespace(
                        iteration=1,
                        metrics={"validation": {"AUC": [0.91]}},
                    )
                )
                lgb_callback = LightGBMProgressCallback(logger, "lgb", 2, 1)
                lgb_callback(
                    SimpleNamespace(
                        begin_iteration=0,
                        iteration=0,
                        evaluation_result_list=[("valid", "auc", 0.92, True)],
                    )
                )

            file_text = logger.path.read_text(encoding="utf-8")
            self.assertIn("validation.AUC=0.910000", file_text)
            self.assertIn("valid.auc=0.920000", file_text)


if __name__ == "__main__":
    unittest.main()

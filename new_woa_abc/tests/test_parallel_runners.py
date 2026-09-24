from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from new_woa_abc.reporting import write_csv, write_json
from new_woa_abc.runners.checkpoints import (
    archive_existing_directory,
    load_typed_rows,
    resume_directory,
    validate_resume_config,
)
from new_woa_abc.runners.run_cec import parser as cec_parser
from new_woa_abc.runners.run_hres2 import parser as hres2_parser


class TestParallelRunnerArguments(unittest.TestCase):
    def test_cec_and_hres2_accept_workers_and_resume_dir(self) -> None:
        for make_parser in (cec_parser, hres2_parser):
            with self.subTest(parser=make_parser.__module__):
                args = make_parser().parse_args(
                    ["--workers", "7", "--resume-dir", "/tmp/campaign"]
                )
                self.assertEqual(args.workers, 7)
                self.assertEqual(args.resume_dir, Path("/tmp/campaign"))


class TestCheckpoints(unittest.TestCase):
    def test_typed_rows_and_resume_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / "configuracion.json", {"domain": "cec", "runs": 2})
            write_csv(
                directory / "todos_los_runs.csv",
                [{"seed": 42, "best_cost": 1.5, "optimum_reached": True}],
            )
            rows = load_typed_rows(
                directory / "todos_los_runs.csv",
                integer_fields={"seed"},
                float_fields={"best_cost"},
                boolean_fields={"optimum_reached"},
            )
            self.assertEqual(rows[0]["seed"], 42)
            self.assertEqual(rows[0]["best_cost"], 1.5)
            self.assertIs(rows[0]["optimum_reached"], True)
            output, config = resume_directory(directory, "cec")
            self.assertEqual(output, directory.resolve())
            validate_resume_config(config, {"runs": 2}, ["runs"])
            with self.assertRaisesRegex(ValueError, "configuración incompatible"):
                validate_resume_config(config, {"runs": 3}, ["runs"])

    def test_incompatible_artifact_is_archived_recoverably(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "mejor_run"
            artifact.mkdir()
            (artifact / "marker.txt").write_text("old", encoding="utf-8")
            archived = archive_existing_directory(artifact)
            self.assertIsNotNone(archived)
            self.assertFalse(artifact.exists())
            self.assertEqual((archived / "marker.txt").read_text(), "old")


if __name__ == "__main__":
    unittest.main()

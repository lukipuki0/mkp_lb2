from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from woa_abc_version_vieja.reporting import write_csv, write_json
from woa_abc_version_vieja.runners.checkpoints import (
    archive_existing_directory,
    load_typed_rows,
    resume_directory,
    validate_resume_config,
)
from woa_abc_version_vieja.runners.run_cec import parser as cec_parser
from woa_abc_version_vieja.runners.run_hres2 import parser as hres2_parser
from woa_abc_version_vieja.runners.run_mkp import parser as mkp_parser
from woa_abc_version_vieja.runners.common import build_config


class TestParallelRunnerArguments(unittest.TestCase):
    def test_mkp_dtw_defaults_use_requested_window(self) -> None:
        args = mkp_parser().parse_args([])
        config = build_config(args, "M1_fire_d2", 42).dtw
        self.assertEqual(
            (config.window, config.band, config.min_slope, config.plateau_max,
             config.patience, config.use_ddtw, config.adapt_thresholds,
             config.p_low, config.p_high),
            (70, 0, 0.1, 15, 8, True, True, 30.0, 70.0),
        )
        self.assertEqual(config.effective_band, 7)

    def test_dtw_overrides_and_other_domain_defaults_remain_available(self) -> None:
        args = mkp_parser().parse_args([
            "--window", "12", "--band", "1", "--min-slope", "3",
            "--plateau-max", "5", "--patience", "2", "--no-ddtw",
        ])
        config = build_config(args, "M1_fire_d2", 42).dtw
        self.assertEqual(
            (config.window, config.band, config.min_slope, config.plateau_max,
             config.patience, config.use_ddtw),
            (12, 1, 3.0, 5, 2, False),
        )
        for make_parser, slope in ((cec_parser, 2.0), (hres2_parser, 0.0)):
            with self.subTest(domain=make_parser.__module__):
                dtw = build_config(make_parser().parse_args([]), "M1_fire_d2", 42).dtw
                self.assertEqual(
                    (dtw.window, dtw.band, dtw.min_slope, dtw.plateau_max, dtw.patience),
                    (20, 2, slope, 4, 2),
                )

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

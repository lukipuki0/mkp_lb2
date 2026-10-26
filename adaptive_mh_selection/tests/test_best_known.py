"""OR-Library best-feasible references map only to the matching local instance set."""
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class BestKnownReferenceTests(unittest.TestCase):
    def test_official_instance_hash_maps_to_best_feasible_reference(self):
        from adaptive_mh_selection.best_known import lookup_best_feasible

        source = ROOT / "instancias" / "mknapcb1.txt"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(lookup_best_feasible(1, 0, digest), 24381)
        self.assertEqual(lookup_best_feasible(1, 29, digest), 59965)

    def test_campaign_protocol_uses_published_reference_for_exact_bundled_instance(self):
        import contextlib
        import io
        import json
        from adaptive_mh_selection.campaign import main

        output = ROOT / "adaptive_mh_selection" / "tests" / ".tmp" / "best-known-dry-run"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--dry-run", "--families", "1", "--indices", "0",
                                   "--runs", "1", "--output", str(output)]), 0)
        protocol = json.loads(stdout.getvalue())
        case = protocol["cases"][0]
        self.assertEqual(case["reference_value"], 24381)
        self.assertIn("best feasible value found", case["reference_source"])
        self.assertFalse(output.exists())

    def test_single_run_protocol_uses_reference_and_explicit_override_wins(self):
        import contextlib
        import io
        import json
        from adaptive_mh_selection.run import main

        for extra, expected, source in (([], 24381, "best feasible value found"),
                                        (["--reference-value", "25000"], 25000, "explicit override")):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["--dry-run", "--instance", str(ROOT / "instancias/mknapcb1.txt"),
                                       "--index", "0", "--output", str(ROOT / "unused"), *extra]), 0)
            protocol = json.loads(stdout.getvalue())
            self.assertEqual(protocol["reference_value"], expected)
            self.assertIn(source, protocol["reference_source"])

    def test_modified_or_unrelated_instance_set_has_no_reference(self):
        from adaptive_mh_selection.best_known import lookup_best_feasible

        self.assertIsNone(lookup_best_feasible(1, 0, "0" * 64))
        self.assertIsNone(lookup_best_feasible(10, 0, "0" * 64))

    def test_reference_values_are_not_mislabeled_as_proven_optima(self):
        from adaptive_mh_selection.best_known import reference_source
        self.assertIn("not certified optimum", reference_source())

    def test_all_nine_families_have_thirty_entries_and_published_values(self):
        from adaptive_mh_selection.best_known import lookup_best_feasible

        for family in range(1, 10):
            source = ROOT / "instancias" / f"mknapcb{family}.txt"
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            for index in range(30):
                self.assertGreater(lookup_best_feasible(family, index, digest), 0,
                                   (family, index))

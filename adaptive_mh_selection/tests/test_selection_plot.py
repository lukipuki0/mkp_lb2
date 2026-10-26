"""Conditional selection probabilities from recorded decisions, not frequencies."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.colors import to_rgba

from hybrid_mkp.orchestrator import COLORES_MH, PipelineResult, SwitchLog

TEMP = Path(__file__).resolve().parent / ".tmp"


def temporary():
    TEMP.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEMP)


def rgb(color):
    return tuple(round(channel, 4) for channel in to_rgba(color)[:3])


def native_result():
    return PipelineResult(
        12., [1, 0], [10., 10., 12., 12.], [9., 10., 11., 12.],
        [float("nan"), .2, float("nan"), .3],
        log_switches=[
            SwitchLog("PSO", "poblacional", 10., 0., .1, 2, [.2], []),
            SwitchLog("SA", "trayectoria", 12., .1, .2, 2, [.3], []),
        ],
    )


def decision(block, pool, solver, start, probabilities):
    return {
        "block_index": block, "pool": pool, "solver": solver,
        "iteration_start": start, "iteration_end": start + 10,
        "probabilities_before": probabilities,
    }


def recorded_decisions():
    """Chosen solver is not the probability: PSO is always picked in its pool."""
    return [
        decision(0, "poblacional", "PSO", 0, {
            "GA": 0.10, "PSO": 0.50, "GWO": 0.10, "WOA": 0.10, "EHO": 0.10, "ACO": 0.10,
        }),
        decision(1, "trayectoria", "SA", 10, {
            "SA": 0.70, "TS": 0.10, "ILS": 0.10, "VNS": 0.10,
        }),
        decision(2, "poblacional", "PSO", 20, {
            "GA": 0.40, "PSO": 0.20, "GWO": 0.05, "WOA": 0.05, "EHO": 0.05,
        }),
    ]


def capture_probability_figure(callback):
    import matplotlib.figure
    found = []
    original = matplotlib.figure.Figure.savefig

    def spy(fig, path, *args, **kwargs):
        if Path(path).name == "selection_probabilities.png":
            found.append({
                "suptitle": fig._suptitle.get_text() if fig._suptitle is not None else "",
                "axes": [{
                    "title": ax.get_title(loc="left"),
                    "xlabel": ax.get_xlabel(),
                    "ylabel": ax.get_ylabel(),
                    "ylim": ax.get_ylim(),
                    "texts": [text.get_text() for text in ax.texts],
                    "lines": [{
                        "label": line.get_label(),
                        "color": rgb(line.get_color()),
                        "x": [float(value) for value in line.get_xdata()],
                        "y": [float(value) for value in line.get_ydata()],
                    } for line in ax.lines],
                } for ax in fig.axes],
            })
        return original(fig, path, *args, **kwargs)

    with patch.object(matplotlib.figure.Figure, "savefig", new=spy):
        result = callback()
    return result, found


class SelectionProbabilityPlotTests(unittest.TestCase):
    def test_plots_recorded_conditional_probabilities_not_frequency_or_global_mix(self):
        from adaptive_mh_selection.plotting import plot_run, plot_selection_probabilities

        records = recorded_decisions()
        with temporary() as tmp:
            folder = Path(tmp)
            _, captured = capture_probability_figure(
                lambda: plot_selection_probabilities(records, folder))
            self.assertEqual(len(captured), 1)
            self.assertTrue((folder / "selection_probabilities.png").is_file())
            self.assertTrue((folder / "selection_probabilities.pdf").is_file())
            self.assertGreater((folder / "selection_probabilities.png").stat().st_size, 1000)
            self.assertGreater((folder / "selection_probabilities.pdf").stat().st_size, 1000)
            figure = captured[0]
            self.assertEqual(len(figure["axes"]), 10)
            disclaimer = " ".join([figure["suptitle"], *[ax["title"] + ax["ylabel"] + ax["xlabel"]
                                                         for ax in figure["axes"]]]).lower()
            self.assertIn("conditional", disclaimer)
            self.assertIn("within pool", disclaimer)
            self.assertIn("not a global probability", disclaimer)
            self.assertIn("not a selection frequency", disclaimer)
            panels = {ax["title"].split(" — ", 1)[1]: ax for ax in figure["axes"]}
            self.assertEqual(set(panels), set(COLORES_MH))
            for axis in figure["axes"]:
                self.assertIn(axis["ylabel"], ("Conditional probability (%)", ""))
                self.assertEqual(axis["xlabel"], "Decision within pool")
                self.assertEqual(axis["ylim"], (0.0, 100.0))
                self.assertLessEqual(len(axis["lines"]), 1)
            pop = {name: panels[name]["lines"][0] for name in ("GA", "PSO", "GWO", "WOA", "EHO", "ACO")
                   if panels[name]["lines"]}
            tray = {name: panels[name]["lines"][0] for name in ("SA", "TS", "ILS", "VNS")
                    if panels[name]["lines"]}
            self.assertEqual(set(pop), {"GA", "PSO", "GWO", "WOA", "EHO", "ACO"})
            self.assertEqual(set(tray), {"SA", "TS", "ILS", "VNS"})
            self.assertEqual(pop["PSO"]["x"], [1.0, 2.0])
            self.assertEqual(pop["PSO"]["y"], [50.0, 20.0])
            self.assertNotEqual(pop["PSO"]["y"], [100.0, 100.0])
            self.assertEqual(pop["GA"]["x"], [1.0, 2.0])
            self.assertEqual(pop["GA"]["y"], [10.0, 40.0])
            self.assertEqual(pop["ACO"]["x"], [1.0])
            self.assertEqual(pop["ACO"]["y"], [10.0])
            self.assertEqual(pop["GWO"]["y"], [10.0, 5.0])
            self.assertEqual(tray["SA"]["x"], [1.0])
            self.assertEqual(tray["SA"]["y"], [70.0])
            self.assertNotIn(10.0, tray["SA"]["x"])
            for label, line in {**pop, **tray}.items():
                self.assertEqual(line["color"], rgb(COLORES_MH[label]), label)
            self.assertTrue(all(figure_axis["title"].startswith(("Poblacional —", "Trayectoria —"))
                                for figure_axis in figure["axes"]))
            rerun, again = capture_probability_figure(
                lambda: plot_run(native_result(), folder / "run", selection_records=records))
            self.assertIn("selection_probabilities.png", {Path(path).name for path in rerun})
            self.assertEqual(again[0]["axes"][0]["lines"][0]["y"], figure["axes"][0]["lines"][0]["y"])

    def test_missing_pool_and_missing_probabilities_are_not_imputed(self):
        from adaptive_mh_selection.plotting import plot_selection_probabilities

        records = [decision(4, "trayectoria", "TS", 40, {"TS": 0.25, "SA": 0.25, "ILS": 0.25, "VNS": 0.25}),
                   {"block_index": 5, "pool": "trayectoria", "solver": "TS",
                    "iteration_start": 50, "iteration_end": 60}]
        with temporary() as tmp:
            _, captured = capture_probability_figure(
                lambda: plot_selection_probabilities(records, Path(tmp)))
        figure = captured[0]
        population_panels = [ax for ax in figure["axes"] if ax["title"].startswith("Poblacional —")]
        self.assertEqual(len(population_panels), 6)
        self.assertTrue(all(ax["lines"] == [] for ax in population_panels))
        self.assertTrue(all(any("No recorded decisions" in text for text in ax["texts"])
                            for ax in population_panels))
        trajectory_ts = next(ax for ax in figure["axes"] if ax["title"] == "Trayectoria — TS")
        self.assertEqual(trajectory_ts["lines"][0]["y"], [25.0])
        self.assertEqual(len(trajectory_ts["lines"][0]["x"]), 1)
        with temporary() as tmp:
            self.assertIsNone(plot_selection_probabilities(None, Path(tmp)))
            self.assertIsNone(plot_selection_probabilities([], Path(tmp) / "empty"))
            self.assertFalse((Path(tmp) / "selection_probabilities.png").exists())
            self.assertFalse((Path(tmp) / "empty").exists())

    def test_trace_csv_roundtrip_plots_the_same_recorded_probabilities(self):
        from adaptive_mh_selection.exports import write_csv
        from adaptive_mh_selection.plotting import load_selection_trace, plot_selection_probabilities

        records = recorded_decisions()
        with temporary() as tmp:
            folder = Path(tmp)
            write_csv(folder / "selection_trace.csv", records, list(records[0]))
            loaded = load_selection_trace(folder / "selection_trace.csv")
            self.assertEqual(loaded[2]["probabilities_before"]["PSO"], 0.20)
            self.assertIsInstance(loaded[0]["probabilities_before"], dict)
            _, direct = capture_probability_figure(
                lambda: plot_selection_probabilities(records, folder / "direct"))
            _, restored = capture_probability_figure(
                lambda: plot_selection_probabilities(loaded, folder / "restored"))
        self.assertEqual(direct[0]["axes"][0]["lines"], restored[0]["axes"][0]["lines"])
        self.assertEqual(direct[0]["axes"][1]["lines"], restored[0]["axes"][1]["lines"])

    def test_adaptive_plot_run_does_not_call_or_delete_instantaneous_renderer_output(self):
        import hybrid_mkp.plots as native_plots
        from adaptive_mh_selection.plotting import plot_run

        result = native_result()
        with temporary() as tmp:
            folder = Path(tmp)
            kept = folder / "instantaneous_fitness_only.png"
            kept.write_bytes(b"preexisting-instantaneous")
            retired = folder / "instantaneous_convergence.png"
            retired.write_bytes(b"obsolete")
            with patch.object(native_plots, "grafico_solo_instantaneo") as renderer:
                paths = plot_run(result, folder, selection_records=recorded_decisions())
            renderer.assert_not_called()
            self.assertEqual(kept.read_bytes(), b"preexisting-instantaneous")
            self.assertFalse(retired.exists())
            names = {Path(path).name for path in paths}
            self.assertNotIn("instantaneous_fitness_only.png", names)
            self.assertIn("selection_probabilities.png", names)
            self.assertIn("selection_probabilities.pdf", names)
            self.assertIn("fig_temporal_diagnostics.png", names)
            self.assertIn("fitness_convergence.png", names)
            self.assertTrue(callable(native_plots.grafico_solo_instantaneo))

    def test_native_instantaneous_renderer_still_writes_for_other_callers(self):
        from hybrid_mkp.plots import grafico_solo_instantaneo

        result = native_result()
        with temporary() as tmp:
            path = grafico_solo_instantaneo(
                historial_inst_global=result.historial_inst_global,
                log_switches=result.log_switches, colores_mh=COLORES_MH,
                valor_optimo=result.valor_optimo, output_dir=tmp)
            self.assertEqual(Path(path).name, "instantaneous_fitness_only.png")
            self.assertGreater(Path(path).stat().st_size, 1000)


class RunExportDeliveryTests(unittest.TestCase):
    def test_winning_run_export_plots_records_without_selection_trace_csv(self):
        from adaptive_mh_selection.exports import write_batch
        import hybrid_mkp.plots as native_plots

        instance = type("Instance", (), {"n": 8, "m": 2, "valor_optimo": 20.0})()
        summary = {
            "gap_pct": 40.0, "requested_iterations": 4, "actual_iterations": 4,
            "iteration_overshoot": 0, "transitions": 1, "best_fitness": 12.0,
            "n_blocks": 2, "elapsed_seconds": 0.01,
        }
        records = recorded_decisions()
        result = {"native_result": native_result(), "summary": summary, "selection_records": records}
        with temporary() as tmp, patch.object(native_plots, "grafico_solo_instantaneo") as renderer:
            best = write_batch([result], Path(tmp), name="tiny", instance=instance, plots=True)
            renderer.assert_not_called()
            self.assertTrue((best / "selection_probabilities.png").is_file())
            self.assertTrue((best / "selection_probabilities.pdf").is_file())
            self.assertFalse((best / "instantaneous_fitness_only.png").exists())
            self.assertEqual(list(Path(tmp).rglob("selection_trace.csv")), [])
            self.assertTrue((best / "fig_temporal_diagnostics.png").is_file())
            report = json.loads((best / "temporal_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rendering"]["panel_c_scale"], "linear")
            self.assertEqual(report["rendering"]["layout"], "reference_7.4x8.6_b_split")

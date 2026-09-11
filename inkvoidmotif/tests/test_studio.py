from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image, ImageDraw

from inkvoidmotif import pipeline, studio
from inkvoidmotif.batch_segment_motifs import batch_segment_paintings
from inkvoidmotif.gradio_app import CSS, build_demo, theme, ui_analyze, ui_compose
from inkvoidmotif.image_ops import decode_b64_image
from inkvoidmotif.openai_api import download_image, nvidia_api_key


def sample_plan(source: Path) -> dict:
    return {
        "source": str(source),
        "source_summary": "A river landscape with a pavilion and distant mountain.",
        "motifs": [
            {
                "name": "pavilion_cluster",
                "description": "A pavilion beneath two pines.",
                "role": "middle distance",
                "bbox": [0.08, 0.48, 0.34, 0.31],
                "must_preserve": ["roof line", "pine needles"],
            },
            {
                "name": "distant_peak",
                "description": "A pale layered mountain peak.",
                "role": "far distance",
                "bbox": [0.54, 0.10, 0.35, 0.32],
                "must_preserve": ["mist edge"],
            },
        ],
    }


class StudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input.png"
        image = Image.new("RGB", (900, 1400), studio.PAPER_RGB)
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [(60, 1100), (400, 620), (850, 1150), (850, 1400), (60, 1400)],
            fill=(60, 70, 62),
        )
        image.save(self.source)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def fake_plan(source_path: Path, output_dir: Path, **_kwargs) -> Path:
        return pipeline.write_json(
            Path(output_dir) / "motif_plan.json", sample_plan(Path(source_path))
        )

    @staticmethod
    def fake_sheet(output_path: Path, **_kwargs) -> Path:
        image = Image.new("RGB", (512, 768), studio.PAPER_RGB)
        draw = ImageDraw.Draw(image)
        draw.ellipse((60, 90, 220, 300), fill=(55, 68, 58))
        draw.polygon([(300, 560), (410, 330), (490, 570)], fill=(82, 88, 76))
        image.save(output_path)
        return Path(output_path)

    def prepared_run(self) -> tuple[dict, list[list[str]]]:
        state, rows, *_ = studio.analyze_painting(self.source, 2, "openai")
        state, *_ = studio.create_motif_bank(
            state, rows, None, "openai", "", "1024x1536", "high", 2
        )
        return state, rows

    def test_layout_preview_tracks_void_for_every_convention(self) -> None:
        for convention in ("top_sky", "diagonal", "river_band"):
            image, achieved = studio.build_composition_guide(
                0.63, convention, 0.7, 0.6, 11, width=180, height=260, label=True
            )
            self.assertEqual(image.size, (180, 260))
            self.assertAlmostEqual(achieved, 0.63, places=2)

    def test_layout_rejects_non_finite_and_out_of_range_values(self) -> None:
        for bad in (float("nan"), float("inf"), 0.1, 0.9):
            with self.assertRaises(studio.StudioError):
                studio.preview_layout(bad, "top_sky", 0.7, 0.6, 1, "1024x1024")

    def test_plan_table_validation_and_unicode_names(self) -> None:
        rows = studio.plan_to_rows(sample_plan(self.source))
        motifs = studio.rows_to_motifs(rows)
        self.assertEqual(motifs[0]["name"], "pavilion_cluster")
        duplicate = [rows[0], rows[0]]
        with self.assertRaises(studio.StudioError):
            studio.rows_to_motifs(duplicate)
        bad_box = [list(rows[0]), list(rows[1])]
        bad_box[1][3] = "nan, 0.1, 0.2, 0.2"
        with self.assertRaises(studio.StudioError):
            studio.rows_to_motifs(bad_box)
        chinese = [list(rows[0]), list(rows[1])]
        chinese[0][0], chinese[1][0] = "松林", "远山"
        self.assertEqual([m["name"] for m in studio.rows_to_motifs(chinese)], ["松林", "远山"])

    @patch("inkvoidmotif.studio.resolve_provider", return_value="openai")
    def test_real_layout_composer_mocked_provider_is_reproducible(self, _provider) -> None:
        reference_counts: list[int] = []

        def fake_edit(image_paths, output_path, **_kwargs):
            reference_counts.append(len(image_paths))
            self.assertTrue(all(Path(path).is_file() for path in image_paths))
            image = Image.new("RGB", (1024, 1024), studio.PAPER_RGB)
            draw = ImageDraw.Draw(image)
            draw.polygon(
                [(0, 720), (430, 330), (1023, 760), (1023, 1023), (0, 1023)],
                fill=(52, 61, 52),
            )
            image.save(output_path)
            return Path(output_path)

        with (
            patch.object(studio, "RUNS_ROOT", self.root / "runs"),
            patch.object(studio, "plan_motifs", side_effect=self.fake_plan),
            patch.object(studio, "generate_motif_panel", side_effect=self.fake_sheet),
            patch.object(pipeline, "edit_image", side_effect=fake_edit),
        ):
            state, rows, summary, crops, plan_file = studio.analyze_painting(
                self.source, 2, "openai"
            )
            self.assertIn("river landscape", summary)
            self.assertEqual(len(crops), 2)
            self.assertTrue(Path(plan_file).exists())
            state, sheet, motifs, bank, _ = studio.create_motif_bank(
                state, rows, None, "openai", "", "1024x1536", "high", 2
            )
            bank_payload = json.loads(Path(bank).read_text(encoding="utf-8"))
            self.assertEqual(bank_payload["image_provider"], "openai")
            self.assertEqual(bank_payload["source_kind"], "generated")
            self.assertTrue(
                all(not Path(item["asset_path"]).is_absolute() for item in bank_payload["motifs"])
            )
            self.assertEqual(len(motifs), 2)
            state, output, gallery, report, archive, status = studio.compose_painting(
                state,
                rows,
                "A new autumn river landscape with mist and an old pavilion.",
                "autumn_test",
                0.62,
                "top_sky",
                0.68,
                0.62,
                7,
                False,
                True,
                "Restrained mineral color.",
                "openai",
                "",
                "1024x1024",
                "high",
            )
            self.assertEqual(reference_counts, [4])
            self.assertTrue(Path(output).exists())
            self.assertEqual(len(gallery), 1)
            self.assertIn("Painting ready", status)
            payload = json.loads(Path(report).read_text(encoding="utf-8"))
            self.assertEqual(payload["target_void"], 0.62)
            self.assertIn("realized_void_region", payload)
            self.assertNotIn("api_key", json.dumps(payload).lower())
            resumed = studio.resume_run(Path(state["run_dir"]).name)
            self.assertEqual(resumed["result"], output)
            self.assertEqual(len(resumed["motifs"]), 2)
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                self.assertIn("MANIFEST.json", names)
                manifest = json.loads(bundle.read("MANIFEST.json"))
                for name, digest in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), digest)
                report_in_zip = json.loads(
                    bundle.read(
                        Path(report)
                        .resolve()
                        .relative_to(Path(state["run_dir"]).resolve())
                        .as_posix()
                    )
                )
            self.assertNotIn(str(Path(state["run_dir"])), json.dumps(report_in_zip))
            self.assertTrue(Path(sheet).is_file())

    @patch("inkvoidmotif.studio.resolve_provider", return_value="openai")
    def test_uploaded_bank_skips_provider_resolution(self, _provider) -> None:
        with (
            patch.object(studio, "RUNS_ROOT", self.root / "runs"),
            patch.object(studio, "plan_motifs", side_effect=self.fake_plan),
        ):
            state, rows, *_ = studio.analyze_painting(self.source, 2, "openai")
            sheet = self.root / "uploaded.png"
            self.fake_sheet(sheet)
            with patch.object(
                studio, "resolve_provider", side_effect=AssertionError("provider call")
            ):
                state, _, _, bank, _ = studio.create_motif_bank(
                    state, rows, sheet, "auto", "", "1024x1536", "high", 2
                )
            payload = json.loads(Path(bank).read_text(encoding="utf-8"))
            self.assertEqual(payload["image_provider"], "uploaded")
            self.assertEqual(state["bank_plan_hash"], state["plan_hash"])

    @patch("inkvoidmotif.studio.resolve_provider", return_value="openai")
    def test_changed_plan_must_rebuild_bank(self, _provider) -> None:
        with (
            patch.object(studio, "RUNS_ROOT", self.root / "runs"),
            patch.object(studio, "plan_motifs", side_effect=self.fake_plan),
            patch.object(studio, "generate_motif_panel", side_effect=self.fake_sheet),
        ):
            state, rows = self.prepared_run()
            changed = [list(row) for row in rows]
            changed[0][0] = "renamed"
            with (
                patch.object(studio, "compose_with_layout") as compose,
                self.assertRaisesRegex(studio.StudioError, "Rebuild"),
            ):
                studio.compose_painting(
                    state,
                    changed,
                    "A sufficiently detailed new mountain scene",
                    "test",
                    0.6,
                    "top_sky",
                    0.7,
                    0.5,
                    2,
                    False,
                    False,
                    "",
                    "openai",
                    "",
                    "1024x1024",
                    "high",
                )
            compose.assert_not_called()

    @patch("inkvoidmotif.studio.resolve_provider", return_value="openai")
    def test_failed_analysis_removes_orphan_run(self, _provider) -> None:
        with (
            patch.object(studio, "RUNS_ROOT", self.root / "runs"),
            patch.object(studio, "plan_motifs", side_effect=RuntimeError("provider failed")),
            self.assertRaises(RuntimeError),
        ):
            studio.analyze_painting(self.source, 2, "openai")
        self.assertEqual(studio.list_run_ids(), [])

    @patch("inkvoidmotif.studio.resolve_provider", return_value="openai")
    def test_failed_compositions_get_unique_diagnostics(self, _provider) -> None:
        with (
            patch.object(studio, "RUNS_ROOT", self.root / "runs"),
            patch.object(studio, "plan_motifs", side_effect=self.fake_plan),
            patch.object(studio, "generate_motif_panel", side_effect=self.fake_sheet),
        ):
            state, rows = self.prepared_run()
            with patch.object(studio, "compose_with_layout", side_effect=TimeoutError("timeout")):
                for _ in range(2):
                    with self.assertRaises(TimeoutError):
                        studio.compose_painting(
                            state,
                            rows,
                            "A sufficiently detailed new mountain scene",
                            "same",
                            0.6,
                            "top_sky",
                            0.7,
                            0.5,
                            2,
                            False,
                            False,
                            "",
                            "openai",
                            "",
                            "1024x1024",
                            "high",
                        )
            failures = list((Path(state["run_dir"]) / "compositions").glob("*.failure.json"))
            self.assertEqual(len(failures), 2)
            self.assertNotEqual(failures[0].name, failures[1].name)
            self.assertNotIn("timeout", failures[0].read_text(encoding="utf-8"))

    def test_region_void_uses_pipeline_metric_exactly(self) -> None:
        from inkvoidmotif.layout import region_void

        expected = region_void(self.source)["void"]
        self.assertEqual(studio.estimate_void_ratio(self.source), expected)

    def test_friendly_errors_do_not_echo_provider_payloads(self) -> None:
        message = studio.friendly_error(RuntimeError("401 secret-token unauthorized"))
        self.assertNotIn("secret-token", message)
        self.assertIn("API key", message)

    def test_state_paths_cannot_escape_runs_root(self) -> None:
        rows = studio.plan_to_rows(sample_plan(self.source))
        state = {
            "run_dir": str(self.root),
            "plan_path": str(self.root / "outside.json"),
        }
        with (
            patch.object(studio, "RUNS_ROOT", self.root / "allowed"),
            self.assertRaises(studio.StudioError),
        ):
            studio.save_table_plan(state, rows)

    def test_download_rejects_non_https_and_private_hosts_without_network(self) -> None:
        for url in ("http://example.com/image.png", "https://127.0.0.1/image.png"):
            with self.assertRaises(RuntimeError):
                download_image(url, self.root / "output.png")

    def test_generated_image_decoder_rejects_invalid_payload(self) -> None:
        import base64

        with self.assertRaises(RuntimeError):
            decode_b64_image(base64.b64encode(b"not an image").decode(), self.root / "bad.png")
        self.assertFalse((self.root / "bad.png").exists())

    def test_provider_auto_honors_environment_override(self) -> None:
        with (
            patch.dict(os.environ, {"INKVOIDMOTIF_PROVIDER": "openai"}, clear=False),
            patch.object(
                studio,
                "provider_readiness",
                return_value={"openai": True, "nvidia": True},
            ),
        ):
            self.assertEqual(studio.resolve_provider("auto"), "openai")

    def test_default_runs_root_is_not_inside_the_installed_package(self) -> None:
        package_root = Path(studio.__file__).resolve().parent
        self.assertFalse(studio.RUNS_ROOT.resolve().is_relative_to(package_root))
        if "INKVOIDMOTIF_STUDIO_RUNS" not in os.environ:
            self.assertEqual(studio.RUNS_ROOT.resolve(), Path.cwd() / "runs" / "studio")

    def test_placeholder_nvidia_keys_are_not_treated_as_configured(self) -> None:
        for value in ("false", "nvapi-your-key-here", "off", "none"):
            with patch.dict(os.environ, {"NVIDIA_API_KEY": value}, clear=False):
                self.assertIsNone(nvidia_api_key())

    def test_batch_rerun_skips_sources_already_in_combined_bank(self) -> None:
        paintings = self.root / "paintings"
        bank_dir = self.root / "combined"
        paintings.mkdir()
        bank_dir.mkdir()
        source = paintings / "Autumn_scene.png"
        self.source.replace(source)
        pipeline.write_json(
            bank_dir / "motif_bank.json",
            {"sources": [source.name], "motifs": [], "image_provider": "openai"},
        )
        with patch("inkvoidmotif.batch_segment_motifs.plan_motifs") as planner:
            output = batch_segment_paintings(paintings, bank_dir, skip_existing=True)
        planner.assert_not_called()
        self.assertEqual(pipeline.read_json(output)["sources"], [source.name])

    def test_custom_theme_can_compare_to_builtin_themes(self) -> None:
        custom = theme()
        self.assertIsInstance(custom == __import__("gradio").themes.Default(), bool)

    def test_gui_places_three_workflow_panels_above_preview(self) -> None:
        with warnings.catch_warnings():
            # Gradio 6.26 allocates short-lived event loops while serializing Blocks.
            warnings.simplefilter("ignore", ResourceWarning)
            demo = build_demo()
            try:
                config = demo.config
                components = {component["id"]: component for component in config["components"]}

                workflow = next(
                    component
                    for component in config["components"]
                    if "workflow-row" in component["props"].get("elem_classes", [])
                )
                preview = next(
                    component
                    for component in config["components"]
                    if "preview-stage" in component["props"].get("elem_classes", [])
                )
                root_children = config["layout"]["children"]
                workflow_layout = next(
                    child for child in root_children if child["id"] == workflow["id"]
                )
                workflow_panels = [
                    components[child["id"]]
                    for child in workflow_layout["children"]
                    if "workflow-panel" in components[child["id"]]["props"].get("elem_classes", [])
                ]

                self.assertEqual(len(workflow_panels), 3)
                root_ids = [child["id"] for child in root_children]
                self.assertLess(root_ids.index(workflow["id"]), root_ids.index(preview["id"]))
                rendered_text = CSS + json.dumps(config, ensure_ascii=False, default=str)
                self.assertNotRegex(rendered_text, r"[\u3400-\u9fff]")
                self.assertNotIn(".studio-hero:after", CSS)
                self.assertNotIn('content: "VOID"', CSS)
            finally:
                demo.close()
                del demo
                gc.collect()

    def test_gui_api_does_not_accept_a_compose_time_sheet(self) -> None:
        names = inspect.signature(ui_compose).parameters
        self.assertNotIn("uploaded_sheet", names)
        self.assertIn("quota_acknowledged", names)

    def test_gui_requests_source_before_quota_acknowledgement(self) -> None:
        with self.assertRaisesRegex(gr.Error, "Upload a source painting first"):
            ui_analyze(None, 8, "auto", "", "", False)

    def test_safe_maintenance_modules_do_nothing_on_import(self) -> None:
        from inkvoidmotif.MotifBank.remove_hallucinations import quarantine_hallucinations
        from inkvoidmotif.remove_extracted_paintings import quarantine_extracted_sources

        review, motifs, sources = (
            self.root / "review",
            self.root / "motifs",
            self.root / "sources",
        )
        for directory in (review, motifs, sources):
            directory.mkdir()
        (review / "01_tree.png").write_bytes(b"review")
        (motifs / "02_tree.png").write_bytes(b"motif")
        (sources / "done.png").write_bytes(b"source")
        (review / "done.png").write_bytes(b"extracted")
        self.assertEqual(len(quarantine_hallucinations(review, motifs, self.root / "q")), 1)
        self.assertEqual(len(quarantine_extracted_sources(review, sources, self.root / "q2")), 1)
        self.assertTrue((motifs / "02_tree.png").exists())
        self.assertTrue((sources / "done.png").exists())


if __name__ == "__main__":
    unittest.main()

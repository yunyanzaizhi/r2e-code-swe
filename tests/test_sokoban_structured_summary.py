import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_system.environments.env_manager import SokobanEnvironmentManager
from agent_system.memory.structured_summary import StructuredSummaryMemory, parse_sokoban_grid
from experiments.analyze_results import summarize_experiment_result


def _grid(*rows: str) -> str:
    return "\n".join(" \t ".join(row) for row in rows)


class DummyEnv:
    mode = "tiny_rgb_array"


class StructuredSummaryTests(unittest.TestCase):
    def test_parse_sokoban_grid_extracts_entities_and_box_on_target(self):
        parsed = parse_sokoban_grid(
            _grid(
                "######",
                "#_S__#",
                "#_√__#",
                "#_X_O#",
                "#____#",
                "######",
            )
        )

        self.assertEqual(parsed["height"], 6)
        self.assertEqual(parsed["width"], 6)
        self.assertEqual(parsed["player"], (2, 3))
        self.assertEqual(parsed["boxes"], [(3, 3), (4, 3)])
        self.assertEqual(parsed["boxes_on_target"], [(3, 3)])
        self.assertEqual(parsed["targets"], [(2, 3), (3, 3), (4, 5)])
        self.assertEqual(parsed["remaining_boxes"], 1)

    def test_structured_summary_memory_outputs_planning_focused_schema(self):
        memory = StructuredSummaryMemory()
        memory.reset(batch_size=1)

        before_push = _grid(
            "######",
            "#____#",
            "#_#__#",
            "#PXO_#",
            "#____#",
            "######",
        )
        after_push = _grid(
            "######",
            "#____#",
            "#_#__#",
            "#_P√_#",
            "#____#",
            "######",
        )

        memory.store({
            "text_obs": [before_push],
            "result_text_obs": [after_push],
            "action": ["Right"],
        })
        memory.store({
            "text_obs": [after_push],
            "result_text_obs": [after_push],
            "action": ["Up"],
        })
        memory.store({
            "text_obs": [after_push],
            "result_text_obs": [after_push],
            "action": ["Up"],
        })

        summaries, valid_lengths = memory.fetch(history_length=100)
        summary = summaries[0]

        self.assertEqual(valid_lengths, [1])
        for label in [
            "[Goal]",
            "[Progress]",
            "[State Snapshot]",
            "[Key State Changes]",
            "[Risks]",
            "[Recent Useful Trace]",
            "[Recommended Focus]",
        ]:
            self.assertIn(label, summary)

        self.assertIn("Step 3", summary)
        self.assertIn("Boxes on targets: 1/1", summary)
        self.assertIn("Invalid / no-op moves: 2", summary)
        self.assertIn("Step 1 / Right", summary)
        self.assertIn("onto target", summary)
        self.assertNotIn("Step 2 / Up", summary.split("[Recent Useful Trace]")[1].split("[Recommended Focus]")[0])
        self.assertTrue("repeated action" in summary.lower() or "up x2" in summary.lower())

    def test_env_manager_builds_prompt_with_structured_summary(self):
        manager = SokobanEnvironmentManager(
            DummyEnv(),
            projection_f=lambda actions: (actions, [True for _ in actions]),
            config=SimpleNamespace(
                env=SimpleNamespace(
                    history_length=100,
                    sokoban=SimpleNamespace(memory_type="structured_summary"),
                )
            ),
        )
        manager.memory = StructuredSummaryMemory()
        manager.memory.reset(batch_size=1)

        before_push = _grid(
            "######",
            "#____#",
            "#_#__#",
            "#PXO_#",
            "#____#",
            "######",
        )
        after_push = _grid(
            "######",
            "#____#",
            "#_#__#",
            "#_P√_#",
            "#____#",
            "######",
        )
        manager.memory.store({
            "text_obs": [before_push],
            "result_text_obs": [after_push],
            "action": ["Right"],
        })

        prompt = manager.build_text_obs(infos=[{}], text_obs=[after_push], init=False)[0]
        self.assertIn("# Structured Task Summary", prompt)
        self.assertIn("[Goal]", prompt)
        self.assertIn("[Recommended Focus]", prompt)
        self.assertIn("not a full history transcript", prompt.lower())

    def test_sokoban_run_scripts_enable_checkpoint_saving(self):
        repo_root = Path(__file__).resolve().parents[1]
        scripts = [
            repo_root / "experiments/run_sokoban_recent_window.sh",
            repo_root / "experiments/run_sokoban_full_history.sh",
            repo_root / "experiments/run_sokoban_structured_summary.sh",
        ]

        for script in scripts:
            script_text = script.read_text()
            self.assertNotIn("trainer.save_freq=-1", script_text, msg=script.name)
            self.assertIn("trainer.save_freq=999999", script_text, msg=script.name)

    def test_sokoban_run_scripts_pin_artifacts_to_repo_paths(self):
        repo_root = Path(__file__).resolve().parents[1]
        expected_snippets = {
            "run_sokoban_recent_window.sh": [
                "mkdir -p experiments/logs experiments/results checkpoints/sokoban_history_exp /tmp/verl-ray",
                'export RAY_TMPDIR="/tmp/verl-ray"',
                'trainer.default_local_dir=$PWD/checkpoints/sokoban_history_exp/recent_window_k${history_length}',
                '2>&1 | tee experiments/logs/recent_window_k${history_length}.log',
            ],
            "run_sokoban_full_history.sh": [
                "mkdir -p experiments/logs experiments/results checkpoints/sokoban_history_exp /tmp/verl-ray",
                'export RAY_TMPDIR="/tmp/verl-ray"',
                'trainer.default_local_dir=$PWD/checkpoints/sokoban_history_exp/full_history',
                '2>&1 | tee experiments/logs/full_history.log',
            ],
            "run_sokoban_structured_summary.sh": [
                "mkdir -p experiments/logs experiments/results checkpoints/sokoban_history_exp /tmp/verl-ray",
                'export RAY_TMPDIR="/tmp/verl-ray"',
                'trainer.default_local_dir=$PWD/checkpoints/sokoban_history_exp/structured_summary',
                '2>&1 | tee experiments/logs/structured_summary.log',
            ],
        }

        for script_name, snippets in expected_snippets.items():
            script_text = (repo_root / "experiments" / script_name).read_text()
            for snippet in snippets:
                self.assertIn(snippet, script_text, msg=f"{script_name} missing {snippet}")
            self.assertNotIn("mkdir -p experiments/logs experiments/results ray checkpoints/sokoban_history_exp", script_text, msg=script_name)

    def test_sokoban_run_scripts_use_short_tmp_ray_dir_outside_repo(self):
        ray_tmpdir = Path("/tmp/verl-ray")
        socket_path = ray_tmpdir / "ray" / "session_2026-04-17_11-03-26_223108_787106" / "sockets" / "plasma_store"
        self.assertLess(len(str(socket_path)), 108)

        repo_root = Path(__file__).resolve().parents[1]
        for script_name in [
            "run_sokoban_recent_window.sh",
            "run_sokoban_full_history.sh",
            "run_sokoban_structured_summary.sh",
        ]:
            script_text = (repo_root / "experiments" / script_name).read_text()
            self.assertIn('export RAY_TMPDIR="/tmp/verl-ray"', script_text, msg=script_name)
            self.assertNotIn('export RAY_TMPDIR="$PWD/ray"', script_text, msg=script_name)
            self.assertNotIn('export RAY_TMPDIR="$PWD/experiments/ray"', script_text, msg=script_name)

    def test_summarize_experiment_result_marks_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "recent_window_k5.log"
            log_path.write_text(
                "\n".join(
                    [
                        "step:0 - val/success_rate:0.125",
                        "step:1 - episode/success_rate:0.250 - val/success_rate:0.500 - episode/length/mean:11.000 - episode/reward/mean:0.100 - prompt_length/mean:100.000 - response_length/mean:25.000 - timing_s/step:10.000 - episode/valid_action_ratio:0.750",
                        "step:2 - episode/success_rate:0.500 - val/success_rate:0.625 - episode/length/mean:9.000 - episode/reward/mean:0.300 - prompt_length/mean:110.000 - response_length/mean:26.000 - timing_s/step:12.000 - episode/valid_action_ratio:0.800",
                    ]
                )
            )

            result = summarize_experiment_result(
                strategy="Recent Window K=5",
                log_path=log_path,
                expected_total_steps=50,
                max_steps=15,
            )

        required_fields = {
            "strategy",
            "source_log",
            "completed_steps",
            "expected_total_steps",
            "is_complete",
            "train_success_rate_last",
            "train_success_rate_max",
            "val_success_rate_last",
            "val_success_rate_max",
            "train_episode_length_mean",
            "train_reward_mean",
            "prompt_tokens_mean",
            "response_tokens_mean",
            "epoch_time_mean",
            "valid_action_ratio_mean",
            "train_tail_mean",
            "train_tail_std",
            "val_tail_mean",
            "val_tail_std",
            "peak_to_late_drop",
            "late_len_mean",
            "horizon_cap_ratio",
            "late_prompt_mean",
            "late_step_time_s",
            "late_valid_action_ratio",
        }

        self.assertTrue(required_fields.issubset(result.keys()))
        self.assertEqual(result["strategy"], "Recent Window K=5")
        self.assertEqual(result["source_log"], str(log_path))
        self.assertEqual(result["completed_steps"], 3)
        self.assertEqual(result["expected_total_steps"], 50)
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["train_success_rate_last"], 0.5)
        self.assertEqual(result["val_success_rate_max"], 0.625)
        self.assertEqual(result["prompt_tokens_mean"], 105.0)


if __name__ == "__main__":
    unittest.main()

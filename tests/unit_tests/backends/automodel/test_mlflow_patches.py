###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the AutoModel MLflow patch.

A real tracking run needs a GPU, a recipe and a server, so what is worth testing
here is everything that fails quietly or dangerously: an environment filter that
lets a credential through, a patch that tracks a run nobody asked to track, a
config section leaking into AutoModel, and a crashed run reported as FINISHED.

No torch, no MLflow and no AutoModel required. ``collect_env_params`` is pure,
and ``install()`` imports the recipe inside the function, so it is exercised
here against stand-ins.
"""

import os
import sys
from types import SimpleNamespace

import pytest

from primus.backends.automodel import options
from primus.backends.automodel.argument_builder import strip_primus_keys
from primus.backends.automodel.patches import mlflow_patches
from primus.backends.automodel.patches.mlflow_patches import collect_env_params
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry


class TestEnvAllowList:
    """The filter that decides what leaves the machine.

    These are the tests that justify the allow-list. The Megatron path filters
    os.environ by name against /(TOKEN|SECRET|KEY)/i, which passes
    MLFLOW_TRACKING_PASSWORD and DATABRICKS_PASSWORD through to the tracking
    server as readable params. Every credential below is obviously fake.
    """

    def test_drops_credentials_a_name_denylist_would_miss(self):
        env = {
            # Neither contains TOKEN, SECRET or KEY: both defeat the Megatron filter.
            "MLFLOW_TRACKING_PASSWORD": "not-a-real-password",
            "DATABRICKS_PASSWORD": "also-not-real",
            # These the deny-list would have caught; they must still go.
            "DATABRICKS_TOKEN": "dapi-fake-value-for-testing-only",
            "AWS_SECRET_ACCESS_KEY": "fake/secret/for/testing",
            # Not a credential, but not on the allow-list either.
            "DATABRICKS_HOST": "https://example.invalid",
        }
        assert collect_env_params(environ=env) == {}

    def test_keeps_the_provenance_we_actually_want(self):
        env = {
            "ROCM_VERSION": "7.1.0",
            "PRIMUS_IMAGE_DIGEST": "sha256:abc",
            "NCCL_DEBUG": "INFO",
            "HIP_VISIBLE_DEVICES": "0,1,2,3",
            "WORLD_SIZE": "8",
            "SLURM_NODELIST": "node[01-04]",
        }
        got = collect_env_params(environ=env)
        assert got == {
            "env__ROCM_VERSION": "7.1.0",
            "env__PRIMUS_IMAGE_DIGEST": "sha256:abc",
            "env__NCCL_DEBUG": "INFO",
            "env__HIP_VISIBLE_DEVICES": "0,1,2,3",
            "env__WORLD_SIZE": "8",
            "env__SLURM_NODELIST": "node[01-04]",
        }

    def test_unlisted_names_are_dropped_rather_than_inspected(self):
        """The allow-list property: an unknown variable never reaches the value check.

        This is the difference from a deny-list. A variable nobody anticipated
        is dropped because it was not named, not because it looked suspicious.
        """
        env = {"SOME_NEW_INTERNAL_VAR": "harmless", "HOME": "/home/someone"}
        assert collect_env_params(environ=env) == {}

    def test_entropy_backstop_catches_a_credential_in_an_allowed_name(self):
        """A token stuffed into an allowed PRIMUS_* name is still dropped."""
        env = {"PRIMUS_SESSION": "aZ3kQ9mP2xR7vL4nB8tY6wS1dF5gH0jK"}
        assert collect_env_params(environ=env) == {}

    def test_entropy_backstop_does_not_eat_ordinary_long_values(self):
        """Entropy alone cannot do this job, which is why value *shape* gates it.

        Measured: this path scores 4.42 bits/char, higher than a real 20-char
        Databricks PAT at 4.13. A pure entropy threshold that catches the token
        also drops the path. Structured values (containing "/", ":",
        ".") never reach the entropy test at all.
        """
        env = {"PRIMUS_CONFIG_PATH": "/workspace/Primus/examples/automodel/configs/MI355X/diffusion"}
        got = collect_env_params(environ=env)
        assert "env__PRIMUS_CONFIG_PATH" in got

    def test_digests_are_kept_despite_looking_like_tokens(self):
        """A 40-char SHA is provenance worth having and scores ~3.9 bits/char.

        Lowercase hex of a digest length is exempted; a credential is rarely
        pure hex, and almost never is when it carries a prefix or mixed case.
        """
        sha = "a" * 7 + "bcdef0123456789abcdef0123456789abc"
        got = collect_env_params(environ={"PRIMUS_COMMIT": sha})
        assert got == {"env__PRIMUS_COMMIT": sha}

    def test_mixed_case_token_of_digest_length_is_still_dropped(self):
        """The digest exemption is hex-only, so it cannot be used as a bypass."""
        env = {"PRIMUS_THING": "aZ3kQ9mP2xR7vL4nB8tY6wS1dF5gH0jKpQ7rT2wX"}
        assert collect_env_params(environ=env) == {}

    def test_secretish_name_dropped_even_when_allow_listed(self):
        """Belt and braces: an allowed prefix plus a credential-shaped name."""
        env = {"PRIMUS_API_KEY": "short", "PRIMUS_AUTH": "short"}
        assert collect_env_params(environ=env) == {}

    def test_extra_names_extend_the_allow_list(self):
        env = {"MY_SITE_VAR": "value"}
        assert collect_env_params(environ=env) == {}
        got = collect_env_params(environ=env, extra_names=["MY_SITE_VAR"])
        assert got == {"env__MY_SITE_VAR": "value"}

    def test_extra_names_cannot_smuggle_a_credential_past_the_backstop(self):
        """Allow-listing a name does not waive the value check."""
        env = {"MY_SITE_TOKEN": "anything"}
        assert collect_env_params(environ=env, extra_names=["MY_SITE_TOKEN"]) == {}


class TestConfigSections:
    """`mlflow:` must reach AutoModel; `primus_mlflow:` must not."""

    def test_primus_mlflow_is_stripped_before_automodel_sees_it(self):
        cleaned = strip_primus_keys({"primus_mlflow": {"enable": True}, "seed": 42})
        assert "primus_mlflow" not in cleaned
        assert cleaned["seed"] == 42

    def test_mlflow_block_survives_because_automodel_owns_it(self):
        """RecipeConfig.mlflow reads a top-level `mlflow:` block.

        Stripping it would silently disable tracking: the patch would run,
        cfg.mlflow would be None, and the run would go untracked with only a
        warning. Guard the distinction so a future change to the strip rules
        cannot make that happen quietly.
        """
        cleaned = strip_primus_keys({"mlflow": {"experiment_name": "x"}, "primus_mlflow": {}})
        assert cleaned["mlflow"] == {"experiment_name": "x"}


class TestPrimusSectionParams:
    """Primus-side feature gates must reach MLflow.

    Regression guard for an observed bug. ``_start_run`` logs
    ``recipe.cfg.to_dict()``, which is the config AFTER ``strip_primus_keys``
    has removed every ``primus_*`` section so AutoModel sees a clean recipe.
    The consequence, observed on a real FP8 run: ``primus_turbo.fp8_linear``
    appeared nowhere in the logged params, so an FP8 run and a BF16 run were
    indistinguishable in MLflow. The only trace was an ambient
    ``env__PRIMUS_TURBO_*`` variable that is set either way.
    """

    def test_feature_gates_are_logged(self):
        options.load(
            SimpleNamespace(
                primus_turbo={"fp8_linear": True, "fp8_linear_granularity": "tensorwise"},
                primus_mlflow={"enable": True},
            )
        )
        got = mlflow_patches._primus_section_params()
        assert got["primus__primus_turbo.fp8_linear"] == "True"
        assert got["primus__primus_turbo.fp8_linear_granularity"] == "tensorwise"
        assert got["primus__primus_mlflow.enable"] == "True"

    def test_an_fp8_run_is_distinguishable_from_a_bf16_one(self):
        """The property that actually matters: the two must not log the same
        inputs. Before the fix they did."""
        options.load(SimpleNamespace(primus_turbo={"fp8_linear": True}))
        fp8 = mlflow_patches._primus_section_params()
        options.load(SimpleNamespace(primus_turbo={"fp8_linear": False}))
        bf16 = mlflow_patches._primus_section_params()
        assert fp8 != bf16

    def test_unset_values_are_omitted_rather_than_logged_as_none(self):
        options.load(SimpleNamespace(primus_turbo={"fp8_linear": True, "fp8_linear_block_size": None}))
        got = mlflow_patches._primus_section_params()
        assert "primus__primus_turbo.fp8_linear_block_size" not in got

    def test_no_sections_is_empty_not_an_error(self):
        options.load(SimpleNamespace())
        assert mlflow_patches._primus_section_params() == {}


class TestDerivedPrecision:
    """`tags.precision` must answer "what precision ran?" in one click.

    The three signals that look like they should answer it, and why each fails:
    the config filename is never logged; the experiment path is a convention we
    chose, so an FP8 run without the override lands in the bf16 experiment; and
    model.compute_dtype stays bfloat16 on an FP8 run. Measured on a real pair
    of runs, compute_dtype / torch_dtype / reduce_dtype were IDENTICAL across
    BF16 and FP8.
    """

    class _Cfg:
        """Stands in for a recipe: _derive_precision reads ``recipe.cfg``."""

        def __init__(self, dtype="bfloat16"):
            self.cfg = SimpleNamespace(
                get=lambda key, default=None: dtype if key == "model.compute_dtype" else default
            )

    def test_bf16_when_no_low_precision_gate_is_set(self):
        options.load(SimpleNamespace())
        assert mlflow_patches._derive_precision(self._Cfg())["precision"] == "bf16"

    def test_fp8_from_the_resolved_gate_not_the_dtype(self):
        """compute_dtype is still bfloat16 here -- that is the whole point."""
        options.load(
            SimpleNamespace(
                primus_turbo={
                    "fp8_linear": True,
                    "fp8_linear_granularity": "tensorwise",
                    "fp8_linear_format": "hybrid",
                }
            )
        )
        got = mlflow_patches._derive_precision(self._Cfg(dtype="bfloat16"))
        assert got["precision"] == "fp8"
        assert got["precision_detail"] == "fp8/tensorwise/hybrid"

    def test_mx_blockwise_granularity_reports_mxfp8(self):
        options.load(
            SimpleNamespace(primus_turbo={"fp8_linear": True, "fp8_linear_granularity": "mx_blockwise"})
        )
        assert mlflow_patches._derive_precision(self._Cfg())["precision"] == "mxfp8"

    @pytest.mark.parametrize("section", ("primus_turbo", "primus_te"))
    def test_mxfp4_wins_over_fp8_from_either_section(self, section):
        """Both gates on means mxfp4 where they overlap, so it is the headline."""
        options.load(
            SimpleNamespace(
                **(
                    {section: {"mxfp4_linear": True}, "primus_turbo": {"fp8_linear": True}}
                    if section == "primus_te"
                    else {section: {"mxfp4_linear": True, "fp8_linear": True}}
                )
            )
        )
        assert mlflow_patches._derive_precision(self._Cfg())["precision"] == "mxfp4"

    def test_attention_precision_is_reported_separately(self):
        """Linear and attention precision are independently settable gates."""
        options.load(SimpleNamespace(primus_turbo={"fp8_linear": True, "fp8_attention": True}))
        got = mlflow_patches._derive_precision(self._Cfg())
        assert got["precision"] == "fp8"
        assert got["precision_attention"] == "fp8"

    def test_bf16_and_fp8_runs_are_distinguishable(self):
        """The property that matters, stated directly."""
        options.load(SimpleNamespace())
        bf16 = mlflow_patches._derive_precision(self._Cfg())
        options.load(SimpleNamespace(primus_turbo={"fp8_linear": True}))
        fp8 = mlflow_patches._derive_precision(self._Cfg())
        assert bf16["precision"] != fp8["precision"]


class TestRegistration:
    def test_patch_is_registered_for_before_train(self):
        """`after_train` is never dispatched by AutoModelPretrainTrainer, so a
        teardown patch registered there would silently never fire."""
        import primus.backends.automodel.patches  # noqa: F401

        names = [p.id for p in PatchRegistry.iter_patches("automodel", "before_train")]
        assert "automodel.mlflow" in names

    def test_disabled_by_default(self):
        """Disabled must mean the patch body never runs -- no client, no network.

        The gate is the patch condition, so run_patches skips it entirely rather
        than calling a body that returns early.
        """
        options.load(SimpleNamespace())
        ctx = PatchContext(backend="automodel", phase="before_train")
        assert mlflow_patches._enabled(ctx) is False

    def test_enabled_only_by_the_primus_section(self):
        options.load(SimpleNamespace(primus_mlflow=SimpleNamespace(enable=True)))
        ctx = PatchContext(backend="automodel", phase="before_train")
        assert mlflow_patches._enabled(ctx) is True

    def test_every_declared_key_is_recognised(self):
        """A misspelled key reads as unset and turns the feature off silently;
        the trainer's undeclared() warning is what surfaces it."""
        options.load(
            SimpleNamespace(
                primus_mlflow=SimpleNamespace(
                    enable=False,
                    per_step=False,
                    env_allowlist_extra=[],
                    upload_logs=True,
                    upload_traces=False,
                    upload_tracelens_report=False,
                    tracelens_ranks=None,
                    tracelens_output_format="xlsx",
                    upload_max_bytes=2147483648,
                    summary=True,
                )
            )
        )
        assert options.undeclared() == []


class _FakeRecipeModule(SimpleNamespace):
    pass


@pytest.fixture
def fake_recipe_class(monkeypatch):
    """Stand in for nemo_automodel's TrainDiffusionRecipe, which needs a GPU."""

    class TrainDiffusionRecipe:
        calls = []

        def run_train_validation_loop(self):
            TrainDiffusionRecipe.calls.append("ran")
            return "done"

    module = _FakeRecipeModule(TrainDiffusionRecipe=TrainDiffusionRecipe)
    monkeypatch.setitem(sys.modules, "nemo_automodel", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "nemo_automodel.recipes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "nemo_automodel.recipes.diffusion", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "nemo_automodel.recipes.diffusion.train", module)
    return TrainDiffusionRecipe


class TestInstall:
    def test_install_is_idempotent(self, fake_recipe_class):
        assert mlflow_patches.install() is True
        wrapped_once = fake_recipe_class.run_train_validation_loop
        assert mlflow_patches.install() is True
        assert fake_recipe_class.run_train_validation_loop is wrapped_once

    def test_untracked_when_run_cannot_start(self, fake_recipe_class, monkeypatch):
        """Tracking is observability: a setup failure must not end the run."""
        monkeypatch.setattr(
            mlflow_patches, "_start_run", lambda recipe: (_ for _ in ()).throw(RuntimeError("no server"))
        )
        mlflow_patches.install()
        fake_recipe_class.calls.clear()
        assert fake_recipe_class().run_train_validation_loop() == "done"
        assert fake_recipe_class.calls == ["ran"]

    def test_successful_run_is_finished(self, fake_recipe_class, monkeypatch):
        statuses = []
        monkeypatch.setattr(mlflow_patches, "_start_run", lambda recipe: SimpleNamespace())
        monkeypatch.setattr(mlflow_patches, "_finish_run", statuses.append)
        mlflow_patches.install()
        fake_recipe_class().run_train_validation_loop()
        assert statuses == ["FINISHED"]

    def test_crashed_run_is_failed_not_finished(self, fake_recipe_class, monkeypatch):
        """The point of the whole exercise: evidence that says a crashed run
        succeeded is worse than no evidence. MLflow's atexit handler defaults to
        FINISHED, so the status has to be set explicitly on the way out."""
        statuses = []

        def boom(self):
            raise RuntimeError("OOM")

        fake_recipe_class.run_train_validation_loop = boom
        monkeypatch.setattr(mlflow_patches, "_start_run", lambda recipe: SimpleNamespace())
        monkeypatch.setattr(mlflow_patches, "_finish_run", statuses.append)
        mlflow_patches.install()

        with pytest.raises(RuntimeError, match="OOM"):
            fake_recipe_class().run_train_validation_loop()
        assert statuses == ["FAILED"]

    def test_keyboard_interrupt_is_also_failed(self, fake_recipe_class, monkeypatch):
        """KeyboardInterrupt is a BaseException, so an `except Exception` would
        let a cancelled run be recorded FINISHED by the atexit handler."""
        statuses = []

        def interrupted(self):
            raise KeyboardInterrupt

        fake_recipe_class.run_train_validation_loop = interrupted
        monkeypatch.setattr(mlflow_patches, "_start_run", lambda recipe: SimpleNamespace())
        monkeypatch.setattr(mlflow_patches, "_finish_run", statuses.append)
        mlflow_patches.install()

        with pytest.raises(KeyboardInterrupt):
            fake_recipe_class().run_train_validation_loop()
        assert statuses == ["FAILED"]


class TestRankExtraction:
    """Why this is not a copy of the Megatron rank filter.

    ``torch_profiler`` writes ``rank{rank}.pt.trace.json`` -- "rank0.", no
    separator before the digit. The Megatron patterns all require one, so a
    copied filter selects nothing and the failure presents as "no traces
    found".
    """

    def test_automodel_trace_name_resolves(self):
        assert mlflow_patches._extract_rank("rank0.pt.trace.json") == 0
        assert mlflow_patches._extract_rank("rank13.pt.trace.json.gz") == 13

    def test_megatron_trace_names_still_resolve(self):
        assert mlflow_patches._extract_rank("rank_0_step_2.json.gz") == 0
        assert mlflow_patches._extract_rank("rank_7.pt.trace.json") == 7
        assert mlflow_patches._extract_rank("primus-exp-rank[3].json") == 3

    def test_unranked_name_is_none_not_zero(self):
        """None must not collapse to rank 0, or an unranked trace joins every
        rank filter."""
        assert mlflow_patches._extract_rank("profile.pt.trace.json") is None

    def test_filter_keeps_only_requested_ranks(self):
        files = ["/t/rank0.pt.trace.json", "/t/rank1.pt.trace.json", "/t/rank2.pt.trace.json"]
        assert mlflow_patches._filter_by_rank(files, [0, 2]) == [
            "/t/rank0.pt.trace.json",
            "/t/rank2.pt.trace.json",
        ]

    def test_none_means_every_rank_and_empty_means_none(self):
        files = ["/t/rank0.pt.trace.json"]
        assert mlflow_patches._filter_by_rank(files, None) == files
        assert mlflow_patches._filter_by_rank(files, []) == []


class TestRankNormalization:
    def test_string_form_from_yaml_is_parsed(self):
        assert mlflow_patches._normalize_ranks("[0, 2]") == [0, 2]

    def test_unparseable_value_uploads_everything_rather_than_nothing(self):
        """An unreadable filter should over-collect, not silently collect
        nothing: too many traces is recoverable, a missing one is not."""
        assert mlflow_patches._normalize_ranks("not-a-list") is None
        assert mlflow_patches._normalize_ranks(42) is None

    def test_junk_entries_dropped_and_result_sorted_unique(self):
        assert mlflow_patches._normalize_ranks([3, "1", -5, True, 3, "x"]) == [1, 3]


class TestSizeCap:
    """Multinode trace sets reach gigabytes and upload from one process."""

    @staticmethod
    def _write(tmp_path, name, size):
        path = tmp_path / name
        path.write_bytes(b"\0" * size)
        return str(path)

    def test_cap_keeps_smallest_first(self, tmp_path):
        small = self._write(tmp_path, "rank0.pt.trace.json", 10)
        medium = self._write(tmp_path, "rank1.pt.trace.json", 100)
        large = self._write(tmp_path, "rank2.pt.trace.json", 10_000)
        kept, dropped = mlflow_patches._cap_total_size([large, small, medium], 200)
        assert kept == sorted([small, medium])
        assert dropped == [large]

    def test_single_oversized_file_is_still_kept(self, tmp_path):
        """A cap smaller than the only file should not yield an empty upload --
        one trace beats none."""
        only = self._write(tmp_path, "rank0.pt.trace.json", 5_000)
        kept, dropped = mlflow_patches._cap_total_size([only], 100)
        assert kept == [only] and dropped == []

    def test_zero_disables_the_cap(self, tmp_path):
        files = [self._write(tmp_path, f"rank{i}.pt.trace.json", 1_000) for i in range(3)]
        kept, dropped = mlflow_patches._cap_total_size(files, 0)
        assert sorted(kept) == sorted(files) and dropped == []


class TestRunSummary:
    """Parsed from the run's own log afterwards, not hooked per step.

    Nothing here may add a GPU sync or a collective to the step; the numbers
    are already in the log, so reading them after the fact costs the run
    nothing.
    """

    LINE = (
        "[TRAIN] step=%d epoch=0 loss=0.5 avg_loss=0.5 lr=1.0e-04 grad_norm=1.0 "
        "step_time=%.3fs samples_per_sec=8.00 samples_per_sec_per_gpu=%.2f mem=%.2fGB"
    )

    def test_first_record_is_excluded_from_every_statistic(self, tmp_path):
        """The first record is warmup, and not merely a slow one.

        On a real Flux run step 0 took 17.9s against a 1.56s steady state, and
        reported 36.46 GB against 47.54 GB -- memory is structurally lower
        because the optimizer state is not allocated yet. Including it both
        distorts the step time and *understates* the peak, which defeats the
        point of reporting a peak.
        """
        log = tmp_path / "info.log"
        log.write_text(
            "\n".join(
                self.LINE % (i, st, sps, mem)
                for i, (st, sps, mem) in enumerate(
                    [(9.0, 0.50, 5.0), (2.0, 2.00, 30.0), (3.0, 6.00, 20.0)]  # first is warmup
                )
            )
        )
        out = mlflow_patches._summarise_from_logs([str(log)])

        # medians over records 2 and 3 only: (2.0 + 3.0)/2 and (2.0 + 6.0)/2
        assert out["perf/iteration_time_ms"] == 2500.0
        assert out["perf/samples_per_sec_per_gpu"] == 4.00
        assert out["perf/max_memory_allocated_gb"] == 30.0  # not the 5.0 warmup
        assert out["perf/logged_steps"] == 3.0  # everything the run logged
        assert out["perf/summary_records"] == 2.0  # what the medians used

    def test_the_excluded_warmup_is_reported_not_discarded(self, tmp_path):
        """A compile step that suddenly doubles is worth seeing."""
        log = tmp_path / "info.log"
        log.write_text(
            "\n".join(self.LINE % (i, st, 1.0, mem) for i, (st, mem) in enumerate([(9.0, 5.0), (2.0, 30.0)]))
        )
        out = mlflow_patches._summarise_from_logs([str(log)])
        assert out["perf/warmup_iteration_time_ms"] == 9000.0
        assert out["perf/warmup_memory_allocated_gb"] == 5.0

    def test_a_single_record_run_keeps_that_record(self, tmp_path):
        """A wrong number beats no number, and summary_records says which case
        you are in."""
        log = tmp_path / "info.log"
        log.write_text(self.LINE % (0, 7.0, 1.5, 12.0))
        out = mlflow_patches._summarise_from_logs([str(log)])
        assert out["perf/iteration_time_ms"] == 7000.0
        assert out["perf/logged_steps"] == 1.0
        assert out["perf/summary_records"] == 1.0

    def test_no_tflops_field_is_emitted(self, tmp_path):
        """The diffusion recipe logs no FLOPs. Inventing one would make a
        published comparison wrong rather than missing."""
        log = tmp_path / "train.log"
        log.write_text(self.LINE % (0, 1.0, 1.0, 1.0))
        assert not any("tflop" in key.lower() for key in mlflow_patches._summarise_from_logs([str(log)]))

    def test_log_without_train_lines_yields_nothing(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("INFO something unrelated\n")
        assert mlflow_patches._summarise_from_logs([str(log)]) == {}

    def test_missing_file_does_not_raise(self):
        assert mlflow_patches._summarise_from_logs(["/nonexistent/train.log"]) == {}


class TestSummaryCountsThisRunOnly:
    """Regression guards for the first real tier-1 run, which reported 16 steps
    for a 4-step job.

    Two independent faults combined. The summary globbed ``*.log``, and a rank
    directory holds info/debug/error/warning logs whose ``[TRAIN]`` lines are
    byte-identical between info and debug -- so every step was counted twice.
    And logs are appended across runs in one workspace, so an earlier run's
    steps were still in the file when the next run summarised it.

    The medians survived in that run by luck (a minority of outliers does not
    move a median) but ``perf/max_memory_allocated_gb`` was already a max across
    two runs, and an A/B sweep puts ten runs in one workspace.

    Nothing in the suite caught this, because nothing compared the steps found
    against the steps the run was asked to take. That comparison is the first
    test here.
    """

    LINE = TestRunSummary.LINE

    @pytest.fixture(autouse=True)
    def _reset_offsets(self):
        """``_log_offsets`` is module state; a leaked value would silently make
        a later test pass for the wrong reason."""
        mlflow_patches._log_offsets = {}
        yield
        mlflow_patches._log_offsets = {}

    def _rank_log(self, tmp_path, rank=0, name="info.log"):
        directory = tmp_path / "logs" / "pre_trainer" / f"rank-{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name

    def _append_run(self, path, steps, step_time=4.3, sps=3.70, mem=36.93):
        """Append one run's worth of [TRAIN] lines, as the recipe would."""
        with open(path, "a", encoding="utf-8") as handle:
            for i in range(steps):
                handle.write(self.LINE % (i, step_time, sps, mem) + "\n")

    def _summarise(self, tmp_path):
        """Drive the exact path ``_upload_artifacts`` uses: same glob patterns,
        same offsets. A test that called ``_summarise_from_logs`` with a
        hand-picked file list would not have caught the original bug, because
        the bug was in which files got picked."""
        files = mlflow_patches._find_files(str(tmp_path / "logs"), mlflow_patches._SUMMARY_LOG_PATTERNS)
        return mlflow_patches._summarise_from_logs(files, mlflow_patches._log_offsets)

    @pytest.mark.parametrize("requested_steps", [1, 4, 20])
    def test_every_record_the_run_emitted_is_counted_exactly_once(self, tmp_path, requested_steps):
        """The run emitted N [TRAIN] records; the summary must report N.

        Named for records rather than steps on purpose. ``perf/logged_steps``
        counts **log records, not training steps**, and the two differ whenever
        ``step_scheduler.log_remote_every_steps`` is above 1 -- which is the
        live case, not a hypothetical: ``wan_synthetic_pre_trainer`` sets 1 and
        ``flux_synth_pre_trainer`` sets 2, so an identical 4-step run yields 4
        records on Wan and 2 on Flux -- the recipe's records are logging
        *windows*, not steps.

        ``test_record_count_follows_the_logging_interval`` pins that
        relationship; this one pins that nothing is double-counted or dropped.
        """
        log = self._rank_log(tmp_path)
        log.touch()
        mlflow_patches._snapshot_log_offsets(str(tmp_path))
        self._append_run(log, requested_steps)

        assert self._summarise(tmp_path)["perf/logged_steps"] == float(requested_steps)

    @pytest.mark.parametrize(
        "max_steps, log_every, expected_records",
        [
            (4, 1, 4),  # wan_synthetic_pre_trainer
            (4, 2, 2),  # flux_synth_pre_trainer -- observed on a real run
            (20, 2, 10),
            (20, 5, 4),
        ],
    )
    def test_record_count_follows_the_logging_interval(
        self, tmp_path, max_steps, log_every, expected_records
    ):
        """Guards against reading perf/logged_steps as a step count.

        Anyone comparing it against max_steps on a Flux run would see half the
        number and reasonably conclude the summary was dropping records -- the
        same shape as the bug this class exists for, with an innocent cause.
        Encoding the real relationship here means the distinction is written
        down somewhere a reader of the numbers will find it.
        """
        log = self._rank_log(tmp_path)
        log.touch()
        mlflow_patches._snapshot_log_offsets(str(tmp_path))
        with open(log, "a", encoding="utf-8") as handle:
            for step in range(0, max_steps, log_every):
                handle.write(self.LINE % (step, 4.3, 3.70, 36.93) + "\n")

        assert self._summarise(tmp_path)["perf/logged_steps"] == float(expected_records)

    def test_requested_steps_still_match_after_earlier_runs_dirtied_the_log(self, tmp_path):
        """The real failure: two previous runs left 8 lines in info.log, and a
        third 4-step run reported 16."""
        log = self._rank_log(tmp_path)
        self._append_run(log, 4)  # run 1
        self._append_run(log, 4)  # run 2

        mlflow_patches._snapshot_log_offsets(str(tmp_path))
        self._append_run(log, 4)  # run 3 -- the one being summarised

        assert self._summarise(tmp_path)["perf/logged_steps"] == 4.0

    def test_summary_reads_one_file_per_rank_not_every_log_level(self, tmp_path):
        """info.log and debug.log carry the same [TRAIN] lines on a real run."""
        for name in ("info.log", "debug.log", "error.log", "warning.log"):
            self._rank_log(tmp_path, name=name).touch()

        selected = mlflow_patches._find_files(str(tmp_path / "logs"), mlflow_patches._SUMMARY_LOG_PATTERNS)

        assert [os.path.basename(p) for p in selected] == ["info.log"]

    def test_identical_debug_log_does_not_double_the_count(self, tmp_path):
        info = self._rank_log(tmp_path, name="info.log")
        debug = self._rank_log(tmp_path, name="debug.log")
        mlflow_patches._snapshot_log_offsets(str(tmp_path))
        self._append_run(info, 4)
        debug.write_text(info.read_text(), encoding="utf-8")  # byte-identical

        assert self._summarise(tmp_path)["perf/logged_steps"] == 4.0

    def test_peak_memory_is_this_runs_not_a_cross_run_max(self, tmp_path):
        """The metric that was genuinely wrong, rather than wrong-but-masked."""
        log = self._rank_log(tmp_path)
        self._append_run(log, 4, mem=99.0)  # an earlier, larger-footprint run
        mlflow_patches._snapshot_log_offsets(str(tmp_path))
        self._append_run(log, 4, mem=36.93)

        assert self._summarise(tmp_path)["perf/max_memory_allocated_gb"] == 36.93

    def test_median_ignores_an_earlier_runs_step_times(self, tmp_path):
        """A median is robust to a few outliers, which is why this hid for so
        long. Make the earlier run big enough that it cannot hide."""
        log = self._rank_log(tmp_path)
        self._append_run(log, 10, step_time=100.0)
        mlflow_patches._snapshot_log_offsets(str(tmp_path))
        self._append_run(log, 4, step_time=4.3)

        out = self._summarise(tmp_path)
        assert out["perf/iteration_time_ms"] == 4300.0
        assert out["perf/logged_steps"] == 4.0

    def test_log_created_after_the_snapshot_is_read_in_full(self, tmp_path):
        """A rank whose log does not exist yet at snapshot time has no offset;
        all of its content belongs to this run."""
        existing = self._rank_log(tmp_path, rank=0)
        self._append_run(existing, 4)
        mlflow_patches._snapshot_log_offsets(str(tmp_path))

        self._append_run(existing, 4)
        self._append_run(self._rank_log(tmp_path, rank=1), 4)  # appeared later

        assert self._summarise(tmp_path)["perf/logged_steps"] == 8.0

    def test_artifact_upload_still_collects_every_log_level(self, tmp_path):
        """Narrowing the summary must not narrow the upload: an error.log is
        exactly what someone debugging a failed run wants attached."""
        for name in ("info.log", "debug.log", "error.log", "warning.log"):
            self._rank_log(tmp_path, name=name).touch()

        uploaded = mlflow_patches._find_files(str(tmp_path / "logs"), ("*.log",))

        assert sorted(os.path.basename(p) for p in uploaded) == [
            "debug.log",
            "error.log",
            "info.log",
            "warning.log",
        ]

    def test_absent_offset_entry_reads_whole_file(self, tmp_path):
        """Guards the default branch directly: no offsets at all means no
        filtering, which is what keeps the existing callers correct."""
        log = self._rank_log(tmp_path)
        self._append_run(log, 3)

        assert mlflow_patches._summarise_from_logs([str(log)])["perf/logged_steps"] == 3.0


class TestPerStepMetrics:
    """Last rank only, host-side values, batched per log window, async.

    The cost being guarded against is blocking I/O, not a GPU sync. The recipe
    already calls .item() on the loss, the elapsed timer and the sample count
    every window for its own [TRAIN] line (``run_train_validation_loop``,
    ``_elapsed_seconds_since``, ``_count_global_samples``), so those
    syncs are paid whether or not anything is tracked. What per-step logging
    adds is an HTTPS POST to a tracking server in another region -- which must
    never happen on the training thread.
    """

    LINE = TestRunSummary.LINE

    @pytest.fixture(autouse=True)
    def _reset(self):
        mlflow_patches._per_step_queue = None
        mlflow_patches._per_step_thread = None
        mlflow_patches._per_step_pending = None
        mlflow_patches._per_step_count = 0
        yield
        mlflow_patches._per_step_queue = None
        mlflow_patches._per_step_thread = None
        mlflow_patches._per_step_pending = None
        mlflow_patches._per_step_count = 0

    def test_per_step_off_still_arms_liveness_but_logs_no_step_metrics(self):
        """per_step off is not "nothing": the heartbeat still needs a way out.

        What per_step decides is whether every log window becomes a network
        call. Liveness is cheaper than that and is wanted either way, so the
        queue and flusher are armed regardless and the heartbeat rides them.
        """
        options.load(SimpleNamespace(primus_mlflow={"per_step": False, "heartbeat_seconds": 0}))
        mlflow_patches._start_per_step("run-abc")
        assert mlflow_patches._per_step_queue is not None  # heartbeat transport

        self._feed(self.LINE % (8, 1.5, 2.56, 47.54), per_step=False)
        assert mlflow_patches._per_step_queue.empty()  # but no per-step metrics
        assert mlflow_patches._previous_step_seen == 8  # progress still tracked
        mlflow_patches._stop_per_step()

    def test_heartbeat_can_be_switched_off(self):
        options.load(SimpleNamespace(primus_mlflow={"per_step": False, "heartbeat_seconds": 0}))
        mlflow_patches._start_heartbeat()
        assert mlflow_patches._heartbeat_thread is None

    def test_heartbeat_is_armed_even_when_per_step_is_on(self, monkeypatch):
        """The per-step stream is not a substitute for the heartbeat.

        A stream that stopped looks the same whether the process was killed or
        is alive and wedged on a collective. Only the timer separates them, so
        arming it is not conditional on per_step. ``heartbeat_seconds: 0``
        remains the off switch.
        """
        armed = []
        monkeypatch.setattr(mlflow_patches, "_start_heartbeat", lambda: armed.append(True))
        options.load(SimpleNamespace(primus_mlflow={"per_step": True}))
        mlflow_patches._start_per_step("run-abc")
        assert armed == [True]
        mlflow_patches._stop_per_step()

    def _drive_heartbeat_once(self, started):
        """Run exactly one heartbeat iteration, with no thread and no sleeping."""

        class _OneShot:
            def __init__(self):
                self.calls = 0

            def wait(self, _interval):
                self.calls += 1
                return self.calls > 1  # False once -> body runs once, then stop

        mlflow_patches._heartbeat_loop(_OneShot(), 0, started)

    def test_heartbeat_reports_seconds_since_previous_step(self):
        """One number that reads correctly on MLflow's default step axis.

        Every heartbeat is logged at the last observed step, so on a wedged run
        they share an x-coordinate and collapse to a single point against step.
        This series carries the same signal without needing the time axis.
        """
        import queue as _queue
        import time as _time

        mlflow_patches._per_step_queue = _queue.Queue(maxsize=16)
        mlflow_patches._previous_step_seen = 7
        mlflow_patches._previous_step_wall = _time.time() - 30.0

        self._drive_heartbeat_once(started=_time.time())

        metrics, step = mlflow_patches._per_step_queue.get_nowait()
        assert step == 7
        assert metrics["primus/previous_step"] == 7.0
        assert 29.0 < metrics["primus/seconds_since_previous_step"] < 31.0

    def test_heartbeat_before_the_first_step_measures_from_run_start(self):
        """Warmup and compile happen before any [TRAIN] line exists.

        Measuring from an unset ``_previous_step_wall`` would report the Unix
        epoch as the previous step, which is worse than useless on a chart. It
        falls back to when the heartbeat armed.
        """
        import queue as _queue
        import time as _time

        mlflow_patches._per_step_queue = _queue.Queue(maxsize=16)
        mlflow_patches._previous_step_seen = -1
        mlflow_patches._previous_step_wall = 0.0

        self._drive_heartbeat_once(started=_time.time() - 5.0)

        metrics, step = mlflow_patches._per_step_queue.get_nowait()
        assert step == 0  # negative step would be rejected by MLflow
        assert 4.0 < metrics["primus/seconds_since_previous_step"] < 6.0

    def _feed(self, line, per_step=True):
        """Push one log record through the live monitor, as logging would."""
        import logging as _logging

        options.load(SimpleNamespace(primus_mlflow={"per_step": per_step}))

        record = _logging.LogRecord("x", _logging.INFO, __file__, 0, line, None, None)
        mlflow_patches._TrainLineHandler().emit(record)

    def test_live_line_yields_both_loss_and_perf_series(self):
        """The reason this replaced the two monkey-patches.

        Those could reach throughput and memory only; loss lives in a local
        dict the recipe never exposes, and loss is the series someone watching
        a long run actually wants.
        """
        import queue as _queue

        mlflow_patches._per_step_queue = _queue.Queue(maxsize=8)
        self._feed(self.LINE % (6, 1.5, 2.56, 47.54))

        metrics, step = mlflow_patches._per_step_queue.get_nowait()
        assert step == 6  # taken from the line, not a counter
        assert metrics["perf/iteration_time_ms"] == 1500.0  # seconds -> ms
        assert metrics["perf/samples_per_sec_per_gpu"] == 2.56
        assert metrics["perf/max_memory_allocated_gb"] == 47.54
        assert metrics["train/loss"] == 0.5
        assert metrics["train/avg_loss"] == 0.5

    def test_step_is_the_recipes_own_step_number(self):
        """Not a count of records: at log_remote_every_steps=2 the recipe emits
        steps 1, 3, 5... and the x-axis must say so."""
        import queue as _queue

        mlflow_patches._per_step_queue = _queue.Queue(maxsize=8)
        for s in (1, 3, 5):
            self._feed(self.LINE % (s, 1.5, 2.56, 47.54))
        assert [mlflow_patches._per_step_queue.get_nowait()[1] for _ in range(3)] == [1, 3, 5]

    def test_non_train_lines_are_ignored(self):
        import queue as _queue

        mlflow_patches._per_step_queue = _queue.Queue(maxsize=8)
        self._feed("[INFO] Epoch 1 complete. avg_loss=1.234")
        self._feed("something with loss= in it but no marker")
        assert mlflow_patches._per_step_queue.empty()

    def test_a_malformed_train_line_never_raises(self):
        """A handler that raises breaks logging itself, taking the run's own
        output with it. Telemetry is never worth that."""
        import queue as _queue

        mlflow_patches._per_step_queue = _queue.Queue(maxsize=8)
        self._feed("[TRAIN] step=nan loss=??? mem=GB")  # must not raise
        assert mlflow_patches._per_step_queue.empty()

    def test_no_tflops_invented_per_step_either(self):
        """The recipe logs no FLOPs; the live path must not imply otherwise."""
        import queue as _queue

        mlflow_patches._per_step_queue = _queue.Queue(maxsize=8)
        self._feed(self.LINE % (2, 1.0, 1.0, 1.0))
        metrics, _ = mlflow_patches._per_step_queue.get_nowait()
        assert not any("tflop" in k.lower() for k in metrics)

    def test_summary_and_live_monitor_share_one_pattern(self):
        """They must agree about what the run did. A second regex would be a
        second thing to keep in step."""
        import queue as _queue

        line = self.LINE % (4, 2.0, 3.25, 12.5)
        mlflow_patches._per_step_queue = _queue.Queue(maxsize=8)
        self._feed(line)
        live, _ = mlflow_patches._per_step_queue.get_nowait()

        import os as _os
        import tempfile

        path = _os.path.join(tempfile.mkdtemp(), "info.log")
        open(path, "w").write(line + "\n")
        summary = mlflow_patches._summarise_from_logs([path])

        assert live["perf/iteration_time_ms"] == summary["perf/iteration_time_ms"]
        assert live["perf/samples_per_sec_per_gpu"] == summary["perf/samples_per_sec_per_gpu"]

    def test_enqueue_is_a_noop_when_disarmed(self):
        """A capture hook left installed after teardown must not raise."""
        mlflow_patches._per_step_queue = None
        mlflow_patches._enqueue_per_step({"perf/x": 1.0}, 1)  # must not raise

    def test_queue_is_bounded_and_drops_rather_than_blocks(self):
        """A stalled tracking server must never apply backpressure to training.

        Blocking here would be the exact failure the design forbids, arriving
        by a different route than a GPU sync.
        """
        import queue as _queue

        mlflow_patches._per_step_queue = _queue.Queue(maxsize=1)
        mlflow_patches._enqueue_per_step({"perf/a": 1.0}, 1)
        mlflow_patches._enqueue_per_step({"perf/b": 2.0}, 2)  # full: dropped, not blocked
        assert mlflow_patches._per_step_queue.qsize() == 1

    def test_stop_without_start_is_safe(self):
        mlflow_patches._stop_per_step()  # must not raise

    def test_worker_logs_against_an_explicit_run_id(self, monkeypatch):
        """Regression: the fluent API cannot be used from the flusher thread.

        mlflow._active_run_stack is a ThreadLocalVariable, so a worker thread
        sees active_run() is None and mlflow.log_metrics drops everything
        silently -- run FINISHED, summary attached, per-step series simply
        absent. The first implementation did exactly that. The worker must
        therefore carry the run id and use MlflowClient.
        """
        import queue as _queue
        import threading

        # Import on the main thread first. Importing mlflow.tracking inside the
        # worker races its own package initialisation and raises a circular
        # ImportError -- an artefact of the test, not the code under test.
        import mlflow.artifacts  # noqa: F401
        import mlflow.tracking

        calls = []

        class FakeClient:
            def log_batch(self, run_id, metrics=None, **kw):
                calls.append(
                    (run_id, {m.key: m.value for m in (metrics or [])}, [m.step for m in (metrics or [])])
                )

        monkeypatch.setattr(mlflow.tracking, "MlflowClient", FakeClient)

        q = _queue.Queue()
        q.put(({"perf/iteration_time_ms": 1500.0}, 4))
        q.put(mlflow_patches._PER_STEP_STOP)
        t = threading.Thread(target=mlflow_patches._per_step_worker, args=(q, "run-abc"), daemon=True)
        t.start()
        t.join(timeout=5)

        assert not t.is_alive()
        assert calls, "worker logged nothing -- the thread-local trap is back"
        run_id, metrics, steps = calls[0]
        assert run_id == "run-abc"
        assert metrics == {"perf/iteration_time_ms": 1500.0}
        assert steps == [4]

    def test_per_step_without_a_run_id_is_declined(self):
        """Arming capture with nowhere to send it is pure cost."""
        options.load(SimpleNamespace(primus_mlflow={"per_step": True}))
        mlflow_patches._start_per_step(None)
        assert mlflow_patches._per_step_queue is None

    def test_worker_drains_and_exits_on_sentinel(self):
        """Teardown must not hang on a server that stopped answering."""
        import queue as _queue
        import threading

        q = _queue.Queue()
        q.put(_queue.Empty)  # junk item: the worker must survive it
        q.put(mlflow_patches._PER_STEP_STOP)
        t = threading.Thread(target=mlflow_patches._per_step_worker, args=(q, "run-xyz"), daemon=True)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()


class TestArtifactUploadIsNonFatal:
    def test_upload_failure_does_not_change_the_run_status(self, monkeypatch):
        """A tracking server having a bad day must not turn a finished training
        run into a failed one."""
        statuses = []
        monkeypatch.setattr(
            mlflow_patches,
            "_upload_artifacts",
            lambda root: (_ for _ in ()).throw(RuntimeError("server down")),
        )
        monkeypatch.setattr(mlflow_patches, "_finish_run", statuses.append)
        mlflow_patches._finalize(None, "FINISHED")
        assert statuses == ["FINISHED"]

    def test_nothing_enabled_means_mlflow_is_never_imported(self, monkeypatch, tmp_path):
        """The no-op path must not reach for MLflow -- which is also what lets
        this suite run without it installed."""
        monkeypatch.setattr(mlflow_patches.options, "flag", lambda path, default=False: False)
        monkeypatch.setitem(sys.modules, "mlflow", None)  # an import would raise
        mlflow_patches._upload_artifacts(str(tmp_path))

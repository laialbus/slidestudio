"""
Milestone 6/8 — CLI tests.

Uses typer.testing.CliRunner. Mocks pipeline_estimate and pipeline_run
so no real API calls are made.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


# ──────────────────────────────────────────────────────────────
# estimate subcommand
# ──────────────────────────────────────────────────────────────

class TestEstimateSubcommand:
    def _mock_cost(self):
        return {
            "input_tokens":   100,
            "output_tokens":  3750,
            "estimated_cost": 0.0567,
        }

    def test_estimate_exits_with_code_zero(self):
        with patch("cli.pipeline_estimate", return_value=self._mock_cost()):
            result = runner.invoke(app, ["estimate", "dummy.pdf"])
        assert result.exit_code == 0

    def test_estimate_prints_input_tokens(self):
        with patch("cli.pipeline_estimate", return_value=self._mock_cost()):
            result = runner.invoke(app, ["estimate", "dummy.pdf"])
        assert "input_tokens" in result.output

    def test_estimate_prints_estimated_cost(self):
        with patch("cli.pipeline_estimate", return_value=self._mock_cost()):
            result = runner.invoke(app, ["estimate", "dummy.pdf"])
        assert "estimated_cost" in result.output

    def test_estimate_calls_pipeline_estimate(self):
        mock_estimate = MagicMock(return_value=self._mock_cost())
        with patch("cli.pipeline_estimate", mock_estimate):
            runner.invoke(app, ["estimate", "dummy.pdf"])
        mock_estimate.assert_called_once()

    def test_estimate_passes_provider_from_config(self):
        import config
        mock_estimate = MagicMock(return_value=self._mock_cost())
        with patch("cli.pipeline_estimate", mock_estimate):
            runner.invoke(app, ["estimate", "dummy.pdf"])
        assert mock_estimate.call_args.kwargs["provider_key"] == config.PROVIDER

    def test_estimate_passes_config_chunk_values(self):
        import config
        mock_estimate = MagicMock(return_value=self._mock_cost())
        with patch("cli.pipeline_estimate", mock_estimate):
            runner.invoke(app, ["estimate", "dummy.pdf"])
        call_kwargs = mock_estimate.call_args.kwargs
        assert call_kwargs["chunk_size"] == config.PIPELINE["chunk_size"]
        assert call_kwargs["overlap_size"] == config.PIPELINE["overlap_size"]


# ──────────────────────────────────────────────────────────────
# run subcommand — --fast, --debug, --open, --max-concurrent flags
# ──────────────────────────────────────────────────────────────

class TestRunSubcommand:
    def _invoke(self, args, output_path=None, mock_provider_cls=None):
        if output_path is None:
            output_path = Path("outputs/test.json")
        mock_pipeline_run = AsyncMock(return_value=(MagicMock(), [], output_path))
        registry_patch = {}
        if mock_provider_cls is not None:
            registry_patch = {"anthropic": mock_provider_cls}

        ctx_managers = [
            patch("cli.pipeline_run", mock_pipeline_run),
        ]
        if registry_patch:
            ctx_managers.append(patch.dict("cli._PROVIDER_REGISTRY", registry_patch))

        from contextlib import ExitStack
        with ExitStack() as stack:
            [stack.enter_context(m) for m in ctx_managers]
            result = runner.invoke(app, ["run"] + args)
        return result, mock_pipeline_run

    def test_fast_passes_max_review_cycles_zero(self):
        _, mock_run = self._invoke(["dummy.pdf", "--fast"])
        assert mock_run.call_args.kwargs["max_review_cycles"] == 0

    def test_debug_passes_debug_true(self):
        _, mock_run = self._invoke(["dummy.pdf", "--debug"])
        assert mock_run.call_args.kwargs["debug"] is True

    def test_max_concurrent_overrides_provider_param(self):
        mock_provider_cls = MagicMock()
        mock_pipeline_run = AsyncMock(return_value=(MagicMock(), [], Path("outputs/test.json")))
        with patch("cli.pipeline_run", mock_pipeline_run), \
             patch.dict("cli._PROVIDER_REGISTRY", {"anthropic": mock_provider_cls}):
            runner.invoke(app, ["run", "dummy.pdf", "--provider", "anthropic", "--max-concurrent", "10"])
        assert mock_provider_cls.call_args.kwargs["config"].max_concurrent == 10

    def test_open_calls_serve_and_open_with_output_path(self):
        output_path = Path("outputs/test.json")
        with patch("cli.pipeline_run", AsyncMock(return_value=(MagicMock(), [], output_path))), \
             patch("exporters.html_server.serve_and_open") as mock_serve:
            runner.invoke(app, ["run", "dummy.pdf", "--open"])
        mock_serve.assert_called_once()
        assert mock_serve.call_args[0][0] == output_path


# ──────────────────────────────────────────────────────────────
# serve subcommand
# ──────────────────────────────────────────────────────────────

class TestServeSubcommand:
    def test_serve_opens_browser_for_existing_output(self, tmp_path):
        output_file = tmp_path / "test.json"
        output_file.write_text("{}", encoding="utf-8")
        with patch("cli.resolve_output_path", return_value=output_file), \
             patch("exporters.html_server.serve_and_open") as mock_serve:
            result = runner.invoke(app, ["serve", "paper.pdf"])
        assert result.exit_code == 0
        mock_serve.assert_called_once()

    def test_serve_passes_output_path_to_serve_and_open(self, tmp_path):
        output_file = tmp_path / "test.json"
        output_file.write_text("{}", encoding="utf-8")
        with patch("cli.resolve_output_path", return_value=output_file), \
             patch("exporters.html_server.serve_and_open") as mock_serve:
            runner.invoke(app, ["serve", "paper.pdf"])
        assert mock_serve.call_args[0][0] == output_file

    def test_serve_exits_with_nonzero_when_no_output(self):
        with patch("cli.resolve_output_path", return_value=None):
            result = runner.invoke(app, ["serve", "paper.pdf"])
        assert result.exit_code == 1

    def test_serve_prints_error_when_no_output(self):
        with patch("cli.resolve_output_path", return_value=None):
            result = runner.invoke(app, ["serve", "paper.pdf"])
        assert "No output found" in result.output

    def test_serve_error_message_includes_run_hint(self):
        with patch("cli.resolve_output_path", return_value=None):
            result = runner.invoke(app, ["serve", "paper.pdf"])
        assert "run" in result.output

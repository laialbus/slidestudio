"""
Milestone 6/8 — CLI tests.

Uses typer.testing.CliRunner. Mocks PDFExtractor, analyze_pdf_cost,
AnalystAgent, and pipeline.route so no real API calls are made.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


# ──────────────────────────────────────────────────────────────
# --estimate flag
# ──────────────────────────────────────────────────────────────

class TestEstimateFlag:
    def _mock_extraction(self):
        return {"headers": ["Intro"], "chunks": ["Some chunk text."]}

    def _mock_cost(self):
        return {
            "input_tokens":   100,
            "output_tokens":  3750,
            "estimated_cost": 0.0567,
        }

    def test_estimate_exits_with_code_zero(self):
        with patch("cli.PDFExtractor") as mock_extractor_cls, \
             patch("cli.analyze_pdf_cost", return_value=self._mock_cost()):
            mock_extractor_cls.return_value.extract.return_value = self._mock_extraction()
            result = runner.invoke(app, ["dummy.pdf", "--estimate"])
        assert result.exit_code == 0

    def test_estimate_prints_input_tokens(self):
        with patch("cli.PDFExtractor") as mock_extractor_cls, \
             patch("cli.analyze_pdf_cost", return_value=self._mock_cost()):
            mock_extractor_cls.return_value.extract.return_value = self._mock_extraction()
            result = runner.invoke(app, ["dummy.pdf", "--estimate"])
        assert "input_tokens" in result.output

    def test_estimate_prints_estimated_cost(self):
        with patch("cli.PDFExtractor") as mock_extractor_cls, \
             patch("cli.analyze_pdf_cost", return_value=self._mock_cost()):
            mock_extractor_cls.return_value.extract.return_value = self._mock_extraction()
            result = runner.invoke(app, ["dummy.pdf", "--estimate"])
        assert "estimated_cost" in result.output

    def test_estimate_calls_analyze_pdf_cost(self):
        mock_analyze = MagicMock(return_value=self._mock_cost())
        with patch("cli.PDFExtractor") as mock_extractor_cls, \
             patch("cli.analyze_pdf_cost", mock_analyze):
            mock_extractor_cls.return_value.extract.return_value = self._mock_extraction()
            runner.invoke(app, ["dummy.pdf", "--estimate"])
        mock_analyze.assert_called_once()

    def test_estimate_passes_provider_from_config(self):
        import config
        mock_analyze = MagicMock(return_value=self._mock_cost())
        with patch("cli.PDFExtractor") as mock_extractor_cls, \
             patch("cli.analyze_pdf_cost", mock_analyze):
            mock_extractor_cls.return_value.extract.return_value = self._mock_extraction()
            runner.invoke(app, ["dummy.pdf", "--estimate"])
        call_args = mock_analyze.call_args
        assert call_args[0][1] == config.PROVIDER

    def test_estimate_instantiates_extractor_with_config_values(self):
        import config
        with patch("cli.PDFExtractor") as mock_extractor_cls, \
             patch("cli.analyze_pdf_cost", return_value=self._mock_cost()):
            mock_extractor_cls.return_value.extract.return_value = self._mock_extraction()
            runner.invoke(app, ["dummy.pdf", "--estimate"])
        mock_extractor_cls.assert_called_once_with(
            chunk_size=config.PIPELINE["chunk_size"],
            overlap_size=config.PIPELINE["overlap_size"],
        )


# ──────────────────────────────────────────────────────────────
# --fast, --debug, --open, --max-concurrent flags
# ──────────────────────────────────────────────────────────────

class TestRunFlags:
    def _mock_extraction(self):
        return {"headers": ["Intro"], "chunks": ["chunk"]}

    def _mock_doc_map(self):
        from schemas.document_map import DocumentMap, Section
        return DocumentMap(
            title="Test",
            document_type="research_paper",
            technical_level="intermediate",
            core_thesis="A thesis.",
            key_concepts=["concept"],
            sections=[Section(heading="Intro", importance="high", summary="Summary.")],
        )

    def _invoke(self, args, output_path=None, mock_provider_cls=None):
        if output_path is None:
            output_path = Path("outputs/test.json")
        mock_route = AsyncMock(return_value=(MagicMock(), [], output_path))
        registry_patch = {}
        if mock_provider_cls is not None:
            registry_patch = {"anthropic": mock_provider_cls}

        ctx_managers = [
            patch("cli.PDFExtractor"),
            patch("cli.AnalystAgent"),
            patch("cli.route", mock_route),
        ]
        if registry_patch:
            ctx_managers.append(patch.dict("cli._PROVIDER_REGISTRY", registry_patch))

        from contextlib import ExitStack
        with ExitStack() as stack:
            mocks = [stack.enter_context(m) for m in ctx_managers]
            mock_extractor = mocks[0]
            mock_analyst   = mocks[1]
            mock_extractor.return_value.extract.return_value = self._mock_extraction()
            mock_analyst.return_value.run = AsyncMock(return_value=self._mock_doc_map())
            result = runner.invoke(app, args)
        return result, mock_route

    def test_fast_passes_max_review_cycles_zero(self):
        _, mock_route = self._invoke(["dummy.pdf", "--fast"])
        assert mock_route.call_args.kwargs["max_review_cycles"] == 0

    def test_debug_passes_debug_true(self):
        _, mock_route = self._invoke(["dummy.pdf", "--debug"])
        assert mock_route.call_args.kwargs["debug"] is True

    def test_max_concurrent_overrides_provider_param(self):
        mock_provider_cls = MagicMock()
        mock_route = AsyncMock(return_value=(MagicMock(), [], Path("outputs/test.json")))
        with patch("cli.PDFExtractor") as mock_extractor, \
             patch("cli.AnalystAgent") as mock_analyst, \
             patch("cli.route", mock_route), \
             patch.dict("cli._PROVIDER_REGISTRY", {"anthropic": mock_provider_cls}):
            mock_extractor.return_value.extract.return_value = self._mock_extraction()
            mock_analyst.return_value.run = AsyncMock(return_value=self._mock_doc_map())
            runner.invoke(app, ["dummy.pdf", "--max-concurrent", "10"])
        assert mock_provider_cls.call_args.kwargs["max_concurrent"] == 10

    def test_open_calls_serve_and_open_with_output_path(self):
        output_path = Path("outputs/test.json")
        with patch("cli.PDFExtractor") as mock_extractor, \
             patch("cli.AnalystAgent") as mock_analyst, \
             patch("cli.route", AsyncMock(return_value=(MagicMock(), [], output_path))), \
             patch("exporters.html_server.serve_and_open") as mock_serve:
            mock_extractor.return_value.extract.return_value = self._mock_extraction()
            mock_analyst.return_value.run = AsyncMock(return_value=self._mock_doc_map())
            runner.invoke(app, ["dummy.pdf", "--open"])
        mock_serve.assert_called_once()
        assert mock_serve.call_args[0][0] == output_path

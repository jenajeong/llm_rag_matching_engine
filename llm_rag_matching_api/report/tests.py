from pathlib import Path
from unittest.mock import Mock

from django.test import SimpleTestCase

from .engine import ReportGenerationError, ReportGenerator, build_report_prompt


def _ranked_professors(count):
    return [
        {
            "professor_info": {"NM": f"Professor {number}"},
            "documents": {},
            "document_scores": {},
        }
        for number in range(1, count + 1)
    ]


class ReportGeneratorInputTests(SimpleTestCase):
    def setUp(self):
        self.generator = ReportGenerator.__new__(ReportGenerator)
        self.ahp_results = {
            "query": "battery",
            "keywords": {},
            "ranked_professors": _ranked_professors(8),
        }

    def test_prepare_input_json_uses_requested_professor_count(self):
        input_data = self.generator._prepare_input_json(self.ahp_results, professor_count=5)

        self.assertEqual(len(input_data["professors"]), 5)
        self.assertEqual(input_data["professors"][-1]["name"], "Professor 5")

    def test_prepare_input_json_supports_increased_professor_count(self):
        input_data = self.generator._prepare_input_json(self.ahp_results, professor_count=7)

        self.assertEqual(len(input_data["professors"]), 7)
        self.assertEqual(input_data["professors"][-1]["name"], "Professor 7")

    def test_prepare_input_json_keeps_all_professors_when_count_is_omitted(self):
        input_data = self.generator._prepare_input_json(self.ahp_results)

        self.assertEqual(len(input_data["professors"]), 8)

    def test_prepare_input_json_rejects_invalid_professor_count(self):
        with self.assertRaisesRegex(ReportGenerationError, "positive integer"):
            self.generator._prepare_input_json(self.ahp_results, professor_count=0)

    def test_generate_from_payload_passes_professor_count_to_input_builder(self):
        self.generator.model = "test-model"
        self.generator._resolve_recommendation = Mock(return_value={"ahp_results": self.ahp_results})
        self.generator._prepare_input_json = Mock(return_value={"professors": []})
        self.generator._generate_report_text = Mock(return_value="# Report")
        self.generator.save_pdf = Mock(return_value=Path("report.pdf"))

        self.generator.generate_from_payload({"professor_count": 7})

        self.generator._prepare_input_json.assert_called_once_with(
            self.ahp_results,
            professor_count=7,
        )

    def test_prompt_instructs_llm_to_include_every_professor(self):
        prompt = build_report_prompt({"professors": []})

        self.assertIn("professors 배열 순서대로 모든 교수를 표시한다", prompt)

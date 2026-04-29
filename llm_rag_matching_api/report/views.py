import json
import logging
from pathlib import Path

from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .engine import ReportGenerationError, ReportGenerator

logger = logging.getLogger(__name__)


@csrf_exempt
def generate(request):
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed. Use POST."}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON body."}, status=400)

    try:
        report_data = ReportGenerator().generate_from_payload(payload)
    except ReportGenerationError as error:
        return JsonResponse({"detail": str(error)}, status=400)
    except Exception as error:
        return JsonResponse({"detail": str(error)}, status=500)

    logger.info("input_data: %s", report_data.get("input_data"))
    logger.info("report_text: %s", report_data.get("report_text"))

    pdf_path = Path(report_data["pdf_path"])
    file_handle = open(pdf_path, "rb")
    response = FileResponse(
        file_handle,
        content_type="application/pdf",
        as_attachment=True,
        filename=report_data["pdf_filename"],
    )
    response._resource_closers.append(lambda: pdf_path.unlink(missing_ok=True))
    response["X-Report-Id"] = report_data["report_id"]
    if report_data.get("search_id"):
        response["X-Search-Id"] = report_data["search_id"]
    response["X-Report-Query"] = report_data["query"].encode("utf-8").hex()
    return response
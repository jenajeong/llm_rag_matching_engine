import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def recommend(request):
    if request.method != "POST":
        return JsonResponse({"detail": "Method not allowed. Use POST."}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON body."}, status=400)

    try:
        from .engine.service import recommend_professors

        result = recommend_professors(payload)
    except ValueError as error:
        return JsonResponse({"detail": str(error)}, status=400)
    except Exception as error:
        return JsonResponse({"detail": str(error)}, status=500)

    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})

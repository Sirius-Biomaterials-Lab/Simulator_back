from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def collection_metrics(app: FastAPI):
    instr = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=False,  # чтобы захватывать пути с path-параметрами
        # excluded_handlers=["/metrics"],     # игнорируем метрики самого /metrics
    )
    instr.instrument(app)  # вешаем сбор
    instr.expose(app, endpoint="/metrics", tags=["monitoring"])  # создаём endpoint /metrics

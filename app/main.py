import logging.config

import sentry_sdk
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette_context import plugins
from starlette_context.middleware import RawContextMiddleware

from app.auth import router as auth_router
from app.logger import LOGGING_CONFIG, logger
from app.modules import routers as models_router

routers = [auth_router, *models_router]

sentry_sdk.init(
    dsn="https://be26a19afc26dfc6a57aab57834b6246@o4509357555515392.ingest.de.sentry.io/4509655652106320",
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
)



def create_app() -> FastAPI:
    logging.config.dictConfig(LOGGING_CONFIG)
    logger.info("Start App")
    app = FastAPI(docs_url='/swagger')

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://localhost:5173", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RawContextMiddleware, plugins=[plugins.CorrelationIdPlugin()])
    for router in routers:
        app.include_router(router)

    instr = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=False,    # чтобы захватывать пути с path-параметрами
        # excluded_handlers=["/metrics"],     # игнорируем метрики самого /metrics
    )
    instr.instrument(app)                   # вешаем сбор
    instr.expose(app, endpoint="/metrics")  # создаём endpoint /metrics

    return app


if __name__ == "__main__":
    uvicorn.run('app.main:create_app', factory=True, host='127.0.0.1', port=8000, reload=True)
    # uvicorn.run('app.main:create_app', factory=True, host='127.0.0.1', port=8000, workers=2)
    # uvicorn.run('app.main:app', host='0.0.0.0', port=8000)

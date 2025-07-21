import logging.config

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette_context import plugins
from starlette_context.middleware import RawContextMiddleware

from app.auth import router as auth_router
from app.logger import LOGGING_CONFIG, logger
from app.modules import routers as models_router
from monitoring.collection_metrics import collection_metrics
from monitoring.sentry import start_sentry

routers = [auth_router, *models_router]


def create_app() -> FastAPI:
    logging.config.dictConfig(LOGGING_CONFIG)
    logger.info("Start App")
    app = FastAPI(docs_url='/swagger')
    start_sentry()

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

    collection_metrics(app)

    return app


if __name__ == "__main__":
    uvicorn.run('app.main:create_app', factory=True, host='127.0.0.1', port=8000, reload=True)
    # uvicorn.run('app.main:create_app', factory=True, host='127.0.0.1', port=8000, workers=2)
    # uvicorn.run('app.main:app', host='0.0.0.0', port=8000)

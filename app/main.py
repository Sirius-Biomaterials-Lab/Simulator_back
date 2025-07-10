import logging.config

import uvicorn
from fastapi import FastAPI
from starlette_context import plugins
from starlette_context.middleware import RawContextMiddleware

from app.auth import router as auth_router
from app.logger import LOGGING_CONFIG, logger
from app.modules import routers as models_router

routers = [auth_router, *models_router]

app = FastAPI(docs_url='/swagger')


def create_app() -> FastAPI:
    logging.config.dictConfig(LOGGING_CONFIG)
    logger.info("Start App")
    app = FastAPI(docs_url='/swagger')
    for router in routers:
        app.include_router(router)

    app.add_middleware(RawContextMiddleware, plugins=[plugins.CorrelationIdPlugin()])
    return app


if __name__ == "__main__":
    uvicorn.run('app.main:create_app', factory=True, host='127.0.0.1', port=8000, reload=True)
    # uvicorn.run('app.main:create_app', factory=True, host='127.0.0.1', port=8000, workers=2)
    # uvicorn.run('app.main:app', host='0.0.0.0', port=8000)

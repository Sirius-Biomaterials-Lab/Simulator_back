from fastapi import FastAPI

from app.models import routers as models_router

routers = [*models_router]

app = FastAPI()

for router in routers:
    app.include_router(router)

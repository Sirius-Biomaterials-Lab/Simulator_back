import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Response, Cookie, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.settings import settings
from app.storage.redis.accessor import get_redis_connection

router = APIRouter(prefix='/auth', tags=["Auth"])

security = HTTPBasic()

redis = get_redis_connection()
SESSION_PREFIX = "session:"


def generate_session_id() -> str:
    return uuid.uuid4().hex


async def get_session_data(
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY),
) -> str:

    key = f"{SESSION_PREFIX}{session_id}"
    user = await redis.get(key)

    if not user:
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    # 🔁 Продление TTL при активности
    # await redis.expire(key, settings.COOKIE_TTL)
    return user.decode("utf-8")


@router.post("/login-cookie/")
async def login_cookie(
        response: Response,
        credentials: Annotated[HTTPBasicCredentials, Depends(security)],
):
    session_id = generate_session_id()
    key = f"{SESSION_PREFIX}{session_id}"

    await redis.set(key, credentials.username, ex=settings.COOKIE_TTL_SECONDS)
    response.set_cookie(
        key=settings.COOKIE_SESSION_ID_KEY,
        value=session_id,
        httponly=True,
        # secure=True,
        samesite="Lax",
        max_age=settings.COOKIE_TTL_SECONDS
    )
    return {"message": f"Welcome, {credentials.username}"}


@router.get("/logout-cookie/")
async def logout_cookie(
        response: Response,
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY),
        user_session_data: str = Depends(get_session_data),
):
    await redis.delete(f"{SESSION_PREFIX}{session_id}")
    response.delete_cookie(settings.COOKIE_SESSION_ID_KEY)
    return {"message": f"Bye, {user_session_data}"}

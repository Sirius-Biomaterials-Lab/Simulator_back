from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Cookie, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse

from app.auth.handlers import get_session_data
from app.exception import DataNotFound
from app.logger import logger
from app.modules.isotropic.isotropic_dependency import get_service
from app.modules.isotropic.server import Service
from app.modules.isotropic.shema import IsotropicUploadRequest, IsotropicResponse
from app.modules.isotropic.solver.shema import IsotropicFitResponse, \
    IsotropicPredictResponse
from app.settings import settings

router = APIRouter(prefix="/modules/isotropic", tags=["isotropic"], dependencies=[Depends(get_session_data)])

ServiceDep = Annotated[Service, Depends(get_service)]


@router.post("/upload_model", response_model=IsotropicResponse,
             responses={
                 200: {"model": IsotropicResponse, "description": "Upload successful"},
                 400: {"model": IsotropicResponse, "description": "Invalid file or parameters"},
             },
             description="Uploads a model file (.csv, .xls, .xlsx) for isotropic processing.",

             )
async def upload_model(
        server: ServiceDep,
        body: IsotropicUploadRequest = Depends(),
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY),
):
    for file in body.files:
        if not file.filename.lower().endswith((".csv", ".xls", ".xlsx")):
            logger.warning("Unsupported file format: %s", file.filename)

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported format. Use .xls, .xlsx or .csv"
            )

    for file in body.files:
        await server.set_data(session_id, file)

    logger.info('server.set_model_and_error_name')
    await server.set_model_and_error_name(session_id,
                                          hyperlastic_model_name=body.hyperlastic_model)
                                          # error_function_name=body.error_function)

    return IsotropicResponse(status="ok")


@router.post("/fit", response_model=IsotropicFitResponse,
             responses={
                 200: {"model": IsotropicFitResponse}
             },
             description="Runs fitting algorithm on the uploaded isotropic data.",
             )
async def fit_model(
        server: ServiceDep,
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY),
) -> IsotropicFitResponse:
    return await server.fit(session_id)


# return IsotropicFitResponse()
# else:
#     return IsotropicResponse(status="error", error='test_error')


@router.post("/predict", response_model=IsotropicPredictResponse,
             description="Performs predictions based on the isotropic model with provided input data.",
             )
async def predict_model(server: ServiceDep,
                        file: UploadFile = File(..., ),
                        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY)
                        ):

    return await server.predict(session_id, file)
    # return IsotropicPredictResponse()


@router.delete("/file/{filename}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(filename: str, server: ServiceDep,
                      session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY)):
    try:
        await server.delete_data(session_id, filename)
    except DataNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.detail)


@router.get("/calculate_energy", response_class=PlainTextResponse,
            responses={
                200: {"content": {"text/plain": {}}},
                404: {"model": IsotropicResponse, "description": "File not found"}
            }
            )
async def calculate_energy(
        server: ServiceDep,
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY),
):
    try:
        energy_text = await server.calculate_energy(session_id)
        return PlainTextResponse(content=energy_text, media_type="text/plain")
    except DataNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error calculating energy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error calculating energy"
        )


@router.delete("/clear_data", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(server: ServiceDep,
                      session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY)):
    await server.delete_all_data(session_id)


@router.get(
    "/download_energy",
    response_class=FileResponse,
    description="Downloads the .energy file as an attachment.",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"model": IsotropicResponse, "description": "File not found"},
    }
)
async def download_energy(
        background_tasks: BackgroundTasks,
        server: ServiceDep,
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY),
):
    """
    Endpoint to generate and download the .energy file as a binary attachment.
    """
    try:
        temp_file_path = await server.download_energy(session_id, background_tasks)

        return FileResponse(
            path=temp_file_path,
            filename="energy.energy",
            media_type="application/octet-stream",
        )
    except Exception as e:
        logger.error(f"Error creating or downloading energy file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating or downloading energy file"
        )

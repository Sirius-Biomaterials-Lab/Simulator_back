from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse, PlainTextResponse

from app.exception import DataNotFound
from app.logger import logger
from app.modules.isotropic.dependecy import get_service
from app.modules.isotropic.server import Service
from app.modules.isotropic.shema import IsotropicResponse, IsotropicUploadRequest, IsotropicFitResponse, \
    IsotropicPredictResponse

router = APIRouter(prefix="/modules/isotropic", tags=["isotropic"])

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
):
    for file in body.files:
        if not file.filename.lower().endswith((".csv", ".xls", ".xlsx")):
            logger.warning("Unsupported file format: %s", file.filename)

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported format. Use .xls, .xlsx or .csv"
            )

    for file in body.files:
        await server.set_data(file)

    logger.info('server.set_model_and_error_name')
    server.set_model_and_error_name(
        hyperlastic_model_name=body.hyperlastic_model,
        error_function_name=body.error_function)

    return IsotropicResponse(status="ok")


@router.post("/fit", response_model=IsotropicFitResponse,
             responses={
                 200: {"model": IsotropicFitResponse}
             },
             description="Runs fitting algorithm on the uploaded isotropic data.",
             )
async def fit_model(server: ServiceDep):
    return server.fit()


# return IsotropicFitResponse()
# else:
#     return IsotropicResponse(status="error", error='test_error')


@router.post("/predict", response_model=IsotropicPredictResponse,
             description="Performs predictions based on the isotropic model with provided input data.",
             )
async def predict_model(server: ServiceDep, file: UploadFile = File(..., )):

    await server.predict(file)
    return IsotropicPredictResponse()


@router.get("/calculate_energy", response_class=PlainTextResponse,
            responses={
                200: {"content": {"text/plain": {}}},
                404: {"model": IsotropicResponse, "description": "File not found"}
            }
            )
async def calculate_energy():
    test_file = Path("tests/test_data/test.energy")
    if not test_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Energy file not found"
        )
    async with aiofiles.open(test_file, mode="r", encoding="utf-8") as f:
        data = await f.read()
    return PlainTextResponse(content=data, media_type="text/plain")


@router.delete("/{filename}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(filename: str, server: ServiceDep):
    try:
        await server.del_data(filename)
    except DataNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.detail)


@router.delete("/clear_data", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(server: ServiceDep):

    await server.clear_data()


@router.get(
    "/download_energy",
    response_class=FileResponse,
    description="Downloads the .energy file as an attachment.",
    responses={
        200: {"content": {"application/octet-stream": {}}},
        404: {"model": IsotropicResponse, "description": "File not found"},
    }
)
async def download_energy():
    """
    Endpoint to download the .energy file as a binary attachment.
    """
    test_file = Path("tests/test_data/test.energy")
    if not test_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Energy file not found"
        )
    return FileResponse(
        path=str(test_file),
        filename=test_file.name,
        media_type="application/octet-stream",
    )

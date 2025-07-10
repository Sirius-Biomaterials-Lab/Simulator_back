from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Cookie

from app.auth.handlers import get_session_data
from app.exception import DataNotFound, DataNotCorrect
from app.logger import logger
from app.modules.anisotropic.anisotropic_dependency import get_anisotropic_service
from app.modules.anisotropic.service import AnisotropicService
from app.modules.anisotropic.shema import (
    AnisotropicUploadRequest, AnisotropicResponse, AnisotropicFitResponse, AnisotropicPredictResponse
)
from app.settings import settings

router = APIRouter(
    prefix="/modules/anisotropic",
    tags=["anisotropic"],
    dependencies=[Depends(get_session_data)]
)

ServiceDep = Annotated[AnisotropicService, Depends(get_anisotropic_service)]


@router.post("/upload_model", response_model=AnisotropicResponse,
             responses={
                 200: {"model": AnisotropicResponse, "description": "Upload successful"},
                 400: {"model": AnisotropicResponse, "description": "Invalid file or parameters"},
             },
             description="Uploads model files (.csv) for anisotropic processing with GOH or HOG models.")
async def upload_model(
        service: ServiceDep,
        body: AnisotropicUploadRequest = Depends(AnisotropicUploadRequest.as_form),
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY),
):
    """Upload data files and set model configuration for anisotropic analysis"""

    try:
        # Upload files
        for file in body.files:
            await service.set_data(session_id, file)

        # Set model configuration
        await service.set_model_config(
            session_id,
            model_type=body.model_type,
            kappa=body.kappa,
            alpha=body.alpha
        )

        logger.info(f"Successfully uploaded {len(body.files)} files and set {body.model_type} model config")
        return AnisotropicResponse(status="ok", detail=f"Uploaded {len(body.files)} files for {body.model_type} model")

    except DataNotCorrect as e:
        logger.error(f"Data validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Upload failed")


@router.post("/fit", response_model=AnisotropicFitResponse,
             responses={
                 200: {"model": AnisotropicFitResponse, "description": "Fit successful"},
                 400: {"model": AnisotropicResponse, "description": "No data or invalid configuration"},
                 500: {"model": AnisotropicResponse, "description": "Fitting failed"},
             },
             description="Runs parameter optimization on uploaded anisotropic data using GOH/HOG models.")
async def fit_model(
        service: ServiceDep,
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY),
) -> AnisotropicFitResponse:
    """Fit anisotropic hyperelastic model to uploaded data"""

    try:
        fit_response = await service.fit(session_id)
        logger.info(f"Successfully fitted anisotropic model for session {session_id}")
        return fit_response

    except DataNotFound as e:
        logger.error(f"Data not found: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DataNotCorrect as e:
        logger.error(f"Data validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Fitting error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model fitting failed"
        )


@router.post("/predict", response_model=AnisotropicPredictResponse,
             responses={
                 200: {"model": AnisotropicPredictResponse, "description": "Prediction successful"},
                 400: {"model": AnisotropicResponse, "description": "Model not fitted or invalid data"},
                 500: {"model": AnisotropicResponse, "description": "Prediction failed"},
             },
             description="Performs predictions using fitted anisotropic model on provided data.")
async def predict_model(
        service: ServiceDep,
        file: UploadFile = File(..., description="Prediction data file (.csv)"),
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY)
) -> AnisotropicPredictResponse:
    """Make predictions using fitted anisotropic model"""

    # Validate file format
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Use .csv files for predictions"
        )

    try:
        predict_response = await service.predict(session_id, file)
        logger.info(f"Successfully made predictions for session {session_id}")
        return predict_response

    except DataNotFound as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except DataNotCorrect as e:
        logger.error(f"Data validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/file/{filename}", status_code=status.HTTP_204_NO_CONTENT,
               description="Deletes a specific uploaded file.")
async def delete_file(
        filename: str,
        service: ServiceDep,
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY)
):
    """Delete specific uploaded file"""

    try:
        await service.delete_data(session_id, filename)
        logger.info(f"Deleted file {filename} for session {session_id}")

    except DataNotFound as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file"
        )


@router.delete("/clear_data", status_code=status.HTTP_204_NO_CONTENT,
               description="Clears all uploaded data and fitted models for the session.")
async def clear_all_data(
        service: ServiceDep,
        session_id: str = Cookie(alias=settings.COOKIE_SESSION_ID_KEY)
):
    """Clear all data for session"""

    try:
        await service.delete_all_data(session_id)
        logger.info(f"Cleared all anisotropic data for session {session_id}")

    except Exception as e:
        logger.error(f"Error clearing data for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear data"
        )

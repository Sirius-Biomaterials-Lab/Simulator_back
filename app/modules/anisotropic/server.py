from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import UploadFile, HTTPException

from app.exception import DataNotCorrect, DataNotFound
from app.logger import logger
from app.modules.anisotropic.anisotropic_cache import AnisotropicCache
from app.modules.anisotropic.shema import AnisotropicFitResponse, AnisotropicPredictResponse
from app.modules.anisotropic.solver import (
    AnisotropicSolver, AnisotropicSolverConfig, AnisotropicModelType
)
from app.modules.exception import UnsupportedFormatFile


@dataclass
class AnisotropicService:
    """Service layer for anisotropic module using SOLID principles"""

    anisotropic_cache: AnisotropicCache

    async def set_data(self, session_id: str, file: UploadFile) -> None:
        """Store uploaded file data"""
        logger.info(f"Setting data for session {session_id}, file: {file.filename}")

        content = await file.read()
        validated_content = self._validate_and_process_file(file.filename, content)
        await file.seek(0)

        await self.anisotropic_cache.set_file_data(session_id, file.filename, validated_content)

    async def delete_data(self, session_id: str, filename: str) -> None:
        """Delete specific file data"""
        await self.anisotropic_cache.del_file_data(session_id, filename)

    async def delete_all_data(self, session_id: str) -> None:
        """Delete all data for session"""
        await self.anisotropic_cache.del_all(session_id)
        logger.info(f'Deleted all anisotropic data for session {session_id}')

    async def set_model_config(self, session_id: str, model_type: str,
                               kappa: Optional[float] = None,
                               alpha: Optional[float] = None) -> None:
        """Store model configuration"""
        await self.anisotropic_cache.set_model_params(
            session_id, model_type, kappa, alpha
        )

    async def fit_model(self, session_id: str) -> AnisotropicFitResponse:
        """Fit anisotropic model to uploaded data"""
        logger.info(f"Starting fit for session {session_id}")

        # Get model configuration
        try:
            params_data = await self.anisotropic_cache.get_model_params(session_id)
        except ValueError as e:
            raise DataNotFound("Model configuration not found. Please upload data first.") from e

        # Get uploaded data
        filename_to_bytes = await self.anisotropic_cache.get_files_data(session_id)
        if not filename_to_bytes:
            raise DataNotFound("No data files found. Please upload data first.")

        # Process data files
        data_frames = []
        for filename, bytes_data in filename_to_bytes.items():
            filename_str = filename.decode('utf-8')
            logger.info(f"Processing file: {filename_str}")


            if filename_str.endswith('.csv'):
                df = pd.read_csv(BytesIO(bytes_data))
            elif filename_str.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(BytesIO(bytes_data))
            else:
                logger.warning(f"Unsupported file format: {filename_str}")
                continue

            data_frames.append(df)



        if not data_frames:
            raise DataNotCorrect("No valid data files found")

        all_data = np.vstack(data_frames)

        # Create solver configuration
        model_type = AnisotropicModelType(params_data['model_type'])
        solver_config = AnisotropicSolverConfig(
            model_type=model_type,
            kappa=params_data.get('kappa'),
            alpha=params_data.get('alpha')
        )

        # Create and run solver
        solver = AnisotropicSolver(solver_config)

        try:
            fit_response = solver.fit(all_data)

            # Store results
            fitted_params = solver.get_fitted_parameters()
            if fitted_params:
                await self.anisotropic_cache.set_fitted_parameters(session_id, fitted_params)

            # Store optimization results
            if solver._optimization_result:
                optimization_info = {
                    'success': solver._optimization_result.success,
                    'final_error': solver._optimization_result.final_error,
                    'iterations': solver._optimization_result.iterations,
                    'message': solver._optimization_result.message,
                }
                await self.anisotropic_cache.set_optimization_result(session_id, optimization_info)

            logger.info(f"Fit completed for session {session_id}")
            return fit_response

        except Exception as e:
            logger.error(f"Error during fitting: {e}")
            raise HTTPException(status_code=500, detail=f"Error during model fitting: {str(e)}")

    async def predict(self, session_id: str, file: UploadFile) -> AnisotropicPredictResponse:
        """Make predictions using fitted model"""
        logger.info(f"Making predictions for session {session_id}")

        # Get model configuration and fitted parameters
        try:
            config_data = await self.anisotropic_cache.get_model_params(session_id)
            fitted_params = await self.anisotropic_cache.get_fitted_parameters(session_id)
        except ValueError as e:
            raise DataNotFound("Model not fitted. Please fit the model first.") from e

        # Process prediction file
        content = await file.read()
        validated_content = self._validate_and_process_file(file.filename, content)
        await file.seek(0)

        if file.filename.endswith('.csv'):
            prediction_df = pd.read_csv(BytesIO(validated_content))
        elif file.filename.endswith(('.xls', '.xlsx')):
            prediction_df = pd.read_excel(BytesIO(validated_content))
        else:
            raise UnsupportedFormatFile

        # Create solver and make predictions
        model_type = AnisotropicModelType(config_data['model_type'])
        solver_config = AnisotropicSolverConfig(
            model_type=model_type,
            kappa=config_data.get('kappa'),
            alpha=config_data.get('alpha')
        )

        solver = AnisotropicSolver(solver_config)

        # Set fitted parameters manually
        from app.modules.anisotropic.solver.models import ModelParameters
        model_params = ModelParameters(
            mu=fitted_params['mu'],
            k1=fitted_params['k1'],
            k2=fitted_params['k2'],
        )
        solver._fitted_parameters = model_params

        try:
            predict_response = solver.predict(prediction_df)
            logger.info(f"Predictions completed for session {session_id}")
            return predict_response

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


    def _validate_and_process_file(self, filename: str, content: bytes) -> bytes:

        buffer = BytesIO(content)

        # Read file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(buffer)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(buffer)
        else:
            raise UnsupportedFormatFile(detail="Неподдерживаемый формат файла")

        if len(df.columns) < 4:
            raise DataNotCorrect(detail="Not enough columns")
        return buffer.getvalue()

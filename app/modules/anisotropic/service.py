from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import pandas as pd
from fastapi import UploadFile, HTTPException

from app.exception import DataNotCorrect, DataNotFound
from app.logger import logger
from app.modules.anisotropic.shema import AnisotropicFitResponse, AnisotropicPredictResponse
from app.modules.anisotropic.solver import (
    AnisotropicSolverConfig, AnisotropicModelType
)
from app.modules.exception import UnsupportedFormatFile
from app.modules.service import ModuleService


@dataclass
class AnisotropicService(ModuleService):
    """Service layer for anisotropic module using SOLID principles"""

    async def set_model_config(self, session_id: str, model_type: str,
                               kappa: Optional[float] = None,
                               alpha: Optional[float] = None) -> None:
        """Store model configuration"""
        params = {"model_type": model_type, "kappa": kappa, "alpha": alpha}
        await self.module_cache.set_model_params(session_id, params)

    async def fit(self, session_id: str) -> AnisotropicFitResponse:
        """Fit anisotropic model to uploaded data"""
        logger.info(f"Starting fit for session {session_id}")

        # Get uploaded data
        filename_to_bytes = await self.module_cache.get_files_data(session_id)
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

        # all_data = np.vstack(data_frames)
        all_data = pd.concat(data_frames, ignore_index=True)
        # Create solver configuration

        try:
            await self.__set_up_solver(session_id)
            fit_response = self.solver.fit(all_data)
            # Store results
            optimization_result = self.solver.get_optimization_result()

            # Store optimization results
            if optimization_result:
                optimization_info = {
                    'fitted_params': optimization_result.parameters.to_dict(),
                    'success': optimization_result.success,
                    'final_error': optimization_result.final_error,
                    'iterations': optimization_result.iterations,
                    'message': optimization_result.message,
                }
                await self.module_cache.set_optimization_params(session_id, optimization_info)

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

            fitted_params = (await self.module_cache.get_optimization_params(session_id))["fitted_params"]
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

        # Set fitted parameters manually
        from app.modules.anisotropic.solver.models import ModelParameters
        model_params = ModelParameters(
            mu=fitted_params['mu'],
            k1=fitted_params['k1'],
            k2=fitted_params['k2'],
            alpha=fitted_params['alpha'],
            kappa=fitted_params['kappa'],
        )

        try:
            await self.__set_up_solver(session_id=session_id)
            predict_response = self.solver.predict(prediction_df, model_params)
            logger.info(f"Predictions completed for session {session_id}")
            return predict_response

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    @staticmethod
    def _validate_and_process_file(filename: str, content: bytes) -> bytes:

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

    async def __set_up_solver(self, session_id: str):
        # Get model configuration
        try:
            config_data = await self.module_cache.get_model_params(session_id)
        except ValueError as e:
            raise DataNotFound("Model configuration not found. Please upload data first.") from e

        # Create solver and make predictions
        model_type = AnisotropicModelType(config_data['model_type'])
        solver_config = AnisotropicSolverConfig(
            model_type=model_type,
            kappa=config_data.get('kappa'),
            alpha=config_data.get('alpha')
        )

        self.solver.setup_solver(solver_config)

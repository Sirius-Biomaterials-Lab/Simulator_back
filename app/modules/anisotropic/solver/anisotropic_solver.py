from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from app.exception import DataNotCorrect
from app.logger import logger
from app.modules.anisotropic.shema import (
    AnisotropicFitResponse, AnisotropicPredictResponse, AnisotropicParameterValue,
    AnisotropicPlotData, AnisotropicPlotLine
)
from app.modules.anisotropic.solver.config import AnisotropicModelType, AnisotropicConstants
from app.modules.anisotropic.solver.evaluator import ModelEvaluator
from app.modules.anisotropic.solver.models import ModelFactory
from app.modules.anisotropic.solver.parameter_optimizer import ParameterOptimizer


@dataclass
class AnisotropicSolverConfig:
    """Configuration for anisotropic solver"""
    model_type: AnisotropicModelType
    kappa: Optional[float]
    alpha: Optional[float]


class AnisotropicSolver:
    """Main solver for anisotropic hyperelastic models following SOLID principles"""

    def __init__(self, config: AnisotropicSolverConfig):
        self.config = config
        self.model = ModelFactory.create_model(config.model_type)
        self.optimizer = ParameterOptimizer(self.model)
        self.evaluator = ModelEvaluator(self.model)

        # Storage for fitted parameters
        self._fitted_parameters = None
        self._optimization_result = None

    def fit(self, data: np.ndarray) -> AnisotropicFitResponse:
        """Fit the anisotropic model to data"""
        logger.info(f"Starting fit for {self.model.get_model_name()} model")

        # Combine and validate data


        # Optimize parameters
        self._optimization_result = self.optimizer.optimize(
            data,
            self.config.alpha,
            self.config.kappa,
        )

        if not self._optimization_result.success:
            logger.warning(f"Optimization failed: {self._optimization_result.message}")

        self._fitted_parameters = self._optimization_result.parameters

        # Evaluate model performance
        metrics = self.evaluator.evaluate_to_metrics(data, self._fitted_parameters)

        # Create parameter response
        parameters = self._create_parameter_response()

        # Generate plot data
        plot_data = self._create_fit_plot_data(data)

        return AnisotropicFitResponse(
            status="ok" if self._optimization_result.success else "warning",
            model_type=self.config.model_type.value,
            parameters=parameters,
            metrics=metrics,
            plot_data=plot_data,
            convergence_info=self._optimization_result.convergence_info
        )

    def predict(self, prediction_data: pd.DataFrame) -> AnisotropicPredictResponse:
        """Make predictions using fitted model"""
        if self._fitted_parameters is None:
            raise ValueError("Model must be fitted before making predictions")

        logger.info("Making predictions with fitted model")



        # Evaluate on prediction data
        metrics = self.evaluator.evaluate_to_metrics(prediction_data, self._fitted_parameters)

        # Generate plot data
        plot_data = self._create_prediction_plot_data(prediction_data)

        # Create detailed predictions
        predictions = self._create_detailed_predictions(prediction_data)

        return AnisotropicPredictResponse(
            status="ok",
            model_type=self.config.model_type.value,
            metrics=metrics,
            plot_data=plot_data,
            predictions=predictions
        )

    def get_fitted_parameters(self) -> Optional[Dict[str, float]]:
        """Get fitted parameters as dictionary"""
        if self._fitted_parameters is None:
            return None
        return self._fitted_parameters.to_dict()



    def _create_parameter_response(self) -> List[AnisotropicParameterValue]:
        """Create parameter response for API"""
        if self._fitted_parameters is None:
            return []

        params = []
        param_dict = self._fitted_parameters.to_dict()

        for name, value in param_dict.items():

            params.append(AnisotropicParameterValue(
                name=name,
                value=value
            ))

        return params

    def _create_fit_plot_data(self, data: np.ndarray) -> AnisotropicPlotData:
        """Create plot data for fit results"""
        # Extract experimental data
        lam1_exp = data[:, 0]
        lam2_exp = data[:, 2]
        p11_exp = data[:, 1]
        p22_exp = data[:, 3]

        # Compute model predictions
        p11_pred, p22_pred = self.evaluator._compute_predictions(
            lam1_exp, lam2_exp, self._fitted_parameters
        )

        lines = [
            AnisotropicPlotLine(
                name="Experimental P11",
                x=lam1_exp.tolist(),
                y=p11_exp.tolist(),
                line_type="markers",
                color="red"
            ),
            AnisotropicPlotLine(
                name=f"Model P11 ({self.config.model_type.value})",
                x=lam1_exp.tolist(),
                y=p11_pred.tolist(),
                line_type="lines",
                color="red"
            ),
            AnisotropicPlotLine(
                name="Experimental P22",
                x=lam2_exp.tolist(),
                y=p22_exp.tolist(),
                line_type="markers",
                color="blue"
            ),
            AnisotropicPlotLine(
                name=f"Model P22 ({self.config.model_type.value})",
                x=lam2_exp.tolist(),
                y=p22_pred.tolist(),
                line_type="lines",
                color="blue"
            ),
        ]

        return AnisotropicPlotData(
            title=f"{self.model.get_model_name()} Model Fit",
            x_label="Stretch λ",
            y_label="1st PK Stress [MPa]",
            lines=lines
        )

    def _create_prediction_plot_data(self, data: np.ndarray) -> AnisotropicPlotData:
        """Create plot data for predictions"""
        return self._create_fit_plot_data(data)  # Same format for now

    def _create_detailed_predictions(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Create detailed prediction results"""
        lam1_exp = data[:, 0]
        lam2_exp = data[:, 2]
        p11_exp = data[:, 1]
        p22_exp = data[:, 3]

        p11_pred, p22_pred = self.evaluator._compute_predictions(
            lam1_exp, lam2_exp, self._fitted_parameters
        )

        predictions = []
        for i in range(len(lam1_exp)):
            predictions.append({
                'lambda1': float(lam1_exp[i]),
                'lambda2': float(lam2_exp[i]),
                'P11_experimental': float(p11_exp[i]),
                'P22_experimental': float(p22_exp[i]),
                'P11_predicted': float(p11_pred[i]),
                'P22_predicted': float(p22_pred[i]),
                'error_P11': float(abs(p11_exp[i] - p11_pred[i])),
                'error_P22': float(abs(p22_exp[i] - p22_pred[i])),
            })

        return predictions

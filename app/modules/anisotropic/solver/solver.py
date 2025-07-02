from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from app.logger import logger
from app.modules.anisotropic.shema import (
    AnisotropicFitResponse, AnisotropicPredictResponse, AnisotropicParameterValue,
    AnisotropicPlotData, AnisotropicPlotLine
)
from app.modules.anisotropic.solver.config import AnisotropicModelType
from app.modules.anisotropic.solver.evaluator import ModelEvaluator
from app.modules.anisotropic.solver.models import ModelFactory, AnisotropicModel, ModelParameters
from app.modules.anisotropic.solver.optimizer import ParameterOptimizer, OptimizationResult
from app.modules.solver import ModuleSolver


@dataclass
class AnisotropicSolverConfig:
    """Configuration for anisotropic solver"""
    model_type: AnisotropicModelType
    kappa: Optional[float]
    alpha: Optional[float]


@dataclass
class AnisotropicSolver(ModuleSolver):
    """Main solver for anisotropic hyperelastic models following SOLID principles"""
    config: AnisotropicSolverConfig = field(init=False)
    model: AnisotropicModel = field(init=False)
    optimizer: ParameterOptimizer = field(init=False)
    evaluator: ModelEvaluator = field(init=False)
    _optimization_result: OptimizationResult = field(init=False)
    _fitted_parameters: ModelParameters = field(init=False)

    # Storage for fitted parametersкак

    def setup_solver(self, config: AnisotropicSolverConfig):
        self.config = config
        self.model = ModelFactory.create_model(config.model_type)
        self.optimizer = ParameterOptimizer(self.model)
        self.evaluator = ModelEvaluator(self.model)

    def fit(self, data: pd.DataFrame) -> AnisotropicFitResponse:
        """Fit the anisotropic model to data"""
        logger.info(f"Starting fit for {self.model.get_model_name()} model")

        # Optimize parameters
        self._optimization_result = self.optimizer.optimize(
            data.to_numpy(dtype=float),
            self.config.alpha,
            self.config.kappa,
        )

        if not self._optimization_result.success:
            logger.error(f"Optimization failed: {self._optimization_result.message}")

        self._fitted_parameters = self._optimization_result.parameters

        # Evaluate model performance
        metrics = self.evaluator.evaluate_to_metrics(data, self._fitted_parameters)

        # Create parameter response
        parameters = self._create_parameter_response()

        # Generate plot data
        plot_data = self._create_fit_plot_data(data)

        return AnisotropicFitResponse(
            status="ok" if self._optimization_result.success else "error",
            parameters=parameters,
            metrics=metrics,
            plot_data=plot_data,
            # convergence_info=self._optimization_result.convergence_info
        )

    def predict(self, prediction_data: pd.DataFrame, optimized_params) -> AnisotropicPredictResponse:

        """Make predictions using fitted model"""
        if optimized_params is None:
            raise ValueError("Model must be fitted before making predictions")
        self._fitted_parameters = optimized_params

        logger.info("Making predictions with fitted model")

        # Evaluate on prediction data
        metrics = self.evaluator.evaluate_to_metrics(prediction_data, self._fitted_parameters)
        logger.info(f'metrics: {metrics}:')
        # Generate plot data
        plot_data = self._create_prediction_plot_data(prediction_data)

        # Create detailed predictions
        # predictions = self._create_detailed_predictions(prediction_data)

        return AnisotropicPredictResponse(
            status="ok",
            metrics=metrics,
            plot_data=plot_data,
            # predictions=predictions
        )

    def get_optimization_result(self) -> Optional[OptimizationResult]:
        """Get fitted parameters as dictionary"""
        if self._optimization_result is None:
            return None
        return self._optimization_result

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

    def _create_fit_plot_data(self, data: pd.DataFrame) -> AnisotropicPlotData:
        """Create plot data for fit results"""

        # Extract experimental data
        lam1_exp = data.iloc[:, 0].to_numpy(dtype=float)
        lam2_exp = data.iloc[:, 2].to_numpy(dtype=float)
        p11_exp = data.iloc[:, 1].to_numpy(dtype=float)
        p22_exp = data.iloc[:, 3].to_numpy(dtype=float)

        # Compute model predictions
        p11_pred, p22_pred = self.evaluator.compute_predictions(
            lam1_exp, lam2_exp, self._fitted_parameters
        )

        lines = [
            AnisotropicPlotLine(
                name="Experimental P11",
                x=lam1_exp.tolist(),
                y=p11_exp.tolist(),

            ),
            AnisotropicPlotLine(
                name=f"Model P11 ({self.config.model_type.value})",
                x=lam1_exp.tolist(),
                y=p11_pred.tolist(),

            ),
            AnisotropicPlotLine(
                name="Experimental P22",
                x=lam2_exp.tolist(),
                y=p22_exp.tolist(),

            ),
            AnisotropicPlotLine(
                name=f"Model P22 ({self.config.model_type.value})",
                x=lam2_exp.tolist(),
                y=p22_pred.tolist(),

            ),
        ]

        return AnisotropicPlotData(
            title=f"{self.model.get_model_name()} Model Fit",
            x_label="Stretch λ",
            y_label="1st PK Stress [MPa]",
            lines=lines
        )

    def _create_prediction_plot_data(self, data: pd.DataFrame) -> AnisotropicPlotData:
        """Create plot data for predictions"""
        logger.info("Create plot data for predictions")
        return self._create_fit_plot_data(data)  # Same format for now

    def _create_detailed_predictions(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Create detailed prediction results"""
        lam1_exp = data[:, 0]
        lam2_exp = data[:, 2]
        p11_exp = data[:, 1]
        p22_exp = data[:, 3]

        p11_pred, p22_pred = self.evaluator.compute_predictions(
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

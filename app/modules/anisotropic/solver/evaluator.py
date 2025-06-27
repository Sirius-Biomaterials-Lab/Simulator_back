from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from app.logger import logger
from app.modules.anisotropic.shema import AnisotropicMetric
from app.modules.anisotropic.solver.models import AnisotropicModel, ModelParameters


@dataclass
class EvaluationResult:
    """Result of model evaluation"""
    r2_p11: float
    r2_p22: float
    rmse_p11: float
    rmse_p22: float
    mae_p11: float
    mae_p22: float
    overall_r2: float


class ModelEvaluator:
    """Evaluator for anisotropic model quality using SOLID principles"""

    def __init__(self, model: AnisotropicModel):
        self.model = model

    def evaluate(self, data: np.ndarray, params: ModelParameters) -> EvaluationResult:
        """Evaluate model performance on given data"""
        logger.info(f"Evaluating {self.model.get_model_name()} model performance")

        # Extract experimental data
        lam1_exp = data[:, 0]
        lam2_exp = data[:, 2]
        p11_exp = data[:, 1]
        p22_exp = data[:, 3]

        # Compute model predictions
        p11_pred, p22_pred = self._compute_predictions(lam1_exp, lam2_exp, params)

        # Calculate metrics
        print(p11_exp, p11_pred)
        r2_p11 = self._safe_r2_score(p11_exp, p11_pred)
        r2_p22 = self._safe_r2_score(p22_exp, p22_pred)

        rmse_p11 = np.sqrt(mean_squared_error(p11_exp, p11_pred))
        rmse_p22 = np.sqrt(mean_squared_error(p22_exp, p22_pred))

        mae_p11 = np.mean(np.abs(p11_exp - p11_pred))
        mae_p22 = np.mean(np.abs(p22_exp - p22_pred))

        # Overall R² combining both directions
        overall_r2 = self._compute_overall_r2(
            np.concatenate([p11_exp, p22_exp]),
            np.concatenate([p11_pred, p22_pred])
        )

        result = EvaluationResult(
            r2_p11=r2_p11,
            r2_p22=r2_p22,
            rmse_p11=rmse_p11,
            rmse_p22=rmse_p22,
            mae_p11=mae_p11,
            mae_p22=mae_p22,
            overall_r2=overall_r2
        )

        logger.info(f"Evaluation results: R²_P11={r2_p11:.4f}, R²_P22={r2_p22:.4f}, "
                    f"Overall_R²={overall_r2:.4f}")

        return result

    def evaluate_to_metrics(self, data: np.ndarray, params: ModelParameters) -> List[AnisotropicMetric]:
        """Evaluate and return metrics in response format"""
        result = self.evaluate(data, params)

        return [
            AnisotropicMetric(name="R² P11", value=result.r2_p11, direction="P11"),
            AnisotropicMetric(name="R² P22", value=result.r2_p22, direction="P22"),
            AnisotropicMetric(name="RMSE P11", value=result.rmse_p11, direction="P11"),
            AnisotropicMetric(name="RMSE P22", value=result.rmse_p22, direction="P22"),
            AnisotropicMetric(name="MAE P11", value=result.mae_p11, direction="P11"),
            AnisotropicMetric(name="MAE P22", value=result.mae_p22, direction="P22"),
            AnisotropicMetric(name="Overall R²", value=result.overall_r2, direction=None),
        ]

    def _compute_predictions(self, lam1: np.ndarray, lam2: np.ndarray,
                             params: ModelParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Compute model predictions for given deformations"""

        p11_pred = np.zeros_like(lam1)
        p22_pred = np.zeros_like(lam2)

        for lam_1, lam_2, i in zip(lam1, lam2, range(len(lam1))):
            try:
                P = self.model.compute_stress(lam_1, lam_2, params)
                p11_pred[i] = P[0, 0]
                p22_pred[i] = P[1, 1]
            except Exception as e:
                logger.warning(f"Error computing stress at point {i}: {e}")
                p11_pred[i] = 0.0
                p22_pred[i] = 0.0

        return p11_pred, p22_pred

    def _safe_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Safely compute R² score with error handling"""
        try:
            # Check for constant values
            if np.var(y_true) == 0:
                # If true values are constant, R² is undefined
                # Return 1.0 if predictions are also constant and equal, 0.0 otherwise
                if np.var(y_pred) == 0 and np.allclose(y_true, y_pred):
                    return 1.0
                else:
                    return 0.0

            r2 = r2_score(y_true, y_pred)

            # Clamp to reasonable range (R² can be negative for very bad fits)
            # return max(-5.0, min(1.0, r2))
            return r2

        except Exception as e:
            logger.warning(f"Error computing R² score: {e}")
            return 0.0

    def _compute_overall_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute overall R² for combined directions"""
        return self._safe_r2_score(y_true, y_pred)


class ComparisonEvaluator:
    """Evaluator for comparing multiple models"""

    def __init__(self, models: List[AnisotropicModel]):
        self.evaluators = [ModelEvaluator(model) for model in models]
        self.models = models

    def compare_models(self, data: np.ndarray,
                       model_params: List[ModelParameters]) -> List[EvaluationResult]:
        """Compare multiple models on the same data"""
        if len(model_params) != len(self.evaluators):
            raise ValueError("Number of parameter sets must match number of models")

        results = []
        for evaluator, params in zip(self.evaluators, model_params):
            result = evaluator.evaluate(data, params)
            results.append(result)

        # Log comparison summary
        logger.info("Model comparison summary:")
        for i, (model, result) in enumerate(zip(self.models, results)):
            logger.info(f"  {model.get_model_name()}: Overall R² = {result.overall_r2:.4f}")

        return results

    def get_best_model_index(self, results: List[EvaluationResult]) -> int:
        """Get index of best performing model based on overall R²"""
        best_r2 = max(result.overall_r2 for result in results)
        best_index = next(i for i, result in enumerate(results) if result.overall_r2 == best_r2)

        logger.info(f"Best model: {self.models[best_index].get_model_name()} "
                    f"(R² = {best_r2:.4f})")

        return best_index

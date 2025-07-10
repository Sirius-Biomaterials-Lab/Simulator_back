from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from app.logger import logger
from app.modules.anisotropic.shema import AnisotropicMetric
from app.modules.anisotropic.solver.models import AnisotropicModel, ModelParameters


@dataclass
class EvaluationResult:
    """Result of model evaluation"""
    r2_p11: float
    r2_p22: float
    rmse_p11: Optional[float] = field(default=None)
    rmse_p22: Optional[float] = field(default=None)
    mae_p11: Optional[float] = field(default=None)
    mae_p22: Optional[float] = field(default=None)
    overall_r2: Optional[float] = field(default=None)


class ModelEvaluator:
    """Evaluator for anisotropic model quality using SOLID principles"""

    def __init__(self, model: AnisotropicModel):
        self.model = model

    def evaluate(self, data: pd.DataFrame, params: ModelParameters) -> EvaluationResult:
        """Evaluate model performance on given data"""
        logger.info(f"Evaluating {self.model.get_model_name()} model performance")
        # Extract experimental data
        lam1_exp = data.iloc[:, 0].to_numpy(dtype=float)
        p11_exp = data.iloc[:, 1].to_numpy(dtype=float)
        lam2_exp = data.iloc[:, 2].to_numpy(dtype=float)
        p22_exp = data.iloc[:, 3].to_numpy(dtype=float)

        # Compute model predictions
        p11_pred, p22_pred = self.compute_predictions(lam1_exp, lam2_exp, params)

        # Calculate metrics

        r2_p11 = self._safe_r2_score(p11_exp, p11_pred)
        r2_p22 = self._safe_r2_score(p22_exp, p22_pred)

        # rmse_p11 = np.sqrt(mean_squared_error(p11_exp, p11_pred))
        # rmse_p22 = np.sqrt(mean_squared_error(p22_exp, p22_pred))
        #
        # mae_p11 = np.mean(np.abs(p11_exp - p11_pred))
        # mae_p22 = np.mean(np.abs(p22_exp - p22_pred))
        #
        # # Overall R² combining both directions
        # overall_r2 = self._compute_overall_r2(
        #     np.concatenate([p11_exp, p22_exp]),
        #     np.concatenate([p11_pred, p22_pred])
        # )

        result = EvaluationResult(
            r2_p11=r2_p11,
            r2_p22=r2_p22,
            # rmse_p11=rmse_p11,
            # rmse_p22=rmse_p22,
            # mae_p11=mae_p11,
            # mae_p22=mae_p22,
            # overall_r2=overall_r2
        )

        logger.info(f"Evaluation results: R²_P11={r2_p11:.4f}, R²_P22={r2_p22:.4f},")

        return result

    def evaluate_to_metrics(self, data: pd.DataFrame, params: ModelParameters) -> List[AnisotropicMetric]:
        """Evaluate and return metrics in response format"""
        logger.info(f"evaluate_to_metrics")
        result = self.evaluate(data, params)

        return [
            AnisotropicMetric(name="R² P11", value=result.r2_p11),
            AnisotropicMetric(name="R² P22", value=result.r2_p22),
            # AnisotropicMetric(name="RMSE P11", value=result.rmse_p11, direction="P11"),
            # AnisotropicMetric(name="RMSE P22", value=result.rmse_p22, direction="P22"),
            # AnisotropicMetric(name="MAE P11", value=result.mae_p11, direction="P11"),
            # AnisotropicMetric(name="MAE P22", value=result.mae_p22, direction="P22"),
            # AnisotropicMetric(name="Overall R²", value=result.overall_r2, direction=None),
        ]

    def compute_predictions(self, lam1: np.ndarray, lam2: np.ndarray,
                            params: ModelParameters) -> Tuple[np.ndarray, np.ndarray]:
        """Compute model predictions for given deformations"""

        p11_pred = np.zeros_like(lam1)
        p22_pred = np.zeros_like(lam2)
        logger.info("Computing predictions")
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

    @staticmethod
    def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

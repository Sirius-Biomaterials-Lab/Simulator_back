from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from autograd import jacobian
from scipy.optimize import minimize, Bounds

from app.logger import logger
from app.modules.anisotropic.solver.config import AnisotropicModelConfig, AnisotropicConstants
from app.modules.anisotropic.solver.models import AnisotropicModel, ModelParameters


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    success: bool
    parameters: ModelParameters
    final_error: float
    iterations: int
    message: str
    convergence_info: Dict[str, Any]


class ObjectiveFunction:
    """Objective function for parameter optimization"""

    def __init__(self, model: AnisotropicModel, data: np.ndarray, alpha: Optional[float] = None,
                 kappa: Optional[float] = None):
        self.model = model
        self.data = data
        self.alpha = alpha
        self.kappa = kappa

        # Extract data columns
        self.lam1 = data[:, 0]
        self.lam2 = data[:, 2]
        self.PE1 = data[:, 1]
        self.PE2 = data[:, 3]

    def __call__(self, params: np.ndarray) -> float:
        """Compute objective function value"""
        try:

            model_params = ModelParameters.from_array(params, self.alpha, self.kappa)

            total_error = 0.0
            n_points = len(self.lam1)

            for i in range(n_points):
                # Compute stress using model
                P = self.model.compute_stress(self.lam1[i], self.lam2[i], model_params)

                # Compute squared error
                error_p11 = (P[0, 0] - self.PE1[i]) ** 2
                error_p22 = (P[1, 1] - self.PE2[i]) ** 2
                total_error += error_p11 + error_p22

            return total_error / n_points

        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1e10  # Large penalty for invalid parameters


class ParameterOptimizer:
    """Parameter optimizer using SOLID principles"""

    def __init__(self, model: AnisotropicModel):
        self.model = model
        self.convergence_tolerance = AnisotropicConstants.CONVERGENCE_TOLERANCE
        self.max_iterations = AnisotropicConstants.MAX_ITERATIONS

    def optimize(self, data: np.ndarray,
                 alpha: Optional[float] = None,
                 kappa: Optional[float] = None
                 ) -> OptimizationResult:
        """Optimize model parameters"""
        logger.info(f"Starting optimization for {self.model.get_model_name()} model")

        # Create objective function
        objective = ObjectiveFunction(self.model, data)

        # Compute jacobian
        jacobian_func = jacobian(objective)

        # Set up optimization parameters
        initial_params = self._get_initial_parameters(alpha, kappa)
        bounds = self._get_parameter_bounds(alpha, kappa)

        logger.info(f"Initial parameters: {initial_params}")
        logger.info(f"Parameter bounds: {bounds}")

        # Perform optimization
        try:
            result = minimize(
                fun=objective,
                x0=initial_params,
                jac=jacobian_func,
                bounds=Bounds(bounds[:, 0], bounds[:, 1]),
                method='L-BFGS-B',
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.convergence_tolerance,
                    'disp': True
                }
            )

            # Create result object
            optimized_params = ModelParameters.from_array(result.x, alpha, kappa)

            optimization_result = OptimizationResult(
                success=result.success,
                parameters=optimized_params,
                final_error=result.fun,
                iterations=result.nit,
                message=result.message,
                convergence_info={
                    'function_evaluations': result.nfev,
                    'jacobian_evaluations': result.njev,
                    'gradient_norm': np.linalg.norm(result.jac) if result.jac is not None else None,
                    'optimization_method': 'L-BFGS-B'
                }
            )

            logger.info(f"Optimization completed: success={result.success}, "
                        f"final_error={result.fun:.6e}, iterations={result.nit}")

            return optimization_result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")

            # Return failed result with default parameters
            default_params = ModelParameters.from_array(
                initial_params, alpha, kappa
            )
            return OptimizationResult(
                success=False,
                parameters=default_params,
                final_error=float('inf'),
                iterations=0,
                message=f"Optimization failed: {str(e)}",
                convergence_info={}
            )

    def _get_initial_parameters(self, alpha: Optional[float], kappa: Optional[float]) -> np.ndarray:
        """Get initial parameter values for optimization"""
        alpha = True if alpha is not None else False
        kappa = True if kappa is not None else False
        return np.array(AnisotropicModelConfig.get_initial_parameters(alpha, kappa))

    def _get_parameter_bounds(self, alpha: Optional[float], kappa: Optional[float]) -> np.ndarray:
        """Get parameter bounds for optimization"""
        alpha = True if alpha is not None else False
        kappa = True if kappa is not None else False
        bounds_list = AnisotropicModelConfig.get_parameter_bounds(alpha, kappa)
        return np.array(bounds_list)

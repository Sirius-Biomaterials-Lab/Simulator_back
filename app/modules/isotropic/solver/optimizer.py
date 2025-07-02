from typing import Sequence, Callable

import numpy as np

from app.modules.isotropic.solver.stress_calculator import StressCalculator


class TissueModelLossEvaluator:
    """
    Evaluator for total mean squared error (MSE) of tissue model parameters
    against experimental uniaxial and biaxial stress data.

    Combines separate error contributions from uniaxial and biaxial tests
    with user-defined weights.
    """

    def __init__(
            self,
            stress_calculator: StressCalculator,
            stretch_data: np.ndarray,
            stress_data: np.ndarray,
            is_uniaxial: np.ndarray,
    ):

        """
        Initialize the loss evaluator.

        Parameters:
            stress_calculator: Instance of StressCalculator to compute model stresses.
            stretch_data: Array of shape (N, 2) with stretch ratios (λ1, λ2) per test.
            stress_data: Array of shape (N, 2) with experimental stresses (P11, P22).
            weight_uniaxial: Relative weight for uniaxial MSE term (default: 0.5).
            weight_biaxial: Relative weight for biaxial MSE term (default: 0.5).
            eps: Tolerance to classify a test as uniaxial when |λ2| < tol.
        """
        self.stress_calculator = stress_calculator
        self._lambdas = stretch_data
        self._exp_stress = stress_data
        self._is_uniaxial = is_uniaxial
        self._n_uni = int(np.sum(self._is_uniaxial))
        self._n_biax = len(stretch_data) - self._n_uni

        self.weight_uni = 1.0 / self._n_uni if self._n_uni else 0.0
        self.weight_biax = 1.0 / self._n_biax if self._n_biax else 0.0

    def compute_loss(self, params: Sequence[float]) -> float:
        """
        Compute the weighted MSE for given model parameters.

        Parameters:
            params: Sequence of model parameters to pass into StressCalculator.

        Returns:
            Total weighted MSE combining uniaxial and biaxial components.
        """
        error_uni, error_biax = 0.0, 0.0

        # Loop over all data points
        for (lam1, lam2), (exp_P11, exp_P22), is_uni in zip(
                self._lambdas, self._exp_stress, self._is_uniaxial
        ):
            # Choose the correct stress computation
            compute_fn: Callable = (
                self.stress_calculator.compute_uniaxial if is_uni else self.stress_calculator.compute_biaxial
            )
            model_P11, model_P22 = compute_fn(params, lam1, lam2)

            # Compute squared differences
            diff11 = model_P11 - exp_P11
            if is_uni:
                error_uni += diff11 ** 2
            else:
                diff22 = model_P22 - exp_P22
                error_biax += diff11 ** 2 + diff22 ** 2

        # Convert sums to mean squared errors and apply weights
        mse_uni = (
            self.weight_uni * error_uni / self._n_uni if self._n_uni > 0 else 0.0
        )
        mse_biax = (
            self.weight_biax * error_biax / self._n_biax if self._n_biax > 0 else 0.0
        )

        return mse_uni + mse_biax

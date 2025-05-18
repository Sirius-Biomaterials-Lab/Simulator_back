from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import minimize, OptimizeResult

from app.exception import DataNotFound
from app.logger import logger
from app.modules.isotropic.solver.hyperelastic_model import HyperelasticModel


@dataclass
class InvariantCalculator:
    """Calculates invariants based on stretch ratios and stress."""

    @staticmethod
    def compute(lam1: float, lam2: float, stress_y: float) -> Tuple[float, float, float, float]:
        eps = 1e-7
        if abs(lam2 - 1.0) < eps and abs(stress_y) < eps:
            lam3 = lam1 ** -0.5
            I1 = lam1 ** 2 + 2 * lam1 ** -1
            I2 = 2 * lam1 + lam1 ** -2
            return I1, I2, lam3, lam3
        else:
            lam3 = 1.0 / (lam1 * lam2)
            I1 = lam1 ** 2 + lam2 ** 2 + lam3 ** 2
            I2 = (lam1 ** 2 * lam2 ** 2 +
                  lam2 ** 2 * lam3 ** 2 +
                  lam3 ** 2 * lam1 ** 2)
            return I1, I2, lam2, lam3


@dataclass
class StressCalculator:
    dW_dI1_func: Callable
    dW_dI2_func: Callable

    def compute(self, params: np.ndarray, lam1: float, lam2: float, stress_y: float) -> Tuple[
        float, float, float, float]:
        I1, I2, lam2_eff, lam3 = InvariantCalculator.compute(lam1, lam2, stress_y)
        args = (I1, I2, *params)

        dW_dI1 = self.dW_dI1_func(*args)
        dW_dI2 = self.dW_dI2_func(*args)

        p = 2.0 * (dW_dI1 * lam3 ** 2 + dW_dI2 * (I1 * lam3 ** 2 - lam3 ** 4))

        sigma_11 = 2.0 * (dW_dI1 * lam1 ** 2 + dW_dI2 * (I1 * lam1 ** 2 - lam1 ** 4)) - p
        sigma_22 = 2.0 * (dW_dI1 * lam2_eff ** 2 + dW_dI2 * (I1 * lam2_eff ** 2 - lam2_eff ** 4)) - p
        sigma_33 = 2.0 * (dW_dI1 * lam3 ** 2 + dW_dI2 * (I1 * lam3 ** 2 - lam3 ** 4)) - p

        return sigma_11, sigma_22, sigma_33, p


@dataclass
class IsotropicSolver:
    data: pd.DataFrame
    error_function_name: str
    error_function: Callable
    hyperelastic_model: HyperelasticModel
    optimization_method: str = 'L-BFGS-B'

    dW_dI1_func: Callable = field(init=False)
    dW_dI2_func: Callable = field(init=False)
    stress_calculator: StressCalculator = field(init=False)

    def __post_init__(self):
        logger.info("Initializing symbolic derivatives...")
        energy_func = self.hyperelastic_model.calculate()
        I1, I2 = sp.symbols('I1 I2')
        self.dW_dI1_func = sp.lambdify((I1, I2, *self.hyperelastic_model.params_sym), energy_func.dW_dI1_sym, 'numpy')
        self.dW_dI2_func = sp.lambdify((I1, I2, *self.hyperelastic_model.params_sym), energy_func.dW_dI2_sym, 'numpy')
        self.stress_calculator = StressCalculator(self.dW_dI1_func, self.dW_dI2_func)

    def fit_model(self) -> OptimizeResult:
        if self.data is None or self.data.empty:
            logger.error("No data to fit. Please load data first.")
            raise DataNotFound

        lambdas = self.data[['lambda_x', 'lambda_y']].values
        experimental_data = self.data[['stress_x_mpa', 'stress_y_mpa']].values
        init_params = np.full(len(self.hyperelastic_model.params_sym), 0.5)

        constraints = [{'type': 'ineq',
                        'fun': self._gent_constraint}] if 'gent' in self.hyperelastic_model.model_name.lower() else None

        logger.info("Starting optimization...")
        result = minimize(
            self.error_function,
            init_params,
            args=(lambdas, experimental_data, self.stress_calculator.compute),
            method=self.optimization_method,
            bounds=self.hyperelastic_model.bounds,
            constraints=constraints
        )
        logger.info("Optimization result: success=%s, params=%s", result.success, result.x)
        return result

    def _gent_constraint(self, params: np.ndarray) -> float:
        Jm = params[1]
        I1_values = [InvariantCalculator.compute(l1, l2, sy)[0] for l1, l2, sy in
                     zip(self.data['lambda_x'], self.data['lambda_y'], self.data['stress_y_mpa'])]
        constraints = [1 - (I1 - 3) / Jm for I1 in I1_values]
        return min(constraints) - 1e-6

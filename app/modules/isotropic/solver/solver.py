from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from app.logger import logger
from app.modules.isotropic.shema import Line, PlotData, Parameter, Metric, IsotropicFitResponse
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


class IsotropicSolver:
    lambdas = np.ndarray[float]
    experimental_data = np.ndarray[float]
    optimization_method: str = 'L-BFGS-B'
    optimization_params: np.ndarray[float]
    stress_calculator: StressCalculator

    def __init__(self, data, error_function_name: str, error_function: Callable, hyperelastic_model: HyperelasticModel):
        logger.info("Initializing symbolic derivatives...")
        self.data = data
        self.error_function_name = error_function_name
        self.error_function = error_function
        self.hyperelastic_model = hyperelastic_model

        energy_func = self.hyperelastic_model.calculate()
        I1, I2 = sp.symbols('I1 I2')
        self.dW_dI1_func = sp.lambdify((I1, I2, *self.hyperelastic_model.params_sym), energy_func.dW_dI1_sym, 'numpy')
        self.dW_dI2_func = sp.lambdify((I1, I2, *self.hyperelastic_model.params_sym), energy_func.dW_dI2_sym, 'numpy')
        self.stress_calculator = StressCalculator(self.dW_dI1_func, self.dW_dI2_func)
        self.lambdas = self.data[['lambda_x', 'lambda_y']].values
        self.experimental_data = self.data[['stress_x_mpa', 'stress_y_mpa']].values

    def fit_model(self):

        init_params = np.full(len(self.hyperelastic_model.params_sym), 0.5)

        constraints = [{'type': 'ineq',
                        'fun': self._gent_constraint}] if 'gent' in self.hyperelastic_model.model_name.lower() else None

        logger.info("Starting optimization...")
        result = minimize(
            self.error_function,
            init_params,
            args=(self.lambdas, self.experimental_data, self.stress_calculator.compute),
            method=self.optimization_method,
            bounds=self.hyperelastic_model.bounds,
            constraints=constraints
        )
        logger.info("Optimization result: success=%s, params=%s", result.success, result.x)
        self.optimization_params = result.x

    def graph_fit(self):
        params = []
        for value in self.optimization_params:
            params.append(Parameter(name='tmp', value=value))

        sigma_model_11 = []
        sigma_model_22 = []
        sigma_exp_11 = []
        sigma_exp_22 = []

        for lams, P_exp in zip(self.lambdas, self.experimental_data):
            lam1, lam2 = lams
            P11_exp, P22_exp = P_exp

            s11_model, s22_model, _, _ = self.stress_calculator.compute(self.optimization_params, lam1, lam2, P22_exp)
            s11_exp = P11_exp * lam1
            s22_exp = P22_exp * lam2

            sigma_model_11.append(s11_model)
            sigma_model_22.append(s22_model)
            sigma_exp_11.append(s11_exp)
            sigma_exp_22.append(s22_exp)

        lines = [
            Line(name="Exp σ11", x=self.lambdas[:, 0], y=sigma_exp_11),
            Line(name="Model σ11", x=self.lambdas[:, 0], y=sigma_model_11),
            Line(name="Exp σ22", x=self.lambdas[:, 1], y=sigma_exp_22),
            Line(name="Model σ22", x=self.lambdas[:, 1], y=sigma_model_22),
        ]
        # # -------- Второй график с инвариантами --------
        # self.plot_invariants()

        metrics = self._calculate_metrics_r2(sigma_model_11, sigma_model_22, sigma_exp_11, sigma_exp_22)

        plot_data = PlotData(
            name=f"{self.hyperelastic_model.model_name} fit ({self.error_function_name}), R²={1222.121212332:.2f}",
            x_label='Stretch λ',
            y_label='Stress (MPa)',
            lines=lines
        )
        return IsotropicFitResponse(
            metrics=metrics,
            parameters=params,
            plot_data=plot_data
        )

    def _gent_constraint(self, params: np.ndarray) -> float:
        Jm = params[1]
        I1_values = [InvariantCalculator.compute(l1, l2, sy)[0] for l1, l2, sy in
                     zip(self.data['lambda_x'], self.data['lambda_y'], self.data['stress_y_mpa'])]
        constraints = [1 - (I1 - 3) / Jm for I1 in I1_values]
        return min(constraints) - 1e-6

    def _calculate_metrics_r2(self, sigma_model_11, sigma_model_22, sigma_exp_11, sigma_exp_22):
        model_combined = np.concatenate([sigma_model_11, sigma_model_22])
        exp_combined = np.concatenate([sigma_exp_11, sigma_exp_22])
        return [Metric(name='r2', value=r2_score(exp_combined, model_combined))]

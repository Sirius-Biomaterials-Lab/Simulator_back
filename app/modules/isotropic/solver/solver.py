from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from app.logger import logger
from app.modules.isotropic.solver.hyperelastic_model import HyperelasticModel
from app.modules.isotropic.solver.model_loss import TissueModelLossEvaluator
from app.modules.isotropic.solver.shema import Line, PlotData, Parameter, Metric, IsotropicFitResponse, \
    IsotropicPredictResponse
from app.modules.isotropic.solver.stress_calculator import StressCalculator


@dataclass
class IsotropicSolver:
    lambdas: np.ndarray[float] = field(init=False)
    experimental_data: np.ndarray[float] = field(init=False)
    is_uniaxial: np.ndarray[bool] = field(init=False)

    stress_calculator: StressCalculator = field(init=False)
    optimization_method: str = 'L-BFGS-B'
    eps: float = 1e-6

    def set_up_solver(self, hyperelastic_model: HyperelasticModel):
        logger.info("Initializing symbolic derivatives...")
        self.hyperelastic_model = hyperelastic_model
        energy_func = self.hyperelastic_model.calculate()
        I1, I2 = sp.symbols('I1 I2')
        self.dW_dI1_func = sp.lambdify((I1, I2, *self.hyperelastic_model.params_sym), energy_func.dW_dI1_sym, 'numpy')
        self.dW_dI2_func = sp.lambdify((I1, I2, *self.hyperelastic_model.params_sym), energy_func.dW_dI2_sym, 'numpy')
        self.stress_calculator = StressCalculator(self.dW_dI1_func, self.dW_dI2_func)

    # def fit_model(self, data: pd.DataFrame, error_function: Callable):
    def fit_model(self, data: pd.DataFrame):
        self.lambdas = data[['lambda_x', 'lambda_y']].values
        self.experimental_data = data[['stress_x_mpa', 'stress_y_mpa']].values
        self.is_uniaxial = np.abs(self.experimental_data[:, 1]) < self.eps

        init_params = np.full(len(self.hyperelastic_model.params_sym), 0.5)

        tissue_model_loss = TissueModelLossEvaluator(self.stress_calculator, self.lambdas, self.experimental_data,
                                                     self.is_uniaxial)

        logger.info("Starting optimization...")
        result = minimize(
            fun=tissue_model_loss.compute_loss,
            x0=init_params,
            method=self.optimization_method,
            bounds=self.hyperelastic_model.bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        logger.info("Optimization result: success=%s, params=%s", result.success, result.x)
        return result.x

    def graph_fit(self, optimization_params: np.ndarray):
        params = []
        for value in optimization_params:
            params.append(Parameter(name='tmp', value=value))

        sigma_model_11, sigma_model_22 = [], []

        for lams, is_uni in zip(self.lambdas, self.is_uniaxial):
            lam1, lam2 = lams
            # Choose the correct stress computation
            compute_fn: Callable = (
                self.stress_calculator.compute_uniaxial if is_uni else self.stress_calculator.compute_biaxial
            )
            p11, p22 = compute_fn(optimization_params, lam1, lam2)

            sigma_model_11.append(p11)
            sigma_model_22.append(p22)

        biax_mask = ~self.is_uniaxial
        lines = [
            Line(name="Exp P_xx", x=self.lambdas[:, 0], y=self.experimental_data[:, 0]),
            Line(name="Model P_xx", x=self.lambdas[:, 0], y=sigma_model_11),
            Line(name="Exp P_yy (biax)", x=self.lambdas[biax_mask, 1], y=self.experimental_data[biax_mask, 1]),
            Line(name="Model P_yy (biax)", x=self.lambdas[biax_mask, 1], y=np.array(sigma_model_22)[biax_mask]),
        ]
        # # -------- Второй график с инвариантами --------
        # self.plot_invariants()

        metric = self._calculate_metrics_r2(
            np.r_[self.experimental_data[:, 0], self.experimental_data[:, 1][biax_mask]],
            np.r_[sigma_model_11, np.array(sigma_model_22)[biax_mask]]
        )

        plot_data = PlotData(
            name=f"{self.hyperelastic_model.model_name} fit",
            x_label='Stretch λ',
            y_label='Stress [MPa]',
            lines=lines
        )
        return IsotropicFitResponse(
            metrics=[metric],
            parameters=params,
            plot_data=plot_data
        )

    def predict(self, prediction_data: pd.DataFrame, optimized_params: np.ndarray) -> IsotropicPredictResponse:
        lam_x = prediction_data['lambda_x'].to_numpy()
        lam_y = prediction_data['lambda_y'].to_numpy()
        stress_x = prediction_data['stress_x_mpa'].to_numpy()
        stress_y = prediction_data['stress_y_mpa'].to_numpy()
        uniaxial_mask = np.abs(stress_y) < self.eps
        biax_mask = ~uniaxial_mask

        predicted_sigma_11, predicted_sigma_22 = [], []

        logger.info("Starting predict...")
        logger.info(f"{optimized_params=}")
        # Векторизованный проход по данным
        for lam1, lam2, is_uni in zip(lam_x, lam_y, uniaxial_mask):
            compute_fn: Callable = (
                self.stress_calculator.compute_uniaxial if is_uni else self.stress_calculator.compute_biaxial
            )
            p11, p22 = compute_fn(optimized_params, lam1, lam2)
            predicted_sigma_11.append(p11)
            predicted_sigma_22.append(p22)

        try:
            r2_val_11 = self._calculate_metrics_r2(stress_x[uniaxial_mask], np.array(predicted_sigma_11)[uniaxial_mask],
                                                   'R²(P_xx)')
            r2_val_11 = 0 if r2_val_11 is None else r2_val_11
        except ValueError:
            r2_val_11 = Metric(name='R²(P_xx)', value=None)

        try:
            r2_val_22 = self._calculate_metrics_r2(stress_y[biax_mask], np.array(predicted_sigma_22)[biax_mask],
                                                   'R²(P_yy)')
        except ValueError:
            r2_val_22 = Metric(name='R²(P_yy)', value=None)

        lines = [
            Line(name="Exp P_xx", x=lam_x.tolist(), y=stress_x.tolist()),
            Line(name="Model P_xx", x=lam_x.tolist(), y=predicted_sigma_11),
            Line(name="Exp P_yy (biax)", x=lam_y[biax_mask].tolist(), y=stress_y[biax_mask].tolist()),
            Line(name="Model P_yy (biax)", x=lam_y[biax_mask].tolist(), y=predicted_sigma_22),
        ]

        plot_data = PlotData(
            name=f"{self.hyperelastic_model.model_name} - Prediction",
            x_label='Stretch λ',
            y_label='Stress (MPa)',
            lines=lines
        )

        metrics = [r2_val_11, r2_val_22]

        return IsotropicPredictResponse(
            status='ok',
            plot_data=plot_data,
            metrics=metrics
        )

    # def _gent_constraint(self, params: np.ndarray) -> float:
    #     Jm = params[1]
    #     I1_values = [InvariantCalculator.compute(l1, l2, sy)[0] for l1, l2, sy in
    #                  zip(self.data['lambda_x'], self.data['lambda_y'], self.data['stress_y_mpa'])]
    #     constraints = [1 - (I1 - 3) / Jm for I1 in I1_values]
    #     return min(constraints) - 1e-6

    @staticmethod
    def _calculate_metrics_r2(y_true, y_pred, name: str = 'r2'):
        return Metric(name=name, value=r2_score(y_true, y_pred))

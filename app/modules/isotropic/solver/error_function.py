from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np

from app.logger import logger





@dataclass
class ErrorFunction:
    """
    Provides factory and calculation methods for different error metrics
    used in hyperelastic model fitting.
    """

    _ERROR_FUNCTIONS: Dict[str, Callable] = None

    @classmethod
    def get_error_function(cls, name_error_function: str) -> Callable:
        """
        Factory method to select the appropriate error function.

        Args:
            name_error_function: User-facing name of the error function.

        Returns:
            Callable error function.

        Raises:
            ValueError: If the error function name is not recognized.
        """
        internal_name = error_functions_name.inverse.get(name_error_function)
        if internal_name is None:
            raise ValueError(f"Unknown error function alias: {name_error_function}")

        error_functions = cls._get_function_map()

        if internal_name not in error_functions:
            raise ValueError(
                f"Unknown model: {internal_name}. Available models: {', '.join(error_functions.keys())}"
            )

        return error_functions[internal_name]

    @classmethod
    def _get_function_map(cls) -> Dict[str, Callable]:
        if cls._ERROR_FUNCTIONS is None:
            cls._ERROR_FUNCTIONS = {
                "R_abs_err_P": cls._abs_error_P,
                "R_abs_err_sigma": cls._abs_error_sigma,
                "R_rel_err": cls._rel_error
            }
        return cls._ERROR_FUNCTIONS

    @staticmethod
    def _abs_error_P(params, lambdas, experimental_data, calculate_sigma):
        total_error = 0.0
        for (lambda_1, lambda_2), (P11_exp, P22_exp) in zip(lambdas, experimental_data):
            try:
                sigma_11, sigma_22, _, _ = calculate_sigma(params, lambda_1, lambda_2, P22_exp)
                P11_model = sigma_11 / lambda_1
                P22_model = sigma_22 / lambda_2
                total_error += (P11_model - P11_exp) ** 2 + (P22_model - P22_exp) ** 2
            except Exception as e:
                logger.warning("[_abs_error_P] Ошибка: %s", e)
                total_error += np.inf
        return total_error

    @staticmethod
    def _abs_error_sigma(params, lambdas, experimental_data, calculate_sigma):
        total_error = 0.0
        for (lambda_1, lambda_2), (P11_exp, P22_exp) in zip(lambdas, experimental_data):
            try:
                sigma_11_exp = P11_exp * lambda_1
                sigma_22_exp = P22_exp * lambda_2
                sigma_11_model, sigma_22_model, _, _ = calculate_sigma(params, lambda_1, lambda_2, P22_exp)
                total_error += (sigma_11_model - sigma_11_exp) ** 2 + (sigma_22_model - sigma_22_exp) ** 2
            except Exception as e:
                logger.warning("[_abs_error_sigma] Ошибка: %s", e)
                total_error += np.inf
        return total_error

    @staticmethod
    def _rel_error(params, lambdas, experimental_data, calculate_sigma):
        total_error = 0.0
        for i, ((lambda_1, lambda_2), (P11_exp, P22_exp)) in enumerate(zip(lambdas, experimental_data)):
            try:
                sigma_11_exp = P11_exp * lambda_1
                sigma_22_exp = P22_exp * lambda_2
                sigma_11_model, sigma_22_model, _, _ = calculate_sigma(params, lambda_1, lambda_2, P22_exp)
                if abs(sigma_11_exp) < 1e-15 or abs(sigma_22_exp) < 1e-15:
                    continue
                rel_11 = ((sigma_11_model / sigma_11_exp) - 1) ** 2
                rel_22 = ((sigma_22_model / sigma_22_exp) - 1) ** 2
                total_error += rel_11 + rel_22
            except Exception as e:
                logger.warning("[_rel_error] Ошибка на итерации %d: %s", i, e)
                total_error += np.inf
        return total_error


if __name__ == "__main__":
    error_func = ErrorFunction.get_error_function("R_abs_err_P")
    logger.info("Selected error function: %s", error_func)

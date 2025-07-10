from enum import Enum
from typing import List, Tuple

import numpy as np


class AnisotropicModelType(str, Enum):
    """Enum for supported anisotropic model types"""
    GOH = "GOH"
    HOG = "HOG"


class AnisotropicConstants:
    """Constants for anisotropic modeling"""

    # Default parameter bounds
    DEFAULT_BOUNDS = {
        'mu': (0.0, 10.0),  # Material parameter mu
        'k1': (0.0, 5000.0),  # Material parameter k1
        'k2': (0.0, 10000.0),  # Material parameter k2
        'kappa': (0.0, 1 / 3),  # Dispersion parameter kappa
        'alpha': (0.0, np.pi),  # Fiber angle alpha
    }

    # Initial parameter values for optimization
    DEFAULT_INITIAL_VALUES = {
        'mu': 0.0001,
        'k1': 1.0,
        'k2': 1.0,
        'kappa': 0.1,
        'alpha': np.pi / 4,
    }

    # Convergence criteria
    CONVERGENCE_TOLERANCE = 1e-9
    MAX_ITERATIONS = 1000

    # Numerical stability
    NUMERICAL_EPSILON = 1e-12


class AnisotropicModelConfig:
    """Configuration for specific anisotropic models"""

    @classmethod
    def get_parameter_bounds(cls, alpha: bool, kappa: bool) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        bound = AnisotropicConstants.DEFAULT_BOUNDS
        bounds = [
            bound['mu'],
            bound['k1'],
            bound['k2'],
        ]

        if not alpha:
            bounds.append(bound['alpha'])
        if not kappa:
            bounds.append(bound['kappa'])

        return bounds

    @classmethod
    def get_initial_parameters(cls, alpha: bool, kappa: bool) -> List[float]:
        """Get initial parameter values for optimization"""

        values = AnisotropicConstants.DEFAULT_INITIAL_VALUES

        initial = [
            values['mu'],
            values['k1'],
            values['k2'],
        ]

        if not alpha:
            initial.append(values['alpha'])
        if not kappa:
            initial.append(values['kappa'])
        return initial


# Model type mapping for validation
MODEL_TYPE_MAPPING = {
    "GOH": AnisotropicModelType.GOH,
    "HOG": AnisotropicModelType.HOG,
}

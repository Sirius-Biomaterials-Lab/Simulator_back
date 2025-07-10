from enum import Enum

import numpy as np
import scipy as sp
from scipy.optimize import Bounds


class IsotropicModelType(str, Enum):
    """Enum for supported anisotropic model types"""
    NeoHookean = "NeoHookean"
    MooneyRivlin = "MooneyRivlin"
    GeneralizedMooneyRivlin = "GeneralizedMooneyRivlin"
    Beda = "Beda"
    Yeoh = "Yeoh"
    Gent = "Gent"
    Carroll = "Carroll"


class IsotropicConstants:
    """Constants for anisotropic modeling"""

    # Default parameter bounds
    BOUNDS = {
        IsotropicModelType.NeoHookean: Bounds([0], [np.inf]),
        IsotropicModelType.MooneyRivlin: Bounds([0, 0], [np.inf, np.inf]),
        IsotropicModelType.GeneralizedMooneyRivlin: Bounds([0] * 5, [np.inf] * 5),
        IsotropicModelType.Beda: Bounds([0, 0, 0, 0, 1, 1, 1], [np.inf] * 7),
        IsotropicModelType.Yeoh: Bounds([0, 0, 0], [np.inf] * 3),
        IsotropicModelType.Carroll: Bounds([1e-6, 1e-6], [np.inf, np.inf])
    }

    # Initial parameter values for optimization

    # Convergence criteria
    CONVERGENCE_TOLERANCE = 1e-9
    MAX_ITERATIONS = 1000
    OPTIMIZER_METHOD = 'L-BFGS-B'

    # Numerical stability
    NUMERICAL_EPSILON = 1e-12



if __name__ == "__main__":
    # print(*list(model_name_mapping.values()))
    print(IsotropicModelType.__members__.keys())

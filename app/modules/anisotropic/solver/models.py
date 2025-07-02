from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

# import numpy as np
import autograd.numpy as np

from app.modules.anisotropic.solver.config import AnisotropicModelType


# import autograd.numpy as anp


@dataclass
class ModelParameters:
    """Data class for model parameters"""
    mu: float
    k1: float
    k2: float
    alpha: Optional[float]
    kappa: Optional[float]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'mu': self.mu,
            'k1': self.k1,
            'k2': self.k2,
            'alpha': self.alpha,
            'kappa': self.kappa
        }

    @classmethod
    def from_array(cls, params: np.ndarray, alpha: Optional[float] = None,
                   kappa: Optional[float] = None) -> 'ModelParameters':
        """Create from parameter array"""

        mu = params[0]
        k1 = params[1]
        k2 = params[2]
        try:
            alpha = params[3]
            kappa = params[4]
        except IndexError:
            alpha = alpha
            kappa = kappa
        return cls(mu=mu, k1=k1, k2=k2, alpha=alpha, kappa=kappa)


class AnisotropicModel(ABC):
    """Abstract base class for anisotropic models"""

    def __init__(self, model_type: AnisotropicModelType):
        self.model_type = model_type

    @abstractmethod
    def compute_stress(self, lam1: float, lam2: float, params: ModelParameters) -> np.ndarray:
        """Compute stress tensor for given deformation and parameters"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass

    @staticmethod
    def _compute_invariants_and_matrices(lam1: float, lam2: float, alpha: float) -> tuple:
        """Compute invariants and transformation matrices (common for both models)"""
        # Compute lambda3 from incompressibility
        lam3 = 1.0 / (lam1 * lam2)

        # Deformation gradient
        F = np.array([
            [lam1, 0.0, 0.0],
            [0.0, lam2, 0.0],
            [0.0, 0.0, lam3]
        ])

        # Right Cauchy-Green tensor
        C = F.T @ F

        # Fiber direction in reference configuration
        a0 = np.array([np.cos(alpha), np.sin(alpha), 0.0])
        M = np.outer(a0, a0)  # Structural tensor

        # Invariants
        I1 = np.trace(C)
        I4 = np.tensordot(C, M)

        # Inverse matrices
        # invF = np.linalg.inv(F)
        invC = np.linalg.inv(C)

        return F, C, M, I1, I4, invC


class GOHModel(AnisotropicModel):
    """Gasser-Ogden-Holzapfel model implementation"""

    def __init__(self):
        super().__init__(AnisotropicModelType.GOH)

    def compute_stress(self, lam1: float, lam2: float, params: ModelParameters) -> np.ndarray:
        """Compute stress using GOH model"""
        F, C, M, I1, I4, invC = self._compute_invariants_and_matrices(
            lam1, lam2, params.alpha
        )

        # Identity tensor
        I = np.eye(3)

        # Pseudo-invariant
        H = params.kappa * I1 + (1 - 3 * params.kappa) * I4
        E = H - 1

        # Exponential term
        exp_term = np.exp(params.k2 * E ** 2)

        # Second Piola-Kirchhoff stress
        S2 = params.mu * I + 2 * params.k1 * exp_term * E * (params.kappa * I + (1 - 3 * params.kappa) * M)

        # Hydrostatic pressure from incompressibility
        p = S2[2, 2] / invC[2, 2]

        # Total second Piola-Kirchhoff stress
        S = -p * invC + S2

        # First Piola-Kirchhoff stress
        P = F @ S

        return P

    def get_model_name(self) -> str:
        return "Gasser-Ogden-Holzapfel"


class HOGModel(AnisotropicModel):
    """Holzapfel-Ogden-Gasser model implementation"""

    def __init__(self):
        super().__init__(AnisotropicModelType.HOG)

    def compute_stress(self, lam1: float, lam2: float, params: ModelParameters) -> np.ndarray:
        """Compute stress using HOG model"""
        F, C, M, I1, I4, invC = self._compute_invariants_and_matrices(
            lam1, lam2, params.alpha
        )

        # Identity tensor
        I = np.eye(3)

        # Pseudo-invariant
        E = (1 - params.kappa) * (I1 - 3) ** 2 + params.kappa * (I4 - 1) ** 2

        # Exponential term
        exp_term = np.exp(params.k2 * E)

        # Second Piola-Kirchhoff stress
        S2 = (params.mu * I +
              2 * params.k1 * exp_term * (
                      2 * (1 - params.kappa) * (I1 - 3) * I +
                      2 * params.kappa * (I4 - 1) * M
              ))

        # Hydrostatic pressure from incompressibility
        p = S2[2, 2] / invC[2, 2]

        # Total second Piola-Kirchhoff stress
        S = -p * invC + S2

        # First Piola-Kirchhoff stress
        P = F @ S

        return P

    def get_model_name(self) -> str:
        return "Holzapfel-Ogden-Gasser"


class ModelFactory:
    """Factory for creating anisotropic models"""

    _models = {
        AnisotropicModelType.GOH: GOHModel,
        AnisotropicModelType.HOG: HOGModel,
    }

    @classmethod
    def create_model(cls, model_type: AnisotropicModelType) -> AnisotropicModel:
        """Create model instance based on type"""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")

        return cls._models[model_type]()

    @classmethod
    def get_available_models(cls) -> List[AnisotropicModelType]:
        """Get list of available model types"""
        return list(cls._models.keys())

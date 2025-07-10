from dataclasses import dataclass
from typing import Callable, Sequence, Tuple


@dataclass
class InvariantCalculator:
    """Calculates invariants based on stretch ratios and stress."""

    @staticmethod
    def compute(lam1: float, lam2: float) -> Tuple[float, float, float]:
        lam3 = 1.0 / (lam1 * lam2)
        I1 = lam1 ** 2 + lam2 ** 2 + lam3 ** 2
        I2 = (lam1 ** 2 * lam2 ** 2 +
              lam2 ** 2 * lam3 ** 2 +
              lam3 ** 2 * lam1 ** 2)
        return I1, I2, lam3


@dataclass
class StressCalculator:
    """
    Compute Cauchy stresses for isotropic hyperelastic materials under
    uniaxial and biaxial loading based on derivatives of the strain-
    energy function with respect to the first two invariants.

    Attributes:
        dW_dI1: Callable[[float, float, *params], float]
            Derivative of strain-energy density W with respect to I₁.
        dW_dI2: Callable[[float, float, *params], float]
            Derivative of strain-energy density W with respect to I₂.
    """

    dW_dI1_func: Callable[..., float]
    dW_dI2_func: Callable[..., float]

    def compute_biaxial(
            self,
            params: Sequence[float],
            lambda1: float,
            lambda2: float,
    ) -> tuple[float, float]:
        """
        Compute P₁₁, P₂₂ and hydrostatic pressure p for
        equal-biaxial tension (σ₃₃ = 0).

        Parameters:
            params: Sequence of material parameters passed to dW_dI1, dW_dI2
            lambda1: Stretch ratio in direction 1
            lambda2: Stretch ratio in direction 2

        Returns:
            P11: First principal Cauchy stress
            P22: Second principal Cauchy stress
            p:  Hydrostatic pressure enforcing σ₃₃ = 0
        """
        I1, I2, lambda3 = InvariantCalculator.compute(lambda1, lambda2)
        psi1 = self.dW_dI1_func(I1, I2, *params)
        psi2 = self.dW_dI2_func(I1, I2, *params)

        # Coefficients from analytic expressions
        coeff1_11 = lambda1 - lambda3 ** 2 / lambda1
        coeff2_11 = lambda1 * lambda2 ** 2 + lambda3 ** 2 * (I1 - lambda1 ** 2)
        P11 = 2 * (coeff1_11 * psi1 + coeff2_11 * psi2)

        coeff1_22 = lambda2 - lambda3 ** 2 / lambda2
        coeff2_22 = lambda2 * lambda1 ** 2 + lambda3 ** 2 * (I1 - lambda2 ** 2)
        P22 = 2 * (coeff1_22 * psi1 + coeff2_22 * psi2)

        # Pressure to enforce σ₃₃ = 0
        # p = 2 * lambda3 * (lambda3 * psi1 + (lambda1 ** 2 + lambda2 ** 2) * psi2)

        return P11, P22

    def compute_uniaxial(
            self,
            params: Sequence[float],
            lambda1: float,
            lambda2: float,
    ) -> tuple[float, float]:
        """
        Compute P₁₁, lateral stress (zero), and pressure p for
        uniaxial tension under free lateral contraction.

        Parameters:
            params: Sequence of material parameters
            lambda1: Stretch ratio in the loading direction

        Returns:
            P11: Axial Cauchy stress
            0.0: Lateral Cauchy stress (by definition)
            p:  Hydrostatic pressure enforcing lateral stress = 0
        """
        I1, I2, lambda3 = InvariantCalculator.compute(lambda1, lambda2)
        psi1 = self.dW_dI1_func(I1, I2, *params)
        psi2 = self.dW_dI2_func(I1, I2, *params)

        # Compute ∂W/∂λ₁ and ∂W/∂λ₂
        dW_dlam1, dW_dlam2 = self._compute_dW_dlam(psi1, psi2, lambda1, lambda2)

        # Pressure to satisfy σ₂₂ = 0
        p = lambda2 * dW_dlam2
        P11 = dW_dlam1 - p / lambda1

        return P11, 0.0

    @staticmethod
    def _compute_dW_dlam(
            psi1: float,
            psi2: float,
            lambda1: float,
            lambda2: float,
    ) -> tuple[float, float]:
        """
        Total derivatives of W with respect to λ₁ and λ₂,
        holding incompressibility.

        Parameters:
            psi1: ∂W/∂I₁ evaluated at (I₁,I₂)
            psi2: ∂W/∂I₂ evaluated at (I₁,I₂)
            lambda1: Stretch λ₁
            lambda2: Stretch λ₂

        Returns:
            dW/dλ₁, dW/dλ₂
        """
        lambda3 = 1.0 / (lambda1 * lambda2)

        # partial derivatives of invariants wrt λ₁, λ₂
        dI1_dlam1 = 2 * lambda1 - 2 * lambda3 ** 2 / lambda1
        dI1_dlam2 = 2 * lambda2 - 2 * lambda3 ** 2 / lambda2
        dI2_dlam1 = 2 * lambda1 * lambda2 ** 2 - 2 * lambda3 ** 2 * lambda2 ** 2 / lambda1
        dI2_dlam2 = 2 * lambda2 * lambda1 ** 2 - 2 * lambda3 ** 2 * lambda1 ** 2 / lambda2

        dW_dlam1 = psi1 * dI1_dlam1 + psi2 * dI2_dlam1
        dW_dlam2 = psi1 * dI1_dlam2 + psi2 * dI2_dlam2
        return dW_dlam1, dW_dlam2

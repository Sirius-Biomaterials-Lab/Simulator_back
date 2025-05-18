import logging
from typing import Callable, Optional, List

import numpy as np
import sympy as sp
from scipy.optimize import Bounds

from app.modules.isotropic.solver import model_name_mapping
from app.modules.isotropic.solver.shema import EnergyFunction

logger = logging.getLogger(__name__)


def energy_model(func: Callable) -> Callable:
    """
    Декоратор для унификации создания энергетических функций моделей
    """

    def wrapper(self):
        logger.debug("Calculating energy function for model: %s", func.__name__)
        return func(self)

    return wrapper


class HyperelasticModel:
    """Class for various hyperelastic material modules calculations."""

    I1_sym, I2_sym = sp.symbols('I1 I2')

    def __init__(self, model_name: str) -> None:
        model_name_internal = model_name_mapping.inverse.get(model_name)
        if model_name_internal is None:
            raise ValueError(f"Unknown model name alias: {model_name}")

        model_map = {
            'neohookean': self.neohookean,
            'mooney_rivlin': self.mooney_rivlin,
            'generalized_mooney_rivlin': self.generalized_mooney_rivlin,
            'beda': self.beda,
            'yeoh': self.yeoh,
            'gent': self.gent,
            'gent_gent': self.gent_gent,
            'mod_gent_gent': self.mod_gent_gent,
            'carroll': self.carroll
        }

        if model_name_internal.lower() not in model_map:
            raise ValueError(f"Unknown model: {model_name_internal}. Available modules: {', '.join(model_map.keys())}")

        self.model_name: str = model_name_internal.lower()
        self.energy_func: Callable = model_map[self.model_name]
        self.params_sym: Optional[List[sp.Expr]] = None
        self.bounds: Optional[Bounds] = None

    def calculate(self):
        return self.energy_func()

    @energy_model
    def neohookean(self):
        self.bounds = Bounds([0], [np.inf])
        self.params_sym = sp.symbols('params0:1')
        mu, = self.params_sym
        W = mu / 2 * (self.I1_sym - 3)
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.Integer(0)
        )

    @energy_model
    def mooney_rivlin(self):
        self.bounds = Bounds([0, 0], [np.inf, np.inf])
        self.params_sym = sp.symbols('params0:2')
        c1, c2 = self.params_sym
        W = c1 * (self.I1_sym - 3) + c2 * (self.I2_sym - 3)
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.diff(W, self.I2_sym)
        )

    @energy_model
    def generalized_mooney_rivlin(self):
        self.bounds = Bounds([0] * 5, [np.inf] * 5)
        self.params_sym = sp.symbols('params0:5')
        C10, C01, C11, C20, C02 = self.params_sym
        W = (C10 * (self.I1_sym - 3) + C01 * (self.I2_sym - 3) +
             C11 * (self.I1_sym - 3) * (self.I2_sym - 3) +
             C20 * (self.I1_sym - 3) ** 2 + C02 * (self.I2_sym - 3) ** 2)
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.diff(W, self.I2_sym)
        )

    @energy_model
    def beda(self):
        self.bounds = Bounds([0, 0, 0, 0, 1, 1, 1], [np.inf] * 7)
        self.params_sym = sp.symbols('params0:7')
        c1, c2, c3, K1, alpha, ksi, beta = self.params_sym
        epsilon = 1e-8
        I1_, I2_ = self.I1_sym - 3 + epsilon, self.I2_sym - 3 + epsilon
        W = ((c1 / alpha) * I1_ ** alpha + c2 * I2_ + (c3 / ksi) * I1_ ** ksi + (K1 / beta) * I2_ ** beta)
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.diff(W, self.I2_sym)
        )

    @energy_model
    def yeoh(self):
        self.bounds = Bounds([0, 0, 0], [np.inf] * 3)
        self.params_sym = sp.symbols('params0:3')
        c1, c2, c3 = self.params_sym
        W = c1 * (self.I1_sym - 3) + c2 * (self.I1_sym - 3) ** 2 + c3 * (self.I1_sym - 3) ** 3
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.Integer(0)
        )

    @energy_model
    def gent(self):
        self.bounds = Bounds([1e-6, 1e-6], [np.inf, np.inf])
        self.params_sym = sp.symbols('params0:2')
        mu, Jm = self.params_sym
        W = -mu * Jm / 2 * sp.log(1 - (self.I1_sym - 3) / Jm)
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.Integer(0)
        )

    @energy_model
    def gent_gent(self):
        self.bounds = Bounds([0, 1e-6, 0], [np.inf] * 3)
        self.params_sym = sp.symbols('params0:3')
        mu, Jm, c2 = self.params_sym
        W = -mu * Jm / 2 * sp.log(1 - (self.I1_sym - 3) / Jm) + c2 * sp.log(self.I2_sym / 3)
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.diff(W, self.I2_sym)
        )

    @energy_model
    def mod_gent_gent(self):
        self.bounds = Bounds([0, 1e-6, 0, 0], [np.inf] * 4)
        self.params_sym = sp.symbols('params0:4')
        mu, Jm, c1, c2 = self.params_sym
        W = (-mu * Jm / 2 * sp.log(1 - (self.I1_sym - 3) / Jm) +
             c1 * (self.I1_sym - 3) ** 2 + c2 * (self.I2_sym - 3))
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.diff(W, self.I2_sym)
        )

    @energy_model
    def carroll(self):
        self.bounds = Bounds([0, 0, 0], [np.inf] * 3)
        self.params_sym = sp.symbols('params0:3')
        A, B, C = self.params_sym
        W = A * self.I1_sym + B * (self.I1_sym ** 4) + C * sp.sqrt(self.I2_sym)
        return EnergyFunction(
            W_sym=W,
            dW_dI1_sym=sp.diff(W, self.I1_sym),
            dW_dI2_sym=sp.diff(W, self.I2_sym)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model = HyperelasticModel('neohookean')
    result = model.calculate()
    print(result)

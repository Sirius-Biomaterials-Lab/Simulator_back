from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class ModuleSolver(ABC):

    @abstractmethod
    def setup_solver(self, *args, **kwargs):
        raise NotImplementedError("Please implement setup_solver")

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        raise NotImplementedError("Please implement fit method")

    @abstractmethod
    def predict(self, data: pd.DataFrame, optimized_params: Any):
        raise NotImplementedError("Please implement predict method")

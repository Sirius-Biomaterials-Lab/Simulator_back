import sympy as sp
from pydantic import BaseModel, ConfigDict


class EnergyFunction(BaseModel):

    W_sym: sp.Expr
    dW_dI1_sym: sp.Expr
    dW_dI2_sym: sp.Expr

    model_config = ConfigDict(arbitrary_types_allowed=True)

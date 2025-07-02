from typing import Literal, List, Optional

from fastapi import UploadFile, File, Form
from pydantic import BaseModel, Field

from app.modules.isotropic.solver import IsotropicModelType


class IsotropicUploadRequest:
    def __init__(
            self,
            hyperlastic_model: Literal[*list(IsotropicModelType.__members__.keys())] = Form(
                ..., description="Hyperlastic model identifier"),
            # error_function: Literal[*list(error_functions_name.values())] = Form(...,
            # description="Error function name"),
            files: List[UploadFile] = File(..., description="Model file (.csv, .xls, .xlsx)")
    ):
        self.hyperlastic_model = hyperlastic_model
        # self.error_function = error_function
        self.files = files


class IsotropicResponse(BaseModel):
    status: str = Field(default="error")
    detail: Optional[str] = None

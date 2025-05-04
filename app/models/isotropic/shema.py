from typing import Optional, Literal, Union

from fastapi import UploadFile, File, Form
from pydantic import BaseModel, Field


class IsotropicUploadRequest:
    def __init__(
            self,
            hyperlastic_model: Literal["Модель Муни-Ривлина", "Модель Нэо-Хукена", "Модель Огдена"] = Form(
                ..., description="Model file (.csv, .xls, .xlsx)"),
            error_function: Literal["Квадратичная ошибка", "Абсолютная ошибка"] = Form(...,
                                                                                       description="Hyperlastic model identifier"),
            file: UploadFile = File(..., description="Error function name")
    ):
        self.hyperlastic_model = hyperlastic_model
        self.error_function = error_function
        self.file = file


class Line(BaseModel):
    name: str = Field(default='test')
    x: list[Union[int, float]] = Field(default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y: list[Union[int, float]] = Field(default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


class PlotData(BaseModel):
    name: str = Field(default='test')
    x_label: str = Field(default='test_x_label')
    y_label: str = Field(default='test_y_label')
    lines: list[Line] = Field(default=[Line()])


class Metric(BaseModel):
    name: str = Field(default="metric1")
    value: float = Field(default=0.5)


class Parameter(BaseModel):
    name: str = Field(default="parameter1")
    value: float = Field(default=0.5)


class IsotropicResponse(BaseModel):
    status: str =  Field(default="error")
    detail: Optional[str] = None


class IsotropicFitResponse(BaseModel):
    status: str = Field(default="ok")
    metrics: list[Metric] = Field(default=[Metric()])
    parameters: list[Parameter] = Field(default=[Parameter()])
    plot_data: PlotData = Field(default=PlotData())


class IsotropicPredictResponse(BaseModel):
    status: str = Field(default="ok")
    plot_data: PlotData = Field(default=PlotData())

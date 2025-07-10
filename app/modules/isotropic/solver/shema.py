from typing import Union, Optional

from pydantic import BaseModel, Field


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
    value: Optional[float] = Field(default=0.5)


class Parameter(BaseModel):
    name: str = Field(default="parameter1")
    value: float = Field(default=0.5)


class IsotropicFitResponse(BaseModel):
    status: str = Field(default="ok")
    metrics: list[Metric] = Field(default=[Metric()])
    parameters: list[Parameter] = Field(default=[Parameter()])
    plot_data: PlotData = Field(default=PlotData())


class IsotropicPredictResponse(BaseModel):
    status: str = Field(default="ok")
    metrics: list[Metric] = Field(default=[Metric()])
    plot_data: PlotData = Field(default=PlotData())

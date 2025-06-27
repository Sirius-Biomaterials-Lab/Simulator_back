from typing import Literal, List, Optional

from fastapi import UploadFile, File, Form
from pydantic import BaseModel, Field


class AnisotropicUploadRequest:
    """Request model for uploading anisotropic data files"""

    def __init__(
            self,
            model_type: Literal["GOH", "HOG"] = Form(
                ..., description="Anisotropic model type (GOH or HOG)"
            ),
            kappa: Optional[float] = Form(
                None, description="Fixed kappa value (0 to 1/3). If None, will be optimized",
                ge=0.0, le=1 / 3
            ),
            alpha: Optional[float] = Form(
                None, description="Fixed alpha value in radians (0 to π). If None, will be optimized",
                ge=0.0, le=3.14159
            ),
            files: List[UploadFile] = File(
                ..., description="Data files (.csv format)"
            )
    ):
        self.model_type = model_type
        self.kappa = kappa
        self.alpha = alpha
        self.files = files


class AnisotropicResponse(BaseModel):
    """Standard response model for anisotropic operations"""

    status: str = Field(default="error", description="Operation status")
    detail: Optional[str] = Field(None, description="Additional details or error message")


class AnisotropicParameterValue(BaseModel):
    """Model for optimized parameter values"""

    name: str = Field(..., description="Parameter name")
    value: float = Field(..., description="Parameter value")


class AnisotropicMetric(BaseModel):
    """Model for evaluation metrics"""

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    direction: Optional[str] = Field(None, description="Direction (P11, P22)")


class AnisotropicPlotLine(BaseModel):
    """Model for plot line data"""

    name: str = Field(..., description="Line name")
    x: List[float] = Field(..., description="X coordinates")
    y: List[float] = Field(..., description="Y coordinates")
    line_type: str = Field("lines", description="Plot line type")
    color: Optional[str] = Field(None, description="Line color")


class AnisotropicPlotData(BaseModel):
    """Model for plot data"""

    title: str = Field(..., description="Plot title")
    x_label: str = Field(..., description="X-axis label")
    y_label: str = Field(..., description="Y-axis label")
    lines: List[AnisotropicPlotLine] = Field(..., description="Plot lines")


class AnisotropicFitResponse(BaseModel):
    """Response model for fit operation"""

    status: str = Field(default="ok", description="Operation status")
    model_type: str = Field(..., description="Model type used")
    parameters: List[AnisotropicParameterValue] = Field(..., description="Optimized parameters")
    metrics: List[AnisotropicMetric] = Field(..., description="Evaluation metrics")
    plot_data: AnisotropicPlotData = Field(..., description="Plot data")
    convergence_info: Optional[dict] = Field(None, description="Optimization convergence information")


class AnisotropicPredictResponse(BaseModel):
    """Response model for prediction operation"""

    status: str = Field(default="ok", description="Operation status")
    model_type: str = Field(..., description="Model type used")
    metrics: List[AnisotropicMetric] = Field(..., description="Prediction metrics")
    plot_data: AnisotropicPlotData = Field(..., description="Plot data")
    predictions: Optional[List[dict]] = Field(None, description="Prediction results")

"""
Anisotropic hyperelastic material modeling module.

This module provides implementations of anisotropic hyperelastic models
(GOH and HOG) with parameter optimization, evaluation, and web API endpoints.
"""

from .handlers import router

__all__ = ['router']

"""
Bridge module for Schrödinger Bridge methods.

This module provides implementations for various Schrödinger Bridge algorithms
including spectral image translation capabilities.
"""

from .trainer_dbdsb import IPF_DBDSB
from .trainer_spectral import SpectralDBDSBTrainer

__all__ = [
    'IPF_DBDSB',
    'SpectralDBDSBTrainer'
]
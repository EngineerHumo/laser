"""Dataset definitions for the spot grading project."""

from .spot_dataset import SpotDataset, SpotSample, create_stratified_folds

__all__ = ["SpotDataset", "SpotSample", "create_stratified_folds"]

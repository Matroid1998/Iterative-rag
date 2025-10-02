"""Shared loaders for advanced cross-system plots."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from cross_plot_utils import load_cross_data
from quality_plot_utils import load_quality_data
from hall_plot_utils import load_hallucination_data, PALETTE


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return joined dataframes from the three analysis layers."""
    run_df, coverage_df, coverage_step_df, quality_step_df, hall_df, late_hit_df = load_cross_data()
    return run_df, coverage_df, coverage_step_df, quality_step_df, hall_df, late_hit_df


__all__ = ["load_all_data", "PALETTE"]

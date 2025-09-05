from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class Analyzer(ABC):
    @abstractmethod
    def analyze_case(self, index: int, verbose: bool = False) -> dict:
        """Analyze a single case and return statistics."""

    @abstractmethod
    def run(
        self, n_processes: int = 8, progressbar: bool = True, verbose: bool = False
    ) -> pd.DataFrame:
        """Run analysis for all cases and return a DataFrame of statistics."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save analysis results to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> pd.DataFrame:
        """Load analysis results from disk."""

    @abstractmethod
    def aggregate(self, path: str | Path | None = None) -> dict:
        """Aggregate statistics across all cases."""

    @abstractmethod
    def plot(self, path: str | Path, name: str = "") -> None:
        """Generate plots from the analysis results."""

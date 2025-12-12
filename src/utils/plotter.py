import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, List


class Plotter:
    """
    Collects (name, series) pairs and plots them on one chart.

    - For each series, a moving average (window) is plotted.
    - Title: "Training Rewards"
    - X label: "Episode"
    - Y label: "Total Reward"
    """

    def __init__(self):
        self.title = "Training Rewards"
        self.xlabel = "Episodes"
        self.ylabel = "Total Reward"

        self._window = 100  # moving average window

        self._runs: List[Tuple[str, np.ndarray]] = []
        self._colors = [
            "#5bc0de",  # light blue
            "#1f77b4",  # blue
            "#f08050",  # light red
            "#d62728",  # red
        ]

    def add_training_run(self, run: Tuple[str, Iterable[float]]) -> None:
        """Add a single run as (name, series). `series` can be list or np.ndarray."""
        if not isinstance(run, tuple) or len(run) != 2:
            raise ValueError("Run must be a tuple: (name: str, series: Iterable[float])")

        name, series = run
        series_arr = np.asarray(series, dtype=float).ravel()
        if series_arr.ndim != 1:
            raise ValueError("Series must be 1-dimensional (per-episode values).")
        self._runs.append((str(name), series_arr))

    def add_many(self, runs: Iterable[Tuple[str, Iterable[float]]]) -> None:
        """Add multiple runs at once."""
        for r in runs:
            self.add_training_run(r)

    def clear(self) -> None:
        """Remove all stored runs."""
        self._runs.clear()

    def _moving_average(self, y: np.ndarray) -> np.ndarray:
        """
        Return a 'same-length' moving average using edge padding.

        If window > len(y), it falls back to window=len(y).
        """
        if self._window <= 1 or y.size == 0:
            return y.copy()

        w = min(int(self._window), int(y.size))
        if w <= 1:
            return y.copy()

        # pad so that the result length equals y length
        left = w // 2
        right = w - 1 - left
        ypad = np.pad(y, (left, right), mode="edge")
        kernel = np.ones(w, dtype=float) / w

        return np.convolve(ypad, kernel, mode="valid")

    def plot(self, save_path: str = None):
        """
        Plot all stored runs with a moving average and optional global-mean hline.

        Parameters
        ----------
        save_path : str or None
            If provided, saves to this path (e.g., 'rewards.png').

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self._runs:
            raise ValueError("No runs to plot. Add runs with add_training_run().")

        fig, ax = plt.subplots(figsize=(10, 6))

        # plot baseline at 0
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=3)

        for i, (name, y) in enumerate(self._runs):
            x = np.arange(len(y))
            y_smooth = self._moving_average(y)
            color = self._colors[i % len(self._colors)]  # cycle through colors
            ax.plot(x, y_smooth, label=f"{name}", linewidth=2, color=color)

        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(ncols=1)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

        plt.close(fig)
        return fig

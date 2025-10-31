"""Light-weight wrapper around Visdom logging."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import torch
from torchvision.utils import make_grid

try:
    from visdom import Visdom  # type: ignore
except Exception:  # pragma: no cover - Visdom may not be installed in CI
    Visdom = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class VisdomLogger:
    def __init__(
        self,
        env: str = "spot_metric_learning",
        server: str = "http://localhost",
        port: int = 8097,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and Visdom is not None
        self.windows: Dict[str, str] = {}
        self.viz: Optional[Visdom]
        if self.enabled:
            self.viz = Visdom(env=env, server=server, port=port)
            checker = getattr(self.viz, "check_connection", lambda: True)
            if not checker():
                LOGGER.warning("Visdom server not reachable at %s:%s", server, port)
                self.enabled = False
                self.viz = None
        else:
            if Visdom is None:
                LOGGER.info("Visdom is not installed; logging disabled")
            self.viz = None

    def log_scalar(self, name: str, x: float, y: float) -> None:
        if not self.enabled or self.viz is None:
            return
        if name not in self.windows:
            self.windows[name] = self.viz.line(X=[x], Y=[y], opts={"title": name, "xlabel": "Step", "ylabel": name})
        else:
            self.viz.line(X=[x], Y=[y], win=self.windows[name], update="append")

    def log_text(self, name: str, text: str) -> None:
        if not self.enabled or self.viz is None:
            return
        if name not in self.windows:
            self.windows[name] = self.viz.text(text, opts={"title": name})
        else:
            self.viz.text(text, win=self.windows[name], append=True)

    def log_images(
        self,
        name: str,
        images: torch.Tensor,
        *,
        nrow: int = 8,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        clamp: bool = True,
    ) -> None:
        """Log an image grid to Visdom."""

        if not self.enabled or self.viz is None:
            return
        if images.ndim != 4:
            raise ValueError("Expected a 4D tensor (batch, channels, height, width)")

        tensor = images.detach().cpu()
        if mean is not None and std is not None:
            mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, -1, 1, 1)
            std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, -1, 1, 1)
            tensor = tensor * std_tensor + mean_tensor

        if clamp:
            tensor = tensor.clamp(0.0, 1.0)

        grid = make_grid(tensor, nrow=nrow)

        if name not in self.windows:
            self.windows[name] = self.viz.image(grid, opts={"title": name})
        else:
            self.viz.image(grid, win=self.windows[name])


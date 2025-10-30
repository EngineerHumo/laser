"""Light-weight wrapper around Visdom logging."""

from __future__ import annotations

import logging
from typing import Dict, Optional

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

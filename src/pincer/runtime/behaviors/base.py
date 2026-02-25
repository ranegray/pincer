"""Base class for runtime behaviors."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pincer.runtime.runtime import PincerRuntime


class Behavior:
    """Base class for behaviors executed by the runtime.

    Subclasses implement ``run(**kwargs)`` and check ``self.should_stop``
    in their loops to support cancellation.
    """

    name: str = "unnamed"

    def __init__(self, runtime: PincerRuntime) -> None:
        self.runtime = runtime
        self._stop = threading.Event()

    @property
    def should_stop(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def run(self, **kwargs) -> None:
        raise NotImplementedError

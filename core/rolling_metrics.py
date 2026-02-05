from collections import deque
from typing import Deque, Dict


class RollingAccuracy:
    def __init__(self, window: int = 200, min_samples: int = 20):
        self.window = max(1, int(window))
        self.min_samples = max(1, int(min_samples))
        self._values: Deque[bool] = deque(maxlen=self.window)

    def update(self, is_correct: bool) -> float:
        self._values.append(bool(is_correct))
        return self.accuracy()

    def accuracy(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def count(self) -> int:
        return len(self._values)

    def ready(self) -> bool:
        return len(self._values) >= self.min_samples

    def snapshot(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy(),
            "count": self.count(),
            "window": self.window,
            "min_samples": self.min_samples,
        }

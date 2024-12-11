import random
from typing import Callable, Any, List, Tuple
from collections import deque
import tictactoe

class ReplayDB:
    def __init__(self, max_steps: int):
        """
        max_steps -- a positive integer for the max size of the database
        """
        self._examples = deque()
        self._max_steps = max_steps

    def add(self, s: Any, a: Any, s2: Any, r: float) -> None:
        """
        Adds (s, a, s', r) entry to this replay database.
        """
        self._examples.append((s, a, s2, r))
        if len(self._examples) > self._max_steps:
            self._examples.popleft()

    def sample(self, n: int) -> List[Tuple[Any, Any, Any, float]]:
        """
        Samples n entries from this replay database with replacement.
        """
        if len(self._examples) == 0:
            return []
        return [random.choice(self._examples) for _ in range(n)]
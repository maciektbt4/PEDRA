# -------------------- MEMORY --------------------------
# Author: rlcode/per (adapted)
# This version adds the standard Python container protocol so you can call
#     len(memory)
# or the convenience alias
#     memory.size()
# without breaking the existing SumTree implementation.
# ------------------------------------------------------

from network.SumTree import SumTree
import random

class Memory:  # stored as (s, a, r, s_) in SumTree
    """Prioritised Experience Replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    # hyper‑parameters for proportional prioritisation
    e: float = 0.01   # small positive constant to ensure non‑zero priority
    a: float = 0.6    # prioritisation exponent (0 ⇒ uniform, 1 ⇒ greedy)

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        # fallback counter for alternate SumTree implementations
        self._counter = 0

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _getPriority(self, error: float) -> float:
        """Convert TD error to a positive priority value."""
        return (error + self.e) ** self.a

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add(self, error: float, sample):
        """Add a transition with its TD‑error‑derived priority."""
        p = self._getPriority(error)
        self.tree.add(p, sample)

        # maintain fallback counter when the SumTree doesn't expose .write
        if not hasattr(self.tree, "write"):
            self._counter = (self._counter + 1) % self.capacity

    def sample(self, n: int):
        """Draw *n* samples, stratified across the priority distribution."""
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx: int, error: float):
        """Update the priority of a previously sampled transition."""
        p = self._getPriority(error)
        self.tree.update(idx, p)

    # ------------------------------------------------------------------
    # length helpers — so `len(memory)` and `memory.size()` work everywhere
    # ------------------------------------------------------------------
    @property
    def _n_entries(self) -> int:
        """Number of valid transitions currently stored."""
        # If SumTree tracks `write` ptr we can deduce length from it
        if hasattr(self.tree, "write"):
            return min(self.tree.write, self.capacity)
        # otherwise fall back to our manual counter
        return self._counter

    def __len__(self) -> int:          # pythonic way: len(memory)
        return self._n_entries

    def size(self) -> int:             # alias for existing code paths
        return len(self)

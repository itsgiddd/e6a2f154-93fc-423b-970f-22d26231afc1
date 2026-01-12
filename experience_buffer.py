import collections
import random
from typing import List, Tuple, Any

class ExperienceBuffer:
    """A lightweight FIFO buffer for reinforcement‑learning experience tuples.
    Stores (observation, action, reward, next_observation, done) entries and allows random sampling.
    """
    def __init__(self, maxlen: int = 5000):
        self.buffer = collections.deque(maxlen=maxlen)

    def add(self, observation: Any, action: Any, reward: float, next_observation: Any, done: bool):
        """Add a new experience to the buffer."""
        self.buffer.append((observation, action, reward, next_observation, done))

    def sample(self, batch_size: int) -> Tuple[List[Any], List[Any], List[float], List[Any], List[bool]]:
        """Return a random batch of experiences.
        Returns five lists: observations, actions, rewards, next_observations, dones.
        """
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, nxt, dn = map(list, zip(*batch))
        return obs, act, rew, nxt, dn

    def __len__(self) -> int:
        return len(self.buffer)

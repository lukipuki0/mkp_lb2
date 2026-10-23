"""EWMA pool selectors; uniform choice deliberately shares the solver RNG."""

from dataclasses import dataclass, field
import math
import random


@dataclass
class AdaptiveSelector:
    names: tuple[str, ...]
    rng: random.Random
    alpha: float = .3
    epsilon: float = .2
    scores: dict[str, float] = field(init=False)
    uses: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.names = tuple(self.names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('A nonempty unique solver pool is required')
        for name in ('alpha', 'epsilon'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be finite and in (0, 1]')
        self.scores = dict.fromkeys(self.names, 0.)
        self.uses = dict.fromkeys(self.names, 0)

    def probabilities(self):
        total = sum(self.scores.values())
        size = len(self.names)
        if total == 0:
            return dict.fromkeys(self.names, 1. / size)
        return {name: self.epsilon / size + (1 - self.epsilon) * self.scores[name] / total
                for name in self.names}

    def choose(self):
        probabilities = self.probabilities()
        selected = self.rng.choices(self.names, weights=list(probabilities.values()), k=1)[0]
        return selected, probabilities

    def _validate_update(self, selected, reward):
        if selected not in self.scores:
            raise ValueError(f'Unknown solver: {selected}')
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError('reward must be finite and in [0, 1]')

    def update(self, selected, reward):
        self._validate_update(selected, reward)
        self.scores[selected] = (1 - self.alpha) * self.scores[selected] + self.alpha * reward
        self.uses[selected] += 1


class UniformSelector(AdaptiveSelector):
    """Native random.choice stream, without reward-dependent probabilities."""

    def choose(self):
        return random.choice(self.names), self.probabilities()

    def probabilities(self):
        return dict.fromkeys(self.names, 1. / len(self.names))

    def update(self, selected, reward):
        self._validate_update(selected, reward)
        self.uses[selected] += 1

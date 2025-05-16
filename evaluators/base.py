# evaluators/base.py

from abc import ABC, abstractmethod


class AbstractEvaluator(ABC):
    @abstractmethod
    def evaluate(self, **kwargs):
        """
        Evaluate a prediction with task-specific parameters.
        Returns metrics such as accuracy, success, distance, etc.
        """
        pass
    
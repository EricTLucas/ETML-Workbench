from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def predict_proba(self, x):
        raise NotImplementedError('This adapter does not supply class probabilities')

    @abstractmethod
    def save(self, directory):
        pass

    @abstractmethod
    def load(self, directory):
        pass

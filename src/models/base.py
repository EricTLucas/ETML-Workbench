from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    def fit_validation(self, x, y, *, validation_data, progress=None, checkpoint=None,
                       resume=None, reset_patience=False, max_bytes=512*1024**2):
        if resume is not None:
            raise ValueError('This model does not support checkpoint resumption')
        return self.fit(x, y)

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

from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
import joblib


class BaseTransformer:
    def __init__(self, config = None):
        self.config = config
        self.transformer = self._create_transformer()

    def _create_transformer(self):
        raise NotImplementedError

    def fit(self, X):
        self.transformer.fit(X)

    def transform(self, X):
        return self.transformer.transform(X)

    def fit_transform(self, X):
        return self.transformer.fit_transform(X)

    @classmethod
    def load(cls, path: str):
        instance = cls({})
        instance.transformer = joblib.load(path)
        return instance


class StandardScaler(BaseTransformer):
    def _create_transformer(self):
        return SklearnStandardScaler()
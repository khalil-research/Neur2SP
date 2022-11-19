from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, *args):
        pass

    @abstractmethod
    def eval_learning_metrics(self):
        pass

    @abstractmethod
    def save_model(self, file_path):
        pass

    @abstractmethod
    def save_results(self, file_path):
        pass

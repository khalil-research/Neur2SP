from abc import ABC, abstractmethod


class DataManager(ABC):
    @abstractmethod
    def generate_instance(self):
        pass

    @abstractmethod
    def generate_dataset_per_scenario(self, n_procs):
        pass

    @abstractmethod
    def generate_dataset_expected(self, n_procs):
        pass

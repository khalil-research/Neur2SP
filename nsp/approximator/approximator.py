from abc import ABC, abstractmethod


class Approximator(ABC):
    @abstractmethod
    def get_master_mip(self):
        pass

    @abstractmethod
    def get_scenario_embedding(self, n_scenarios, test_set="0"):
        # TODO: Think of a more appropriate name
        pass

    @abstractmethod
    def approximate(self, n_scenarios, gap=0.02, time_limit=600, threads=1, log_dir=None, test_set="0"):
        pass

    @abstractmethod
    def get_first_stage_solution(self, model):
        pass

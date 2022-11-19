from abc import ABC, abstractmethod


class TwoStageStocProg(ABC):
    """
    Class for Two staged Stochastic Integer Programming Problem
    """

    def _make_surrogate_scenario_model(self, scenario_id=None):
        pass

    @abstractmethod
    def _make_extensive_model(self):
        """ Creates a two-stage extensive form. """
        pass

    @abstractmethod
    def _make_second_stage_model(self, *args):
        """ Creates a second stage problem for a given scenario. """
        pass

    @abstractmethod
    def solve_extensive(self, n_scenarios, gap=0.02, time_limit=600, threads=1, verbose=1,
                        log_dir=None, node_file_start=None, node_file_dir=None,
                        test_set="0"):
        """ Solves the two-stage extensive form. """
        pass

    def solve_surrogate_scenario_model(self, *args):
        pass

    @abstractmethod
    def get_second_stage_objective(self, sol, xi, gap=0.0001, time_limit=1e7,
                                   threads=1, verbose=0):
        """ Solves the second stage problem for a given scenario. """
        pass

    @abstractmethod
    def evaluate_first_stage_sol(self, sol, n_scenarios, gap=0.0001, time_limit=600, threads=1, verbose=0,
                                 test_set="0", n_procs=1):
        """ Evaluates the first stage solution across all scenarios. """
        pass

    @abstractmethod
    def get_first_stage_extensive_solution(self, model):
        pass

    @abstractmethod
    def get_scenarios(self, n_scenarios, test_set):
        pass
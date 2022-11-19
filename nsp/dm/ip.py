import multiprocessing as mp
import pickle as pkl
import time

import numpy as np
import numpy.random as rng

from nsp.two_sp.ip import InvestmentProblem
from nsp.utils.ip import Sampler
from nsp.utils.ip import get_path
from .dm import DataManager


# logger = logging.getLogger(__file__)

class InvestmentProblemDataManager(DataManager):
    """Data manager for the problems used in

    Carøe, C. C. (1999). Decomposition in stochastic integer programming.
    Institute of Mathematical Sciences, Department of Operations Research,
    University of Copenhagen.

    Schultz, R. (1995). On structure and stability in stochastic
    programs with random technology matrix and complete integer recourse.
    Mathematical Programming, 70(1), 73-89.
    """

    def __init__(self, problem_config):
        self.cfg = problem_config
        self.rng = np.random.RandomState()

        self.instance_path = get_path(self.cfg.data_path, self.cfg, "inst")
        self.ml_data_p_path = get_path(self.cfg.data_path, self.cfg, "ml_data_p")
        self.ml_data_e_path = get_path(self.cfg.data_path, self.cfg, "ml_data_e")

        self.instances = None
        self.second_stage_optimal_solutions = []

    def generate_instance(self):
        print("Generating instance...")
        rng.seed(self.cfg.seed)

        self.instances = {}
        self._get_static_data()
        for i in range(self.cfg.n_instances):
            inst = {}
            self._get_deterministic_data(inst)
            self.instances[i] = inst

        pkl.dump(self.instances, open(self.instance_path, 'wb'))

    def generate_dataset_per_scenario(self, n_procs):
        """ Generate scenario wise optimal solutions for each problem and store in a list. """
        print("Generating NN-P dataset for machine learning...")
        print(f" PROBLEM: Investment Problem")
        print(f" Num processes: {n_procs}")
        if self.instances is None:
            self._load_instances()

        n_scenarios_to_sample = self.cfg.n_samples_p // self.cfg.n_samples_per_scenario
        sampler = Sampler()
        scenarios = sampler.get_scenarios(n_scenarios_to_sample)

        self.instances[0].update(self.instances[-1])
        instance = self.instances[0]
        inv_prob = InvestmentProblem(instance, scenarios)
        pool = mp.Pool() if n_procs == -1 else mp.Pool(n_procs)

        total_time = time.time()
        results = []
        for scenario_id, scenario in enumerate(scenarios):
            for i in range(self.cfg.n_samples_per_scenario):
                sol = self._get_random_sol()
                results.append(pool.apply_async(self._worker_generate_per_scenario_dataset,
                                                (inv_prob, scenario_id, sol,)))
        results = [r.get() for r in results]
        total_time = time.time() - total_time

        data = []
        for r in results:
            r.update({"features": list(r["sol"]) + list(scenarios[r["scenario_id"]])})
            data.append(r)

        tr_data, val_data = self._get_data_split(data)
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": data,
            "total_time": total_time
        }

        print("Time for data generation:", total_time)
        pkl.dump(ml_data, open(self.ml_data_p_path, 'wb'))

    def generate_dataset_expected(self, n_procs):
        """Generate dataset for training ML models. """
        print("Generating NN-E dataset for machine learning...")
        print(f" PROBLEM: Investment Problem")
        print(f" Num processes: {n_procs}")
        if self.instances is None:
            self._load_instances()

        n_samples_generated = 0
        sampler = Sampler()
        scenarios = sampler.get_support()

        self.instances[0].update(self.instances[-1])
        instance = self.instances[0]
        inv_prob = InvestmentProblem(instance, scenarios)

        total_time = time.time()
        # Generate random data using random first-stage solution and scenario subsets
        fss_scenario_subset_pairs = []
        n_second_stage_problems = 0
        while n_samples_generated < self.cfg.n_samples_e:
            # Generate random first-stage solution
            x_subopt = self._get_random_sol()

            # Sample a random subset of scenarios
            _n_scenarios = self.rng.randint(1, self.cfg.n_max_scenarios_in_tr)
            scenario_perm = self.rng.permutation(scenarios.shape[0])
            scenario_idxs = scenario_perm[:_n_scenarios]
            fss_scenario_subset_pairs.append((x_subopt, scenario_idxs))

            n_samples_generated += 1
            n_second_stage_problems += _n_scenarios

        # Solve for a (first-stage, scenario-subset) combination
        results = []
        pool = mp.Pool() if n_procs == -1 else mp.Pool(n_procs)
        manager = mp.Manager()
        mp_count = manager.Value('i', 0)

        for fss_id, (fss, scenario_subset) in enumerate(fss_scenario_subset_pairs):
            for scenario_id in scenario_subset:
                results.append(pool.apply_async(self._worker_get_second_stage_objective,
                                                (inv_prob, fss, fss_id, scenario_id,
                                                 n_second_stage_problems, mp_count,)))

        results = [r.get() for r in results]
        results_dict = {f'{r["fss_id"]}_{r["scenario_id"]}': r for r in results}

        data = []
        self._extract_data_expected(fss_scenario_subset_pairs,
                                      results_dict,
                                      scenarios,
                                      data)

        total_time = time.time() - total_time

        mp_time = np.sum([d["time"] for d in data])

        tr_data, val_data = self._get_data_split(data)
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": data,
            "total_time": total_time
        }

        pkl.dump(ml_data, open(self.ml_data_e_path, 'wb'))

    @staticmethod
    def _extract_data_expected(fss_scenario_subset_pairs,
                                 results_dict,
                                 scenarios,
                                 data,
                                 fs_results_dict=None):
        for fss_id, (fss, scenario_subset) in enumerate(fss_scenario_subset_pairs):
            objs_lst, times_lst = [], []
            if fs_results_dict is not None:
                times_lst.append(fs_results_dict[fss_id]['time'])

            for scenario_id in scenario_subset:
                objs_lst.append(results_dict[f"{fss_id}_{scenario_id}"]["obj"])
                times_lst.append(results_dict[f"{fss_id}_{scenario_id}"]["time"])

            data.append({
                "x": fss,
                "scenario": list(scenarios[scenario_subset]),
                "n_scenarios": len(scenario_subset),
                "obj_mean": np.mean(objs_lst),
                "obj_vals": objs_lst,
                "scenario_ids": scenario_subset,
                "time": np.sum(times_lst),
                "times": times_lst
            })

    def _worker_generate_per_scenario_dataset(self, inv_prob, scenario_id, fs_sol):
        _time = time.time()
        if fs_sol is None:
            info = inv_prob.solve_surrogate_scenario_model(scenario_id,
                                                           gap=self.cfg.mip_gap,
                                                           time_limit=self.cfg.time_limit,
                                                           verbose=self.cfg.verbose)
            fs_sol = list(info['sol'])
            # Subtract first stage cost to get the second stage cost
            qval = info['obj_val'] - np.dot(inv_prob.c_fs, info['sol'])
        else:
            qval = inv_prob.get_second_stage_objective(fs_sol,
                                                       scenario_id,
                                                       gap=self.cfg.mip_gap,
                                                       time_limit=self.cfg.time_limit,
                                                       verbose=self.cfg.verbose)

        _time = time.time() - _time

        return {"sol": fs_sol, "obj": qval, "time": _time, "scenario_id": scenario_id}

    def _worker_get_first_stage_sol(self, inv_prob, scenario_id):
        _time = time.time()
        info = inv_prob.solve_surrogate_scenario_model(scenario_id,
                                                       gap=self.cfg.mip_gap,
                                                       time_limit=self.cfg.time_limit,
                                                       verbose=self.cfg.verbose)
        _time = time.time() - _time

        return {"sol": list(info['sol']),
                "scenario_id": scenario_id,
                "time": _time}

    def _worker_get_second_stage_objective(self, inv_prob, fss, fss_id, scenario_id,
                                           n_second_stage_problems, mp_count):
        _time = time.time()
        obj = inv_prob.get_second_stage_objective(fss, scenario_id,
                                                  gap=self.cfg.mip_gap,
                                                  time_limit=self.cfg.time_limit,
                                                  verbose=self.cfg.verbose)
        _time = time.time() - _time

        mp_count.value += 1
        count = mp_count.value

        if count % 1000 == 0:
            print(f'Solving LP {count}/{n_second_stage_problems}, '
                  f'{(count / n_second_stage_problems) * 100:.2f}%')

        return {"fss_id": fss_id,
                "scenario_id": scenario_id,
                "obj": obj,
                "time": _time}

    def _get_static_data(self):
        instances = self.instances
        instances[-1] = {
            'c_fs': np.asarray([-1.5, -4]),
            'c_ss': np.asarray([-16, -19, -23, -28]),
            'W': np.asarray([[2, 3, 4, 5], [6, 1, 3, 2]]),
            'T_v1': np.asarray([[1, 0], [0, 1]]),
            'T_v2': np.asarray([[2 / 3, 1 / 3], [1 / 3, 2 / 3]]),
            'time_limit': self.cfg.time_limit,
            'mip_gap': self.cfg.mip_gap,
            'verbose': self.cfg.verbose,
            'seed': self.cfg.seed,
            'first_stage_vtype': self.cfg.first_stage_vtype,
            'second_stage_vtype': self.cfg.second_stage_vtype
        }

    def _get_deterministic_data(self, inst):
        if self.cfg.technology_identity == 1:
            inst['T'] = self.instances[-1]['T_v1']
        else:
            inst['T'] = self.instances[-1]['T_v2']

    def _load_instances(self):
        """ Loads instances files. """
        self.instances = pkl.load(open(self.instance_path, 'rb'))

    def _perturb_sol(self, x=None, perturb_threshold=None, instance=None):
        if instance['first_stage_vtype'] == "I":
            perturb_probs = self.rng.rand(x.shape[0])
            return [self.rng.randint(0, 6) if pp <= perturb_threshold else elem
                    for elem, pp in zip(x, perturb_probs)]

        elif instance['first_stage_vtype'] == "C":
            return (5 * self.rng.rand(2)).tolist()

    def _get_random_sol(self):
        return (5 * self.rng.rand(2)).tolist()

    def _get_data_split(self, data):
        """ Gets train/validation splits for the data. """
        perm = self.rng.permutation(len(data))

        split_idx = int(self.cfg.tr_split * (len(data)))
        tr_idx = perm[:split_idx].tolist()
        val_idx = perm[split_idx:].tolist()

        tr_data = [data[i] for i in tr_idx]
        val_data = [data[i] for i in val_idx]

        return tr_data, val_data

import pickle as pkl
import time
from multiprocessing import Manager, Pool

import numpy as np

from nsp.two_sp.cflp import FacilityLocationProblem
from nsp.utils.cflp import get_path
from .dm import DataManager


def hash_expected(sol, scenario, scenario_subset):
    hashed_scenario_subset = list(map(lambda x: tuple(x), scenario_subset))
    return frozenset(sol.items()), tuple(hashed_scenario_subset), tuple(scenario)


class FacilityLocationDataManager(DataManager):
    def __init__(self, problem_config):

        self.cfg = problem_config

        self.rng = np.random.RandomState()
        self.rng.seed(self.cfg.seed)

        self.instance_path = get_path(self.cfg.data_path, self.cfg, "inst")
        self.ml_data_p_path = get_path(self.cfg.data_path, self.cfg, "ml_data_p")
        self.ml_data_e_path = get_path(self.cfg.data_path, self.cfg, "ml_data_e")

    def generate_instance(self):
        """
        Generate a Capacited Facility Location problem following
            Cornuejols G, Sridharan R, Thizy J-M (1991)
            A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
            European Journal of Operations Research 50:280-297.
        Outputs as a gurobi model.
        """
        print("Generating instance...")

        self.instance = {}
        self._get_problem_data(self.cfg, self.instance)
        self._generate_first_stage_data(self.cfg, self.instance, self.rng)

        pkl.dump(self.instance, open(self.instance_path, 'wb'))

    def generate_dataset_per_scenario(self, n_procs):
        """ Generate dataset for training ML models. """
        self._load_instances()

        print("Generating NN-P dataset for machine learning... ")
        print(f"  PROBLEM: cflp_{self.instance['n_facilities']}_{self.instance['n_customers']}")

        data = []
        total_time = time.time()
        probs = np.linspace(0.1, 0.9, 9)

        two_sp = FacilityLocationProblem(self.instance)

        # sample a set of scenarios
        n_scenarios_to_sample = self.instance['n_samples_p'] // self.instance['n_samples_per_scenario']
        scenarios = self._sample_n_scenarios(n_scenarios_to_sample, self.instance, self.rng)

        with Manager() as manager:

            mp_data_list = manager.list()

            # Get costs for suboptimal solutions for each scenario
            procs_to_run = []

            for scenario in scenarios:
                for j in range(self.instance['n_samples_per_scenario']):
                    p = self.rng.choice(probs)  # prob. of zero
                    x_subopt = self._get_pure_random_x(prob=p, size=self.instance['n_facilities'])
                    procs_to_run.append((np.array(scenario), x_subopt))

            if n_procs == -1:
                pool = Pool()
            else:
                pool = Pool(n_procs)
            for scenario, x_subopt in procs_to_run:
                pool.apply_async(self.solve_second_stage_subopt_mp,
                                 args=(
                                     scenario,
                                     x_subopt,
                                     two_sp,
                                     self.instance,
                                     mp_data_list))

            pool.close()
            pool.join()

            data = list(mp_data_list)

        total_time = time.time() - total_time

        # get train/validation split, then store
        tr_data, val_data = self._get_data_split(data, self.instance)
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": data,
            "total_time": total_time
        }

        print("Total Time:         ", total_time)
        print("Dataset size:       ", len(data))
        print("Train dataset size: ", len(tr_data))
        print("Valid dataset size: ", len(val_data))

        pkl.dump(ml_data, open(self.ml_data_p_path, 'wb'))

    def solve_second_stage_subopt_mp(self, scenario, x_subopt, two_sp, instance, mp_data_list):
        """ Obtains the cost of the suboptimal first stage solution.  """
        try:
            time_ = time.time()

            x_subopt_obj = two_sp.get_second_stage_objective(
                x_subopt,
                scenario,
                gap=instance['mip_gap'],
                time_limit=instance['time_limit'],
                verbose=instance['verbose'])

            x_subopt_features = self._get_feature_vector(x_subopt, scenario)

            time_ = time.time() - time_

            mp_data_list.append({
                "demands": scenario,
                "x": x_subopt,
                "obj": x_subopt_obj,
                "features": x_subopt_features,
                "time": time_})

        except Exception as e:
            print(f"Failed to get second stage objective for suboptimal x in time limit ({instance['time_limit']}s)")
            print(f"  x: {x_subopt_obj}")
            print(f"  demands: {scenario}")
            print(f"  exception: {e}")

        return

    def generate_dataset_expected(self, n_procs):
        """ Generate dataset for training ML models. """
        self._load_instances()

        print("Generating NN-E dataset for machine learning... ")
        print(f"  PROBLEM: cflp_{self.instance['n_facilities']}_{self.instance['n_customers']}")

        data = []
        total_time = time.time()
        probs = np.linspace(0.0, 0.9, 10)

        two_sp = FacilityLocationProblem(self.instance)

        with Manager() as manager:

            # Get costs from suboptimal first stage solutions. 
            print("  Getting objective for first varying first stage. ")

            # get set of all processes to run
            procs_to_run = []
            n_procs_to_run = 0
            n_samples = self.instance['n_samples_e']

            for _ in range(n_samples):
                # choose a random first stage optimal solution and perturb it
                p = self.rng.choice(probs)  # prob. of swapping bits in sol
                first_stage_sol = self._get_pure_random_x(prob=p, size=self.instance['n_facilities'])

                # choose a random subset of demands
                n_second_stage = self.rng.randint(1, self.instance['n_max_scenarios_in_tr'])
                scenario_subset = self._sample_n_scenarios(n_second_stage, self.instance, self.rng)

                # update
                procs_to_run.append((first_stage_sol, scenario_subset))
                n_procs_to_run += n_second_stage

            # initialize data structures for storing solutions
            mp_cost_dict = manager.dict()
            mp_time_dict = manager.dict()
            mp_count = manager.Value('i', 0)

            if n_procs == -1:
                pool = Pool()
            else:
                pool = Pool(n_procs)

            for first_stage_sol, scenario_subset in procs_to_run:
                for scenario in scenario_subset:
                    pool.apply_async(self.solve_subset_second_stage_cost_mp_expected,
                                     args=(
                                         first_stage_sol,
                                         scenario,
                                         scenario_subset,
                                         two_sp,
                                         self.instance,
                                         mp_cost_dict,
                                         mp_time_dict,
                                         mp_count,
                                         n_procs_to_run))

            pool.close()
            pool.join()

            print("Storing reuslts... ", end="")

            for first_stage_sol, scenario_subset in procs_to_run:
                objs, scens, times = [], [], []
                for scenario in scenario_subset:
                    # add items to subset if and only if no errors occured
                    scenario_hash = hash_expected(first_stage_sol, scenario, scenario_subset)
                    if scenario_hash in mp_cost_dict:
                        objs.append(mp_cost_dict[scenario_hash])
                        times.append(mp_time_dict[scenario_hash])
                        scens.append(scenario)

                data.append({
                    "x": first_stage_sol,
                    "obj_vals": objs,
                    "obj_mean": np.mean(objs),
                    "demands": scens,
                    "time": np.sum(times),
                    "times": times})

            print("Done")

        total_time = time.time() - total_time

        mp_time = list(map(lambda x: x['time'], data))
        mp_time = np.sum(mp_time)

        # get train/validation split, then store
        tr_data, val_data = self._get_data_split(data, self.instance)
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": data,
            "total_time": total_time,
            "mp_time": mp_time,
        }

        print("Time (No MP):       ", mp_time)
        print("Total Time:         ", total_time)
        print("Dataset size:       ", len(data))
        print("Train dataset size: ", len(tr_data))
        print("Valid dataset size: ", len(val_data))

        pkl.dump(ml_data, open(self.ml_data_e_path, 'wb'))

    def solve_subset_second_stage_cost_mp_expected(self, first_stage_sol, scenario, scenario_subset, two_sp, instance,
                                                     mp_cost_dict, mp_time_dict, mp_count, n_procs_to_run):
        """ Obtains the cost of the suboptimal first stage solution.  """
        try:

            time_ = time.time()

            second_stage_obj = two_sp.get_second_stage_objective(
                first_stage_sol,
                scenario,
                gap=instance['mip_gap'],
                time_limit=instance['time_limit'],
                verbose=instance['verbose'])

            time_ = time.time() - time_

            mp_count.value += 1
            count = mp_count.value

            if count % 1000 == 0:
                print(f'Solving LP {count} / {n_procs_to_run}')

            mp_cost_dict[hash_expected(first_stage_sol, scenario, scenario_subset)] = second_stage_obj
            mp_time_dict[hash_expected(first_stage_sol, scenario, scenario_subset)] = time_

        except Exception as e:
            print(f"Failed to get second stage objective for suboptimal x in time limit ({instance['time_limit']}s)")
            print(f"  x: {first_stage_sol}")
            print(f"  demands: {scenario}")
            print(f"  exception: {e}")

        return

    def _get_data_split(self, data, instance):
        """ Gets train/validation splits for the data.  """
        perm = self.rng.permutation(len(data))

        split_idx = int(instance['tr_split'] * (len(data)))
        tr_idx = perm[:split_idx].tolist()
        val_idx = perm[split_idx:].tolist()

        tr_data = [data[i] for i in tr_idx]
        val_data = [data[i] for i in val_idx]

        return tr_data, val_data

    def _load_instances(self):
        """ Loads instances files. """
        self.instance = pkl.load(open(self.instance_path, 'rb'))

    def _sample_second_stage(self, instance):
        """ Samples second stage demands for a single problem. """
        return self.rng.randint(5, 35 + 1, size=instance['n_customers'])

    def _sol_dict_to_vect(self, x_sol):
        """ Converts a solution dictionary to a vector. """
        x_vect = np.zeros(len(x_sol))
        for k, v in x_sol.items():
            index = int(k.split('_')[1])
            x_vect[index] = v
        return x_vect

    def _sol_vect_to_dict(self, x_vect):
        """ Converts a solution vector to a vector. """
        x_sol = {}
        for index in range(x_vect.size):
            x_sol[f"x_{index}"] = x_vect[index]
        return x_sol

    def _get_feature_vector(self, x_sol, demand):
        """ Gets the simple feature vector (x, deamnds). """
        x_vect = self._sol_dict_to_vect(x_sol)
        features = x_vect.tolist() + demand.tolist()
        return features

    def _get_random_subopt_x(self, x_sol, p):
        """ Modeify bits in a solution x with probability p.  """
        x_vect = self._sol_dict_to_vect(x_sol)
        probs = self.rng.rand(x_vect.shape[0])
        swap_bits = probs <= p
        x_sub = np.abs(x_vect - swap_bits)
        x_sub_dict = self._sol_vect_to_dict(x_sub)
        return x_sub_dict

    def _get_pure_random_x(self, prob, size):
        """ Modeify bits in a solution x with probability p.  """
        x_sub = self.rng.choice([0.0, 1.0], p=[prob, 1 - prob], size=size)
        x_sub_dict = self._sol_vect_to_dict(x_sub)
        return x_sub_dict

    @staticmethod
    def _get_problem_data(cfg, inst):
        """ Stores generic problem information. """
        inst['n_customers'] = cfg.n_customers
        inst['n_facilities'] = cfg.n_facilities
        inst['integer_second_stage'] = cfg.flag_integer_second_stage
        inst['bound_tightening_constrs'] = cfg.flag_bound_tightening
        inst['n_samples_p'] = cfg.n_samples_p
        inst['n_samples_per_scenario'] = cfg.n_samples_per_scenario
        inst['n_samples_e'] = cfg.n_samples_e
        inst['n_max_scenarios_in_tr'] = cfg.n_max_scenarios_in_tr
        inst['tr_split'] = cfg.tr_split
        inst['time_limit'] = cfg.time_limit
        inst['mip_gap'] = cfg.mip_gap
        inst['verbose'] = cfg.verbose

    @staticmethod
    def _generate_first_stage_data(cfg, inst, rng):
        """ Computes and stores information for first stage problem. """
        inst['c_x'] = rng.rand(cfg.n_customers)
        inst['c_y'] = rng.rand(cfg.n_customers)

        inst['f_x'] = rng.rand(cfg.n_facilities)
        inst['f_y'] = rng.rand(cfg.n_facilities)

        inst['demands'] = rng.randint(5, 35 + 1, size=cfg.n_customers)
        inst['capacities'] = rng.randint(10, 160 + 1, size=cfg.n_facilities)
        inst['fixed_costs'] = rng.randint(100, 110 + 1, size=cfg.n_facilities) * np.sqrt(inst['capacities']) \
                              + rng.randint(90 + 1, size=cfg.n_facilities)
        inst['fixed_costs'] = inst['fixed_costs'].astype(int)

        inst['total_demand'] = inst['demands'].sum()
        inst['total_capacity'] = inst['capacities'].sum()

        # adjust capacities according to ratio
        inst['capacities'] = inst['capacities'] * cfg.ratio * inst['total_demand'] / inst['total_capacity']
        inst['capacities'] = inst['capacities'].astype(int)
        inst['total_capacity'] = inst['capacities'].sum()

        # transportation costs
        inst['trans_costs'] = np.sqrt(
            (inst['c_x'].reshape((-1, 1)) - inst['f_x'].reshape((1, -1))) ** 2
            + (inst['c_y'].reshape((-1, 1)) - inst['f_y'].reshape((1, -1))) ** 2) \
                              * 10 * inst['demands'].reshape((-1, 1))
        inst['trans_costs'] = inst['trans_costs'].transpose()

    @staticmethod
    def _sample_n_scenarios(n, inst, rng):
        scenarios = []
        for _ in range(n):
            scenarios.append(rng.randint(5, 35 + 1, size=inst['n_customers']))
        return scenarios

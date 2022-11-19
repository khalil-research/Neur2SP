# import itertools
import multiprocessing as mp
import pickle as pkl
import time

import gurobipy as gp
import numpy as np
import numpy.random as rng

from nsp.two_sp.pp import PoolingProblem
from nsp.utils.pp import Sampler
from nsp.utils.pp import get_path
from .dm import DataManager


class PoolingProblemDataManager(DataManager):
    def __init__(self, problem_config):
        self.cfg = problem_config
        self.rng = np.random.RandomState()

        self.instance_path = get_path(self.cfg.data_path, self.cfg, "inst")
        self.ml_data_p_path = get_path(self.cfg.data_path, self.cfg, "ml_data_p")
        self.ml_data_e_path = get_path(self.cfg.data_path, self.cfg, "ml_data_e")

        self.instances = None

    def generate_instance(self):
        print("Generating instance...")
        rng.seed(self.cfg.seed)

        self.instances = {}
        for i in range(self.cfg.n_instances):
            inst = {}
            self._get_deterministic_data(inst)
            self.instances[i] = inst

        pkl.dump(self.instances, open(self.instance_path, 'wb'))

    def generate_dataset_per_scenario(self, n_procs):
        """ Generate scenario wise optimal solutions for each problem
        and store in a list. """
        print("Generating NN-P dataset for machine learning...")
        print(f" PROBLEM: Pooling Problem")
        print(f" Num processes: {n_procs}")
        if self.instances is None:
            self._load_instances()

        n_scenarios_to_sample = self.cfg.n_samples_p // self.cfg.n_samples_per_scenario
        sampler = Sampler(self.cfg)
        scenarios = sampler.get_scenarios(n_scenarios_to_sample)

        instance = self.instances[0]
        pool_prob = PoolingProblem(instance, scenarios)
        pool = mp.Pool() if n_procs == -1 else mp.Pool(n_procs)

        total_time = time.time()
        results = []
        probs = scenarios['probs']
        for scen_id, prob in enumerate(probs):
            for _ in range(self.cfg.n_samples_per_scenario):
                sol = self._get_random_sol()
                results.append(pool.apply_async(self._worker_generate_dataset,
                                                (pool_prob, scen_id, sol,)))
        results = [r.get() for r in results]

        total_time = time.time() - total_time

        tr_data, val_data = self._get_data_split(results)
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": results,
            "total_time": total_time
        }

        print("Time for data generation:", total_time)
        pkl.dump(ml_data, open(self.ml_data_p_path, 'wb'))

    def generate_dataset_expected(self, n_procs):
        """Generate dataset for training ML models. """
        print("Generating NN-E dataset for machine learning...")
        print(f" PROBLEM: Pooling Problem")
        print(f" Num processes: {n_procs}")
        if self.instances is None:
            self._load_instances()

        n_samples_generated = 0
        sampler = Sampler(self.cfg)
        scenarios = sampler.get_support()
        mp_manager = mp.Manager()
        # shared_scenarios = mp_manager.dict(scenarios)

        instance = self.instances[0]
        pool_prob = PoolingProblem(instance, scenarios)
        pool_prob.scenarios = mp_manager.dict(scenarios['scenarios'])
        pool_prob.demand = mp_manager.dict(scenarios['scenarios']['demand'])
        pool_prob.sulfur = mp_manager.dict(scenarios['scenarios']['sulfur'])
        pool_prob.probs = mp_manager.list(scenarios['probs'])

        total_time = time.time()
        # Generate random data using random first-stage solution and scenario subsets
        fss_scenario_subset_pairs = []
        n_second_stage_problems = 0
        while n_samples_generated < self.cfg.n_samples_e:
            # Generate random first-stage solution
            x_subopt = self._get_random_sol()

            # Sample a random subset of scenarios
            _n_scenarios = self.rng.randint(1, self.cfg.n_max_scenarios_in_tr)
            scenario_idxs = sampler.get_scenario_idxs(_n_scenarios)
            fss_scenario_subset_pairs.append((x_subopt, scenario_idxs))

            n_samples_generated += 1
            n_second_stage_problems += _n_scenarios

        print("Created fs-scenario pairs", time.time() - total_time)

        # Solve for a (first-stage, scenario-subset) combination
        results = []
        pool = mp.Pool() if n_procs == -1 else mp.Pool(n_procs)
        manager = mp.Manager()
        mp_count = manager.Value('i', 0)

        for fss_id, (fss, scenario_subset) in enumerate(fss_scenario_subset_pairs):
            for scenario_id in scenario_subset:
                results.append(pool.apply_async(self._worker_get_second_stage_objective,
                                                (pool_prob, fss, fss_id, scenario_id,
                                                 n_second_stage_problems, mp_count,)))
        results = [r.get() for r in results]
        results_dict = {f'{r["fss_id"]}_{r["scenario_id"]}': r for r in results}

        data = []
        self._extract_data_expected(fss_scenario_subset_pairs,
                                      results_dict,
                                      scenarios['scenarios'],
                                      data,
                                      probs=scenarios['probs'])

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

    def _worker_generate_dataset(self, pool_prob, scenario_id, fs_sol):
        _time = time.time()

        if fs_sol is None:
            info = pool_prob.solve_surrogate_scenario_model(scenario_id,
                                                            gap=self.cfg.mip_gap,
                                                            time_limit=self.cfg.time_limit,
                                                            verbose=self.cfg.verbose)

            # First-stage profits
            fs_cost = sum([pool_prob.fc_sources[i] * info['z_s'][i]
                           for i in pool_prob.sources])
            fs_cost += sum([pool_prob.fc_pools[j] * info['z_p'][j]
                            for j in pool_prob.pools])
            fs_cost += sum([pool_prob.fc_terminals[k] * info['z_t'][k]
                            for k in pool_prob.terminals])
            fs_cost += sum([pool_prob.fc_connections[conn] * info['z_e'][conn]
                            for conn in pool_prob.connections])
            qval = info['obj_val'] - fs_cost
            fs_sol = {
                "z_s": info["z_s"],
                "z_p": info["z_p"],
                "z_t": info["z_t"],
                "z_e": info["z_e"]
            }
        else:
            qval = pool_prob.get_second_stage_objective(fs_sol, scenario_id,
                                                        gap=self.cfg.mip_gap,
                                                        time_limit=self.cfg.time_limit,
                                                        verbose=self.cfg.verbose)

        features = self._get_feature_vector(pool_prob, scenario_id, fs_sol)
        _time = time.time() - _time

        return {"sol": fs_sol, "obj": qval, "time": _time, "scenario_id": scenario_id,
                "features": features}

    def _worker_get_first_stage_sol(self, pool_prob, scenario_id):
        _time = time.time()
        info = pool_prob.solve_surrogate_scenario_model(scenario_id=scenario_id,
                                                        gap=self.cfg.mip_gap,
                                                        time_limit=self.cfg.time_limit,
                                                        verbose=self.cfg.verbose)
        _time = time.time() - _time
        fs_sol = {
            "z_s": info["z_s"],
            "z_p": info["z_p"],
            "z_t": info["z_t"],
            "z_e": info["z_e"]
        }
        return {"sol": fs_sol, "time": _time, "scenario_id": scenario_id}

    def _worker_get_second_stage_objective(self, pool_prob, fss, fss_id, scenario_id,
                                           n_second_stage_problems, mp_count):
        _time = time.time()
        obj = pool_prob.get_second_stage_objective(fss, scenario_id,
                                                   gap=self.cfg.mip_gap,
                                                   time_limit=self.cfg.time_limit,
                                                   verbose=self.cfg.verbose)
        _time = time.time() - _time

        mp_count.value += 1
        count = mp_count.value

        # if count % 1000 == 0:
        print(f'Solving LP {count}/{n_second_stage_problems}, '
              f'{(count / n_second_stage_problems) * 100:.2f}%')

        return {"fss_id": fss_id,
                "scenario_id": scenario_id,
                "obj": obj,
                "time": _time}

    @staticmethod
    def _extract_data_expected(fss_scenario_subset_pairs,
                                 results_dict,
                                 scenarios,
                                 data,
                                 fs_results_dict=None,
                                 probs=None):
        # Extract results
        for fss_id, (fss, scenario_subset) in enumerate(fss_scenario_subset_pairs):
            objs_lst, times_lst = [], []
            if fs_results_dict is not None:
                times_lst.append(fs_results_dict[fss_id]["time"])

            scenario_probs = []
            scenario_subset_result = []
            has_result = False
            for scenario_id in scenario_subset:
                key = f"{fss_id}_{scenario_id}"
                if key in results_dict:
                    has_result = True
                    objs_lst.append(results_dict[key]["obj"])
                    times_lst.append(results_dict[key]["time"])
                    scenario_probs.append(probs[scenario_id])
                    scenario_subset_result.append(scenario_id)

            if has_result:
                scenario_probs = np.asarray(scenario_probs)
                scenario_probs = scenario_probs / np.sum(scenario_probs)

                data.append({
                    "sol": fss,
                    "scenario": [[scenarios['sulfur']['D', sid],
                                  scenarios['demand']['X', sid],
                                  scenarios['demand']['Y', sid],
                                  scenario_probs[idx]]
                                 for idx, sid in enumerate(scenario_subset_result)],
                    "n_scenarios": len(scenario_subset_result),
                    "obj_mean": np.dot(objs_lst, scenario_probs),
                    "obj_vals": objs_lst,
                    "scenario_ids": scenario_subset_result,
                    "obj_probs": scenario_probs,
                    "time": np.sum(times_lst),
                    "times": times_lst
                })

    @staticmethod
    def _get_deterministic_data(inst):
        # Define the crude oil sources, cost per unit
        sources, fixed_costs_sources, var_costs = gp.multidict({
            "D": [-110, -8],
            "A": [-120, -6],
            "B": [-300, -15],
            "C": [-135, -9]
        })
        inst['sources'] = list(sources)
        inst['fixed_cost_sources'] = dict(fixed_costs_sources)
        inst['variable_cost_sources'] = dict(var_costs)

        # Define the crude oil sinks, market price, maximum sulfur percentage (quality), and maximum satisfiable demand
        terminals, fixed_costs_terminals, price, max_sulfur, penalty_sulfur, penalty_demand = gp.multidict({
            "X": [-10, 9, 2.5, -500, -20],
            "Y": [-10, 15, 1.5, -700, -30]
        })
        inst['terminals'] = list(terminals)
        inst['fixed_cost_terminals'] = dict(fixed_costs_terminals)
        inst['variable_price_terminals'] = dict(price)
        inst['max_sulfur'] = dict(max_sulfur)
        inst['penalty_sulfur'] = dict(penalty_sulfur)
        inst['penalty_product'] = dict(penalty_demand)

        # Define intermediate pools
        pools, fixed_costs_pools = gp.multidict({
            "P": -10
        })
        inst['pools'] = list(pools)
        inst['fixed_cost_pools'] = dict(fixed_costs_pools)

        # Node connections (i.e., graph edges)
        s2t, s2t_cap = gp.multidict({
            # Source to Terminal
            ("D", "X"): 150,
            ("C", "X"): 150,
            ("C", "Y"): 200
        })
        inst['s2t'] = list(s2t)
        inst['s2t_cap'] = dict(s2t_cap)

        p2t, p2t_cap = gp.multidict({
            # Pool to Terminal
            ("P", "X"): 150,
            ("P", "Y"): 200
        })
        inst['p2t'] = list(p2t)
        inst['p2t_cap'] = dict(p2t_cap)

        s2p, s2p_cap = gp.multidict({
            # Source to Pool
            ("D", "P"): 350,
            ("A", "P"): 350,
            ("B", "P"): 350,
            ("C", "P"): 350
        })
        inst['s2p'] = list(s2p)
        inst['s2p_cap'] = dict(s2p_cap)

        connections = s2t.copy()
        connections.extend(s2p)
        connections.extend(p2t)
        fixed_costs_connections = {conn: -10 for conn in connections}
        inst['connections'] = list(connections)
        inst['fixed_cost_edges'] = dict(fixed_costs_connections)

    def _load_instances(self):
        """ Loads instances files. """
        self.instances = pkl.load(open(self.instance_path, 'rb'))

    @staticmethod
    def _get_feature_vector(pool_prob, sid, info):
        return [info['z_s']['A'], info['z_s']['B'], info['z_s']['C'], info['z_s']['D'],
                info['z_p']['P'], info['z_t']['X'], info['z_t']['Y'],
                info['z_e']['D', 'X'], info['z_e']['C', 'X'], info['z_e']['C', 'Y'],
                info['z_e']['P', 'X'], info['z_e']['P', 'Y'],
                info['z_e']['D', 'P'], info['z_e']['A', 'P'], info['z_e']['B', 'P'], info['z_e']['C', 'P'],
                pool_prob.sulfur['D', sid],
                pool_prob.demand['X', sid],
                pool_prob.demand['Y', sid]]

    def _select_edges_randomly(self, solution, edges_lst):
        # Select all other edges with a prop of 0.5
        edge_sel_prob = self.rng.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        edge_sel = [1 - edge_sel_prob, edge_sel_prob]
        for (i, j) in edges_lst:
            solution['z_e'][i, j] = self.rng.choice([0, 1], p=edge_sel) \
                if solution['z_e'][i, j] == 0 else solution['z_e'][i, j]

    def _get_random_sol(self):
        solution = {'z_e': {(i, j): 0 for (i, j) in self.instances[0]['connections']},
                    'z_s': {i: 0 for i in self.instances[0]['sources']},
                    'z_p': {j: 0 for j in self.instances[0]['pools']},
                    'z_t': {k: 0 for k in self.instances[0]['terminals']}}

        # 0: Source to terminal only
        # 1: Pool to terminal only
        # 2: Source and pool to terminal
        network_type = self.rng.choice([0, 1, 2], p=[0.25, 0.25, 0.5])
        if network_type == 0 or network_type == 2:
            # Select at least one direct edge
            s2t_id = self.rng.randint(len(self.instances[0]['s2t']))
            solution['z_e'][self.instances[0]['s2t'][s2t_id]] = 1

            is_sparse = self.rng.choice([0, 1], p=[0.7, 0.3])
            if not is_sparse:
                self._select_edges_randomly(solution, self.instances[0]['s2t'])

        elif network_type == 1 or network_type == 2:
            # Select at least one pool to terminal
            p2t_id = self.rng.randint(len(self.instances[0]['p2t']))
            solution['z_e'][self.instances[0]['p2t'][p2t_id]] = 1

            # Select at least one source to pool
            s2p_id = self.rng.randint(len(self.instances[0]['s2p']))
            solution['z_e'][self.instances[0]['s2p'][s2p_id]] = 1

            is_sparse = self.rng.choice([0, 1], p=[0.7, 0.3])
            if not is_sparse:
                self._select_edges_randomly(solution, self.instances[0]['p2t'])
                self._select_edges_randomly(solution, self.instances[0]['s2p'])

        # Select relevant nodes
        for (i, j) in self.instances[0]['s2p']:
            if solution['z_e'][i, j] == 1:
                solution['z_s'][i] = 1
                solution['z_p'][j] = 1

        for (i, k) in self.instances[0]['s2t']:
            if solution['z_e'][i, k] == 1:
                solution['z_s'][i] = 1
                solution['z_t'][k] = 1

        for (j, k) in self.instances[0]['p2t']:
            if solution['z_e'][j, k] == 1:
                solution['z_p'][j] = 1
                solution['z_t'][k] = 1

        return solution

    def _get_data_split(self, data):
        """ Gets train/validation splits for the data. """
        tr_data, val_data = None, None
        if len(data):
            perm = self.rng.permutation(len(data))

            split_idx = int(self.cfg.tr_split * (len(data)))
            tr_idx = perm[:split_idx].tolist()
            val_idx = perm[split_idx:].tolist()

            tr_data = [data[i] for i in tr_idx]
            val_data = [data[i] for i in val_idx]

        return tr_data, val_data

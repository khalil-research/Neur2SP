import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from nsp.two_sp.sslp import SSLP
from nsp.utils.sslp import get_path
from .dm import DataManager



def hash_expected(sol, scenario, scenario_subset):
    hashed_scenario_subset = list(map(lambda x: tuple(x), scenario_subset))
    return frozenset(sol), tuple(hashed_scenario_subset), tuple(scenario)


class SSLPDataManager(DataManager):
    def __init__(self, problem_config):

        self.cfg = problem_config

        self.rng = np.random.RandomState()
        self.seed_used = None

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
        self.seed_used = self.cfg.seed
        self.rng.seed(self.cfg.seed)

        self.inst = {}
        self._get_problem_data(self.cfg, self.inst)
        self._create_siplib_lp_files(self.cfg, self.inst)

        self._get_obj_data(self.inst)
        self._get_constr_data(self.inst)
        self._get_siplib_scenarios(self.inst)

        pkl.dump(self.inst, open(self.instance_path, 'wb'))

    def generate_dataset_per_scenario(self, n_procs):
        """ Generate dataset for training ML models. """
        self._load_instance()

        print("Generating NN-P dataset for machine learning... ")
        print(f"  PROBLEM: sslp_{self.inst['n_locations']}_{self.inst['n_clients']}")

        data = []
        total_time = time.time()
        probs = np.linspace(0.1, 0.9, 9)

        two_sp = SSLP(self.inst)

        with Manager() as manager:

            # Get all optimal first stage costs for each scneario.
            mp_data_list = manager.list()

            # Get costs for suboptimal solutions for each scenario
            n_scenarios = self.inst['n_samples_p'] // self.inst['n_samples_per_scenario']
            scenarios = self._sample_scenarios(n_scenarios, self.inst, self.rng)

            procs_to_run = []

            for scenario in scenarios:
                for j in range(self.inst['n_samples_per_scenario']):
                    p = self.rng.choice(probs)  # prob. of swapping bits in sol
                    x_subopt = self._get_pure_random_x(p, self.inst["n_locations"])
                    procs_to_run.append((scenario, x_subopt))

            # Solve all suboptimal LPs
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
                                     self.inst,
                                     mp_data_list))

            pool.close()
            pool.join()

            data = list(mp_data_list)

        total_time = time.time() - total_time

        # get train/validation split, then store
        tr_data, val_data = self._get_data_split(data, self.inst)
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
            print(f"  x: {x_subopt}")
            print(f"  scenario: {scenario}")
            print(f"  exception: {e}")

        return

    def generate_dataset_expected(self, n_procs):
        """ """
        self._load_instance()

        print("Generating NN-E dataset for machine learning... ")
        print(f"  PROBLEM: sslp_{self.inst['n_locations']}_{self.inst['n_clients']}")

        data = []
        total_time = time.time()
        probs = np.linspace(0.0, 0.9, 10)

        two_sp = SSLP(self.inst)

        with Manager() as manager:

            # Get costs from suboptimal first stage solutions. 
            print("  Getting objective for first varying first stage. ")

            # get set of all processes to run
            procs_to_run = []
            n_procs_to_run = 0
            n_samples = self.inst['n_samples_e']

            for _ in range(n_samples):
                # choose a random first stage optimal solution and perturb it
                p = self.rng.choice(probs)  # prob. of swapping bits in sol
                first_stage_sol = self._get_pure_random_x(p, self.inst["n_locations"])

                # choose a random subset of demands
                n_second_stage = self.rng.randint(1, self.inst['n_max_scenarios_in_tr'])
                scenario_subset = self._sample_scenarios(n_second_stage, self.inst, self.rng)

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
                                         self.inst,
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
        tr_data, val_data = self._get_data_split(data, self.inst)
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
            print(f"  demands: {demands}")
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

    def _sol_dict_to_vect(self, x_sol):
        """ Converts a solution dictionary to a vector. """
        x_vect = np.zeros(len(x_sol))
        for k, v in x_sol.items():
            index = int(k.split('_')[1])
            x_vect[index - 1] = v
        return x_vect

    def _sol_vect_to_dict(self, x_vect):
        """ Converts a solution vector to a vector. """
        x_sol = {}
        for index in range(x_vect.size):
            x_sol[f"x_{index + 1}"] = x_vect[index]
        return x_sol

    def _get_feature_vector(self, x_sol, demand):
        """ Gets the simple feature vector (x, deamnds). """
        x_vect = self._sol_dict_to_vect(x_sol)
        features = x_vect.tolist() + demand
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

    def _load_instance(self):
        """ Loads instance file. """
        self.inst = pkl.load(open(self.instance_path, 'rb'))

    @staticmethod
    def _get_problem_data(cfg, inst):
        """ Stores generic problem information. """
        inst['n_locations'] = cfg.n_locations
        inst['n_clients'] = cfg.n_clients
        inst['n_samples_p'] = cfg.n_samples_p
        inst['n_samples_per_scenario'] = cfg.n_samples_per_scenario
        inst['n_samples_e'] = cfg.n_samples_e
        inst['n_max_scenarios_in_tr'] = cfg.n_max_scenarios_in_tr
        inst['tr_split'] = cfg.tr_split
        inst['time_limit'] = cfg.time_limit
        inst['mip_gap'] = cfg.mip_gap
        inst['verbose'] = cfg.verbose
        inst['siplib_instance_names'] = cfg.siplib_instance_names

        inst['seed'] = cfg.seed
        inst['data_path'] = cfg.data_path

    @staticmethod
    def _create_siplib_lp_files(cfg, inst):
        """ Gets the EF file form the SMPS and MPS data. """
        lp_dir = cfg.data_path + '/sslp/sslp_data/'
        inst['siplib_instance_path_dict'] = {}

        for sslp_instance in inst['siplib_instance_names']:

            lp_smps_file = lp_dir + sslp_instance + '.smps'
            lp_mps_file = lp_dir + sslp_instance + '.mps'

            # create .mps file if it does not exist
            if not os.path.isfile(lp_smps_file):
                f_content = [f"{sslp_instance}.cor\n", f"{sslp_instance}.tim\n", f"{sslp_instance}.sto"]
                with open(lp_smps_file, 'w') as f:
                    f.writelines(f_content)

            # load smps file to scip
            if not os.path.isfile(lp_mps_file):
                import pyscipopt
                model = pyscipopt.Model()
                model.readProblem(lp_smps_file)
                model.writeProblem(lp_mps_file)

            inst['siplib_instance_path_dict'][sslp_instance] = lp_mps_file

    @staticmethod
    def _get_obj_data(inst):
        """ Gets the objective function information from the extensive form. """
        sslp_instance = inst['siplib_instance_names'][0]
        sslp_instance_fp = inst['siplib_instance_path_dict'][sslp_instance]
        n_scenarios_in_instance = int(sslp_instance.split("_")[-1])

        model = gp.read(sslp_instance_fp)

        # initialize data structure to store scenario costs
        first_stage_costs = np.zeros(inst['n_locations'])
        second_stage_costs = np.zeros((inst['n_clients'], inst['n_locations']))
        recourse_costs = np.zeros(inst['n_locations'])

        # recover costs for second stage variables
        for var in model.getVars():
            if "y" in var.varName:
                v = var.varName.split("_")
                client = int(v[1]) - 1
                location = int(v[2]) - 1
                scenario = int(v[4])
                second_stage_costs[client][location] = var.obj * n_scenarios_in_instance

            elif "x" in var.varName and var.varName.count("_") > 1:
                v = var.varName.split("_")
                location = int(v[1]) - 1
                scenario = int(v[4])
                recourse_costs[location] = var.obj * n_scenarios_in_instance

            else:
                v = var.varName.split("_")
                location = int(v[1]) - 1
                first_stage_costs[location] = var.obj

        inst['first_stage_costs'] = first_stage_costs
        inst['second_stage_costs'] = second_stage_costs
        inst['recourse_costs'] = recourse_costs

    @staticmethod
    def _get_constr_data(inst):
        """ Gets constraint coefficients. """

        sslp_instance = inst['siplib_instance_names'][0]
        sslp_instance_fp = inst['siplib_instance_path_dict'][sslp_instance]
        n_scenarios_in_instance = int(sslp_instance.split("_")[-1])

        model = gp.read(sslp_instance_fp)

        # create dicts for all variables
        x_vars, y_vars, r_vars = {}, {}, {}
        for var in model.getVars():
            if "y" in var.varName:
                y_vars[var.varname] = var
            elif "x" in var.varName and var.varName.count("_") > 1:
                r_vars[var.varname] = var
            else:
                x_vars[var.varname] = var

        # initialize data structures for constraint data
        loc_limit = 0
        client_coeffs = np.zeros((inst['n_clients'], inst['n_locations']))
        location_coeffs = np.zeros(inst['n_locations'])
        recourse_coeffs = np.zeros(inst['n_locations'])

        # collect cconstraint data
        for constr in model.getConstrs():
            if is_fs_constr(model, constr, x_vars, inst['n_locations']):
                loc_limit = - constr.RHS

            elif is_client_constr(constr):
                pass
                # scenario, client, rhs = get_client_constr_rhs(model, constr, y_vars, inst['n_locations'],
                #                                              inst['n_clients'])
                # client_active[scenario][client] = rhs

            elif is_demand_constr(constr):

                scenario, location, location_coeff, client_coeff, recourse_coeff = get_demand_constr_data(
                    model, constr, x_vars, y_vars, r_vars, inst['n_locations'], inst['n_clients'])

                location_coeffs[location] = location_coeff
                client_coeffs[:, location] = client_coeff
                recourse_coeffs[location] = recourse_coeff

            else:
                raise Exception(f'Constraint {constr} not handled.')

        # store cconstraint data
        inst['location_limit'] = loc_limit
        inst['client_coeffs'] = client_coeffs
        inst['location_coeffs'] = location_coeffs
        inst['recourse_coeffs'] = recourse_coeffs

    @staticmethod
    def _get_siplib_scenarios(inst):

        inst['siplib_scenario_dict'] = {}

        for sslp_instance in inst['siplib_instance_names']:

            sslp_instance_fp = inst['siplib_instance_path_dict'][sslp_instance]
            n_scenarios = int(sslp_instance.split('_')[-1])
            clients_active = np.zeros((n_scenarios, inst['n_clients']))

            model = gp.read(sslp_instance_fp)

            # create dicts for all variables
            x_vars, y_vars, r_vars = {}, {}, {}
            for var in model.getVars():
                if "y" in var.varName:
                    y_vars[var.varname] = var
                elif "x" in var.varName and var.varName.count("_") > 1:
                    r_vars[var.varname] = var
                else:
                    x_vars[var.varname] = var

            for constr in model.getConstrs():
                if is_fs_constr(model, constr, x_vars, inst['n_locations']):
                    pass
                elif is_client_constr(constr):
                    scenario, client, rhs = get_client_constr_rhs(model, constr, y_vars, inst['n_locations'],
                                                                  inst['n_clients'])
                    clients_active[scenario][client] = rhs

            inst['siplib_scenario_dict'][sslp_instance] = clients_active

    @staticmethod
    def _sample_scenarios(n, inst, rng):
        """ Samples n scenarios based on sampling procedure for SSLP. """
        # scenarios = []
        # for _ in range(n):
        #    scenarios.append(rng.randint(0, 2, size=inst['n_clients']).tolist())
        scenarios = rng.randint(0, 2, size=(n, inst['n_clients'])).tolist()
        return scenarios


##
# Some utilitiy functions for collecting data from the constraint matrix
##
def is_fs_constr(model, constr, x_vars, n_locations):
    for location_index in range(n_locations):
        if model.getCoeff(constr, x_vars[f'x_{location_index + 1}']) == 0:
            return False
    return True


def is_client_constr(constr):
    return constr.sense == '='


def get_client_constr_rhs(model, constr, y_vars, n_locations, n_clients):
    scenario = int(constr.constrName.split("_")[-1])

    client = -1
    for c in range(1, n_clients + 1):
        if model.getCoeff(constr, y_vars[f"y_{c}_1_1_{scenario}"]) == 1:
            rhs = constr.RHS
            client = c - 1
            break

    return scenario, client, rhs


def is_demand_constr(constr):
    return constr.sense == '>' and constr.rhs == 0


def get_demand_constr_data(model, constr, x_vars, y_vars, r_vars, n_locations, n_clients):
    # get scenaio
    scenario = int(constr.constrName.split("_")[-1])

    # get location
    location = -1
    for l in range(1, n_locations + 1):
        if model.getCoeff(constr, x_vars[f"x_{l}"]) != 0:
            location = l - 1
            break
    location_coeff = model.getCoeff(constr, x_vars[f"x_{location + 1}"])

    # get client coeffs
    client_coeffs = np.zeros(n_clients)
    for client in range(n_clients):
        v_name = f"y_{client + 1}_{location + 1}_1_{scenario}"
        client_coeffs[client] = model.getCoeff(constr, y_vars[v_name])

    v_name = f"x_{location + 1}_0_1_{scenario}"
    recourse_coeff = model.getCoeff(constr, r_vars[v_name])

    return scenario, location, location_coeff, client_coeffs, recourse_coeff

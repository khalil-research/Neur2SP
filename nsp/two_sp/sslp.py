from multiprocessing import Pool, Manager

import gurobipy as gp
import numpy as np

from .two_sp import TwoStageStocProg


class SSLP(TwoStageStocProg):

    def __init__(self, inst, relative_dir=None):
        self.inst = inst

    def _make_extensive_model(self, scenarios):
        """ Formulates two stage extensive form. """

        n_scenarios = len(scenarios)
        scenario_prob = 1 / n_scenarios


        model = gp.Model()

        # ADD VARIABLES
        x_vars, y_vars, r_vars = {}, {}, {}

        for loc in range(self.inst['n_locations']):
            v_name = f"x_{loc + 1}"
            obj = self.inst['first_stage_costs'][loc]
            x_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for scen in range(n_scenarios):
            for clnt in range(self.inst['n_clients']):
                for loc in range(self.inst['n_locations']):
                    v_name = f"y_{clnt + 1}_{loc + 1}_{scen}"
                    obj = self.inst['second_stage_costs'][clnt][loc] * scenario_prob
                    y_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for scen in range(n_scenarios):
            for loc in range(self.inst['n_locations']):
                v_name = f"r_{loc + 1}_{scen}"
                obj = self.inst['recourse_costs'][loc] * scenario_prob
                r_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="C")

        # ADD CONSTRAINTS
        # location limit constraints
        eq_ = 0
        for loc in range(self.inst['n_locations']):
            eq_ += x_vars[f"x_{loc + 1}"]

        model.addConstr(eq_ <= self.inst['location_limit'], name="location_limit")

        # location capacity constraints
        for scen in range(n_scenarios):
            for loc in range(self.inst['n_locations']):
                eq_ = self.inst['location_coeffs'][loc] * x_vars[f"x_{loc + 1}"]
                eq_ += self.inst['recourse_coeffs'][loc] * r_vars[f"r_{loc + 1}_{scen}"]
                for clnt in range(self.inst['n_clients']):
                    eq_ += self.inst['client_coeffs'][clnt][loc] * y_vars[f"y_{clnt + 1}_{loc + 1}_{scen}"]
                model.addConstr(eq_ >= 0, name=f"capacity_{loc + 1}_{scen}")

        # client constrints
        for scen in range(n_scenarios):
            for clnt in range(self.inst['n_clients']):
                eq_ = 0
                for loc in range(self.inst['n_locations']):
                    eq_ += y_vars[f"y_{clnt + 1}_{loc + 1}_{scen}"]
                model.addConstr(eq_ == scenarios[scen][clnt], name=f'active_{clnt + 1}_{scen}')

        return model

    def _make_second_stage_model(self, scenario):
        """ Initializes a second stage model. """
        model = gp.Model()

        # ADD VARIABLES
        x_vars, y_vars, r_vars = {}, {}, {}

        for loc in range(self.inst['n_locations']):
            v_name = f"x_{loc + 1}"
            obj = self.inst['first_stage_costs'][loc]
            x_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for clnt in range(self.inst['n_clients']):
            for loc in range(self.inst['n_locations']):
                v_name = f"y_{clnt + 1}_{loc + 1}"
                obj = self.inst['second_stage_costs'][clnt][loc]
                y_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for loc in range(self.inst['n_locations']):
            v_name = f"r_{loc + 1}"
            obj = self.inst['recourse_costs'][loc]
            r_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="C")

        # ADD CONSTRAINTS
        constrs = {}

        # location limit constraints
        eq_ = 0
        for loc in range(self.inst['n_locations']):
            eq_ += x_vars[f"x_{loc + 1}"]

        c_name = "location_limit"
        constrs[c_name] = model.addConstr(eq_ <= self.inst['location_limit'], name=c_name)

        # location capacity constraints
        for loc in range(self.inst['n_locations']):
            eq_ = self.inst['location_coeffs'][loc] * x_vars[f"x_{loc + 1}"]
            eq_ += self.inst['recourse_coeffs'][loc] * r_vars[f"r_{loc + 1}"]
            for clnt in range(self.inst['n_clients']):
                eq_ += self.inst['client_coeffs'][clnt][loc] * y_vars[f"y_{clnt + 1}_{loc + 1}"]
            c_name = f"capacity_{loc + 1}"
            constrs[c_name] = model.addConstr(eq_ >= 0, name=c_name)

        # client constrints
        for clnt in range(self.inst['n_clients']):
            eq_ = 0
            for loc in range(self.inst['n_locations']):
                eq_ += y_vars[f"y_{clnt + 1}_{loc + 1}"]
            c_name = f'active_{clnt + 1}'
            constrs[c_name] = model.addConstr(eq_ == scenario[clnt], name=c_name)

        model.update()

        return model

    def solve_extensive(self, n_scenarios, gap=0.02, time_limit=600, threads=1, log_dir=None, node_file_start=None,
                        node_file_dir=None, test_set="0"):
        """ Solves the extensive form. """

        def callback(model, where):
            """ Callback function to log time, bounds, and first stage sol. """
            if where == gp.GRB.Callback.MIPSOL:
                self.ef_solving_results['time'].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                self.ef_solving_results['primal'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                self.ef_solving_results['dual'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                self.ef_solving_results['incumbent'].append(model.cbGetSolution(model._x))

        scenarios = self.get_scenarios(n_scenarios, test_set)

        model = self._make_extensive_model(scenarios)

        # get variables for callback
        model.update()
        self.ef_solving_results = {'primal': [], 'dual': [], 'incumbent': [], 'time': []}
        ef_fs_vars = []
        for i in range(1, self.inst['n_locations'] + 1):
            ef_fs_vars.append(model.getVarByName(f"x_{i}"))
        model._x = ef_fs_vars

        # solve two_sp
        if log_dir is not None:
            model.setParam("LogFile", log_dir)
        if node_file_start is not None:
            model.setParam("NodefileStart", node_file_start)
            model.setParam("NodefileDir", node_file_dir)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam('Threads', threads)
        model.optimize(callback)

        return model

    def get_second_stage_objective(self, sol, scenario, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        """ Gets the second stage model for an objective. """
        model = self._make_second_stage_model(scenario)
        model = self.fix_first_stage(model, sol)

        # optimize model
        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam('Threads', threads)
        model.optimize()

        return self.get_second_stage_cost(model)

    def evaluate_first_stage_sol(self, sol, n_scenarios, gap=0.0001, time_limit=600, threads=1, verbose=0, test_set="0", n_procs=1):
        """ Gets the objective function value for a given solution. """
        scenarios = self.get_scenarios(n_scenarios, test_set)
        n_scenarios = len(scenarios)
        scenario_prob = 1/n_scenarios

        # get first stage objective
        fs_obj = 0
        for v_name, value in sol.items():
            index = int(v_name.split("_")[-1]) - 1
            fs_obj += self.inst['first_stage_costs'][index] * value

        # second stage objective
        pool = Pool(n_procs)

        results = [pool.apply_async(self.mp_get_second_stage_obj,
                                   args=(sol, scenario, scenario_prob, gap, time_limit, threads, verbose))
                  for scenario in scenarios]

        results = [r.get() for r in results]

        second_stage_obj_val = np.sum(results)

        return fs_obj + second_stage_obj_val

    def mp_get_second_stage_obj(self, sol, scenario, scenario_prob, gap, time_limit, threads, verbose):
        """ Multiprocessing """
        second_stage_obj = scenario_prob * self.get_second_stage_objective(sol, scenario, gap=gap, time_limit=time_limit,
                                                                           threads=threads, verbose=verbose)
        return second_stage_obj

    def get_first_stage_extensive_solution(self, model):
        """ Recovers the solution from a LR/NN embedded model. """
        x_sol = {}
        for var in model.getVars():
            if "x" in var.varName and len(var.varName.split("_")) == 2:
                x_sol[var.varName] = np.abs(var.x)
        return x_sol

    def fix_first_stage(self, model, sol):
        """ Fixes the first stage solution of a given model"""
        for var in model.getVars():
            if var.varName in sol:
                var.ub = sol[var.varName]
                var.lb = sol[var.varName]
        model.update()
        return model

    def get_second_stage_cost(self, model):
        """ Gets the second stage cost of a given model.  """
        ss_obj = 0
        for var in model.getVars():
            if "y" in var.varName or "r" in var.varName:
                ss_obj += var.obj * var.x
        return ss_obj

    def get_scenario_optimal_first_stage(self, scenario, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        """ Gets the first stage optimal solution for scenario. """
        model = self._make_second_stage_model(scenario)
        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam('Threads', threads)
        model.optimize()

        # recover first stage solution
        sol = {}
        for var in model.getVars():
            v_name = var.varName
            if "x" in var.varName:
                sol[var.varName] = float(int(var.x))

        # recover second stage objective value
        ss_obj = self.get_second_stage_cost(model)

        return sol, ss_obj

    def get_scenarios(self, n_scenarios, test_set):
        """ Gets n_scenario sceanrios.  Randomization based on test_set. """
        if test_set == "siplib":
            sslp_instance = f'sslp_{self.inst["n_locations"]}_{self.inst["n_clients"]}_{n_scenarios}'
            scenarios = self.inst['siplib_scenario_dict'][sslp_instance]

        else:
            test_set = int(test_set)
            rng = np.random.RandomState()
            rng.seed(n_scenarios + test_set)

            scenarios = rng.randint(0, 2, size=(n_scenarios, self.inst['n_clients'])).tolist()


        return scenarios
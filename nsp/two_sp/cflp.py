from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from .two_sp import TwoStageStocProg


class FacilityLocationProblem(TwoStageStocProg):

    def __init__(self, inst):
        self.tol = 1e-6
        self.inst = inst

        self.n_customers = self.inst['n_customers']
        self.n_facilities = self.inst['n_facilities']
        self.integer_second_stage = self.inst['integer_second_stage']
        self.bound_tightening_constrs = self.inst['bound_tightening_constrs']
        self.capacities = self.inst['capacities']
        self.fixed_costs = self.inst['fixed_costs']
        self.trans_costs = self.inst['trans_costs']
        self.recourse_costs = 2 * np.max([np.max(self.fixed_costs), np.max(self.trans_costs)])

    def _make_extensive_model(self, scenarios):
        """ Formulates two stage extensive form. """
        demands = scenarios
        n_scenarios = len(scenarios)
        scenario_prob = 1 / n_scenarios

        model = gp.Model()
        var_dict = {}

        # binary variables for each location
        for i in range(self.n_facilities):
            var_name = f"x_{i}"
            var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.fixed_costs[i], vtype="B", name=var_name)

        # add either continous or binary second stage serving costs
        for s in range(n_scenarios):
            for i in range(self.n_facilities):
                for j in range(self.n_customers):
                    var_name = f"y_{i}_{j}_{s}"
                    if self.integer_second_stage:
                        var_dict[var_name] = model.addVar(lb=0.0, ub=1.0,
                                                          obj=self.trans_costs[i, j] * scenario_prob, vtype="B",
                                                          name=var_name)
                    else:
                        var_dict[var_name] = model.addVar(lb=0.0, ub=1.0,
                                                          obj=self.trans_costs[i, j] * scenario_prob, vtype="C",
                                                          name=var_name)

        # add either continous or binary second stage recourse costs
        for s in range(n_scenarios):
            for j in range(self.n_customers):
                var_name = f"z_{j}_{s}"
                if self.integer_second_stage:
                    var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.recourse_costs * scenario_prob,
                                                      vtype="B", name=var_name)
                else:
                    var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.recourse_costs * scenario_prob,
                                                      vtype="C", name=var_name)

        # add demand constraints
        for s in range(n_scenarios):
            for j in range(self.n_customers):
                cons = var_dict[f"z_{j}_{s}"]
                for i in range(self.n_facilities):
                    cons += var_dict[f"y_{i}_{j}_{s}"]
                model.addConstr(cons >= 1, name=f"d_{j}_{s}")

        # capacity constraints
        for s in range(n_scenarios):
            for i in range(self.n_facilities):
                cons = - self.capacities[i] * var_dict[f"x_{i}"]
                for j in range(self.n_customers):
                    cons += demands[s][j] * var_dict[f"y_{i}_{j}_{s}"]
                model.addConstr(cons <= 0, name=f"c_{i}")

        # bound tightening constraints
        if self.bound_tightening_constrs:
            for s in range(n_scenarios):
                for i in range(self.n_facilities):
                    for j in range(self.n_customers):
                        model.addConstr(- var_dict[f"x_{i}"] + var_dict[f"y_{i}_{j}_{s}"] <= 0, name=f"t_{i}_{j}_{s}")

        model.update()

        return model

    def _make_second_stage_model(self, demands):
        """ Creates the second stage model. """
        model = gp.Model()
        var_dict = {}

        # binary variables for each location
        for i in range(self.n_facilities):
            var_name = f"x_{i}"
            # bound lower and upper to solution
            var_dict[var_name] = model.addVar(obj=self.fixed_costs[i], vtype="B", name=var_name)

        # add either continous or binary second stage serving costs
        for i in range(self.n_facilities):
            for j in range(self.n_customers):
                var_name = f"y_{i}_{j}"
                if self.integer_second_stage:
                    var_dict[var_name] = model.addVar(obj=self.trans_costs[i, j], vtype="B", name=var_name)
                else:
                    var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.trans_costs[i, j], vtype="C",
                                                      name=var_name)

        # add either continous or binary second stage recourse costs
        for j in range(self.n_customers):
            var_name = f"z_{j}"
            if self.integer_second_stage:
                var_dict[var_name] = model.addVar(obj=self.recourse_costs, vtype="B", name=var_name)
            else:
                var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.recourse_costs, vtype="C", name=var_name)

        model.update()

        # add demand constraints
        for j in range(self.n_customers):
            cons = var_dict[f"z_{j}"]
            for i in range(self.n_facilities):
                cons += var_dict[f"y_{i}_{j}"]
            model.addConstr(cons >= 1, name=f"d_{j}")

        # capacity constraints
        for i in range(self.n_facilities):
            cons = - self.capacities[i] * var_dict[f"x_{i}"]
            for j in range(self.n_customers):
                cons += demands[j] * var_dict[f"y_{i}_{j}"]
            model.addConstr(cons <= 0, name=f"c_{i}")

        # bound tightening constraints
        if self.bound_tightening_constrs:
            for i in range(self.n_facilities):
                for j in range(self.n_customers):
                    model.addConstr(- var_dict[f"x_{i}"] + var_dict[f"y_{i}_{j}"] <= 0, name=f"t_{i}_{j}")

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
                self.ef_solving_results['incumbent'].append(model.cbGetSolution(model._x ))

        # make extensive form 
        scenarios = self.get_scenarios(n_scenarios, test_set)
        model = self._make_extensive_model(scenarios)

        # get variables for callback
        model.update()
        self.ef_solving_results = {'primal': [], 'dual': [], 'incumbent': [], 'time': []}
        ef_fs_vars = []
        for i in range(self.n_facilities):
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
        model.setParam("Threads", threads)

        model.optimize(callback)
        return model

    def get_second_stage_objective(self, sol, demands, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        """ Gets the second stage model for an objective. """

        model = self._make_second_stage_model(demands)

        # fix first stage solution
        model = self.fix_first_stage(model, sol)

        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)

        model.optimize()

        second_stage_obj = self.get_second_stage_cost(model)
        return second_stage_obj

    def evaluate_first_stage_sol(self, sol, n_scenarios, gap=0.0001, time_limit=600, threads=1, verbose=0, test_set="0", n_procs=1):
        """ Gets the objective function value for a given solution. """

        scenarios = self.get_scenarios(n_scenarios, test_set)
        n_scenarios = len(scenarios)
        scenario_prob = 1 / len(scenarios)

        # evaluate first stage values
        first_stage_obj_val = 0
        for i in range(self.n_facilities):
            first_stage_obj_val += sol[f"x_{i}"] * self.fixed_costs[i]

        with Manager() as manager:

            mp_list = manager.list()

            pool = Pool(n_procs)
            for demand in scenarios:
                pool.apply_async(self.mp_get_second_stage_obj,
                                 args=(sol, demand, scenario_prob, gap, time_limit, verbose, mp_list))
            pool.close()
            pool.join()

            second_stage_costs = list(mp_list)

        second_stage_obj_val = np.sum(second_stage_costs)

        return first_stage_obj_val + second_stage_obj_val

    def mp_get_second_stage_obj(self, sol, demand, scenario_prob, gap, time_limit, verbose, mp_list):
        """ Multiprocessing """
        second_stage_obj = scenario_prob * self.get_second_stage_objective(sol, demand, gap=gap, time_limit=time_limit,
                                                                           verbose=verbose)
        mp_list.append(second_stage_obj)

    def get_scenario_optimal_first_stage(self, demands, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        """ Gets the first stage optimal solution for scenario. """
        model = self._make_second_stage_model(demands)
        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)
        model.optimize()

        # recover first stage solution and second stage objective value
        sol = self.get_first_stage_solution(model)
        second_stage_obj = self.get_second_stage_cost(model)

        return sol, second_stage_obj

    def fix_first_stage(self, model, sol):
        """ Fixes the first stage solution of a given model. """
        for sol_var_name, sol_var_val in sol.items():
            model.getVarByName(sol_var_name).lb = sol_var_val
            model.getVarByName(sol_var_name).ub = sol_var_val
        model.update()
        return model

    def get_second_stage_cost(self, model):
        """ Gets the second stage cost of a given model.  """
        second_stage_obj = 0
        for var in model.getVars():
            if "x" not in var.varName:
                second_stage_obj += var.obj * var.x
        return second_stage_obj

    def get_first_stage_solution(self, model):
        """ Recovers the first stage solution.  """
        sol = {}
        for var in model.getVars():
            if "x" not in var.varName:
                continue
            sol[var.varName] = var.x
        return sol

    def get_first_stage_extensive_solution(self, model):
        """ Gets the first stage solution for the extensive model.  """
        return self.get_first_stage_solution(model)

    def get_scenarios(self, n_scenarios, test_set):
        """ Gets n_scenario sceanrios.  Randomization based on test_set. """
        test_set = int(test_set)
        rng = np.random.RandomState()
        rng.seed(n_scenarios + test_set)
        scenarios = [] 
        for _ in range(n_scenarios):
            scenarios.append(rng.randint(5, 35 + 1, size=self.n_customers))

        return scenarios
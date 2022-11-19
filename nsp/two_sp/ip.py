import multiprocessing as mp

import gurobipy as gp
import numpy as np

from nsp.two_sp import TwoStageStocProg


class InvestmentProblem(TwoStageStocProg):
    def __init__(self, inst, scenarios):
        self.inst = inst
        self.c_fs = inst['c_fs']
        self.c_ss = inst['c_ss']
        self.W = inst['W']
        self.T = inst['T']

        self.scenarios = scenarios
        self.n_scenarios = self.scenarios.shape[0]

        # callback info
        self.ef_solving_results = None
        self.ef_fs_vars = None

    def _make_surrogate_scenario_model(self, scenario_id=None):
        if scenario_id is None:
            # Make extensive model
            _scenario = self.scenarios
            _num_scenario = self.n_scenarios
        else:
            # Make a surrogate single scenario model
            _scenario = [self.scenarios[scenario_id]]
            _num_scenario = 1
        prob = 1 / _num_scenario

        m = gp.Model()
        # Add variables
        x, y = None, {}
        x = m.addMVar(2, lb=0, ub=5, vtype=self.inst['first_stage_vtype'], obj=self.c_fs, name='x')
        for s in range(_num_scenario):
            y[s] = m.addMVar(4, vtype=self.inst['second_stage_vtype'], lb=0, obj=prob * self.c_ss, name=f'y_{s}')

        # Add constraints
        for s in range(_num_scenario):
            m.addConstr(self.W @ y[s] <= _scenario[s] - self.T @ x)
        m._x = x
        m.update()

        return m

    def _make_extensive_model(self):
        """Make extensive form of the model"""
        return self._make_surrogate_scenario_model()

    def _make_second_stage_model(self, sol, scenario):
        m = gp.Model()
        # Add variables
        y = m.addMVar(4, vtype=self.inst['second_stage_vtype'], lb=0, obj=self.c_ss, name='y')
        # Add constraints
        m.addConstr(self.W @ y <= scenario - self.T @ sol)

        return m, y

    def solve_extensive(self,
                        n_scenarios,
                        gap=0.01,
                        time_limit=600,
                        threads=1,
                        verbose=1,
                        log_dir=None,
                        node_file_start=None,
                        node_file_dir=None,
                        test_set="0"):
        """ Solves the extensive form. """

        def callback(model, where):
            """ Callback function to log time, bounds, and first stage sol. """
            if where == gp.GRB.Callback.MIPSOL:
                self.ef_solving_results['time'].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                self.ef_solving_results['primal'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                self.ef_solving_results['dual'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                self.ef_solving_results['incumbent'].append(model.cbGetSolution(model._x))

        self.ef_solving_results = {'primal': [], 'dual': [], 'incumbent': [], 'time': []}

        m = self._make_surrogate_scenario_model()
        # self.ef_fs_vars = [m.getVarByName("x[0]"), m.getVarByName("x[1]")]

        # solve two_sp
        m.setParam("OutputFlag", verbose)
        m.setParam('MIPGap', gap)
        m.setParam('Threads', threads)
        m.setParam('TimeLimit', time_limit)
        if log_dir is not None:
            m.setParam("LogFile", log_dir)
        if node_file_start is not None:
            m.setParam("NodefileStart", node_file_start)
            m.setParam("NodefileDir", node_file_dir)
        m.optimize(callback)
        # info = self._extract_mip_solve_info(m, get_var=True)

        return m

    def solve_surrogate_scenario_model(self,
                                       scenario_id,
                                       gap=0.01,
                                       time_limit=600,
                                       threads=1,
                                       verbose=0,
                                       log_dir=None,
                                       node_file_start=None,
                                       node_file_dir=None,
                                       test_set="0"):
        """ Solves the surrogate form."""
        m = self._make_surrogate_scenario_model(scenario_id)
        # Set params
        m.setParam("OutputFlag", verbose)
        m.setParam('MIPGap', gap)
        m.setParam('Threads', threads)
        m.setParam('TimeLimit', time_limit)
        if log_dir is not None:
            m.setParam("LogFile", log_dir)
        if node_file_start is not None:
            m.setParam("NodefileStart", node_file_start)
            m.setParam("NodefileDir", node_file_dir)
        # Solve
        m.optimize()

        info = self._extract_mip_solve_info(m, get_var=True)

        return info

    def get_second_stage_objective(self,
                                   sol,
                                   scenario_id,
                                   gap=0.0001,
                                   time_limit=600,
                                   verbose=0,
                                   threads=1):
        """Get the objective value of a second problem for a given
        first-stage solution and scenario

        sol: numpy.ndarray
            First-stage solution

        scenario: numpy.ndarray
            Scenario vector
        """
        m, _ = self._make_second_stage_model(sol, self.scenarios[scenario_id])
        m.setParam("OutputFlag", verbose)
        m.setParam('MIPGap', gap)
        m.setParam('TimeLimit', time_limit)
        m.setParam('Threads', threads)
        m.optimize()
        info = self._extract_mip_solve_info(m)

        return info["obj_val"]

    def evaluate_first_stage_sol(self,
                                 sol,
                                 n_scenarios=4,
                                 gap=0.0001,
                                 time_limit=600,
                                 threads=1,
                                 verbose=0,
                                 test_set="0",
                                 n_procs=1):
        """Find cx + E_xi [ Q(x, xi) ]

        sol: numpy.ndarray
            A numpy array of the solution values
        """
        if sol is None:
            return None

        print()
        print(f'  Evaluating first stage solution: {sol}')
        print()

        first_stage_obj_val = self.c_fs @ sol

        pool = mp.Pool(n_procs)
        results = [pool.apply_async(self.get_second_stage_objective,
                                    (sol, scenario_id,),
                                    dict(gap=gap, time_limit=time_limit, verbose=verbose))
                   for scenario_id in range(self.n_scenarios)]
        results = [r.get() for r in results]
        expected_second_stage_obj_val = np.mean(results)

        return first_stage_obj_val + expected_second_stage_obj_val

    def get_first_stage_extensive_solution(self, model):
        status = model.getAttr('status')
        if not (status == 3 or status == 4 or status == 5):
            return model._x.x

        return None

    @staticmethod
    def _extract_mip_solve_info(m, get_var=False):
        """Helper function to extract useful information after a MIP solve

        m: gurobipy.Model
            Optimized Gurobi model

        get_var: bool
            Flag to indicate if the solution values are required
        """
        info = {"sol": None, "obj_val": gp.GRB.INFINITY}
        status = m.getAttr('status')
        if not (status == 3 or status == 4 or status == 5):
            info["obj_val"] = m.objVal
            info["sol"] = m._x.x if get_var else None

        return info

    def get_scenarios(self, n_scenarios, test_set):
        pass

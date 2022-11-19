import gurobipy as gp
import numpy as np

from .two_sp import TwoStageStocProg


class PoolingProblem(TwoStageStocProg):
    def __init__(self, inst, scenarios):
        self.inst = inst
        # Nodes
        self.sources = inst['sources']
        self.pools = inst['pools']
        self.terminals = inst['terminals']
        # Edges
        self.connections = inst['connections']
        self.s2t = inst['s2t']
        self.s2p = inst['s2p']
        self.p2t = inst['p2t']
        # Cost
        self.fc_connections = inst['fixed_cost_edges']
        self.fc_sources = inst['fixed_cost_sources']
        self.vc_sources = inst['variable_cost_sources']
        self.fc_pools = inst['fixed_cost_pools']
        self.fc_terminals = inst['fixed_cost_terminals']
        self.vp_terminals = inst['variable_price_terminals']
        # Capacity
        self.s2t_cap = inst['s2t_cap']
        self.s2p_cap = inst['s2p_cap']
        self.p2t_cap = inst['p2t_cap']
        # Max sulfur
        self.max_sulfur = inst['max_sulfur']
        # Penalty
        self.penalty_sulfur = inst['penalty_sulfur']
        self.penalty_product = inst['penalty_product']
        # Scenarios
        self.scenarios = scenarios['scenarios']
        self.demand = scenarios['scenarios']['demand']
        self.sulfur = scenarios['scenarios']['sulfur']
        self.probs = scenarios['probs']
        self.num_scenarios = len(self.probs)

        # callback info
        self.ef_solving_results = None
        self.ef_fs_vars = None
        self.fs_vars_dict = {}

    def solve_extensive(self,
                        n_scenarios,
                        gap=0.02,
                        time_limit=600,
                        threads=1,
                        verbose=1,
                        log_dir=None,
                        node_file_start=None,
                        node_file_dir=None,
                        test_set="0"):
        """Solves the extensive form"""

        def callback(model, where):
            """ Callback function to log time, bounds, and first stage sol. """
            if where == gp.GRB.Callback.MIPSOL:
                self.ef_solving_results['time'].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                self.ef_solving_results['primal'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                self.ef_solving_results['dual'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                self.ef_solving_results['incumbent'].append({
                    'z_s': dict(model.cbGetSolution(model._z_s)),
                    'z_p': dict(model.cbGetSolution(model._z_p)),
                    'z_t': dict(model.cbGetSolution(model._z_t)),
                    'z_e': dict(model.cbGetSolution(model._z_e))
                })

        m, self.fs_vars_dict = self._make_extensive_model()

        # get info for callback
        m.update()
        self.ef_solving_results = {'primal': [], 'dual': [], 'incumbent': [], 'time': []}
        # solve two_sp
        m.setParam("OutputFlag", verbose)
        m.setParam('MIPGap', gap)
        m.setParam('TimeLimit', time_limit)
        m.setParam('Threads', threads)
        if log_dir is not None:
            m.setParam("LogFile", log_dir)
        if node_file_start is not None:
            m.setParam("NodefileStart", node_file_start)
            m.setParam("NodefileDir", node_file_dir)

        m.params.NonConvex = 2
        m.optimize(callback)
        # info = self._extract_mip_solve_info(m, get_var=True)

        # return info
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

        m, self.fs_vars_dict = self._make_surrogate_scenario_model(scenario_id)
        m.setParam("OutputFlag", verbose)
        m.setParam('MIPGap', gap)
        m.setParam('TimeLimit', time_limit)
        m.setParam('Threads', threads)
        if log_dir is not None:
            m.setParam("LogFile", log_dir)
        if node_file_start is not None:
            m.setParam("NodefileStart", node_file_start)
            m.setParam("NodefileDir", node_file_dir)

        m.params.NonConvex = 2
        m.optimize()
        info = self._extract_mip_solve_info(m, get_var=True)

        return info

    def get_second_stage_objective(self,
                                   fs_sol,
                                   scenario_id,
                                   gap=0.01,
                                   time_limit=600,
                                   threads=1,
                                   verbose=0):
        m = self._make_second_stage_model(fs_sol, scenario_id)
        m.setParam("OutputFlag", verbose)
        m.setParam('MIPGap', gap)
        m.setParam('TimeLimit', time_limit)
        m.setParam('Threads', threads)
        m.params.NonConvex = 2
        m.optimize()
        info = self._extract_mip_solve_info(m)

        return info["obj_val"]

    def evaluate_first_stage_sol(self,
                                 fs_sol,
                                 n_scenarios,
                                 gap=0.0001,
                                 time_limit=600,
                                 threads=1,
                                 verbose=0,
                                 test_set="0",
                                 n_procs=1):
        if fs_sol is None:
            return None

        # First-stage profits
        fs_revenue = sum([self.fc_sources[i] * fs_sol['z_s'][i] for i in self.sources])
        fs_revenue += sum([self.fc_pools[j] * fs_sol['z_p'][j] for j in self.pools])
        fs_revenue += sum([self.fc_terminals[k] * fs_sol['z_t'][k] for k in self.terminals])
        fs_revenue += sum([self.fc_connections[conn] * fs_sol['z_e'][conn] for conn in self.connections])

        # Second-stage profits
        scen_revenue = 0
        for sid, prob in enumerate(self.probs):
            scen_revenue += prob * (self.get_second_stage_objective(fs_sol, sid, gap=gap,
                                                                    time_limit=time_limit, verbose=verbose))

        return fs_revenue + scen_revenue

    def get_first_stage_extensive_solution(self, model):
        sol = {}

        status = model.getAttr('status')
        if not (status == 3 or status == 4 or status == 5):
            sol["z_s"] = {i: np.floor(self.fs_vars_dict['z_s'][i].x + 0.5) for i in self.sources}
            sol["z_p"] = {j: np.floor(self.fs_vars_dict['z_p'][j].x + 0.5) for j in self.pools}
            sol["z_t"] = {k: np.floor(self.fs_vars_dict['z_t'][k].x + 0.5) for k in self.terminals}
            sol["z_e"] = {(i, j): np.floor(self.fs_vars_dict['z_e'][i, j].x + 0.5) for i, j in self.connections}

            return sol

        return None

    def _extract_mip_solve_info(self, m, get_var=False):
        """Helper function to extract useful information after a MIP solve

        m: gurobipy.Model
            Optimized Gurobi model

        get_var: bool
            Flag to indicate if the solution values are required
        """
        info = {"obj_val": gp.GRB.INFINITY}
        status = m.getAttr('status')
        if not (status == 3 or status == 4 or status == 5):
            info["obj_val"] = m.objVal
            if get_var:
                info["z_s"] = {i: np.floor(self.fs_vars_dict['z_s'][i].x + 0.5) for i in self.sources}
                info["z_p"] = {j: np.floor(self.fs_vars_dict['z_p'][j].x + 0.5) for j in self.pools}
                info["z_t"] = {k: np.floor(self.fs_vars_dict['z_t'][k].x + 0.5) for k in self.terminals}
                info["z_e"] = {(i, j): np.floor(self.fs_vars_dict['z_e'][i, j].x + 0.5) for i, j in self.connections}

        return info

    def _make_surrogate_scenario_model(self, scenario_id=None):
        """Make a two-stage stochastic optimization model for the Pooling Problem
        - If a scenario_id is not provided, we solve the extensive form of the problem
        with all scenarios in the generated instance.
        - If a scenario_id is provided, we construct a surrogate problem with
        first-stage and provided scenario constraints.
        """
        # Fetch relevant scenario/s
        if scenario_id is not None:
            _probs = [1]
            _scenario_ids = [scenario_id]
        else:
            _probs = self.probs
            _scenario_ids = range(self.num_scenarios)
        _num_scenarios = len(_scenario_ids)

        #####################################
        #    Initialize model variables     #
        #####################################
        m = gp.Model('StochasticPoolingProblem')

        #####################################
        #         Define variables          #
        #####################################
        # First-stage variables. Used to design the network
        z_s = m.addVars(self.sources, vtype="B", name="z_s")
        z_p = m.addVars(self.pools, vtype="B", name="z_p")
        z_t = m.addVars(self.terminals, vtype="B", name="z_t")
        z_e = m.addVars(self.connections, vtype="B", name="z_e")
        m._z_s, m._z_p, m._z_t, m._z_e = z_s, z_p, z_t, z_e
        fs_vars_dict = {'z_s': z_s, 'z_p': z_p, 'z_t': z_t, 'z_e': z_e}

        # Flow variables
        _flow_names = self.s2t[:]
        _flow_names.extend(self.p2t)
        flow = m.addVars(_flow_names, _scenario_ids, name='flow', lb=0)

        # Proportion of total flow at pool j from source i under each scenario
        prop = m.addVars(self.s2p, _scenario_ids, name='prop', lb=0)

        # To ensure relatively complete recourse
        surplus_sulfur = m.addVars(self.terminals, _scenario_ids, name='surplus_sulfur', lb=0)
        surplus_product = m.addVars(self.terminals, _scenario_ids, name='surplus_product', lb=0)

        #######################################
        #          Define constraints         #
        #######################################

        # Using edge enforces use of nodes
        for i, j in self.s2p:
            m.addConstr(z_e[i, j] <= z_s[i])
            m.addConstr(z_e[i, j] <= z_p[j])

        for i, k in self.s2t:
            m.addConstr(z_e[i, k] <= z_s[i])
            m.addConstr(z_e[i, k] <= z_t[k])

        for j, k in self.p2t:
            m.addConstr(z_e[j, k] <= z_p[j])
            m.addConstr(z_e[j, k] <= z_t[k])

        # Using node enforces at least one outgoing edge to be selected
        m.addConstrs((z_s[i] <= z_e.sum(i, '*') for i in self.sources))
        m.addConstrs((z_p[j] <= z_e.sum(j, '*') for j in self.pools))

        # Define second stage constraints
        for sid in _scenario_ids:
            # Flow capacity constraints based on first-stage decision
            m.addConstrs((flow[i, k, sid] <= self.s2t_cap[i, k] * z_e[i, k] for i, k in self.s2t))
            m.addConstrs((flow[j, k, sid] <= self.p2t_cap[j, k] * z_e[j, k] for j, k in self.p2t))
            m.addConstrs((prop[i, j, sid] * flow.sum(j, '*', sid) <= self.s2p_cap[i, j] * z_e[i, j]
                          for i, j in self.s2p))

            # Sum of fractions of inflow to a pool equals 1 if the pool is selected,
            # 0 otherwise.
            m.addConstrs((prop.sum('*', j, sid) == z_p[j] for j in self.pools))

            # Inflow at terminal must be less than demand + surplus product
            m.addConstrs((flow.sum('*', k, sid) <= self.demand[k, sid] + surplus_product[k, sid]
                          for k in self.terminals))

            # Sulphur at terminal must be less than the maximum allowed quantity + surplus sulphur
            for k in self.terminals:
                # Sulphur from source to terminal
                _lhs = sum([self.sulfur[i, sid] * flow[i, k, sid] for i, k2 in self.s2t if k2 == k])
                # For all pools with inflow to sink k
                for j, k2 in self.p2t:
                    if k2 == k:
                        # For all sources with inflow to pool j
                        _lhs += flow[j, k, sid] * sum([self.sulfur[i, sid] * prop[i, j, sid]
                                                       for i, j2 in self.s2p if j2 == j])

                m.addConstr(_lhs - surplus_sulfur[k, sid] <= self.max_sulfur[k] * flow.sum('*', k, sid))

        #####################################
        #         Define objective          #
        #####################################
        # First-stage profits
        fs_revenue = sum([self.fc_sources[i] * z_s[i] for i in self.sources])
        fs_revenue += sum([self.fc_pools[j] * z_p[j] for j in self.pools])
        fs_revenue += sum([self.fc_terminals[k] * z_t[k] for k in self.terminals])
        fs_revenue += sum([self.fc_connections[conn] * z_e[conn] for conn in self.connections])

        # Second-stage profits
        scen_revenue = 0
        for sid, prob in zip(_scenario_ids, _probs):
            # Terminal revenues
            _revenue = sum([self.vp_terminals[k] * flow.sum('*', k, sid) for k in self.terminals])
            # Source to terminal costs
            _cost = sum([self.vc_sources[i] * flow[i, k, sid] for i, k in self.s2t])
            # Source to pool costs
            _cost += sum([self.vc_sources[i] * prop[i, j, sid] * flow.sum(j, '*', sid) for i, j in self.s2p])
            # Overage costs
            _cost += sum([self.penalty_sulfur[k] * surplus_sulfur[k, sid] for k in self.terminals])
            _cost += sum([self.penalty_product[k] * surplus_product[k, sid] for k in self.terminals])

            scen_revenue += prob * (_revenue + _cost)

        m.setObjective(fs_revenue + scen_revenue, gp.GRB.MAXIMIZE)
        m.update()

        return m, fs_vars_dict

    def _make_extensive_model(self):
        return self._make_surrogate_scenario_model(scenario_id=None)

    def _make_second_stage_model(self, fs_sol, scenario_id):
        assert 0 <= scenario_id < self.num_scenarios

        m = gp.Model('SecondStagePoolingProblem')

        #####################################
        #         Define variables          #
        #####################################
        # Flow variables
        _flow_names = self.s2t[:]
        _flow_names.extend(self.p2t)
        flow = m.addVars(_flow_names, name='flow', lb=0)

        # Proportion of total flow at pool j from source i under each scenario
        prop = m.addVars(self.s2p, name='prop', lb=0)

        # To ensure relatively complete recourse
        surplus_sulfur = m.addVars(self.terminals, name='surplus_sulfur', lb=0)
        surplus_product = m.addVars(self.terminals, name='surplus_product', lb=0)

        #######################################
        #          Define constraints         #
        #######################################
        # Define second stage constraints
        # Flow capacity constraints based on first-stage decision
        m.addConstrs((flow[i, k] <= self.s2t_cap[i, k] * fs_sol['z_e'][i, k] for i, k in self.s2t))
        m.addConstrs((flow[j, k] <= self.p2t_cap[j, k] * fs_sol['z_e'][j, k] for j, k in self.p2t))
        m.addConstrs((prop[i, j] * flow.sum(j, '*') <= self.s2p_cap[i, j] * fs_sol['z_e'][i, j]
                      for i, j in self.s2p))

        # Sum of fractions of inflow to a pool equals 1 if the pool is selected.
        # 0 otherwise.
        m.addConstrs((prop.sum('*', j) == fs_sol['z_p'][j] for j in self.pools))

        # Inflow at terminal must be less than demand + surplus product
        m.addConstrs((flow.sum('*', k) <= self.demand[k, scenario_id] + surplus_product[k]
                      for k in self.terminals))

        for k in self.terminals:
            # Sulphur from source to terminal
            _lhs = sum([self.sulfur[i, scenario_id] * flow[i, k] for i, k2 in self.s2t if k2 == k])
            # For all pools with inflow to sink k
            for j, k2 in self.p2t:
                if k2 == k:
                    # For all sources with inflow to pool j
                    _lhs += flow[j, k] * sum([self.sulfur[i, scenario_id] * prop[i, j]
                                              for i, j2 in self.s2p if j2 == j])

            m.addConstr(_lhs - surplus_sulfur[k] <= self.max_sulfur[k] * flow.sum('*', k))

        #####################################
        #         Define objective          #
        #####################################
        # Terminal revenues
        _revenue = sum([self.vp_terminals[k] * flow.sum('*', k) for k in self.terminals])
        # Source to terminal costs
        _cost = sum([self.vc_sources[i] * flow[i, k] for i, k in self.s2t])
        # Source to pool costs
        _cost += sum([self.vc_sources[i] * prop[i, j] * flow.sum(j, '*') for i, j in self.s2p])
        # Overage costs
        _cost += sum([self.penalty_sulfur[k] * surplus_sulfur[k] for k in self.terminals])
        _cost += sum([self.penalty_product[k] * surplus_product[k] for k in self.terminals])
        scen_revenue = (_revenue + _cost)

        m.setObjective(scen_revenue, gp.GRB.MAXIMIZE)
        m.update()

        return m

    def get_scenarios(self, n_scenarios, test_set):
        pass

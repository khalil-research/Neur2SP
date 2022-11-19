import time

import gurobipy as gp
import numpy as np
import torch

from .approximator import Approximator


class PoolingProblemApproximator(Approximator):
    def __init__(self, pool_problem, model, model_type, mipper):
        self.two_sp = pool_problem
        self.model = model
        self.model_type = model_type
        self.mipper = mipper

        self.inst = self.two_sp.inst

    def get_master_mip(self):
        mip = gp.Model('mipQ')
        z_s = mip.addVars(self.inst['sources'], vtype="B", name="z_s")
        z_p = mip.addVars(self.inst['pools'], vtype="B", name="z_p")
        z_t = mip.addVars(self.inst['terminals'], vtype="B", name="z_t")
        z_e = mip.addVars(self.inst['connections'], vtype="B", name="z_e")

        # Using an edge selects nodes
        for i, j in self.inst['s2p']:
            mip.addConstr(z_e[i, j] <= z_s[i])
            mip.addConstr(z_e[i, j] <= z_p[j])

        for i, k in self.inst['s2t']:
            mip.addConstr(z_e[i, k] <= z_s[i])
            mip.addConstr(z_e[i, k] <= z_t[k])

        for j, k in self.inst['p2t']:
            mip.addConstr(z_e[j, k] <= z_p[j])
            mip.addConstr(z_e[j, k] <= z_t[k])

        # Using node enforces at least one outgoing edge to be selected
        mip.addConstrs((z_s[i] <= z_e.sum(i, '*') for i in self.inst['sources']))
        mip.addConstrs((z_p[j] <= z_e.sum(j, '*') for j in self.inst['pools']))

        # First-stage profits
        fs_revenue = sum([self.inst['fixed_cost_sources'][i] * z_s[i] for i in self.inst['sources']])
        fs_revenue += sum([self.inst['fixed_cost_pools'][j] * z_p[j] for j in self.inst['pools']])
        fs_revenue += sum([self.inst['fixed_cost_terminals'][k] * z_t[k] for k in self.inst['terminals']])
        fs_revenue += sum([self.inst['fixed_cost_edges'][conn] * z_e[conn] for conn in self.inst['connections']])
        mip.setObjective(fs_revenue, gp.GRB.MAXIMIZE)
        mip.update()

        return mip

    def get_scenario_embedding(self, n_scenarios, test_set="0"):
        """Ignore n_scenarios as we already have provided the scenarios via sampler"""
        if test_set != "0":
            raise ValueError('Test set not defined for PP')

        if self.model_type == 'nn_p' or self.model_type == 'lr':
            scenario_embedding = [
                [self.two_sp.sulfur['D', scen_id],
                 self.two_sp.demand['X', scen_id],
                 self.two_sp.demand['Y', scen_id]]
                for scen_id, _ in enumerate(self.two_sp.probs)]

        # Get embedding if NN-E model.
        elif self.model_type == 'nn_e':
            scenario_embedding = [
                [self.two_sp.sulfur['D', scen_id],
                 self.two_sp.demand['X', scen_id],
                 self.two_sp.demand['Y', scen_id],
                 prob]
                for scen_id, prob in enumerate(self.two_sp.probs)]

            x_scen = np.array(scenario_embedding)
            x_scen = torch.from_numpy(x_scen).float()
            x_scen = torch.reshape(x_scen, (1, x_scen.shape[0], x_scen.shape[1]))
            scenario_embedding = self.model.embed_scenarios(x_scen).detach().numpy().reshape(-1)

        else:
            raise ValueError('Invalid model type!')

        return scenario_embedding

    def approximate(self, n_scenarios, gap=0.02, time_limit=600, threads=1, log_dir=None, test_set="0"):
        def callback(model, where):
            """ Callback function to log time, bounds, and first stage sol. """
            if where == gp.GRB.Callback.MIPSOL:
                solving_results['time'].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                solving_results['primal'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                solving_results['dual'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                solving_results['incumbent'].append(model.cbGetSolution(first_stage_vars))

        total_time = time.time()

        master_mip = self.get_master_mip()
        first_stage_vars = self.get_first_stage_variables(master_mip)
        scenario_embedding = self.get_scenario_embedding(n_scenarios)

        approximator_mip = self.mipper(master_mip,
                                       first_stage_vars,
                                       self.model,
                                       scenario_embedding,
                                       scenario_probs=self.two_sp.probs).get_mip()
        solving_results = {'time': [], 'primal': [], 'dual': [], 'incumbent': []}

        approximator_mip.setParam('TimeLimit', time_limit)
        approximator_mip.setParam('MipGap', gap)
        approximator_mip.setParam('Threads', threads)
        approximator_mip.setParam('LogFile', log_dir)
        approximator_mip.optimize(callback)
        total_time = time.time() - total_time

        solving_results['incumbent'] = [dict(x) for x in solving_results['incumbent']]
        # Calculate the results here
        try:
            first_stage_sol = self.get_first_stage_solution(approximator_mip)
        except: 
            first_stage_sol = None
        results = {
            'time': total_time,
            'predicted_obj': approximator_mip.objVal,
            'sol': first_stage_sol,
            'solving_results': solving_results,
            'solving_time': approximator_mip.runtime
        }

        return results

    def get_first_stage_solution(self, model):
        x_sol = {'z_p': {}, 'z_s': {}, 'z_t': {}, 'z_e': {}}
        for var in model.getVars():
            if 'z_p' in var.varName or 'z_s' in var.varName or 'z_t' in var.varName:
                x_sol[var.varName[:3]][var.varName[4]] = np.floor(var.x + 0.5)
            elif 'z_e' in var.varName:
                x_sol['z_e'][var.varName[4], var.varName[6]] = np.floor(var.x + 0.5)

        return x_sol

    @staticmethod
    def get_first_stage_variables(mip):
        return {0: mip.getVarByName('z_s[A]'),
                1: mip.getVarByName('z_s[B]'),
                2: mip.getVarByName('z_s[C]'),
                3: mip.getVarByName('z_s[D]'),
                4: mip.getVarByName('z_p[P]'),
                5: mip.getVarByName('z_t[X]'),
                6: mip.getVarByName('z_t[Y]'),
                7: mip.getVarByName('z_e[D,X]'),
                8: mip.getVarByName('z_e[C,X]'),
                9: mip.getVarByName('z_e[C,Y]'),
                10: mip.getVarByName('z_e[P,X]'),
                11: mip.getVarByName('z_e[P,Y]'),
                12: mip.getVarByName('z_e[D,P]'),
                13: mip.getVarByName('z_e[A,P]'),
                14: mip.getVarByName('z_e[B,P]'),
                15: mip.getVarByName('z_e[C,P]')}

import time

import gurobipy as gp
import numpy as np
import torch

from .approximator import Approximator


class InvestmentProblemApproximator(Approximator):
    def __init__(self, inv_problem, model, model_type, mipper):
        self.two_sp = inv_problem
        self.model = model
        self.model_type = model_type
        self.mipper = mipper

        self.inst = self.two_sp.inst

    def get_master_mip(self):
        """Initialize MIP model with first-stage variables and constraints """
        # initialize first stage variables
        mip = gp.Model('mipQ')
        x_in = mip.addVars(2, vtype=self.inst['first_stage_vtype'], lb=0, ub=5,
                           obj=self.inst['c_fs'], name='x_in')
        mip.update()

        return mip

    def get_scenario_embedding(self, n_scenarios, test_set="0"):
        """ Gets the set of scenarios.  """
        if test_set == "0":
            scenario_embedding = self.two_sp.scenarios
        else:
            raise ValueError('Test set not defined for INVP')

        # Get embedding if NN-E model.
        if self.model_type == 'nn_e':
            x_scen = torch.from_numpy(scenario_embedding).float()
            x_scen = torch.reshape(x_scen, (1, x_scen.shape[0], x_scen.shape[1]))
            scenario_embedding = self.model.embed_scenarios(x_scen).detach().numpy().reshape(-1)

        return scenario_embedding

    def approximate(self,
                    n_scenarios=4,
                    gap=0.02,
                    time_limit=600,
                    threads=1,
                    log_dir=None,
                    test_set="0"):
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
        scenario_embedding = self.get_scenario_embedding(test_set)

        approximator_mip = self.mipper(master_mip,
                                       first_stage_vars,
                                       self.model,
                                       scenario_embedding).get_mip()
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
        x_sol = np.zeros(2)
        for var in model.getVars():
            if "x_in" in var.varName:
                idx = int(var.varName.split('[')[-1][:-1])
                x_sol[idx] = np.abs(var.x)

        return x_sol

    @staticmethod
    def get_first_stage_variables(mip):
        return {0: mip.getVarByName('x_in[0]'),
                1: mip.getVarByName('x_in[1]')}

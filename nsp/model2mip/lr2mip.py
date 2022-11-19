import gurobipy as gp
import numpy as np


class LR2MIP(object):
    """
        Take a learned neural representation of the Q(x, k) and fuse 
        it into the parent representation

    Params
    ------
    """

    def __init__(self,
                 first_stage_mip,
                 first_stage_vars,
                 lr,
                 scenario_representations,
                 scenario_probs=None):

        self.gp_model = first_stage_mip
        self.gp_vars = first_stage_vars
        self.lr = lr
        self.scenario_representations = scenario_representations
        self.scenario_probs = scenario_probs

        # if no scenario probabilities are given, assume equal probability
        self.n_scenarios = len(scenario_representations)
        if self.scenario_probs is None:
            self.scenario_probs = np.ones(self.n_scenarios) / self.n_scenarios

        self.Q_var_lst = []
        self.scenario_index = 0

    def get_mip(self):
        """ Gets MIP embedding of NN. """
        for scenario_prob, scenario in zip(self.scenario_probs,
                                           self.scenario_representations):
            Q_var = self._add_scenario_to_mip(scenario)
            Q_var.setAttr("obj", scenario_prob)
            self.Q_var_lst.append(Q_var)
            self.scenario_index += 1

        self.gp_model.update()

        return self.gp_model

    def _add_scenario_to_mip(self, scenario):
        """
        Take a learned neural representation of the Q(x, k) and fuse 
        it into the parent representation.
        """

        nVar = len(self.gp_vars.keys())

        # Extract learned weights and bias from the neural network
        wt = self.lr.coef_
        b = self.lr.intercept_

        v_name = f"Q_s_{self.scenario_index}"
        Q_var = self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name=v_name)

        _eq = 0
        for i in range(wt.shape[0]):
            if i < nVar:
                _eq += wt[i] * self.gp_vars[i]
            else:
                _eq += wt[i] * scenario[i - nVar]
        _eq += b
        # Equality constraint works for both maximization and minimization problems
        self.gp_model.addConstr(Q_var == _eq, name=f"mult_{v_name}")
        self.gp_model.update()

        return Q_var

import gurobipy as gp
import numpy as np


class Net2MIPPerScenario(object):
    """
        Take a learned neural representation of the Q(x, k) and fuse
        it into the parent representation

    Params
    ------

    """

    def __init__(self,
                 first_stage_mip,
                 first_stage_vars,
                 network,
                 scenario_representations,
                 scenario_probs=None,
                 M_plus=1e5,
                 M_minus=1e5):

        self.gp_model = first_stage_mip
        self.gp_vars = first_stage_vars
        self.network = network
        self.scenario_representations = scenario_representations
        self.scenario_probs = scenario_probs
        self.M_plus = M_plus
        self.M_minus = M_minus

        # if no scenario probabilities are given, assume equal probability
        self.n_scenarios = len(self.scenario_representations)
        if self.scenario_probs is None:  # 'sulfur' in scenario_representations:
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
        W, B = [], []
        for name, param in self.network.named_parameters():
            if 'weight' in name:
                W.append(param.cpu().detach().numpy())
            if 'bias' in name:
                B.append(param.cpu().detach().numpy())

        XX = []
        for k, (wt, b) in enumerate(zip(W, B)):

            outSz, inpSz = wt.shape

            X, S, Z = [], [], []
            for j in range(outSz):
                x_name = f'x_{self.scenario_index}_{k + 1}_{j}'
                s_name = f's_{self.scenario_index}_{k + 1}_{j}'
                z_name = f'z_{self.scenario_index}_{k + 1}_{j}'

                if k < len(W) - 1:
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=x_name))
                    S.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=s_name))
                    Z.append(self.gp_model.addVar(vtype=gp.GRB.BINARY, name=z_name))
                else:
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name=x_name))

                # W out-by-in
                # x in-by-1
                # _eq = W . x
                _eq = 0
                for i in range(inpSz):
                    # First layer weights are partially multiplied by gp.var and features
                    if k == 0:
                        # Multiply gp vars
                        if i < nVar:
                            _eq += wt[j][i] * self.gp_vars[i]
                        else:
                            _eq += wt[j][i] * scenario[i - nVar]
                    else:
                        _eq += wt[j][i] * XX[-1][i]

                # Add bias
                _eq += b[j]

                # Add constraint for each output neuron
                if k < len(W) - 1:
                    self.gp_model.addConstr(_eq == X[-1] - S[-1], name=f"mult_{x_name}__{s_name}")
                    self.gp_model.addConstr(X[-1] <= self.M_plus * (1 - Z[-1]), name=f"bm_{x_name}")
                    self.gp_model.addConstr(S[-1] <= self.M_minus * (Z[-1]), name=f"bm_{s_name}")

                else:
                    self.gp_model.addConstr(_eq == X[-1], name=f"mult_out_{x_name}__{s_name}")

                # Save current layers gurobi vars
                XX.append(X)

        self.gp_model.update()
        Q_var = XX[-1][-1]

        return Q_var


class Net2MIPExpected(object):
    """
        Take a learned neural representation of the sum_k Q(x, k) and fuse 
        it into the parent representation

    Params
    ------

    """

    def __init__(self,
                 first_stage_mip,
                 first_stage_vars,
                 network,
                 scenario_embedding,
                 scenario_probs=None,
                 M_plus=1e5,
                 M_minus=1e5):

        self.gp_model = first_stage_mip
        self.gp_vars = first_stage_vars
        self.network = network
        self.scenario_embedding = scenario_embedding
        self.M_plus = M_plus
        self.M_minus = M_minus

        self.Q_var_lst = []
        self.scenario_index = 0

        self.extract_weights()

    def get_mip(self):
        """ Gets MIP embedding of NN. """
        self._create_mip()
        return self.gp_model

    def extract_weights(self):
        """ Extract weights from model.  """
        # Extract learned weights and bias from the neural network

        self.W = [self.network.relu_input.weight.cpu().detach().numpy(),
                  self.network.relu_output.weight.cpu().detach().numpy()]
        self.B = [self.network.relu_input.bias.cpu().detach().numpy(),
                  self.network.relu_output.bias.cpu().detach().numpy()]

    def _create_mip(self):
        """
        Take a learned neural representation of the Q(x, k) and fuse 
        it into the parent representation.
        """

        nVar = len(self.gp_vars.keys())

        XX = []
        for k, (wt, b) in enumerate(zip(self.W, self.B)):

            outSz, inpSz = wt.shape

            X, S, Z = [], [], []
            for j in range(outSz):
                x_name = f'x_{self.scenario_index}_{k + 1}_{j}'
                s_name = f's_{self.scenario_index}_{k + 1}_{j}'
                z_name = f'z_{self.scenario_index}_{k + 1}_{j}'

                if k < len(self.W) - 1:
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=x_name))
                    S.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=s_name))
                    Z.append(self.gp_model.addVar(vtype=gp.GRB.BINARY, name=z_name))
                else:
                    X.append(self.gp_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, obj=1, name=x_name))

                # W out-by-in
                # x in-by-1
                # _eq = W . x 
                _eq = 0
                for i in range(inpSz):
                    # First layer weights are partially multiplied by gp.var and features
                    if k == 0:
                        # Multiply gp vars
                        if i < nVar:
                            _eq += wt[j][i] * self.gp_vars[i]
                        else:
                            _eq += wt[j][i] * self.scenario_embedding[i - nVar]
                    else:
                        _eq += wt[j][i] * XX[-1][i]

                # Add bias
                _eq += b[j]

                # Add constraint for each output neuron 
                if k < len(self.W) - 1:
                    self.gp_model.addConstr(_eq == X[-1] - S[-1], name=f"mult_{x_name}__{s_name}")
                    self.gp_model.addConstr(X[-1] <= self.M_plus * (1 - Z[-1]), name=f"bm_{x_name}")
                    self.gp_model.addConstr(S[-1] <= self.M_minus * (Z[-1]), name=f"bm_{s_name}")

                else:
                    self.gp_model.addConstr(_eq == X[-1], name=f"mult_out_{x_name}__{s_name}")

                # Save current layers gurobi vars
                XX.append(X)

        self.gp_model.update()
        Q_var = XX[-1][-1]

        return Q_var

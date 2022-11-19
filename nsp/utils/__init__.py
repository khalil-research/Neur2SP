import pickle

import numpy as np
import torch
import torch.nn as nn

import nsp.params as params
from .cflp import get_path as get_path_cflp
from .consts import DataManagerModes
from .consts import LearningModelTypes
from .consts import LossPenaltyTypes
from .ip import get_path as get_path_ip
from .pp import get_path as get_path_pp
from .sslp import get_path as get_path_sslp


def factory_get_path(args):
    if 'cflp' in args.problem:
        from .cflp import get_path
        return get_path

    elif 'ip' in args.problem:
        from .ip import get_path
        return get_path

    elif 'sslp' in args.problem:
        from .sslp import get_path
        return get_path

    elif 'pp' in args.problem:
        from .pp import get_path
        return get_path

    else:
        raise Exception(f"nsp.utils not defined for problem class {args.problem}")


def factory_sampler(args, cfg=None):
    if 'ip' in args.problem[:2]:
        from .ip import Sampler
        try:
            return Sampler(samples_per_dist=int(np.sqrt(args.n_scenarios)))
        except:
            return Sampler(samples_per_dist=int(np.sqrt(args.n_scenarios_for_eval)))

    elif 'cflp' in args.problem[:4]:
        return None
    elif 'sslp' in args.problem[:4]:
        return None
    elif 'pp' in args.problem[:2]:
        from .pp import Sampler
        try:
            return Sampler(cfg=cfg,
                           samples_per_dist=int(np.cbrt(args.n_scenarios)))
        except:
            return Sampler(cfg=cfg,
                           samples_per_dist=int(np.cbrt(args.n_scenarios_for_eval)))


def load_instance(args, cfg):
    """ Loads instance file from the cfg. """
    get_path = factory_get_path(args)
    inst_fp = get_path(cfg.data_path, cfg, "inst")
    inst = pickle.load(open(inst_fp, 'rb'))

    if 'ip' in args.problem:
        insts = inst
        inst = insts[0]
        inst.update(insts[-1])

    elif 'pp' in args.problem:
        inst = inst[0]

    return inst


def get_scenario_sets_for_problem(problem):
    """ Gets all scenario sets for each problem.  """
    if 'cflp' in problem:
        return 100, 500, 1000

    elif 'sslp' in problem:
        if problem == 'sslp_5_25':
            return 50, 100
        elif problem == 'sslp_10_50':
            return 50, 100, 500, 1000, 2000
        elif problem == 'sslp_15_45':
            return 5, 10, 15

    elif 'ip' in problem:
        return 4, 9, 36, 121, 441, 1681, 10000

    elif 'pp' in problem:
        return 125, 216, 343, 512, 729, 1000

    else:
        raise Exception("Not a valid problem class")


def get_problem_from_param():
    """ Gets all problems form params"""

    def is_problem(p):
        if "ip" in p or "sslp" in p or "cflp" in p or "pp" in p:
            return True
        return False

    problems = list(filter(lambda x: is_problem(x), dir(params)))
    return problems


def get_large_scenario_size(problem):
    if 'cflp' in problem or 'sslp' in problem:
        return 10000

    elif 'ip' in problem:
        # 140^2
        return 19600

    elif 'pp' in problem:
        # 22^3
        return 10648

    else:
        raise Exception("Not a valid problem class")


def get_test_sets(problem, n_scenarios):
    """ Gets all test sets for each problem.  """
    if 'cflp' in problem:
        test_sets = list(range(10))
        test_sets = list(map(lambda x: str(x), test_sets))
        return test_sets

    elif 'sslp' in problem:
        test_sets = list(range(10))
        test_sets = list(map(lambda x: str(x), test_sets))
        test_sets = ['siplib'] + test_sets
        return test_sets

    elif 'ip' in problem:
        return ['0']

    elif 'pp' in problem:
        return ['0']

    else:
        raise Exception("Not a valid problem class")


def get_results_for_problem(args, problem, n_scenarios):
    """ Imports correct get_path file based on args.problem.  """
    cfg = getattr(params, problem)

    if 'cflp' in problem:
        fp = get_path_cflp(args.data_dir, cfg, f"combined_results_s{n_scenarios}")
    elif 'ip' in problem:
        fp = get_path_ip(args.data_dir, cfg, f"combined_results_s{n_scenarios}")
    elif 'sslp' in problem:
        fp = get_path_sslp(args.data_dir, cfg, f"combined_results_s{n_scenarios}")
    elif 'pp' in problem:
        fp = get_path_pp(args.data_dir, cfg, f"combined_results_s{n_scenarios}")
    else:
        raise Exception(f"nsp.utils not defined for problem class {problem}")

    with open(fp, 'rb') as p:
        problem_results = pickle.load(p)

    return problem_results


class LossFunction:
    def __init__(self, criterion, weights=None):
        self.loss_fn = getattr(nn, criterion)()
        self.weights = {'lasso': 1, 'ridge': 1} if weights is None else weights
        self.penalty_fn = self.elastic_penalty

    def get_loss(self, model, y_pred, y):
        loss = self.loss_fn(y_pred, y)
        loss += self.penalty_fn(model)
        return loss

    def lasso_penalty(self, model):
        penalty = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                penalty += torch.norm(param, 1)

        return self.weights['lasso'] * penalty

    def ridge_penalty(self, model):
        penalty = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                penalty += torch.pow(torch.norm(param, 2), 2)

        return self.weights['ridge'] * penalty

    def elastic_penalty(self, model):
        l1_loss = self.lasso_penalty(model)
        l2_loss = self.ridge_penalty(model)
        return l1_loss + l2_loss

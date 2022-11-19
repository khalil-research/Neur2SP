import argparse
import hashlib

import numpy as np

problem_types = {
    "cflp": ["cflp_10_10", "cflp_25_25", "cflp_50_50"],
    "sslp": ["sslp_5_25", "sslp_10_50", "sslp_15_45"],
    "ip": ["ip_b_H", "ip_i_H", "ip_b_E", "ip_i_E"],
    "pp": ["pp"],
}

class ContinuousValueSampler(object):
    """ A class to sample uniformly at random in the range of [lb,ub].
        Additionally includes a probability of sampling zero if needed.  """

    def __init__(self, lb, ub, prob_zero=0.0):
        self.lb = lb
        self.ub = ub
        self.prob_zero = prob_zero

    def sample(self):
        if np.random.rand() < self.prob_zero:
            return 0
        return np.round(np.random.uniform(self.lb, self.ub), 5)


class DiscreteSampler(object):
    """ A class to sample uniformly at random in the range of [lb,ub]. """
    def __init__(self, choices):
        self.choices = choices

    def sample(self):
        return np.random.choice(self.choices)


def get_nn_p_config():
    """ Defines params space for nn_single_cut. """
    LR_LB, LR_UB = 1e-5, 1e-1
    L1_LB, L1_UB = 1e-5, 1e-1
    L2_LB, L2_UB = 1e-5, 1e-1
    L1_ZERO, L2_ZERO = 0.25, 0.25

    config = {
        # general parameters
        "batch_size": DiscreteSampler([16, 32, 64, 128]),
        "lr": ContinuousValueSampler(LR_LB, LR_UB),
        "wt_lasso": ContinuousValueSampler(L1_LB, L1_UB, L1_ZERO),
        "wt_ridge": ContinuousValueSampler(L2_LB, L2_UB, L2_ZERO),
        "n_epochs": DiscreteSampler([1000]),
        "loss_fn": DiscreteSampler(["MSELoss"]),
        "dropout": ContinuousValueSampler(0.0, 0.5),
        "optimizer": DiscreteSampler(['Adam', 'Adagrad', 'RMSprop']),

        # NN-P specific parameters
        "hidden_dims": DiscreteSampler([32, 64])
    }

    return config


def get_nn_e_config():
    """ Defines params space for NN-E. """
    LR_LB, LR_UB = 1e-5, 1e-1
    L1_LB, L1_UB = 1e-5, 1e-1
    L2_LB, L2_UB = 1e-5, 1e-1
    L1_ZERO, L2_ZERO = 0.25, 0.25

    config = {
        # general parameters
        "batch_size": DiscreteSampler([16, 32, 64, 128]),
        "lr": ContinuousValueSampler(LR_LB, LR_UB),
        "wt_lasso": ContinuousValueSampler(L1_LB, L1_UB, L1_ZERO),
        "wt_ridge": ContinuousValueSampler(L2_LB, L2_UB, L2_ZERO),
        "n_epochs": DiscreteSampler([2000]),
        "loss_fn": DiscreteSampler(["MSELoss"]),
        "dropout": ContinuousValueSampler(0.0, 0.5),
        "optimizer": DiscreteSampler(['Adam', 'Adagrad', 'RMSprop']),

        # single-cut specific parameters
        "embed_hidden_dim": DiscreteSampler([64, 128, 256, 512]),
        "embed_dim1": DiscreteSampler([16, 32, 64, 128]),
        "embed_dim2": DiscreteSampler([8, 16, 32, 64]),
        "relu_hidden_dim": DiscreteSampler([64, 128, 256, 512]),
        "agg_type": DiscreteSampler(["mean"]),
    }

    return config


def get_config(model_type):
    """ Gets the config for the given model_type. """
    if model_type == "nn_p":
        return get_nn_p_config()
    elif model_type == "nn_e":
        return get_nn_e_config()
    else:
        raise Exception(f"Config not defined for model_type [{model_type}]")


def sample_config(problem, model_type, config):
    """ Samples a confiuration for NN-E. """
    config_cmd = f"python -m nsp.scripts.train_model --use_wandb 0 --problem {problem} --model_type {model_type}"
    for param_name, param_sampler in config.items():
        param_val = param_sampler.sample()
        config_cmd += f" --{param_name} {param_val}"

    return config_cmd


def main(args):
    cmds = []

    for problem in args.problems:
        for ptypes in problem_types[problem]:
            for model_type in args.model_type:
                config = get_config(model_type)
                for i in range(args.n_configs):
                    p_hash = int(hashlib.md5(b'{ptypes}').hexdigest(), 16)
                    np.random.seed((args.seed + i + p_hash) % (2 ** 32 - 1))
                    cmds.append(sample_config(ptypes, model_type, config))

    # write to text file
    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmds[:-1]):
        textfile.write(f"{i + args.start_idx} {cmd}\n")
    textfile.write(f"{i + 2} {cmds[-1]}")
    textfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a list of configs to run for random search.')
    parser.add_argument('--problems', type=str, nargs='+', default=["cflp", "sslp", "ip", "pp"])
    parser.add_argument('--model_type', type=str, default=["nn_p", "nn_e"])
    parser.add_argument('--n_configs', type=int, default=100)
    parser.add_argument('--file_name', type=str, default='table.dat')
    parser.add_argument('--start_idx', type=int, default=1)
    parser.add_argument('--use_problem_for_rng', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    main(args)

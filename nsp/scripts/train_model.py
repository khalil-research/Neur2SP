import argparse
import pickle

import numpy as np
import torch

import nsp.params as params
from nsp.models import factory_learning_model
from nsp.utils import factory_get_path
from nsp.utils import load_instance


def factory_scenario_padder(args, split, max_scenarios):
    x_scen, x_n_scen = None, None

    def pad_scenario_ip(scenario):
        n_scenarios = len(scenario)
        padded_scenario = np.vstack((scenario,
                                     np.zeros((max_scenarios - n_scenarios, 2))))

        return padded_scenario, n_scenarios

    def pad_scenario_cflp_sslp(t, n_scenarios):
        pd = (0, 0, 0, n_scenarios - t.shape[0])
        p = torch.nn.ZeroPad2d(pd)
        return p(t)

    def pad_scenario_pp(scenario, n_scenarios):
        padded_scenarios = np.asarray(scenario)
        padded_scenarios = np.vstack((padded_scenarios,
                                      np.zeros((max_scenarios - n_scenarios, 4))))

        return padded_scenarios

    if 'cflp' in args.problem or 'sslp' in args.problem:
        max_n_scenarios = max(list(map(lambda x: len((x['demands'])), split)))
        x_scen = list(map(lambda x: np.array(x['demands']), split))
        x_scen = list(map(lambda x: torch.from_numpy(x).float(), x_scen))
        x_scen = list(map(lambda x: pad_scenario_cflp_sslp(x, max_n_scenarios), x_scen))
        x_scen = torch.stack(x_scen).numpy()
        x_n_scen = np.array(list(map(lambda x: len(x['demands']), split))).reshape(-1, 1)

    elif 'ip' in args.problem:
        scen_size_lst = [pad_scenario_ip(x['scenario']) for x in split]
        x_scen, x_n_scen = zip(*scen_size_lst)
        x_scen, x_n_scen = np.asarray(x_scen), np.asarray(x_n_scen)

    elif 'pp' in args.problem:
        x_scen = np.asarray([pad_scenario_pp(x['scenario'], x['n_scenarios'])
                             for x in split])
        x_n_scen = np.asarray([x['n_scenarios'] for x in split])

    return x_scen, x_n_scen


def load_split_expected(args, split, n_scenarios):
    """ For a split (returned from get_data_split), provides features/labels as tensors.  """

    # first-stage decision
    x_fs = None
    if 'ip' in args.problem:
        x_fs = np.array([x['x'] for x in split])
    elif 'pp' in args.problem:
        x_fs = np.array([[x['sol']['z_s']['A'],
                          x['sol']['z_s']['B'],
                          x['sol']['z_s']['C'],
                          x['sol']['z_s']['D'],
                          x['sol']['z_p']['P'],
                          x['sol']['z_t']['X'],
                          x['sol']['z_t']['Y'],
                          x['sol']['z_e']['D', 'X'],
                          x['sol']['z_e']['C', 'X'],
                          x['sol']['z_e']['C', 'Y'],
                          x['sol']['z_e']['P', 'X'],
                          x['sol']['z_e']['P', 'Y'],
                          x['sol']['z_e']['D', 'P'],
                          x['sol']['z_e']['A', 'P'],
                          x['sol']['z_e']['B', 'P'],
                          x['sol']['z_e']['C', 'P']]
                         for x in split])
    else:
        x_fs = np.array(list(map(lambda x: np.array(list(x['x'].values())), split)))
    x_scen, x_n_scen = factory_scenario_padder(args, split, n_scenarios)
    y = np.array(list(map(lambda x: x['obj_mean'], split))).reshape(-1, 1)

    return x_fs, x_scen, x_n_scen, y


def load_data_expected(args, cfg, n_scenarios):
    """ Loads training and validation data for NN-E case """
    ml_data_fp = get_path(cfg.data_path, cfg, ptype="ml_data_e")
    with open(ml_data_fp, 'rb') as p:
        ml_data = pickle.load(p)

    x_fs_tr, x_scen_tr, x_n_scen_tr, y_tr = load_split_expected(args,
                                                                  ml_data['tr_data'],
                                                                  n_scenarios)
    x_fs_val, x_scen_val, x_n_scen_val, y_val = load_split_expected(args,
                                                                      ml_data['val_data'],
                                                                      n_scenarios)

    return {'x_fs_tr': x_fs_tr, 'x_scen_tr': x_scen_tr, 'x_n_scen_tr': x_n_scen_tr, 'y_tr': y_tr,
            'x_fs_val': x_fs_val, 'x_scen_val': x_scen_val, 'x_n_scen_val': x_n_scen_val, 'y_val': y_val}


def load_split_per_scenario(split):
    x = np.array(list(map(lambda x: x['features'], split)))
    y = np.array(list(map(lambda x: x['obj'], split)))

    return x, y


def load_data_per_scenario(cfg):
    """ Loads training and validation data for the NN-P case """
    ml_data_fp = get_path(cfg.data_path, cfg, ptype="ml_data_p")
    with open(ml_data_fp, 'rb') as p:
        ml_data = pickle.load(p)

    x_tr, y_tr = load_split_per_scenario(ml_data['tr_data'])
    x_val, y_val = load_split_per_scenario(ml_data['val_data'])

    return {'x_tr': x_tr, 'y_tr': y_tr, 'x_val': x_val, 'y_val': y_val}


def factory_load_data(args, cfg, n_scenarios):
    """ Factory to load data for the single or a NN-P case """
    if args.model_type == 'nn_e':
        return load_data_expected(args, cfg, n_scenarios)
    else:
        return load_data_per_scenario(cfg)


def process_data(args, inst, data):
    """ Problem specific data processing. """
    # if sslp reduce magnitude of labels for bad solutions
    if 'sslp' in args.problem:
        cap_val = 500
        data['y_tr'][data['y_tr'] > cap_val] = cap_val
        data['y_val'][data['y_val'] > cap_val] = cap_val

    elif 'cflp' in args.problem:
        pass


def main(args):
    print(f"TRAINING MODELS: problem = {args.problem}\n")

    torch.manual_seed(args.seed)

    global get_path
    get_path = factory_get_path(args)

    # load config and instance
    cfg = getattr(params, args.problem)
    inst = load_instance(args, cfg)

    # load data
    data = factory_load_data(args, cfg, cfg.n_max_scenarios_in_tr)
    process_data(args, inst, data)

    # initialize and train model
    model = factory_learning_model(args, inst)
    model.train(data)
    model.eval_learning_metrics()

    # save results
    if "nn" in args.model_type:
        results_fp = get_path(cfg.data_path, cfg, ptype=f"{args.model_type}_results", suffix=f"{model.run_name}.pkl")
        model_fp = get_path(cfg.data_path, cfg, ptype=f"{args.model_type}", suffix=f"{model.run_name}.pt")
    else:
        results_fp = get_path(cfg.data_path, cfg, ptype=f"{args.model_type}_results")
        model_fp = get_path(cfg.data_path, cfg, ptype=f"{args.model_type}")

    model.save_results(results_fp)
    model.save_model(model_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train linear regression and neural network on data from data manager.')

    parser.add_argument('--problem', type=str, default="cflp_10")
    parser.add_argument('--model_type', type=str, default='lr')

    # wandb parameters
    parser.add_argument('--use_wandb', type=int, default=0, help='Indicator to use weights and biases (wandb).')

    # General NN parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_fn', type=str, default='MSELoss')
    parser.add_argument('--wt_lasso', type=float, default=0)
    parser.add_argument('--wt_ridge', type=float, default=0)
    parser.add_argument('--log_freq', type=int, default=10, help='Frequency to evaluate model.')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of training epochs.')

    # NN-P parameters
    parser.add_argument('--hidden_dims', nargs="+", type=int, default=[64], help='List for hidden dimension.')

    # NN-E parameters
    parser.add_argument('--agg_type', type=str, default="mean",
                        help='Type of aggregation for scenario representations (mean/sum).')
    parser.add_argument('--embed_hidden_dim', type=int, default=256,
                        help='Dimension of first layer for scenario embedding (before sum).')
    parser.add_argument('--embed_dim1', type=int, default=64,
                        help='Dimension of intermediate embedding layer (before sum).')
    parser.add_argument('--embed_dim2', type=int, default=8, help='Dimension of final embedding layer (after sum).')
    parser.add_argument('--relu_hidden_dim', type=int, default=512, help='Dimension of last layer.')

    # Random seed 
    parser.add_argument('--seed', type=int, default=1234, help='Seed.')

    args = parser.parse_args()

    main(args)

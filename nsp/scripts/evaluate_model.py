import argparse
import pickle
import numpy as np

import torch

import nsp.params as params
from nsp.approximator import factory_approximator
from nsp.two_sp import factory_two_sp
from nsp.utils import LearningModelTypes
from nsp.utils import factory_get_path
from nsp.utils import factory_sampler
from nsp.utils import load_instance


def load_learning_model(args, cfg):
    """ Loads trained model """
    if "nn" in args.model_type:  # pytorch model
        model_fp = get_path(cfg.data_path, cfg, ptype=args.model_type, suffix=".pt")
        model = torch.load(model_fp, map_location=torch.device('cpu'))
    else:  # scikit-learn model
        model_fp = get_path(cfg.data_path, cfg, ptype=args.model_type, suffix=".pkl")
        with open(model_fp, 'rb') as p:
            model = pickle.load(p)

    return model


def main(args):
    print(f"EVALUATING MODELS: problem={args.problem}, model_type={args.model_type}, "
          f"n_scenarios={args.n_scenarios}, test_set={args.test_set}\n")
    problem_str = f"s{args.n_scenarios}_ts{args.test_set}"

    # get imports and configs
    global get_path
    get_path = factory_get_path(args)

    # load config
    cfg = getattr(params, args.problem)

    # load instance 
    inst = load_instance(args, cfg)

    # load sampler
    sampler = factory_sampler(args, cfg=cfg)

    two_sp = factory_two_sp(args.problem, inst, sampler=sampler)

    # load models
    trained_model = load_learning_model(args, cfg)

    # Get approximator
    ptype = f"grb_log_{args.model_type}_{problem_str}"
    log_dir = get_path(cfg.data_path, cfg, ptype=ptype, suffix='.log', as_str=True)
    approximator_mip = factory_approximator(args, two_sp, trained_model, args.model_type)

    # Evaluate first-stage decision obtained by solving approximator mip
    results = approximator_mip.approximate(
        n_scenarios=args.n_scenarios,
        gap=args.mip_gap,
        time_limit=args.time_limit,
        threads=args.mip_threads,
        log_dir=log_dir,
        test_set=args.test_set)

    if results['sol'] is not None:
        true_obj = two_sp.evaluate_first_stage_sol(results['sol'],
                                                   n_scenarios=args.n_scenarios,
                                                   verbose=0,
                                                   test_set=args.test_set,
                                                   n_procs=args.n_procs)
        results['true_obj'] = true_obj
    else:
        results['true_obj'] = np.nan

    # combine and save results
    fp_results = get_path(cfg.data_path, cfg, ptype=f"embedding_results_{args.model_type}_{problem_str}")

    with open(fp_results, 'wb') as p:
        pickle.dump(results, p)

    print(f"Results for {args.model_type}")
    print(f'    True:         {results["true_obj"]}')
    print(f'    Predicted:    {results["predicted_obj"]}')
    print(f'    Time:         {results["time"]}')
    print(f'    Sol:          {results["sol"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates linear regression, neural network, or scenario embedding neural network.')

    parser.add_argument('--problem', type=str, default='cflp_10_10')
    parser.add_argument('--n_scenarios', type=int, default=100)

    parser.add_argument('--model_type',
                        type=str,
                        choices=list([m.name for m in LearningModelTypes]),
                        default=LearningModelTypes.lr.name,
                        help='Type of model to evaluate.')

    # Optimization parameters
    parser.add_argument('--time_limit', type=int, default=3 * 3600, help='Time limit for solver.')
    parser.add_argument('--mip_gap', type=float, default=0.0001, help='Gap limit for solver.')
    parser.add_argument('--mip_threads', type=int, default=1, help='Number of threads for MIP solver.')
    parser.add_argument('--n_procs', type=int, default=1,
                        help='Number of processes for evaluting the first stage solution.')
    parser.add_argument('--test_set', type=str, default="0",
                        help='Evaluate a the test set (unseen scenarios). Only implemented for CFLP.')
    parser.add_argument('--hidden_dims', nargs="+", type=int, default=[64], help='List for hidden dimension.')
    args = parser.parse_args()
    main(args)

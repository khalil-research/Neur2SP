import argparse
import pickle as pkl
import shutil

import numpy as np

import nsp.params as params
from nsp.utils import factory_get_path


def parse_run_name(run_name):
    pass


def get_best_model(args):
    cfg = getattr(params, args.problem)
    get_path = factory_get_path(args)
    results_fp = get_path(cfg.data_path, cfg, ptype=f'{args.model_type}_results', suffix='.pkl')
    results_fp_prefix = str(results_fp.stem)
    model_suffix = '.pt' if 'nn' in args.model_type else '.pkl'
    model_fp = get_path(cfg.data_path, cfg, ptype=f'{args.model_type}',
                        suffix=model_suffix)

    # Find best result
    best_criterion, best_results_path = np.infty, None
    results_paths = [x for x in model_fp.parent.iterdir()
                     if results_fp_prefix in str(x.stem)]
    print(f'Checking {len(results_paths)} model files...')
    for rp in results_paths:
        rdict = pkl.load(open(rp, 'rb'))
        if best_criterion > rdict[args.criterion]:
            best_criterion = rdict[args.criterion]
            best_results_path = rp

    # Generate best model path
    parts = str(best_results_path.stem).split('_')
    parts.remove('results')
    best_model_path = "_".join(parts) + model_suffix
    best_model_path = best_results_path.parent.joinpath(best_model_path)

    print(f'Best model :{best_results_path}')
    print(f'Best {args.criterion}  :{best_criterion}')

    shutil.copy(str(best_results_path), str(results_fp))
    shutil.copy(str(best_model_path), str(model_fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='pp_5')
    parser.add_argument('--model_type', type=str, default='nn_p')
    parser.add_argument('--criterion', type=str, default='val_mae')
    args = parser.parse_args()

    get_best_model(args)

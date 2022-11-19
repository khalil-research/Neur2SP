import argparse
import pickle as pkl

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

    rs_results = {}
    rs_results['tr_mse'] = []
    rs_results['tr_mae'] = []
    rs_results['val_mse'] = []
    rs_results['val_mae'] = []

    for rp in results_paths:
        rdict = pkl.load(open(rp, 'rb'))

        rs_results['tr_mse'].append(rdict['tr_mse'])
        rs_results['tr_mae'].append(rdict['tr_mae'])
        rs_results['val_mse'].append(rdict['val_mse'])
        rs_results['val_mae'].append(rdict['val_mae'])

    print(f"Results for {args.problem}")
    print("  tr mae:", np.mean(rs_results['tr_mae']), np.std(rs_results['tr_mae']))
    print("  tr mse:", np.mean(rs_results['tr_mse']), np.std(rs_results['tr_mse']))
    print("  val mae:", np.mean(rs_results['val_mae']), np.std(rs_results['val_mae']))
    print("  val mse:", np.mean(rs_results['val_mse']), np.std(rs_results['val_mse']))

    results_fp = get_path(cfg.data_path, cfg, ptype=f'{args.model_type}_rs_config_results', suffix='.pkl')
    with open(results_fp, "wb") as p:
        pkl.dump(rs_results, p )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='cflp_10_10')
    parser.add_argument('--model_type', type=str, default='nn_single_cut')
    #parser.add_argument('--criterion', type=str, default='val_mae')
    args = parser.parse_args()

    get_best_model(args)
